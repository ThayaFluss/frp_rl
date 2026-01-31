"""
Model definitions for PPO with different architectures.

This module contains:
- GRUCore: GRU-based core for stateful temporal processing
- GRURepModel: GRU-based representation model (ObsEncoder + GRUCore)
- S5RepModel: S5-based representation model (ObsEncoder + S5Core)
- ActorCriticBase: Base class for actor-critic networks
- ActorCriticContinuous: Actor-critic for continuous action spaces
- ActorCriticDiscrete: Actor-critic for discrete action spaces
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import functools
from flax.linen.initializers import constant, orthogonal
from typing import Dict
import distrax
from .s5 import StackedEncoderModel
from .agalite import BatchedAGaLiTe


class GRUCore(nn.Module):
    """GRU Core for stateful temporal processing."""

    @functools.partial(
        nn.scan,
        variable_broadcast='params',
        in_axes=0,
        out_axes=0,
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        """Applies the GRU core to process temporal sequences."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state
        )
        features = rnn_state[0].shape[-1]
        new_rnn_state, y = nn.GRUCell(features)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """Initialize the hidden state for GRU Core."""
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size))


# ============================================================================
# New Refactored Architecture: RepModel = ObsEncoder + Core
# ============================================================================


class GRURepModel(nn.Module):
    """
    GRU-based encoder for ActorCritic networks.

    This encoder handles sequence encoding using GRU, independent of
    the action space type (continuous/discrete).
    """

    config: Dict

    @staticmethod
    def initialize_carry(batch_size, config):
        """
        Initialize GRU hidden state with shape compatible with S5.

        Args:
            batch_size: Number of environments
            config: Configuration dict (unused for GRU, kept for consistency)

        Returns:
            Initial GRU hidden state with shape (1, batch_size, hidden_size)
            Wrapped in a list for compatibility with S5's multi-layer structure
        """
        import jax.numpy as jnp
        hidden_size = 256  # GRU hidden size
        carry = GRUCore.initialize_carry(batch_size, hidden_size)
        # Add time dimension to match S5 format: (batch, hidden) -> (1, batch, hidden)
        # Return as single-element list to match S5's multi-layer structure
        return [jnp.expand_dims(carry, axis=0)]

    @nn.compact
    def __call__(self, hidden, obs, dones):
        """
        Encode observations using GRU.

        Args:
            hidden: GRU hidden state (list with single element for S5 compatibility)
            obs: Observations [seq_len, batch, obs_dim]
            dones: Done flags [seq_len, batch]

        Returns:
            (new_hidden, embedding): Updated hidden state (list format) and embeddings [seq_len, batch, 256]
        """
        if self.config.get("NO_RESET"):
            dones = jnp.zeros_like(dones)

        # Extract hidden state from list and remove time dimension
        # hidden is [(1, batch, hidden_size)]
        h = hidden[0].squeeze(0)  # (1, batch, hidden) -> (batch, hidden)

        # Encoder layers
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.leaky_relu(embedding)
        embedding = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.leaky_relu(embedding)

        # GRU processing
        rnn_in = (embedding, dones)
        h, embedding = GRUCore()(h, rnn_in)

        # Add time dimension back and wrap in list
        new_hidden = [jnp.expand_dims(h, axis=0)]  # (batch, hidden) -> [(1, batch, hidden)]

        return new_hidden, embedding


class S5RepModel(nn.Module):
    """
    S5-based encoder for ActorCritic networks.

    This encoder handles sequence encoding using S5 state space model,
    independent of the action space type (continuous/discrete).
    """

    config: Dict

    @staticmethod
    def initialize_carry(batch_size, config):
        """
        Initialize S5 hidden state.

        Args:
            batch_size: Number of environments
            config: Configuration dict containing S5_SSM_SIZE and S5_N_LAYERS

        Returns:
            Initial S5 hidden state
        """
        ssm_size = config["S5_SSM_SIZE"] // 2
        n_layers = config["S5_N_LAYERS"]
        return StackedEncoderModel.initialize_carry(batch_size, ssm_size, n_layers)

    @staticmethod
    def _create_ssm_init_fn(config):
        """
        Create SSM initialization function from config.

        Args:
            config: Configuration dict containing S5 parameters

        Returns:
            SSM initialization function for StackedEncoderModel
        """
        from .s5 import init_S5SSM, make_DPLR_HiPPO

        d_model = config["S5_D_MODEL"]
        ssm_size = config["S5_SSM_SIZE"]
        blocks = config["S5_BLOCKS"]
        block_size = int(ssm_size / blocks)

        Lambda, _, _, V, _ = make_DPLR_HiPPO(ssm_size)
        block_size = block_size // 2
        ssm_size = ssm_size // 2
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vinv = V.conj().T

        return init_S5SSM(
            H=d_model,
            P=ssm_size,
            Lambda_re_init=Lambda.real,
            Lambda_im_init=Lambda.imag,
            V=V,
            Vinv=Vinv,
            C_init="lecun_normal",
            discretization="zoh",
            dt_min=0.001,
            dt_max=0.1,
            conj_sym=True,
            clip_eigs=False,
            bidirectional=False
        )

    def setup(self):
        """Setup S5 encoder layers."""
        # Encoder layers
        self.rep_model_0 = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.rep_model_1 = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )

        # S5 state space model
        ssm_init_fn = self._create_ssm_init_fn(self.config)
        self.s5 = StackedEncoderModel(
            ssm=ssm_init_fn,
            d_model=self.config["S5_D_MODEL"],
            n_layers=self.config["S5_N_LAYERS"],
            activation=self.config["S5_ACTIVATION"],
            do_norm=self.config["S5_DO_NORM"],
            prenorm=self.config["S5_PRENORM"],
            do_gtrxl_norm=self.config["S5_DO_GTRXL_NORM"],
        )

    def __call__(self, hidden, obs, dones):
        """
        Encode observations using S5.

        Args:
            hidden: S5 hidden state
            obs: Observations [seq_len, batch, obs_dim]
            dones: Done flags [seq_len, batch]

        Returns:
            (new_hidden, embedding): Updated hidden state and embeddings [seq_len, batch, d_model]
        """
        if self.config.get("NO_RESET"):
            dones = jnp.zeros_like(dones)

        # Encoder layers
        embedding = self.rep_model_0(obs)
        embedding = nn.leaky_relu(embedding)
        embedding = self.rep_model_1(embedding)
        embedding = nn.leaky_relu(embedding)

        # S5 processing
        hidden, embedding = self.s5(hidden, embedding, dones)

        return hidden, embedding


class AGaLiTeRepModel(nn.Module):
    """
    AGaLiTe-based encoder following RepModel interface.

    Key: Hidden state format is (1, batch, ...) to match GRU/S5,
    enabling use with ppo_standard.py without modifications.

    The format conversion:
    - initialize_carry: returns (1, batch, ...) format
    - __call__ entry: converts (1, batch, ...) -> (batch, ...) via squeeze(0)
    - __call__ exit: converts (batch, ...) -> (1, batch, ...) via x[None, :]
    """

    config: Dict

    def setup(self):
        """Setup AGaLiTe module."""
        self.agalite = BatchedAGaLiTe(
            n_layers=self.config.get("AGALITE_N_LAYERS", 4),
            d_model=self.config.get("AGALITE_D_MODEL", 64),
            d_head=self.config.get("AGALITE_D_HEAD", 64),
            d_ffc=self.config.get("AGALITE_D_FFC", 64),
            n_heads=self.config.get("AGALITE_N_HEADS", 4),
            eta=self.config.get("AGALITE_ETA", 4),
            r=self.config.get("AGALITE_R", 2),
            reset_on_terminate=True
        )

    @staticmethod
    def initialize_carry(batch_size, config) -> dict:
        """
        Initialize AGaLiTe memory state with leading 1 dimension.

        This matches GRU/S5 format: (1, batch, ...)

        Args:
            batch_size: Number of environments
            config: Configuration dict containing AGaLiTe parameters

        Returns:
            Initial memory state with format (1, batch, ...)
        """
        import jax
        memory = BatchedAGaLiTe.initialize_carry(
            batch_size=batch_size,
            n_layers=config.get("AGALITE_N_LAYERS", 4),
            n_heads=config.get("AGALITE_N_HEADS", 4),
            d_head=config.get("AGALITE_D_HEAD", 64),
            eta=config.get("AGALITE_ETA", 4),
            r=config.get("AGALITE_R", 2)
        )
        # Add leading 1 to match GRU/S5 format: (batch, ...) -> (1, batch, ...)
        return jax.tree_map(lambda x: x[None, :], memory)

    @nn.compact
    def __call__(self, hidden, obs, dones):
        """
        Encode observations using AGaLiTe.

        Args:
            hidden: Memory state with format (1, batch, ...) for GRU/S5 compatibility
            obs: Observations [seq_len, batch, obs_dim]
            dones: Done flags [seq_len, batch]

        Returns:
            (new_hidden, embedding): Updated hidden state (1, batch, ...) and
                                     embeddings [seq_len, batch, d_model]
        """
        if self.config.get("NO_RESET"):
            dones = jnp.zeros_like(dones)

        # Remove leading 1: (1, batch, ...) -> (batch, ...)
        hidden_inner = jax.tree_map(lambda x: x.squeeze(0), hidden)

        # Pre-embedding to d_model dimension
        d_model = self.config.get("AGALITE_D_MODEL", 64)
        embedding = nn.Dense(
            d_model,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        # AGaLiTe processing
        rnn_in = (embedding, dones)
        new_hidden_inner, embedding = self.agalite(hidden_inner, rnn_in)

        # Add leading 1 back: (batch, ...) -> (1, batch, ...)
        new_hidden = jax.tree_map(lambda x: x[None, :], new_hidden_inner)

        return new_hidden, embedding


class ActorCriticBase(nn.Module):
    """
    Base ActorCritic network with pluggable RepModel.

    This class provides the common actor/critic heads and delegates
    representation learning to a separate RepModel (GRU or S5).
    RepModel = ObsEncoder (stateless) + Core (stateful).
    """

    rep_model: nn.Module  # GRURepModel or S5RepModel
    action_dim: int
    config: Dict

    def initialize_core_hidden_state(self, batch_size):
        """
        Initialize Core hidden state.

        This method delegates to the RepModel's initialize_carry method,
        providing a unified interface regardless of Core type (GRU/S5).

        Args:
            batch_size: Number of environments

        Returns:
            Initial Core hidden state
        """
        return self.rep_model.initialize_carry(batch_size, self.config)

    def forward_rep_model(self, hidden, obs, dones):
        """
        Forward pass through RepModel to obtain representations.

        Args:
            hidden: Core hidden state
            obs: Observations
            dones: Done flags

        Returns:
            (new_hidden, representation): Updated hidden state and learned representations
        """
        return self.rep_model(hidden, obs, dones)

    def decode_actor(self, embedding):
        """
        Actor head (before distribution).

        Returns logits/mean for action distribution.
        """
        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.leaky_relu(actor_mean)
        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.leaky_relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        return actor_mean

    def decode_critic(self, embedding):
        """Critic head (value function)."""
        critic = nn.Dense(
            128, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.leaky_relu(critic)
        critic = nn.Dense(
            128, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.leaky_relu(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)
        return jnp.squeeze(critic, axis=-1)


class ActorCriticContinuous(ActorCriticBase):
    """
    ActorCritic for continuous action spaces.

    Uses Gaussian (MultivariateNormalDiag) distribution.
    Works with any encoder (GRU, S5, etc.).
    """

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        # Encode (GRU or S5)
        hidden, embedding = self.forward_rep_model(hidden, obs, dones)

        # Actor - Continuous (Gaussian)
        actor_mean = self.decode_actor(embedding)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(log_std))

        # Critic
        critic = self.decode_critic(embedding)

        return hidden, pi, critic


class ActorCriticDiscrete(ActorCriticBase):
    """
    ActorCritic for discrete action spaces.

    Uses Categorical distribution.
    Works with any encoder (GRU, S5, etc.).
    """

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        # Encode (GRU or S5)
        hidden, embedding = self.forward_rep_model(hidden, obs, dones)

        # Actor - Discrete (Categorical)
        actor_logits = self.decode_actor(embedding)
        pi = distrax.Categorical(logits=actor_logits)

        # Critic
        critic = self.decode_critic(embedding)

        return hidden, pi, critic
