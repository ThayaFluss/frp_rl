"""
Model definitions for PPO with different architectures.

This module contains:
- RecurrentCore classes: GRUCore, S5Core, AGaLiTeCore with unified interface
- PreCoreEncoder: Shared pre-recurrent encoder (2-layer Dense + leaky_relu)
- ActorCriticBase: Base class for actor-critic networks with integrated encoder and core
- ActorCriticContinuous: Actor-critic for continuous action spaces
- ActorCriticDiscrete: Actor-critic for discrete action spaces

Core Interface:
    All cores implement:
    - __call__(hidden, embedding, dones) -> (new_hidden, output)
    - initialize_carry(batch_size, config) -> hidden
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import functools
from flax.linen.initializers import constant, orthogonal
from typing import Dict, Tuple
import distrax
from .s5 import StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from .agalite import BatchedAGaLiTe


# =============================================================================
# Core Classes with Unified Interface
# =============================================================================


class GRUCore(nn.Module):
    """GRU Core with unified interface.

    Wraps nn.GRUCell with scan for sequence processing.
    Hidden state format: [(1, batch, hidden_size)]
    """

    @functools.partial(
        nn.scan,
        variable_broadcast='params',
        in_axes=0,
        out_axes=0,
        split_rngs={'params': False})
    @nn.compact
    def _scan_fn(self, carry, x):
        """Internal scan function for GRU."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self._initialize_carry_inner(ins.shape[0], ins.shape[1]),
            rnn_state
        )
        features = rnn_state[0].shape[-1]
        new_rnn_state, y = nn.GRUCell(features)(rnn_state, ins)
        return new_rnn_state, y

    def __call__(self, hidden, embedding, dones):
        """Forward pass with unified interface.

        Args:
            hidden: Hidden state [(1, batch, hidden_size)]
            embedding: Encoded observations [seq_len, batch, embed_dim]
            dones: Done flags [seq_len, batch]

        Returns:
            (new_hidden, output): Updated hidden state and core output
        """
        # Extract from list and remove time dimension
        h = hidden[0].squeeze(0)  # (1, batch, hidden) -> (batch, hidden)
        rnn_in = (embedding, dones)
        h, out = self._scan_fn(h, rnn_in)
        # Wrap back to list format
        new_hidden = [jnp.expand_dims(h, axis=0)]
        return new_hidden, out

    @staticmethod
    def _initialize_carry_inner(batch_size, hidden_size):
        """Initialize GRU cell carry (internal use)."""
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size))

    @staticmethod
    def initialize_carry(batch_size: int, config: Dict):
        """Initialize GRU hidden state.

        Args:
            batch_size: Number of environments
            config: Configuration dictionary (unused for GRU, kept for interface)

        Returns:
            Hidden state [(1, batch, hidden_size)]
        """
        hidden_size = 256  # Encoder output size
        carry = GRUCore._initialize_carry_inner(batch_size, hidden_size)
        # (batch, hidden) -> [(1, batch, hidden)]
        return [jnp.expand_dims(carry, axis=0)]


class S5Core(nn.Module):
    """S5 Core wrapper with unified interface.

    Encapsulates S5-specific initialization (HiPPO, SSM) and provides
    the same interface as GRUCore.

    Attributes:
        config: Configuration dictionary with S5 parameters
    """

    config: Dict

    def setup(self):
        """Setup S5 StackedEncoderModel with HiPPO initialization."""
        d_model = self.config["S5_D_MODEL"]
        ssm_size = self.config["S5_SSM_SIZE"]
        blocks = self.config["S5_BLOCKS"]
        block_size = int(ssm_size / blocks)

        Lambda, _, _, V, _ = make_DPLR_HiPPO(ssm_size)
        block_size = block_size // 2
        ssm_size_half = ssm_size // 2
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vinv = V.conj().T

        ssm_init_fn = init_S5SSM(
            H=d_model,
            P=ssm_size_half,
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

        self.s5 = StackedEncoderModel(
            ssm=ssm_init_fn,
            d_model=d_model,
            n_layers=self.config["S5_N_LAYERS"],
            activation=self.config["S5_ACTIVATION"],
            do_norm=self.config["S5_DO_NORM"],
            prenorm=self.config["S5_PRENORM"],
            do_gtrxl_norm=self.config["S5_DO_GTRXL_NORM"],
        )

    def __call__(self, hidden, embedding, dones):
        """Forward pass with unified interface.

        Args:
            hidden: S5 hidden state (list of layer states)
            embedding: Encoded observations [seq_len, batch, embed_dim]
            dones: Done flags [seq_len, batch]

        Returns:
            (new_hidden, output): Updated hidden state and core output
        """
        return self.s5(hidden, embedding, dones)

    @staticmethod
    def initialize_carry(batch_size: int, config: Dict):
        """Initialize S5 hidden state.

        Args:
            batch_size: Number of environments
            config: Configuration dictionary with S5 parameters

        Returns:
            S5 hidden state (list of layer states)
        """
        ssm_size = config["S5_SSM_SIZE"] // 2
        n_layers = config["S5_N_LAYERS"]
        return StackedEncoderModel.initialize_carry(batch_size, ssm_size, n_layers)


class AGaLiTeCore(nn.Module):
    """AGaLiTe Core wrapper with unified interface.

    Encapsulates AGaLiTe-specific hidden state format conversion
    ((1, batch, ...) <-> (batch, ...)) and provides the same interface.

    Attributes:
        config: Configuration dictionary with AGaLiTe parameters
    """

    config: Dict

    def setup(self):
        """Setup BatchedAGaLiTe module."""
        self.agalite = BatchedAGaLiTe(
            n_layers=self.config.get("AGALITE_N_LAYERS", 4),
            d_model=self.config.get("AGALITE_D_MODEL", 256),
            d_head=self.config.get("AGALITE_D_HEAD", 64),
            d_ffc=self.config.get("AGALITE_D_FFC", 256),
            n_heads=self.config.get("AGALITE_N_HEADS", 4),
            eta=self.config.get("AGALITE_ETA", 4),
            r=self.config.get("AGALITE_R", 2),
            reset_on_terminate=True
        )

    def __call__(self, hidden, embedding, dones):
        """Forward pass with hidden state format conversion.

        Args:
            hidden: AGaLiTe hidden state with (1, batch, ...) format
            embedding: Encoded observations [seq_len, batch, embed_dim]
            dones: Done flags [seq_len, batch]

        Returns:
            (new_hidden, output): Updated hidden state and core output
        """
        # Remove leading 1: (1, batch, ...) -> (batch, ...)
        hidden_inner = jax.tree_map(lambda x: x.squeeze(0), hidden)
        rnn_in = (embedding, dones)
        new_hidden_inner, out = self.agalite(hidden_inner, rnn_in)
        # Add leading 1 back: (batch, ...) -> (1, batch, ...)
        new_hidden = jax.tree_map(lambda x: x[None, :], new_hidden_inner)
        return new_hidden, out

    @staticmethod
    def initialize_carry(batch_size: int, config: Dict):
        """Initialize AGaLiTe hidden state with (1, batch, ...) format.

        Args:
            batch_size: Number of environments
            config: Configuration dictionary with AGaLiTe parameters

        Returns:
            AGaLiTe hidden state with (1, batch, ...) format
        """
        memory = BatchedAGaLiTe.initialize_carry(
            batch_size=batch_size,
            n_layers=config.get("AGALITE_N_LAYERS", 4),
            n_heads=config.get("AGALITE_N_HEADS", 4),
            d_head=config.get("AGALITE_D_HEAD", 64),
            eta=config.get("AGALITE_ETA", 4),
            r=config.get("AGALITE_R", 2)
        )
        # (batch, ...) -> (1, batch, ...)
        return jax.tree_map(lambda x: x[None, :], memory)


# =============================================================================
# Core Factory
# =============================================================================


def get_core_class(core_type: str):
    """Get the Core class for a given core type.

    Args:
        core_type: One of "gru", "s5", "agalite"

    Returns:
        The corresponding Core class
    """
    cores = {
        "gru": GRUCore,
        "s5": S5Core,
        "agalite": AGaLiTeCore,
    }
    if core_type not in cores:
        raise ValueError(f"Unknown core_type: {core_type}")
    return cores[core_type]


class PreCoreEncoder(nn.Module):
    """
    Shared Pre-Recurrent Encoder (2-layer Dense + leaky_relu).

    This encoder is used by all core types (GRU, S5, AGaLiTe) to transform
    observations before feeding into the recurrent core.

    Attributes:
        hidden_dims: Tuple of hidden dimensions for the two Dense layers.
                     Default is (128, 256) to match existing encoder structure.
    """

    hidden_dims: Tuple[int, int] = (128, 256)

    @nn.compact
    def __call__(self, obs):
        """
        Encode observations through 2-layer MLP.

        Args:
            obs: Input observations with shape [..., obs_dim]

        Returns:
            Encoded observations with shape [..., hidden_dims[-1]]
        """
        x = obs
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name=f"encoder_{i}"
            )(x)
            x = nn.leaky_relu(x)
        return x


# =============================================================================
# Actor-Critic Base
# =============================================================================


class ActorCriticBase(nn.Module):
    """
    Base ActorCritic network with integrated encoder and core.

    This class provides:
    - Shared PreCoreEncoder for all core types
    - Unified core interface via GRUCore/S5Core/AGaLiTeCore
    - Common actor/critic heads

    Attributes:
        core_type: One of "gru", "s5", or "agalite"
        action_dim: Dimension of action space
        config: Configuration dictionary
    """

    core_type: str  # "gru", "s5", "agalite"
    action_dim: int
    config: Dict

    def setup(self):
        """Setup encoder and core based on core_type."""
        # Shared pre-recurrent encoder
        self.encoder = PreCoreEncoder()

        # Create core with unified interface
        CoreClass = get_core_class(self.core_type)
        if self.core_type == "gru":
            # GRU doesn't need config in setup (uses nn.compact internally)
            self.core = CoreClass()
        else:
            # S5 and AGaLiTe need config for setup
            self.core = CoreClass(config=self.config)

    @staticmethod
    def initialize_carry(batch_size: int, core_type: str, config: Dict):
        """
        Initialize hidden state for the specified core type.

        Delegates to the core-specific initialize_carry method.

        Args:
            batch_size: Number of environments
            core_type: One of "gru", "s5", "agalite"
            config: Configuration dictionary

        Returns:
            Initial hidden state appropriate for the core type
        """
        CoreClass = get_core_class(core_type)
        return CoreClass.initialize_carry(batch_size, config)

    def forward_core(self, hidden, embedding, dones):
        """
        Forward pass through the recurrent core.

        Uses unified core interface.

        Args:
            hidden: Hidden state from initialize_carry or previous step
            embedding: Encoded observations [seq_len, batch, embed_dim]
            dones: Done flags [seq_len, batch]

        Returns:
            (new_hidden, output): Updated hidden state and core output
        """
        return self.core(hidden, embedding, dones)

    def initialize_core_hidden_state(self, batch_size):
        """
        Initialize core hidden state (convenience method).

        Args:
            batch_size: Number of environments

        Returns:
            Initial hidden state for this network's core type
        """
        return self.initialize_carry(batch_size, self.core_type, self.config)

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
    Works with any core type (GRU, S5, AGaLiTe).
    """

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        # Handle NO_RESET config option
        if self.config.get("NO_RESET"):
            dones = jnp.zeros_like(dones)

        # Encode observations
        embedding = self.encoder(obs)

        # Process through recurrent core
        hidden, embedding = self.forward_core(hidden, embedding, dones)

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
    Works with any core type (GRU, S5, AGaLiTe).
    """

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        # Handle NO_RESET config option
        if self.config.get("NO_RESET"):
            dones = jnp.zeros_like(dones)

        # Encode observations
        embedding = self.encoder(obs)

        # Process through recurrent core
        hidden, embedding = self.forward_core(hidden, embedding, dones)

        # Actor - Discrete (Categorical)
        actor_logits = self.decode_actor(embedding)
        pi = distrax.Categorical(logits=actor_logits)

        # Critic
        critic = self.decode_critic(embedding)

        return hidden, pi, critic
