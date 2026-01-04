import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from flax import struct
import chex
from typing import Tuple, Optional, Any, Dict, List
from frp.orthogonal_lazy import (
    create_base_matrices,
    sample_word_index,
    build_word_from_index,
)
from flax.core import freeze

# IMPORTANT: Gymnax Auto-Reset Behavior (Lazy Mode)
# This environment inherits from gymnax.Environment, which provides automatic
# reset functionality when done=True. See docs/dev/GYMNAX_AUTO_RESET_MECHANISM.md
# for detailed documentation on how this affects env_index resampling behavior.
#
# Key points:
# - env.step() automatically calls reset_env() when done=True
# - reset_env() samples a NEW env_index on each call
# - Therefore, env_index is resampled at meta-episode boundaries via gymnax auto-reset
# - Tests must use env.step() NOT env.step_env() to observe this behavior
# - Lazy mode uses on-demand word building (obs_word) instead of storing all words

@struct.dataclass
class MetaEnvStateLazy:
    """
    Lazy evaluation version of MetaEnvState.

    Key differences from original:
    - obs_bases: stores base matrices (B, D, D) instead of all words
    - obs_word: stores current single word matrix (input_dim, output_dim) instead of all words
    """
    env_index: int
    obs_bases: chex.Array  # Base matrices (B, D, D)
    obs_word: chex.Array   # Current word matrix (input_dim, output_dim)
    trial_num: int
    total_steps: int
    env_state: Any
    init_state: Optional[chex.Array]
    init_obs: Optional[chex.Array]

@struct.dataclass
class MetaEnvParams:
    num_trials_per_episode: int = 16
    env_params: Any = None

class MetaEnvironmentLazy(environment.Environment):
    """
    Lazy evaluation version of MetaEnvironment.

    Main changes:
    - Only computes one word matrix at reset time (not all words)
    - Base matrices stored in env_state for PPO updates
    - Uses build_word_from_index instead of precomputed word table
    """

    def __init__(self, env_class, env_kwargs: Dict[str, Any], meta_kwargs: Dict[str, Any]):
        super().__init__()
        self.env = env_class(**env_kwargs)
        self.input_dim = self.env.observation_space(self.env.default_params).shape[0]
        self.meta_kwargs = meta_kwargs

        # Meta-learning specific parameters
        self.meta_dim = meta_kwargs.get('meta_dim', 4)
        self.meta_truncate_aug = meta_kwargs.get("meta_truncate_aug", 0)
        self.meta_depth = meta_kwargs.get('meta_depth', 1)

        # Also check for keys with 'meta_' prefix
        if 'meta_depth' in meta_kwargs:
            self.meta_depth = meta_kwargs['meta_depth']
        if 'meta_dim' in meta_kwargs:
            self.meta_dim = meta_kwargs['meta_dim']
        if 'meta_max_depth' in meta_kwargs:
            self.meta_max_depth = meta_kwargs['meta_max_depth']
        if 'meta_with_adjoint' in meta_kwargs:
            self.meta_with_adjoint = meta_kwargs['meta_with_adjoint']
        if 'meta_rng' in meta_kwargs:
            self.rng = meta_kwargs['meta_rng']
        if 'meta_const_aug' in meta_kwargs:
            self.meta_const_aug = meta_kwargs['meta_const_aug']

        self.meta_max_depth = meta_kwargs.get('meta_max_depth', 2)
        self.meta_with_adjoint = meta_kwargs.get('meta_with_adjoint', False)
        self.rng = meta_kwargs.get('meta_rng', jax.random.PRNGKey(42))
        self.meta_const_aug = meta_kwargs.get('meta_const_aug', False)

        # Lazy-specific: compute num_base and total_words
        # Using backward compatibility formula: num_base = 2^(max_depth // depth)
        self.num_base = 2 ** (self.meta_max_depth // self.meta_depth)
        self.total_words = self.num_base ** self.meta_depth
        self.encoding_mode = 'base_b'  # Fixed during execution

        # Set up evaluation method if specified
        if self.meta_const_aug == "padding":
            self.eval_weight = jnp.eye(self.meta_dim)[:self.input_dim,:]
        elif self.meta_const_aug == "tiling":
            from .environments.metaaug.padding import create_periodic_weight
            self.eval_weight = create_periodic_weight(
                input_dim=self.input_dim,
                output_dim=self.meta_dim,
                period=round(self.meta_dim/2)
            )
        else:
            # Don't prepare eval_weight for reducing memory
            pass

        ### obs_shape is output dimension of this class
        if self.meta_const_aug == "identity":
            self.aug_output_dim = self.input_dim
        elif self.meta_truncate_aug == 1:
            self.aug_output_dim = self.input_dim
        else:
            self.aug_output_dim = self.meta_dim

        self.obs_shape = (self.aug_output_dim + 3,)

        # Initialize meta-augmentation (lazy version)
        self._initialize_meta_augmentation()

    def _initialize_meta_augmentation(self):
        """
        Lazy version: only create base matrices, not all words.
        """
        # Create base matrices only
        self.initial_bases = create_base_matrices(
            self.rng,
            self.num_base,
            size=self.meta_dim,
            with_adjoint=self.meta_with_adjoint
        )

        # Option B: No identity detection in lazy evaluation
        # (Identity detection would require creating all words)
        self.exclude = jnp.array([], dtype=jnp.int32)

    @property
    def default_params(self) -> MetaEnvParams:
        return MetaEnvParams(
            num_trials_per_episode=self.meta_kwargs.get('num_trials_per_episode', 16),
            env_params=self.env.default_params
        )

    def step_env(
        self, key: chex.PRNGKey, state: MetaEnvStateLazy, action: Any, params: MetaEnvParams
    ) -> Tuple[chex.Array, MetaEnvStateLazy, float, bool, dict]:
        key, key_reset = jax.random.split(key)

        env_obs_st, env_state_st, reward, env_done, info = self.env.step_env(
            key, state.env_state, action, params.env_params
        )
        env_obs_re, env_state_re = state.init_obs, state.init_state

        env_state = jax.tree_map(
            lambda x, y: jax.lax.select(env_done, x, y), env_state_re, env_state_st
        )
        env_obs = jax.lax.select(env_done, env_obs_re, env_obs_st)

        if self.meta_const_aug == "identity":
            # For identity, use observation directly
            env_obs = env_obs
        elif self.meta_const_aug in ["padding", "tiling"]:
            env_obs = (env_obs[None,:] @ self.eval_weight)[0]
        else:
            # Lazy version: use pre-computed single word from state
            # No need for indexing - the word is already selected
            weight = jnp.sqrt(2) * state.obs_word
            env_obs = (env_obs[None,:] @ weight)[0]

        # trial num increases when env has done
        trial_num = state.trial_num + env_done
        total_steps = state.total_steps + 1
        done = trial_num >= params.num_trials_per_episode

        state = MetaEnvStateLazy(
            env_index=state.env_index,
            obs_bases=state.obs_bases,
            obs_word=state.obs_word,  # Keep the same word (no recomputation)
            trial_num=trial_num,
            total_steps=total_steps,
            env_state=env_state,
            init_state=state.init_state,
            init_obs=state.init_obs,
        )

        # Handle both discrete and continuous actions
        action_value = action[0] if isinstance(action, jnp.ndarray) and len(action.shape) > 0 else action
        obs = jnp.concatenate([env_obs, jnp.array([jnp.float32(action_value), jnp.float32(env_done), 0.0])])

        return obs, state, reward, done, info

    def reset_env(
        self, key: chex.PRNGKey, params: MetaEnvParams
    ) -> Tuple[chex.Array, MetaEnvStateLazy]:
        env_key, obs_key, index_key = jax.random.split(key, 3)
        env_obs, env_state = self.env.reset_env(env_key, params.env_params)

        # Sample word index (no exclusion in lazy version - Option B)
        env_index = sample_word_index(index_key, self.total_words, exclude=self.exclude)

        # Lazy evaluation: compute ONE word matrix at reset time
        # Use initial_bases (will be overridden by PPO if RESET_WORDS is enabled)
        word = build_word_from_index(
            self.initial_bases,
            env_index,
            self.num_base,
            self.meta_depth,
            self.input_dim,
            self.aug_output_dim,
            self.encoding_mode
        )

        if self.meta_const_aug == "identity":
            augmented_obs = env_obs
        elif self.meta_const_aug in ["padding", "tiling"]:
            augmented_obs = (env_obs[None,:] @ self.eval_weight)[0]
        else:
            # Apply the computed word
            weight = jnp.sqrt(2) * word
            augmented_obs = (env_obs[None,:] @ weight)[0]

        state = MetaEnvStateLazy(
            env_index=env_index,
            obs_bases=self.initial_bases,  # Store base matrices (will be updated by PPO)
            obs_word=word,  # Store the single computed word
            trial_num=0,
            total_steps=0,
            env_state=env_state,
            init_state=env_state,
            init_obs=env_obs,
        )
        obs = jnp.concatenate([augmented_obs, jnp.array([0.0, 0.0, 1.0])])

        return obs, state

    def action_space(
        self, params: Optional[MetaEnvParams] = None
    ) -> spaces.Space:
        return self.env.action_space(params.env_params if params else None)

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        action_space = self.action_space(None)
        if isinstance(action_space, spaces.Box):
            return action_space.shape[0]
        elif isinstance(action_space, spaces.Discrete):
            return action_space.n
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

    def observation_space(self, params: MetaEnvParams) -> spaces.Box:
        env_obs_space = self.env.observation_space(params.env_params)
        high = jnp.ones([self.obs_shape[0]])
        return spaces.Box(-high, high, (self.obs_shape[0],), dtype=env_obs_space.dtype)


def create_gymnax_environment(env_name: str, env_kwargs: Dict[str, Any], meta_kwargs: Dict[str, Any], norm_kwargs: Dict[str, Any] = None):
    """Create a gymnax environment wrapped in MetaEnvironmentLazy."""
    from .meta_environment_factory import create_gymnax_environment_internal
    return create_gymnax_environment_internal(env_name, env_kwargs, meta_kwargs, norm_kwargs, MetaEnvironmentLazy)


def create_meta_environment(env_name: str, env_kwargs: Dict[str, Any], meta_kwargs: Dict[str, Any], norm_kwargs: Dict[str, Any] = None):
    """Create a meta environment (lazy evaluation version).

    This is the dispatcher function that routes to the appropriate environment class.
    """
    from .meta_environment_factory import create_meta_environment_internal
    return create_meta_environment_internal(env_name, env_kwargs, meta_kwargs, norm_kwargs, MetaEnvironmentLazy)
