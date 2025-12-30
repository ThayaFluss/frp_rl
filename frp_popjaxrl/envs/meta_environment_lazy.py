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
    try:
        from gymnax import make as gymnax_make
        from .wrappers import GymnaxRewardNormWrapper

        # Format the environment name to match gymnax's expected format
        if env_name.lower() == "cartpole":
            env_name = "CartPole-v1"
        elif env_name.lower() == "pendulum":
            env_name = "Pendulum-v1"
        elif env_name.lower() == "acrobot":
            env_name = "Acrobot-v1"
        elif env_name.lower() == "mountaincar":
            env_name = "MountainCar-v0"
        elif env_name.lower() == "mountaincarcontinuous":
            env_name = "MountainCarContinuous-v0"
        elif "-" not in env_name and not any(suffix in env_name.lower() for suffix in ["minatar", "bsuite", "misc"]):
            if env_name.lower() in ["asterix", "breakout", "freeway", "seaquest", "spaceinvaders"]:
                env_name = f"{env_name.capitalize()}-MinAtar"
            elif env_name.lower() in ["catch", "deepsea", "memorychain", "umbrellachain",
                                     "discountingchain", "mnistbandit", "simplebandit"]:
                env_name = f"{env_name.capitalize()}-bsuite"
            elif env_name.lower() in ["fourrooms", "metamaze", "pointrobot", "bernoullibandit",
                                     "gaussianbandit", "reacher", "swimmer", "pong"]:
                env_name = f"{env_name.capitalize()}-misc"

        # Get the base environment
        env, _ = gymnax_make(env_name)

        # Create a wrapper class that applies reward normalization
        class NormalizedEnv(GymnaxRewardNormWrapper):
            def __init__(self, **kwargs):
                if norm_kwargs is not None:
                    strategy = norm_kwargs.get('strategy', 'dynamic')
                    max_steps = norm_kwargs.get('max_steps', 200)
                else:
                    strategy = 'dynamic'
                    max_steps = 200
                super().__init__(env.__class__(**kwargs), strategy=strategy, max_steps=max_steps)

        # Return the meta environment with the normalized env
        return MetaEnvironmentLazy(NormalizedEnv, env_kwargs, meta_kwargs)
    except Exception as e:
        raise ValueError(f"Error creating gymnax environment {env_name}: {e}")


def create_meta_environment(env_name: str, env_kwargs: Dict[str, Any], meta_kwargs: Dict[str, Any], norm_kwargs: Dict[str, Any] = None):
    """
    Create a meta environment (lazy evaluation version).

    This is the dispatcher function that routes to the appropriate environment class.
    """
    # Handle popgym environments
    if env_name == "cartpole":
        from .environments.popgym_cartpole import NoisyStatelessCartPole
        return MetaEnvironmentLazy(NoisyStatelessCartPole, env_kwargs, meta_kwargs)
    if env_name == "cartpole_origin":
        # Note: cartpole_origin uses different base class, not compatible with lazy version
        raise NotImplementedError("cartpole_origin is not compatible with lazy evaluation")
    if env_name == "s_cartpole_hard":
        from .environments.popgym_cartpole import StatelessCartPoleHard
        return MetaEnvironmentLazy(StatelessCartPoleHard, env_kwargs, meta_kwargs)
    if env_name == "ns_cartpole_hard":
        from .environments.popgym_cartpole import NoisyStatelessCartPoleHard
        return MetaEnvironmentLazy(NoisyStatelessCartPoleHard, env_kwargs, meta_kwargs)
    elif env_name == "minesweeper":
        from .environments.popgym_minesweeper import MineSweeper
        return MetaEnvironmentLazy(MineSweeper, env_kwargs, meta_kwargs)
    elif env_name == "minesweeper_hard":
        from .environments.popgym_minesweeper import MineSweeperHard
        return MetaEnvironmentLazy(MineSweeperHard, env_kwargs, meta_kwargs)
    elif env_name == "multiarmedbandit":
        from .environments.popgym_multiarmedbandit import MultiarmedBandit
        return MetaEnvironmentLazy(MultiarmedBandit, env_kwargs, meta_kwargs)
    elif env_name == "higherlower":
        from .environments.popgym_higherlower import HigherLower
        return MetaEnvironmentLazy(HigherLower, env_kwargs, meta_kwargs)
    elif env_name == "higherlower_easy":
        from .environments.popgym_higherlower import HigherLowerEasy
        return MetaEnvironmentLazy(HigherLowerEasy, env_kwargs, meta_kwargs)
    elif env_name == "higherlower_medium":
        from .environments.popgym_higherlower import HigherLowerMedium
        return MetaEnvironmentLazy(HigherLowerMedium, env_kwargs, meta_kwargs)
    elif env_name == "higherlower_hard":
        from .environments.popgym_higherlower import HigherLowerHard
        return MetaEnvironmentLazy(HigherLowerHard, env_kwargs, meta_kwargs)
    elif env_name == "pendulum":
        from .environments.popgym_pendulum import NoisyStatelessPendulum
        return MetaEnvironmentLazy(NoisyStatelessPendulum, env_kwargs, meta_kwargs)
    elif env_name == "pendulum_easy":
        from .environments.popgym_pendulum import NoisyStatelessPendulumEasy
        return MetaEnvironmentLazy(NoisyStatelessPendulumEasy, env_kwargs, meta_kwargs)
    elif env_name == "pendulum_medium":
        from .environments.popgym_pendulum import NoisyStatelessPendulumMedium
        return MetaEnvironmentLazy(NoisyStatelessPendulumMedium, env_kwargs, meta_kwargs)
    elif env_name == "pendulum_hard":
        from .environments.popgym_pendulum import NoisyStatelessPendulumHard
        return MetaEnvironmentLazy(NoisyStatelessPendulumHard, env_kwargs, meta_kwargs)
    elif env_name == "autoencode":
        from .environments.popgym_autoencode import Autoencode
        return MetaEnvironmentLazy(Autoencode, env_kwargs, meta_kwargs)
    elif env_name == "battleship":
        from .environments.popgym_battleship import Battleship
        return MetaEnvironmentLazy(Battleship, env_kwargs, meta_kwargs)
    elif env_name == "concentration":
        from .environments.popgym_concentration import Concentration
        return MetaEnvironmentLazy(Concentration, env_kwargs, meta_kwargs)
    elif env_name == "count_recall":
        from .environments.popgym_count_recall import CountRecall
        return MetaEnvironmentLazy(CountRecall, env_kwargs, meta_kwargs)
    elif env_name == "repeat_first":
        from .environments.popgym_repeat_first import RepeatFirst
        return MetaEnvironmentLazy(RepeatFirst, env_kwargs, meta_kwargs)
    elif env_name == "repeat_first_hard":
        from .environments.popgym_repeat_first import RepeatFirstHard
        return MetaEnvironmentLazy(RepeatFirstHard, env_kwargs, meta_kwargs)
    elif env_name == "repeat_previous_hard":
        from .environments.popgym_repeat_previous import RepeatPreviousHard
        return MetaEnvironmentLazy(RepeatPreviousHard, env_kwargs, meta_kwargs)
    # Check if it's a gymnax environment
    elif env_name.startswith("gymnax_"):
        base_env_name = env_name[7:]
        return create_gymnax_environment(base_env_name, env_kwargs, meta_kwargs, norm_kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
