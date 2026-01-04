import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from flax import struct
import chex
from typing import Tuple, Optional, Any, Dict, List
from flax.core import freeze

# IMPORTANT: Gymnax Auto-Reset Behavior (Separated Mode)
# This environment inherits from gymnax.Environment, which provides automatic
# reset functionality when done=True. See docs/dev/GYMNAX_AUTO_RESET_MECHANISM.md
# for detailed documentation.
#
# Key differences from Legacy mode:
# - env_index is NOT stored in MetaEnvState (managed externally by algorithm)
# - reset_env() does NOT sample env_index (FRP is externalized)
# - Gymnax auto-reset still runs, but only affects trial_num and env_state
# - env_index resampling happens in algorithm code (ppo_in_context.py)
# - Tests must use env.step() NOT env.step_env() to match training loop behavior

@struct.dataclass
class MetaEnvState:
    trial_num: int
    total_steps: int
    env_state: Any
    init_state: Optional[chex.Array]
    init_obs: Optional[chex.Array]

@struct.dataclass
class MetaEnvParams:
    num_trials_per_episode: int = 16
    env_params: Any = None

class MetaEnvironment(environment.Environment):
    def __init__(self, env_class, env_kwargs: Dict[str, Any], meta_kwargs: Dict[str, Any]):
        super().__init__()
        self.env = env_class(**env_kwargs)
        self.input_dim = self.env.observation_space(self.env.default_params).shape[0]
        self.meta_kwargs = meta_kwargs  # Store meta_kwargs for use in default_params

        # MetaEnvironment returns raw observations (input_dim + 3 metadata)
        # FRP transformation happens externally in PPO
        self.obs_shape = (self.input_dim + 3,)

    @property
    def default_params(self) -> MetaEnvParams:
        return MetaEnvParams(
            num_trials_per_episode=self.meta_kwargs.get('num_trials_per_episode', 16),
            env_params=self.env.default_params
        )
    
    def step_env(
        self, key: chex.PRNGKey, state: MetaEnvState, action: Any, params: MetaEnvParams
    ) -> Tuple[chex.Array, MetaEnvState, float, bool, dict]:
        key, key_reset = jax.random.split(key)

        env_obs_st, env_state_st, reward, env_done, info = self.env.step_env(key, state.env_state, action, params.env_params)
        env_obs_re, env_state_re = state.init_obs, state.init_state

        env_state = jax.tree_map(
            lambda x, y: jax.lax.select(env_done, x, y), env_state_re, env_state_st
        )
        env_obs = jax.lax.select(env_done, env_obs_re, env_obs_st)

        # No FRP transformation here - will be done externally by FRPManager
        # env_obs is returned as-is

        # trial num increases when env has done
        trial_num = state.trial_num + env_done
        total_steps = state.total_steps + 1
        done = trial_num >= params.num_trials_per_episode

        state = MetaEnvState(
            trial_num=trial_num,
            total_steps=total_steps,
            env_state=env_state,
            init_state=state.init_state,
            init_obs=state.init_obs,
        )

        # Handle both discrete and continuous actions
        action_value = action[0] if isinstance(action, jnp.ndarray) and len(action.shape) > 0 else action

        # Return raw observation without padding
        # FRP transformation will be applied externally by PPO
        obs = jnp.concatenate([env_obs, jnp.array([jnp.float32(action_value), jnp.float32(env_done), 0.0])])

        return obs, state, reward, done, info
    
    def reset_env(
        self, key: chex.PRNGKey, params: MetaEnvParams
    ) -> Tuple[chex.Array, MetaEnvState]:
        # IMPORTANT: Split RNG to match legacy implementation's RNG consumption pattern.
        # Legacy splits key into 3 parts (env_key, obs_key, index_key).
        # In separated mode, FRP is handled externally, so we don't sample env_index here.
        # However, we still split to match the RNG consumption pattern.
        # The actual env_index sampling happens in PPO (maybe_resample_env_index),
        # which uses the same RNG derivation to ensure identical random sequences.
        env_key, _, _ = jax.random.split(key, 3)

        env_obs, env_state = self.env.reset_env(env_key, params.env_params)

        state = MetaEnvState(
            trial_num=0,
            total_steps=0,
            env_state=env_state,
            init_state=env_state,
            init_obs=env_obs,
        )

        # Return raw observation without padding
        # FRP transformation will be applied externally by PPO
        obs = jnp.concatenate([env_obs, jnp.array([0.0, 0.0, 1.0])])

        return obs, state

    def action_space(
        self, params: Optional[MetaEnvParams] = None
    ) -> spaces.Space:
        return self.env.action_space(params.env_params if params else None)


    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        action_space = self.action_space(None)  # Get the action space object
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
    """Create a gymnax environment wrapped in MetaEnvironment.
    
    This function uses gymnax.make to create the base environment, applies
    reward normalization, and then wraps it with MetaEnvironment.
    """
    from .meta_environment_factory import create_gymnax_environment_internal
    return create_gymnax_environment_internal(env_name, env_kwargs, meta_kwargs, norm_kwargs, MetaEnvironment)

def create_meta_environment(env_name: str, env_kwargs: Dict[str, Any], meta_kwargs: Dict[str, Any], norm_kwargs: Dict[str, Any] = None):
    """Create a meta environment (separated FRP mode).

    This is the dispatcher function that routes to the appropriate environment class.
    """
    from .meta_environment_factory import create_meta_environment_internal
    return create_meta_environment_internal(env_name, env_kwargs, meta_kwargs, norm_kwargs, MetaEnvironment)
