import jax.numpy as jnp
import jax
import flax.linen as nn
from gymnax.environments import environment, spaces
from flax import struct
import chex
from typing import Tuple, Optional, Dict, Any
from flax.linen.initializers import constant, orthogonal
import numpy as np
from .popgym_cartpole import NoisyStatelessCartPole, EnvParams, EnvState



class MetaAugNetwork(nn.Module):
    out_size: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.out_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        return x


@struct.dataclass
class MetaEnvState:
    # Fields actually used by this environment
    obs_params: chex.Array
    trial_num: int
    total_steps: int
    env_state: EnvState
    init_state: Optional[chex.Array]
    init_obs: Optional[chex.Array]
    # Fields added for compatibility with MetaEnvironment (unused but needed for algorithm code)
    env_index: int = 0
    obs_words: Optional[chex.Array] = None

@struct.dataclass
class MetaEnvParams:
    num_trials_per_episode: int = 16
    env_params: EnvParams = EnvParams()
    #aug_network_params: Dict[str, Any] = struct.field(default_factory=dict)

class NoisyStatelessMetaCartPole(environment.Environment):

    def __init__(self, **meta_kwargs):
        super().__init__()
        self.env = NoisyStatelessCartPole(max_steps_in_episode=200, noise_sigma=0.0)
        self.obs_shape = (7,)
        self.obs_aug = MetaAugNetwork(4)

        # Store meta_kwargs attributes for compatibility with algorithm code
        # These are used by _create_words() in ppo_gru/s5_in_context.py
        # but not actually used by this environment (it uses MetaAugNetwork)
        self.meta_depth = meta_kwargs.get('meta_depth', 1)
        self.meta_dim = meta_kwargs.get('meta_dim', 4)
        self.meta_max_depth = meta_kwargs.get('meta_max_depth', 2)
        self.meta_with_adjoint = meta_kwargs.get('meta_with_adjoint', False)
        if 'meta_const_aug' in meta_kwargs:
            self.meta_const_aug = meta_kwargs['meta_const_aug']

        # Store meta_kwargs for use in default_params
        self.meta_kwargs = meta_kwargs

    @property
    def default_params(self) -> MetaEnvParams:
        return MetaEnvParams(
            num_trials_per_episode=self.meta_kwargs.get('num_trials_per_episode', 16)
        )

    def step_env(
        self, key: chex.PRNGKey, state: MetaEnvState, action: int, params: MetaEnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        key, key_reset = jax.random.split(key)

        env_obs_st, env_state_st, reward, env_done, info = self.env.step_env(key, state.env_state, action, params.env_params)
        # env_obs_re, env_state_re = self.env.reset_env(key_reset, params.env_params)
        env_obs_re, env_state_re = state.init_obs, state.init_state

        env_state = jax.tree_map(
            lambda x, y: jax.lax.select(env_done, x, y), env_state_re, env_state_st
        )
        env_obs = jax.lax.select(env_done, env_obs_re, env_obs_st)
        env_obs = self.obs_aug.apply(state.obs_params, env_obs[None,:])[0]

        trial_num = state.trial_num + env_done
        total_steps = state.total_steps + 1
        done = trial_num >= params.num_trials_per_episode

        state = MetaEnvState(
            obs_params=state.obs_params,
            trial_num=trial_num,
            total_steps=total_steps,
            env_state=env_state,
            init_state=state.init_state,
            init_obs=state.init_obs,
            # Keep compatibility fields unchanged
            env_index=state.env_index,
            obs_words=state.obs_words,
        )

        obs = jnp.concatenate([env_obs, jnp.array([action, env_done, 0.0])])

        return (
            obs,
            state,
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: MetaEnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        env_key, obs_key = jax.random.split(key)
        env_obs, env_state = self.env.reset_env(env_key, params.env_params)

        # Initialize MetaAugNetwork with the provided parameters
        self.obs_aug = MetaAugNetwork()
        #self.obs_aug = create_meta_aug_network(**params.aug_network_params)
        obs_params = self.obs_aug.init(obs_key, env_obs[None,:])
        test = self.obs_aug.apply(obs_params, env_obs[None,:])[0]

        # Create a dummy obs_words array for compatibility (not actually used)
        # Shape should match what MetaEnvironment uses
        dummy_obs_words = jnp.zeros((1, self.env.observation_space(params.env_params).shape[0], self.meta_dim))

        state = MetaEnvState(
            obs_params=obs_params,
            trial_num=0,
            total_steps=0,
            env_state=env_state,
            init_state = env_state,
            init_obs = env_obs,
            # Add compatibility fields with dummy values
            env_index=0,
            obs_words=dummy_obs_words,
        )
        obs = jnp.concatenate([test, jnp.array([0.0, 0.0, 1.0])])

        return obs, state

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(
        self, params: Optional[MetaEnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.ones([4+3])
        return spaces.Box(-high, high, (7,), dtype=jnp.float32)
