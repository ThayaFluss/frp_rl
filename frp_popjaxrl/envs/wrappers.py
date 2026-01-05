import jax
import jax.numpy as jnp
from gymnax.wrappers.purerl import GymnaxWrapper, environment, Optional, partial, Tuple, chex, spaces, Union
from flax import struct
from typing import Dict, Any

class AliasPrevAction(GymnaxWrapper):
    """Adds a t0 flag and the last action."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        action_space = self._env.action_space(params)
        og_observation_space = self._env.observation_space(params)
        if type(action_space) == spaces.Discrete:
            low = jnp.concatenate([og_observation_space.low, jnp.array([0.0, 0.0])])
            high = jnp.concatenate([og_observation_space.high, jnp.array([action_space.n - 1, 1.0])])
        elif type(action_space) == spaces.Box:
            low = jnp.concatenate([og_observation_space.low, jnp.array([action_space.low]), jnp.array([0.0])])
            high = jnp.concatenate([og_observation_space.high, jnp.array([action_space.high]), jnp.array([1.0])])
        else:
            raise NotImplementedError
        return spaces.Box(
            low=low,
            high=high,
            shape=(self._env.observation_space(params).shape[-1]+2,), # NOTE: ASSUMES FLAT RIGHT NOW
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.concatenate([obs, jnp.array([0.0, 1.0])])
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        action_space = self._env.action_space(params)
        if isinstance(action_space, spaces.Discrete):
            obs = jnp.concatenate([obs, jnp.array([action, 0.0])])
        else:
            obs = jnp.concatenate([obs, action, jnp.array([0.0])])
        return obs, state, reward, done, info

class AliasPrevActionV2(GymnaxWrapper):
    """Adds a t0 flag and the last action."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        action_space = self._env.action_space(params)
        og_observation_space = self._env.observation_space(params)
        if type(action_space) == spaces.Discrete:
            low = jnp.concatenate([og_observation_space.low, jnp.zeros((action_space.n+1,))])
            high = jnp.concatenate([og_observation_space.high, jnp.ones((action_space.n+1,))])
            return spaces.Box(
                low=low,
                high=high,
                shape=(self._env.observation_space(params).shape[-1]+action_space.n+1,), # NOTE: ASSUMES FLAT RIGHT NOW
                dtype=self._env.observation_space(params).dtype,
            )
        elif type(action_space) == spaces.Box:
            low = jnp.concatenate([og_observation_space.low, jnp.array([action_space.low]), jnp.array([0.0])])
            high = jnp.concatenate([og_observation_space.high, jnp.array([action_space.high]), jnp.array([1.0])])
            return spaces.Box(
                low=low,
                high=high,
                shape=(self._env.observation_space(params).shape[-1]+2,), # NOTE: ASSUMES FLAT RIGHT NOW
                dtype=self._env.observation_space(params).dtype,
            )
        else:
            raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        action_space = self._env.action_space(params)
        obs, state = self._env.reset(key, params)
        if isinstance(action_space, spaces.Box):
            obs = jnp.concatenate([obs, jnp.array([0.0, 1.0])])
        elif isinstance(action_space, spaces.Discrete):
            obs = jnp.concatenate([obs, jnp.zeros((action_space.n,)), jnp.array([1.0])])
        else:
            raise NotImplementedError
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        action_space = self._env.action_space(params)
        if isinstance(action_space, spaces.Discrete):
            # obs = jnp.concatenate([obs, jnp.array([action, 0.0])])
            action_in = jnp.zeros((action_space.n,))
            action_in = action_in.at[action].set(1.0)
            obs = jnp.concatenate([obs, action_in, jnp.array([0.0])])
        else:
            obs = jnp.concatenate([obs, action, jnp.array([0.0])])
        return obs, state, reward, done, info

class GymnaxFlagWrapper(GymnaxWrapper):
    """Wrapper for gymnax environments that adds flag dimensions.
    
    This wrapper adds 3 flag dimensions to the observation:
    - action: The last action taken
    - done: Whether the episode is done
    - reset: Whether this is the first observation after a reset
    
    This makes the observation space compatible with meta environments.
    """

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        og_observation_space = self._env.observation_space(params)
        
        # Add 3 flag dimensions: [action, done, reset]
        low = jnp.concatenate([og_observation_space.low, jnp.zeros(3)])
        high = jnp.concatenate([og_observation_space.high, jnp.ones(3)])
        
        return spaces.Box(
            low=low,
            high=high,
            shape=(og_observation_space.shape[0] + 3,),
            dtype=og_observation_space.dtype,
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        """Reset the environment and add flag dimensions."""
        obs, state = self._env.reset(key, params)
        # Add flag dimensions: [action, done, reset]
        # action=0, done=0, reset=1
        obs = jnp.concatenate([obs, jnp.array([0.0, 0.0, 1.0])])
        return obs, state
        

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        """Step the environment and add flag dimensions."""
        obs, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        # Add flag dimensions: [action, done, reset]
        # Convert action to float if it's an integer
        action_value = action[0] if isinstance(action, jnp.ndarray) and len(action.shape) > 0 else action
        obs = jnp.concatenate([obs, jnp.array([jnp.float32(action_value), jnp.float32(done), 0.0])])
        return obs, state, reward, done, info
@struct.dataclass
class GymnaxRewardNormState:
    """State for the GymnaxRewardNormWrapper.
    
    This state includes the original environment state plus additional fields
    for tracking reward normalization statistics.
    """
    env_state: environment.EnvState
    running_mean: float  # Running mean of discounted returns
    running_var: float   # Running variance of discounted returns
    discounted_return: float  # Current episode's discounted return accumulator

@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int

class GymnaxRewardNormWrapper(GymnaxWrapper):
    """Wrapper for gymnax environments that normalizes rewards.
    
    This wrapper applies reward normalization in a variety of ways:
    - 'dynamic': Learns a running variance of returns and normalizes reward accordingly (default)
    - 'minmax': Scales rewards to a range [-1, 1] and divides by max_steps
    - 'fixed': Maps rewards to fixed values (e.g., -1 for failure, +1/max_steps for success)
    - 'custom': Uses environment-specific logic
    
    The 'dynamic' strategy is the most general and recommended approach, as it adapts
    to any environment without requiring prior knowledge of reward ranges.
    """

    def __init__(
        self,
        env: environment.Environment,
        strategy: str = 'dynamic',
        reward_range: Optional[Tuple[float, float]] = None,
        max_steps: Optional[int] = None,
        discount_factor: float = 0.99,
        update_rate: float = 0.01,
        init_var: float = 1.0,
        init_mean: float = 0.0,
        eps: float = 1e-8,
        update_stats: bool = True
    ):
        """Initialize the reward normalization wrapper.
        
        Args:
            env: The environment to wrap.
            strategy: Normalization strategy ('dynamic', 'minmax', 'fixed', or 'custom').
            reward_range: (min, max) for 'minmax' scaling. Defaults to (-1, 1) if None.
            max_steps: Used for step-based scaling in 'minmax' and 'fixed'.
            discount_factor: Gamma for discounted return in 'dynamic' mode.
            update_rate: Learning rate for updating running mean/var in 'dynamic' mode.
            init_var: Initial variance for 'dynamic' mode, to avoid division by zero early on.
            init_mean: Initial mean for 'dynamic' mode.
            eps: Small constant for numeric stability in division.
            update_stats: Whether to update running statistics (set to False during evaluation).
        """
        super().__init__(env)
        
        self.strategy = strategy
        self.reward_range = reward_range or (-16.2736044, 0) # pendulum default

        self.max_steps = max_steps or 200 # cartpole default
        
        # For dynamic normalization
        self.discount_factor = discount_factor
        self.update_rate = update_rate
        self.eps = eps
        self.init_var = init_var
        self.init_mean = init_mean
        self.update_stats = update_stats

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, GymnaxRewardNormState]:
        """Reset the environment and initialize normalization state."""
        obs, env_state = self._env.reset(key, params)
        
        # Initialize the normalization state
        norm_state = GymnaxRewardNormState(
            env_state=env_state,
            running_mean=self.init_mean,
            running_var=self.init_var,
            discounted_return=0.0
        )
        
        return obs, norm_state
        
    @partial(jax.jit, static_argnums=(0,))
    def reset_env(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, GymnaxRewardNormState]:
        """Environment-specific reset with normalization state initialization.
        
        This method is called by MetaEnvironment.reset_env and needs to be implemented
        to ensure reward normalization works with MetaEnvironment.
        """
        obs, env_state = self._env.reset_env(key, params)
        
        # Initialize the normalization state
        norm_state = GymnaxRewardNormState(
            env_state=env_state,
            running_mean=self.init_mean,
            running_var=self.init_var,
            discounted_return=0.0
        )
        
        return obs, norm_state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: GymnaxRewardNormState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, GymnaxRewardNormState, float, bool, dict]:
        """Perform a step in the environment and normalize the reward."""
        # Step the underlying environment
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        
        # Apply the chosen normalization method
        normalized_reward, new_norm_state = self._normalize_reward(reward, state, done)
        
        # Update the environment state in the normalization state
        new_norm_state = GymnaxRewardNormState(
            env_state=env_state,
            running_mean=new_norm_state.running_mean,
            running_var=new_norm_state.running_var,
            discounted_return=new_norm_state.discounted_return
        )
        
        # Add original reward to info for logging
        info = dict(info)
        info["original_reward"] = reward
        
        return obs, new_norm_state, normalized_reward, done, info
        
    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: GymnaxRewardNormState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, GymnaxRewardNormState, float, bool, dict]:
        """Environment-specific step transition with reward normalization.
        
        This method is called by MetaEnvironment.step_env and needs to be implemented
        to ensure reward normalization works with MetaEnvironment.
        """
        # Step the underlying environment
        obs, env_state, reward, done, info = self._env.step_env(
            key, state.env_state, action, params
        )
        
        # Apply the chosen normalization method
        normalized_reward, new_norm_state = self._normalize_reward(reward, state, done)
        
        # Update the environment state in the normalization state
        new_norm_state = GymnaxRewardNormState(
            env_state=env_state,
            running_mean=new_norm_state.running_mean,
            running_var=new_norm_state.running_var,
            discounted_return=new_norm_state.discounted_return
        )
        
        # Add original reward to info for logging
        info = dict(info)
        info["original_reward"] = reward
        
        return obs, new_norm_state, normalized_reward, done, info

    def _normalize_reward(
        self, reward: float, state: GymnaxRewardNormState, done: bool
    ) -> Tuple[float, GymnaxRewardNormState]:
        """Normalize the reward based on the selected strategy."""
        if self.strategy == 'minmax':
            # Similar to Pendulum normalization
            low, high = self.reward_range
            shifted = reward + (high - low) / 2
            scaled = shifted / ((high - low) / 2)
            normalized_reward = scaled / self.max_steps
            
            # State doesn't change for minmax strategy
            new_state = state
            
        elif self.strategy == 'fixed':
            # Similar to CartPole normalization
            normalized_reward = jnp.where(
                jnp.isclose(reward, 0.0), 
                -1.0, 
                1.0 / self.max_steps
            )
            
            # State doesn't change for fixed strategy
            new_state = state
            
        elif self.strategy == 'custom':
            # Environment-specific normalization (user-provided logic)
            normalized_reward = reward
            
            # State doesn't change for custom strategy
            new_state = state
            
        elif self.strategy == 'dynamic':
            # Update discounted return
            new_discounted_return = (
                self.discount_factor * state.discounted_return * (1.0 - done) + reward
            )
            
            # Update running statistics if enabled
            if self.update_stats:
                # Update running mean first
                mean_delta = new_discounted_return - state.running_mean
                new_running_mean = state.running_mean + self.update_rate * mean_delta
                
                # Update running variance using the squared difference from the mean
                # This is a more accurate way to update variance
                delta_squared = (new_discounted_return - new_running_mean) ** 2
                new_running_var = state.running_var + self.update_rate * (delta_squared - state.running_var)
            else:
                # Don't update statistics during evaluation
                new_running_var = state.running_var
                new_running_mean = state.running_mean
            
            # Scale the immediate reward by the sqrt of the running var
            # We do not shift reward by mean to preserve sign
            normalized_reward = reward / jnp.sqrt(new_running_var + self.eps)
        
            # For compatibility with fixed/minmax.
            normalized_reward /= self.max_steps
        
            # Create new state with updated statistics
            new_state = GymnaxRewardNormState(
                env_state=state.env_state,
                running_mean=new_running_mean,
                running_var=new_running_var,
                discounted_return=new_discounted_return
            )
        
        else:
            # Default: no normalization
            normalized_reward = reward
            new_state = state
            
        return normalized_reward, new_state

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        """_summary_
        Args:
            key (chex.PRNGKey): _description_
            state (environment.EnvState): _description_
            action (Union[int, float]): _description_
            params (Optional[environment.EnvParams], optional): _description_. Defaults to None.

        Returns:
            Tuple[chex.Array, environment.EnvState, float, bool, dict]: _description_

        Example:
            Episode 1 starts: returned_episode_returns = 0
            Episode 1 in progress: episode_returns increases, returned_episode_returns stays 0
            Episode 1 completes (total return 100): returned_episode_returns = 100
            Episode 2 starts: returned_episode_returns stays 100, episode_returns resets to 0
            Episode 2 in progress: episode_returns increases, returned_episode_returns stays 100
            Episode 2 completes (total return 150): returned_episode_returns = 150        
            
        """
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state = env_state,
            episode_returns = new_episode_return * (1 - done),
            episode_lengths = new_episode_length * (1 - done),
            returned_episode_returns = state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths = state.returned_episode_lengths * (1 - done) + new_episode_length * done,
            timestep = state.timestep + 1,
        )
        # info["returned_episode_returns"] = state.returned_episode_returns
        # info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        info["return_info"] = jnp.stack([state.timestep, state.returned_episode_returns])
        # info["timestep"] = state.timestep
        return obs, state, reward, done, info
