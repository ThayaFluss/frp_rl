import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from flax import struct
import chex
from typing import Tuple, Optional, Any, Dict, List
from frp.orthogonal import create_orthogonal_matrices, create_words, random_choice, detect_identity_matrices, get_weight_matrix
from flax.core import freeze

@struct.dataclass
class MetaEnvState:
    env_index: int
    obs_words: chex.Array
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

        # Meta-learning specific parameters
        self.meta_dim = meta_kwargs.get('meta_dim', 4)
        self.meta_truncate_aug = meta_kwargs.get("meta_truncate_aug", 0)
        # For FRP
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

        # Set up evaluation method if specified
        if self.meta_const_aug == "padding":
            self.eval_weight = jnp.eye(self.meta_dim)[:self.input_dim,:]       
        elif self.meta_const_aug == "tiling":
            from .environments.metaaug.padding import create_periodic_weight
            self.eval_weight = create_periodic_weight(input_dim=self.input_dim, output_dim=self.meta_dim, period=round(self.meta_dim/2))
        else:
            ...
            #do not prepare the self.eval weight for reducing memory.

        ### obs_shape is output dimension of this class
        if self.meta_const_aug == "identity":
            self.aug_output_dim = self.input_dim
        elif self.meta_truncate_aug ==1:
            self.aug_output_dim = self.input_dim
        else:
            self.aug_output_dim = self.meta_dim

        self.obs_shape = (self.aug_output_dim + 3,)

        # Initialize meta-augmentation
        self._initialize_meta_augmentation()

    def _initialize_meta_augmentation(self):
        matrices = create_orthogonal_matrices(
            self.rng, 
            self.meta_depth, 
            size=self.meta_dim, 
            max_depth=self.meta_max_depth, 
            with_adjoint=self.meta_with_adjoint
        )
        self.words = create_words(
            matrices, 
            self.meta_depth, 
            out_size=self.meta_dim, 
            max_depth=self.meta_max_depth
        )
        # Exclude identity matrices before cutoff
        self.exclude = detect_identity_matrices(self.words)
        if self.meta_truncate_aug==1:
            ### Truncate output of augmentation
            self.words = self.words[:, :self.input_dim, :self.aug_output_dim]
        else:
            self.words = self.words[:, :self.input_dim, ]
        
        self.total_words = self.words.shape[0]
        #self.obs_aug = MetaAugNetwork(out_size=self.meta_dim, words=self.words)

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

        if self.meta_const_aug == "identity":
            # For identity, we directly use the observation vector without transformation
            env_obs = env_obs
        elif self.meta_const_aug in ["padding", "tiling"]:
            env_obs = (env_obs[None,:] @ self.eval_weight)[0]
        else:
            # Default behaviour. Choose a word from obs_words.
            #old_weight = jnp.sqrt(2)*jax.lax.dynamic_slice(state.obs_words, (state.env_index, 0, 0), (1, self.input_dim, self.aug_output_dim))[0]
            weight = get_weight_matrix(state.obs_words, state.env_index, self.input_dim, self.aug_output_dim)
            
            # Use the weight from our new function regardless of the match result
            env_obs = (env_obs[None,:] @ weight)[0]

        # trail num increases when env has done 
        trial_num = state.trial_num + env_done
        total_steps = state.total_steps + 1
        done = trial_num >= params.num_trials_per_episode

        state = MetaEnvState(
            env_index=state.env_index,
            obs_words=state.obs_words,
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
    ) -> Tuple[chex.Array, MetaEnvState]:
        env_key, obs_key, index_key = jax.random.split(key, 3)
        env_obs, env_state = self.env.reset_env(env_key, params.env_params)

        if len(self.exclude) == 0:
            env_index = jax.random.randint(index_key, (), 0, self.total_words).astype(jnp.int32)
        else:
            env_index = random_choice(index_key, 
                                  total_words=self.total_words, 
                                  exclude=self.exclude).astype(jnp.int32)

        if self.meta_const_aug == "identity":
            # For identity, we directly use the observation vector without transformation
            augmented_obs = env_obs
        elif self.meta_const_aug in ["padding", "tiling"]:
            augmented_obs = (env_obs[None,:] @ self.eval_weight)[0]
        else:
            weight = get_weight_matrix(self.words, env_index, self.input_dim, self.aug_output_dim)
            
            # Use the weight from our new function regardless of the match result
            augmented_obs = (env_obs[None,:] @ weight)[0]

        state = MetaEnvState(
            env_index=env_index,
            obs_words=self.words,
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
    try:
        from gymnax import make as gymnax_make
        from .wrappers import GymnaxRewardNormWrapper
        
        # Format the environment name to match gymnax's expected format
        # Convert names like "cartpole" to "CartPole-v1"
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
            # Add appropriate suffix for environments without one
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
                # Get normalization parameters from norm_kwargs if provided, otherwise use defaults
                if norm_kwargs is not None:
                    strategy = norm_kwargs.get('strategy', 'dynamic')
                    max_steps = norm_kwargs.get('max_steps', 200)
                else:
                    strategy = 'dynamic'
                    max_steps = 200
                super().__init__(env.__class__(**kwargs), strategy=strategy, max_steps=max_steps)

        # Return the meta environment with the normalized env
        return MetaEnvironment(NormalizedEnv, env_kwargs, meta_kwargs)
    except Exception as e:
        raise ValueError(f"Error creating gymnax environment {env_name}: {e}")

def create_meta_environment(env_name: str, env_kwargs: Dict[str, Any], meta_kwargs: Dict[str, Any], norm_kwargs: Dict[str, Any] = None):
    # Handle popgym environments
    if env_name == "cartpole":
        from .environments.popgym_cartpole import NoisyStatelessCartPole
        return MetaEnvironment(NoisyStatelessCartPole, env_kwargs, meta_kwargs)
    if env_name == "cartpole_origin":
        from .environments.meta_cartpole_origin_like import NoisyStatelessMetaCartPole
        return NoisyStatelessMetaCartPole(**meta_kwargs)
    if env_name == "s_cartpole_hard":
        from .environments.popgym_cartpole import StatelessCartPoleHard
        return MetaEnvironment(StatelessCartPoleHard, env_kwargs, meta_kwargs)
    if env_name == "ns_cartpole_hard":
        from .environments.popgym_cartpole import NoisyStatelessCartPoleHard
        return MetaEnvironment(NoisyStatelessCartPoleHard, env_kwargs, meta_kwargs)


    elif env_name == "minesweeper":
        from .environments.popgym_minesweeper import MineSweeper
        return MetaEnvironment(MineSweeper, env_kwargs, meta_kwargs)
    elif env_name == "minesweeper_hard":
        from .environments.popgym_minesweeper import MineSweeperHard
        return MetaEnvironment(MineSweeperHard, env_kwargs, meta_kwargs)

    elif env_name == "multiarmedbandit":
        from .environments.popgym_multiarmedbandit import MultiarmedBandit
        return MetaEnvironment(MultiarmedBandit, env_kwargs, meta_kwargs)
    elif env_name == "higherlower":
        from .environments.popgym_higherlower import HigherLower
        return MetaEnvironment(HigherLower, env_kwargs, meta_kwargs)
    elif env_name == "higherlower_easy":
        from .environments.popgym_higherlower import HigherLowerEasy
        return MetaEnvironment(HigherLowerEasy, env_kwargs, meta_kwargs)
    elif env_name == "higherlower_medium":
        from .environments.popgym_higherlower import HigherLowerMedium
        return MetaEnvironment(HigherLowerMedium, env_kwargs, meta_kwargs)
    elif env_name == "higherlower_hard":
        from .environments.popgym_higherlower import HigherLowerHard
        return MetaEnvironment(HigherLowerHard, env_kwargs, meta_kwargs)
    elif env_name == "pendulum":
        from .environments.popgym_pendulum import NoisyStatelessPendulum
        return MetaEnvironment(NoisyStatelessPendulum, env_kwargs, meta_kwargs)
    elif env_name == "pendulum_easy":
        from .environments.popgym_pendulum import NoisyStatelessPendulumEasy
        return MetaEnvironment(NoisyStatelessPendulumEasy, env_kwargs, meta_kwargs)
    elif env_name == "pendulum_medium":
        from .environments.popgym_pendulum import NoisyStatelessPendulumMedium
        return MetaEnvironment(NoisyStatelessPendulumMedium, env_kwargs, meta_kwargs)
    elif env_name == "pendulum_hard":
        from .environments.popgym_pendulum import NoisyStatelessPendulumHard
        return MetaEnvironment(NoisyStatelessPendulumHard, env_kwargs, meta_kwargs)
    elif env_name == "autoencode":
        from .environments.popgym_autoencode import Autoencode
        return MetaEnvironment(Autoencode, env_kwargs, meta_kwargs)
    elif env_name == "battleship":
        from .environments.popgym_battleship import Battleship
        return MetaEnvironment(Battleship, env_kwargs, meta_kwargs)
    elif env_name == "concentration":
        from .environments.popgym_concentration import Concentration
        return MetaEnvironment(Concentration, env_kwargs, meta_kwargs)
    elif env_name == "count_recall":
        from .environments.popgym_count_recall import CountRecall
        return MetaEnvironment(CountRecall, env_kwargs, meta_kwargs)
    elif env_name == "repeat_first":
        from .environments.popgym_repeat_first import RepeatFirst
        return MetaEnvironment(RepeatFirst, env_kwargs, meta_kwargs)
    elif env_name == "repeat_first_hard":
        from .environments.popgym_repeat_first import RepeatFirstHard
        return MetaEnvironment(RepeatFirstHard, env_kwargs, meta_kwargs)
    elif env_name == "repeat_previous_hard":
        from .environments.popgym_repeat_previous import RepeatPreviousHard
        return MetaEnvironment(RepeatPreviousHard, env_kwargs, meta_kwargs)


    # Check if it's a gymnax environment
    elif env_name.startswith("gymnax_"):
        # [Duplicated]
        # Extract the base environment name
        base_env_name = env_name[7:]  # Remove "gymnax_" prefix
        return create_gymnax_environment(base_env_name, env_kwargs, meta_kwargs, norm_kwargs)

    else:
        raise ValueError(f"Unknown environment: {env_name}")
