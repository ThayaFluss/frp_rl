from .environments import (
    NoisyStatelessMetaCartPole,
    AutoencodeEasy,
    AutoencodeMedium,
    AutoencodeHard,
    BattleshipEasy,
    BattleshipMedium,
    BattleshipHard,
    StatelessCartPoleEasy,
    StatelessCartPoleMedium,
    StatelessCartPoleHard,
    NoisyStatelessCartPoleEasy,
    NoisyStatelessCartPoleMedium,
    NoisyStatelessCartPoleHard,
    ConcentrationEasy,
    ConcentrationMedium,
    ConcentrationHard,
    CountRecallEasy,
    CountRecallMedium,
    CountRecallHard,
    HigherLowerEasy,
    HigherLowerMedium,
    HigherLowerHard,
    MineSweeperEasy,
    MineSweeperMedium,
    MineSweeperHard,
    MultiarmedBanditEasy,
    MultiarmedBanditMedium,
    MultiarmedBanditHard,
    StatelessPendulumEasy,
    StatelessPendulumMedium,
    StatelessPendulumHard,
    NoisyStatelessPendulumEasy,
    NoisyStatelessPendulumMedium,
    NoisyStatelessPendulumHard,
    RepeatFirstEasy,
    RepeatFirstMedium,
    RepeatFirstHard,
    RepeatPreviousEasy,
    RepeatPreviousMedium,
    RepeatPreviousHard,
)
from .meta_environment import create_meta_environment

def make(env_id: str, **kwargs):
    """Create and register an environment by ID.

    This function provides a unified interface for creating all types of environments:
    - Meta-wrapped environments (MetaGymnax*, Meta*)
    - Direct gymnax environments (Gymnax*)
    - Special meta environments (NoisyStatelessMetaCartPole)
    - Standard popgym environments

    Special Environments:
        NoisyStatelessMetaCartPole (cartpole_origin): This is a complete meta-environment
        that uses a learnable MetaAugNetwork instead of FRP transformations. It cannot be
        wrapped with MetaEnvironment and should only be accessed via registration.py.

    Args:
        env_id: Environment identifier string
        **kwargs: Environment-specific keyword arguments (env_kwargs + meta_kwargs)

    Returns:
        Tuple of (environment, default_params)

    Examples:
        >>> env, params = make("MetaCartPole", meta_dim=64, meta_depth=2)
        >>> env, params = make("NoisyStatelessMetaCartPole", meta_dim=4)
        >>> env, params = make("GymnaxCartPole-v1")
    """
    # Case 1: Meta environment with gymnax base (MetaGymnax*)
    if env_id.startswith("MetaGymnax"):
        # [Duplicated]
        # Extract the base environment name
        base_env_name = env_id[10:]  # Remove "MetaGymnax" prefix
        print(f"Creating meta-gymnax environment: {base_env_name}")
        
        # Separate env_kwargs and meta_kwargs
        env_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('meta_')}
        
        # Keep the 'meta_' prefix for meta_kwargs
        meta_kwargs = {k: v for k, v in kwargs.items() if k.startswith('meta_')}
        
        # Use the existing create_meta_environment function with gymnax prefix
        env = create_meta_environment(f"gymnax_{base_env_name}", env_kwargs, meta_kwargs)
        return env, env.default_params
    
    # Case 2: Meta environment with popgym base (Meta*)
    elif env_id.startswith("Meta"):
        # Extract the base environment name
        base_env_name = env_id[4:].lower()
        print(f"Creating meta-popgym environment: {base_env_name}")
        
        # Separate env_kwargs and meta_kwargs
        env_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('meta_')}
        
        # Keep the 'meta_' prefix for meta_kwargs
        meta_kwargs = {k: v for k, v in kwargs.items() if k.startswith('meta_')}
        
        env = create_meta_environment(base_env_name, env_kwargs, meta_kwargs)
    
    # Case 3: Direct gymnax environment (Gymnax*)
    elif env_id.startswith("Gymnax"):
        base_env_name = env_id[6:]  # Remove "Gymnax" prefix
        print(f"Creating direct gymnax environment: {base_env_name}")
        
        try:
            from gymnax import make as gymnax_make
            from .wrappers import GymnaxFlagWrapper, GymnaxRewardNormWrapper
            
            # Pass kwargs to gymnax_make
            env, env_params = gymnax_make(base_env_name, **kwargs)
            
            # First apply reward normalization, then add flags
            env = GymnaxRewardNormWrapper(env, strategy='dynamic')
            wrapped_env = GymnaxFlagWrapper(env)
            
            return wrapped_env, env_params
        except Exception as e:
            raise ValueError(f"Error creating gymnax environment {base_env_name}: {e}")
    
    # Existing popgym environments
    elif env_id == "NoisyStatelessMetaCartPole":
        env = NoisyStatelessMetaCartPole(**kwargs)
    elif env_id == "AutoencodeEasy":
        env = AutoencodeEasy(**kwargs)
    elif env_id == "AutoencodeMedium":
        env = AutoencodeMedium(**kwargs)
    elif env_id == "AutoencodeHard":
        env = AutoencodeHard(**kwargs)
    elif env_id == "BattleshipEasy":
        env = BattleshipEasy(**kwargs)
    elif env_id == "BattleshipMedium":
        env = BattleshipMedium(**kwargs)
    elif env_id == "BattleshipHard":
        env = BattleshipHard(**kwargs)
    elif env_id == "StatelessCartPoleEasy":
        env = StatelessCartPoleEasy(**kwargs)
    elif env_id == "StatelessCartPoleMedium":
        env = StatelessCartPoleMedium(**kwargs)
    elif env_id == "StatelessCartPoleHard":
        env = StatelessCartPoleHard(**kwargs)
    elif env_id == "NoisyStatelessCartPoleEasy":
        env = NoisyStatelessCartPoleEasy(**kwargs)
    elif env_id == "NoisyStatelessCartPoleMedium":
        env = NoisyStatelessCartPoleMedium(**kwargs)
    elif env_id == "NoisyStatelessCartPoleHard":
        env = NoisyStatelessCartPoleHard(**kwargs)
    elif env_id == "ConcentrationEasy":
        env = ConcentrationEasy(**kwargs)
    elif env_id == "ConcentrationMedium":
        env = ConcentrationMedium(**kwargs)
    elif env_id == "ConcentrationHard":
        env = ConcentrationHard(**kwargs)
    elif env_id == "CountRecallEasy":
        env = CountRecallEasy(**kwargs)
    elif env_id == "CountRecallMedium":
        env = CountRecallMedium(**kwargs)
    elif env_id == "CountRecallHard":
        env = CountRecallHard(**kwargs)
    elif env_id == "HigherLowerEasy":
        env = HigherLowerEasy(**kwargs)
    elif env_id == "HigherLowerMedium":
        env = HigherLowerMedium(**kwargs)
    elif env_id == "HigherLowerHard":
        env = HigherLowerHard(**kwargs)
    elif env_id == "MinesweeperEasy":
        env = MineSweeperEasy(**kwargs)
    elif env_id == "MinesweeperMedium":
        env = MineSweeperMedium(**kwargs)
    elif env_id == "MinesweeperHard":
        env = MineSweeperHard(**kwargs)
    elif env_id == "MultiArmedBanditEasy":
        env = MultiarmedBanditEasy(**kwargs)
    elif env_id == "MultiArmedBanditMedium":
        env = MultiarmedBanditMedium(**kwargs)
    elif env_id == "MultiArmedBanditHard":
        env = MultiarmedBanditHard(**kwargs)
    elif env_id == "StatelessPendulumEasy":
        env = StatelessPendulumEasy(**kwargs)
    elif env_id == "StatelessPendulumMedium":
        env = StatelessPendulumMedium(**kwargs)
    elif env_id == "StatelessPendulumHard":
        env = StatelessPendulumHard(**kwargs)
    elif env_id == "NoisyStatelessPendulumEasy":
        env = NoisyStatelessPendulumEasy(**kwargs)
    elif env_id == "NoisyStatelessPendulumMedium":
        env = NoisyStatelessPendulumMedium(**kwargs)
    elif env_id == "NoisyStatelessPendulumHard":
        env = NoisyStatelessPendulumHard(**kwargs)
    elif env_id == "RepeatFirstEasy":
        env = RepeatFirstEasy(**kwargs)
    elif env_id == "RepeatFirstMedium":
        env = RepeatFirstMedium(**kwargs)
    elif env_id == "RepeatFirstHard":
        env = RepeatFirstHard(**kwargs)
    elif env_id == "RepeatPreviousEasy":
        env = RepeatPreviousEasy(**kwargs)
    elif env_id == "RepeatPreviousMedium":
        env = RepeatPreviousMedium(**kwargs)
    elif env_id == "RepeatPreviousHard":
        env = RepeatPreviousHard(**kwargs)
    else:
        
        raise ValueError(f"Environment ID: {env_id} is not registered.")

    return env, env.default_params
