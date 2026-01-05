"""
Environment factory functions shared across all meta_environment implementations.

This module provides common factory functions for creating environments that are
used by all three modes (separated, lazy, legacy).

IMPORTANT: This factory only handles environments that need to be wrapped with
MetaEnvironment/MetaEnvironmentLazy. Special meta-environments like
NoisyStatelessMetaCartPole (cartpole_origin) that are already complete
meta-environments should be accessed via registration.py instead.
"""

from typing import Dict, Any, Type


def _get_environment_mapping():
    """Get mapping from environment names to their classes.

    Returns a dict mapping environment names to tuples of:
    (module_path, class_name)

    Note: All environments listed here are wrapped with MetaEnvironment/MetaEnvironmentLazy.
    Special environments like NoisyStatelessMetaCartPole (cartpole_origin) that are
    already complete meta-environments should be accessed via registration.py instead.
    """
    return {
        # CartPole variants
        "cartpole": (".environments.popgym_cartpole", "NoisyStatelessCartPole"),
        "s_cartpole_hard": (".environments.popgym_cartpole", "StatelessCartPoleHard"),
        "ns_cartpole_hard": (".environments.popgym_cartpole", "NoisyStatelessCartPoleHard"),

        # MineSweeper variants
        "minesweeper": (".environments.popgym_minesweeper", "MineSweeper"),
        "minesweeper_hard": (".environments.popgym_minesweeper", "MineSweeperHard"),

        # Other environments
        "multiarmedbandit": (".environments.popgym_multiarmedbandit", "MultiarmedBandit"),

        # HigherLower variants
        "higherlower": (".environments.popgym_higherlower", "HigherLower"),
        "higherlower_easy": (".environments.popgym_higherlower", "HigherLowerEasy"),
        "higherlower_medium": (".environments.popgym_higherlower", "HigherLowerMedium"),
        "higherlower_hard": (".environments.popgym_higherlower", "HigherLowerHard"),

        # Pendulum variants
        "pendulum": (".environments.popgym_pendulum", "NoisyStatelessPendulum"),
        "pendulum_easy": (".environments.popgym_pendulum", "NoisyStatelessPendulumEasy"),
        "pendulum_medium": (".environments.popgym_pendulum", "NoisyStatelessPendulumMedium"),
        "pendulum_hard": (".environments.popgym_pendulum", "NoisyStatelessPendulumHard"),

        # Memory/recall environments
        "autoencode": (".environments.popgym_autoencode", "Autoencode"),
        "battleship": (".environments.popgym_battleship", "Battleship"),
        "concentration": (".environments.popgym_concentration", "Concentration"),
        "count_recall": (".environments.popgym_count_recall", "CountRecall"),
        "repeat_first": (".environments.popgym_repeat_first", "RepeatFirst"),
        "repeat_first_hard": (".environments.popgym_repeat_first", "RepeatFirstHard"),
        "repeat_previous_hard": (".environments.popgym_repeat_previous", "RepeatPreviousHard"),
    }


def _format_gymnax_env_name(env_name: str) -> str:
    """Format environment name to match gymnax's expected format.

    Args:
        env_name: Base environment name (e.g., "cartpole", "pendulum")

    Returns:
        Formatted name for gymnax (e.g., "CartPole-v1", "Pendulum-v1")
    """
    # Convert common names to gymnax format
    if env_name.lower() == "cartpole":
        return "CartPole-v1"
    elif env_name.lower() == "pendulum":
        return "Pendulum-v1"
    elif env_name.lower() == "acrobot":
        return "Acrobot-v1"
    elif env_name.lower() == "mountaincar":
        return "MountainCar-v0"
    elif env_name.lower() == "mountaincarcontinuous":
        return "MountainCarContinuous-v0"
    elif "-" not in env_name and not any(suffix in env_name.lower() for suffix in ["minatar", "bsuite", "misc"]):
        # Add appropriate suffix for environments without one
        if env_name.lower() in ["asterix", "breakout", "freeway", "seaquest", "spaceinvaders"]:
            return f"{env_name.capitalize()}-MinAtar"
        elif env_name.lower() in ["catch", "deepsea", "memorychain", "umbrellachain",
                                 "discountingchain", "mnistbandit", "simplebandit"]:
            return f"{env_name.capitalize()}-bsuite"
        elif env_name.lower() in ["fourrooms", "metamaze", "pointrobot", "bernoullibandit",
                                 "gaussianbandit", "reacher", "swimmer", "pong"]:
            return f"{env_name.capitalize()}-misc"

    # If no special handling needed, return as-is
    return env_name


def create_gymnax_environment_internal(
    env_name: str,
    env_kwargs: Dict[str, Any],
    meta_kwargs: Dict[str, Any],
    norm_kwargs: Dict[str, Any],
    meta_env_class: Type
):
    """Create a gymnax environment wrapped in a MetaEnvironment class.

    This is an internal function used by all meta_environment implementations.

    Args:
        env_name: Name of the gymnax environment
        env_kwargs: Keyword arguments for the base environment
        meta_kwargs: Keyword arguments for the meta environment wrapper
        norm_kwargs: Keyword arguments for reward normalization
        meta_env_class: The MetaEnvironment class to use for wrapping
                       (MetaEnvironment, MetaEnvironmentLazy, or MetaEnvironmentLegacy)

    Returns:
        Wrapped meta environment instance

    Raises:
        ValueError: If environment creation fails
    """
    try:
        from gymnax import make as gymnax_make
        from .wrappers import GymnaxRewardNormWrapper

        # Format the environment name to match gymnax's expected format
        formatted_name = _format_gymnax_env_name(env_name)

        # Get the base environment
        env, _ = gymnax_make(formatted_name)

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
        return meta_env_class(NormalizedEnv, env_kwargs, meta_kwargs)
    except Exception as e:
        raise ValueError(f"Error creating gymnax environment {env_name}: {e}")


def create_meta_environment_internal(
    env_name: str,
    env_kwargs: Dict[str, Any],
    meta_kwargs: Dict[str, Any],
    norm_kwargs: Dict[str, Any],
    meta_env_class: Type
    ):
    """Create a meta environment using the appropriate wrapper class.

    This is an internal function used by all meta_environment implementations.

    Args:
        env_name: Name of the environment
        env_kwargs: Keyword arguments for the base environment
        meta_kwargs: Keyword arguments for the meta environment wrapper
        norm_kwargs: Keyword arguments for reward normalization (for gymnax envs)
        meta_env_class: The MetaEnvironment class to use for wrapping
        lazy_mode: Not used anymore, kept for API compatibility

    Returns:
        Meta environment instance

    Raises:
        ValueError: If environment name is unknown

    Note:
        Special environments like NoisyStatelessMetaCartPole (cartpole_origin) that are
        already complete meta-environments should be accessed via registration.py instead.
    """
    # Check if it's a gymnax environment
    if env_name.startswith("gymnax_"):
        # Extract the base environment name
        base_env_name = env_name[7:]  # Remove "gymnax_" prefix
        return create_gymnax_environment_internal(
            base_env_name, env_kwargs, meta_kwargs, norm_kwargs, meta_env_class
        )

    # Get environment mapping
    env_mapping = _get_environment_mapping()

    if env_name not in env_mapping:
        raise ValueError(f"Unknown environment: {env_name}")

    module_path, class_name = env_mapping[env_name]

    # Import class and wrap with meta_env_class
    from importlib import import_module
    module = import_module(module_path, package='envs')
    env_class = getattr(module, class_name)
    return meta_env_class(env_class, env_kwargs, meta_kwargs)
