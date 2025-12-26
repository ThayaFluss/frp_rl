"""
Checkpoint saving and loading utilities for model persistence.

This module provides functions for:
- Saving model checkpoints with metadata
- Saving training configurations
- Loading checkpoints for evaluation
- Managing experiment directories
"""

import os
import pickle
import time
from typing import Dict, Any, Optional, Tuple
import yaml
from flax.core import unfreeze


# Configuration keys to exclude from serialization
EXCLUDED_CONFIG_KEYS = ["ENV", "ENV_PARAMS", "EVAL_ENV", "EVAL_ENV_PARAMS"]
EXCLUDED_META_KEYS = ["meta_rng"]


def create_experiment_directory(base_dir: str = "exp") -> str:
    """
    Create a timestamped experiment directory.

    Args:
        base_dir: Base directory for experiments (default: "exp")

    Returns:
        str: Path to created experiment directory

    Example:
        >>> exp_dir = create_experiment_directory()
        >>> print(exp_dir)
        exp/20251226_114959
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_config_yaml(
    config: Dict[str, Any],
    arch: str,
    env_name: str,
    env_kwargs: Dict[str, Any],
    meta_kwargs: Dict[str, Any],
    norm_kwargs: Dict[str, Any],
    seed: int,
    exp_dir: str,
) -> str:
    """
    Save training configuration as YAML file.

    Args:
        config: Full training configuration dictionary
        arch: Architecture name ("gru" or "s5")
        env_name: Environment name
        env_kwargs: Environment-specific kwargs
        meta_kwargs: Meta-environment kwargs (meta_rng will be filtered out)
        norm_kwargs: Normalization kwargs
        seed: Random seed used for training
        exp_dir: Experiment directory path

    Returns:
        str: Path to saved config.yaml file

    Example:
        >>> config_path = save_config_yaml(
        ...     config=config,
        ...     arch="s5",
        ...     env_name="cartpole",
        ...     env_kwargs={},
        ...     meta_kwargs={"meta_depth": 4},
        ...     norm_kwargs={"strategy": "fixed"},
        ...     seed=42,
        ...     exp_dir="exp/20251226_114959"
        ... )
        >>> print(config_path)
        exp/20251226_114959/config.yaml
    """
    # Prepare config for YAML serialization
    config_to_save = _prepare_config_for_checkpoint(
        config, arch, env_name, env_kwargs, meta_kwargs, norm_kwargs, seed
    )

    # Save config.yaml
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_to_save, f, default_flow_style=False)

    print(f"Config saved to {config_path}")
    return config_path


def save_checkpoint(
    params,
    config: Dict[str, Any],
    arch: str,
    env_name: str,
    env_kwargs: Dict[str, Any],
    meta_kwargs: Dict[str, Any],
    norm_kwargs: Dict[str, Any],
    eval_metric: float,
    num_updates: int,
    exp_dir: str,
) -> str:
    """
    Save a model checkpoint with all necessary metadata.

    Args:
        params: Model parameters (will be unfrozen for serialization)
        config: Full training configuration dictionary
        arch: Architecture name ("gru" or "s5")
        env_name: Environment name
        env_kwargs: Environment-specific kwargs
        meta_kwargs: Meta-environment kwargs (meta_rng will be filtered out)
        norm_kwargs: Normalization kwargs
        eval_metric: Evaluation metric value (e.g., in_context_metric)
        num_updates: Number of training updates completed
        exp_dir: Experiment directory path (e.g., "exp/20251226_114959")

    Returns:
        str: Path to saved checkpoint file

    Example:
        >>> checkpoint_path = save_checkpoint(
        ...     params=train_state.params,
        ...     config=config,
        ...     arch="s5",
        ...     env_name="cartpole",
        ...     env_kwargs={},
        ...     meta_kwargs={"meta_depth": 4, "meta_rng": rng},
        ...     norm_kwargs={"strategy": "fixed"},
        ...     eval_metric=0.95,
        ...     num_updates=1000,
        ...     exp_dir="exp/20251226_114959"
        ... )
        >>> print(checkpoint_path)
        exp/20251226_114959/model_1000_iter.pkl
    """
    # Convert params to serializable format
    params_dict = unfreeze(params)

    # Filter meta_kwargs to remove PRNG key
    filtered_meta_kwargs = {k: v for k, v in meta_kwargs.items() if k not in EXCLUDED_META_KEYS}

    # Extract timestamp from exp_dir
    timestamp = os.path.basename(exp_dir)

    # Prepare checkpoint dictionary
    checkpoint = {
        "params": params_dict,
        "config": _prepare_config_for_checkpoint(config, arch, env_name, env_kwargs,
                                                   filtered_meta_kwargs, norm_kwargs),
        "arch": arch,
        "env_name": env_name,
        "env_kwargs": env_kwargs,
        "meta_kwargs": filtered_meta_kwargs,
        "norm_kwargs": norm_kwargs,
        "eval_metric": eval_metric,
        "num_updates": num_updates,
        "timestamp": timestamp,
    }

    # Save checkpoint file
    checkpoint_path = os.path.join(exp_dir, f"model_{num_updates}_iter.pkl")
    try:
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint to {checkpoint_path}")
        print(f"Error details: {e}")
        raise

    print(f"Model saved to {checkpoint_path}")
    print(f"Eval metric (in_context): {eval_metric}")

    return checkpoint_path


def load_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load a model checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint .pkl file

    Returns:
        Tuple of (checkpoint_dict, metadata_dict) where:
        - checkpoint_dict: Full checkpoint containing all data
        - metadata_dict: Extracted metadata for convenience

    Example:
        >>> checkpoint, metadata = load_checkpoint("exp/20251226_114959/model_1000_iter.pkl")
        >>> params = checkpoint["params"]
        >>> arch = metadata["arch"]
        >>> env_name = metadata["env_name"]
    """
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    # Extract commonly used metadata
    metadata = {
        "arch": checkpoint["arch"],
        "env_name": checkpoint["env_name"],
        "env_kwargs": checkpoint["env_kwargs"],
        "meta_kwargs": checkpoint["meta_kwargs"],
        "norm_kwargs": checkpoint["norm_kwargs"],
        "config": checkpoint["config"],
        "eval_metric": checkpoint.get("eval_metric"),
        "num_updates": checkpoint.get("num_updates"),
        "timestamp": checkpoint.get("timestamp"),
    }

    return checkpoint, metadata


def _prepare_config_for_checkpoint(
    config: Dict[str, Any],
    arch: str,
    env_name: str,
    env_kwargs: Dict[str, Any],
    meta_kwargs: Dict[str, Any],
    norm_kwargs: Dict[str, Any],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare config dictionary for serialization by filtering out non-serializable objects.

    This is an internal helper function that:
    1. Removes environment objects (ENV, EVAL_ENV, etc.)
    2. Adds metadata (arch, env_name, etc.)
    3. Optionally adds seed

    Args:
        config: Original config dictionary
        arch: Architecture name
        env_name: Environment name
        env_kwargs: Environment kwargs
        meta_kwargs: Meta-environment kwargs (already filtered)
        norm_kwargs: Normalization kwargs
        seed: Optional random seed

    Returns:
        Dict[str, Any]: Serializable config dictionary
    """
    # Filter out non-serializable keys
    config_to_save = {k: v for k, v in config.items() if k not in EXCLUDED_CONFIG_KEYS}

    # Add metadata
    config_to_save.update({
        "arch": arch,
        "env_name": env_name,
        "env_kwargs": env_kwargs,
        "meta_kwargs": meta_kwargs,
        "norm_kwargs": norm_kwargs,
    })

    # Add seed if provided
    if seed is not None:
        config_to_save["seed"] = seed

    return config_to_save
