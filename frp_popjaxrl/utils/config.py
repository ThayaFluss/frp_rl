"""Configuration file loading and merging utilities (flat format, W&B sweep compatible).

This module provides utilities for loading YAML configuration files and merging them
with CLI arguments. The flat YAML format uses parameter names that match CLI arguments
exactly, making it compatible with W&B sweep configuration.

Example YAML config:
    # configs/meta_cartpole_s5.yaml
    env: cartpole
    arch: s5
    dim: 128
    lr: 5.0e-5

Example usage:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--env", type=str, default="cartpole")
    # ... other arguments ...
    args = parser.parse_args()
    args = load_and_merge_config(parser, args)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from envs.wrappers import AliasPrevActionV2


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration values.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return config or {}


def get_explicitly_set_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> set[str]:
    """Detect CLI arguments that were explicitly specified by the user.

    This function compares the actual argument values with their defaults
    to determine which arguments were explicitly provided on the command line.

    Limitation: If a user explicitly specifies a value that matches the default,
    it will not be detected as explicitly set.

    Args:
        parser: The ArgumentParser instance used to parse arguments.
        args: The parsed arguments namespace.

    Returns:
        Set of argument names that were explicitly specified.
    """
    defaults = vars(parser.parse_args([]))
    actual = vars(args)
    explicitly_set = set()

    for key, default_value in defaults.items():
        if key == "config":
            continue
        actual_value = actual.get(key)
        if actual_value != default_value:
            explicitly_set.add(key)

    return explicitly_set


def merge_configs(
    yaml_config: dict[str, Any],
    cli_args: argparse.Namespace,
    explicitly_set_args: set[str],
) -> argparse.Namespace:
    """Merge YAML configuration with CLI arguments (CLI takes priority).

    Priority order: CLI explicit > YAML config > argparse defaults

    Args:
        yaml_config: Configuration dictionary loaded from YAML.
        cli_args: Parsed CLI arguments namespace.
        explicitly_set_args: Set of argument names explicitly specified on CLI.

    Returns:
        New Namespace with merged configuration values.
    """
    merged = argparse.Namespace(**vars(cli_args))

    for key, yaml_value in yaml_config.items():
        # Skip keys that don't exist as CLI arguments
        if not hasattr(merged, key):
            continue
        # Skip keys that were explicitly specified on CLI
        if key in explicitly_set_args:
            continue
        # Handle null values in YAML (convert to None)
        if yaml_value is None:
            setattr(merged, key, None)
        else:
            setattr(merged, key, yaml_value)

    return merged


def load_and_merge_config(
    parser: argparse.ArgumentParser,
    cli_args: argparse.Namespace,
) -> argparse.Namespace:
    """Main entry point: merge YAML config with CLI args if --config is specified.

    This function checks if the --config argument was provided. If so, it loads
    the YAML configuration and merges it with CLI arguments, with CLI arguments
    taking priority over YAML values.

    Args:
        parser: ArgumentParser instance (used to detect explicitly set args).
        cli_args: Parsed arguments from parse_args() (should include config attribute).

    Returns:
        Namespace with merged configuration values. If --config was not specified,
        returns the original cli_args unchanged.

    Example:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default=None)
        parser.add_argument("--env", type=str, default="cartpole")
        args = parser.parse_args()
        args = load_and_merge_config(parser, args)
    """
    if not hasattr(cli_args, "config") or cli_args.config is None:
        return cli_args

    yaml_config = load_yaml_config(cli_args.config)
    explicitly_set = get_explicitly_set_args(parser, cli_args)

    return merge_configs(yaml_config, cli_args, explicitly_set)


def load_config_with_debug_support(
    parser: argparse.ArgumentParser,
    cli_args: argparse.Namespace,
    debug_config_path: str = "configs/debug.yaml",
) -> argparse.Namespace:
    """Load config with automatic debug.yaml support.

    When debug=1 is set and no --config is specified, this function automatically
    loads the debug configuration file. This eliminates the need for hardcoded
    debug parameter values in training scripts.

    Args:
        parser: ArgumentParser instance (used to detect explicitly set args).
        cli_args: Parsed arguments from parse_args().
        debug_config_path: Path to debug config file (relative to script location).

    Returns:
        Namespace with merged configuration values.

    Example:
        args = parser.parse_args()
        args = load_config_with_debug_support(parser, args)
        # If --debug 1 was passed without --config, debug.yaml values are applied
    """
    # If debug=1 and no config specified, auto-load debug.yaml
    if getattr(cli_args, "debug", 0) == 1 and cli_args.config is None:
        # Resolve debug config path relative to the script's directory
        script_dir = Path(__file__).parent.parent
        resolved_path = script_dir / debug_config_path
        if resolved_path.exists():
            cli_args.config = str(resolved_path)

    return load_and_merge_config(parser, cli_args)


def build_training_config(
    args: argparse.Namespace,
    env: AliasPrevActionV2,
    env_params: Any,
    eval_env: AliasPrevActionV2 | None = None,
    eval_env_params: Any | None = None,
    meta_kwargs: dict[str, Any] | None = None,
    eval_meta_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build unified training configuration from parsed arguments.

    This function constructs the training config dictionary from CLI arguments,
    eliminating the need for debug/non-debug branching in training scripts.
    All parameters are read from args, which should already have debug.yaml
    values merged via load_config_with_debug_support().

    Args:
        args: Parsed and merged arguments namespace.
        env: Training environment (wrapped with AliasPrevActionV2).
        env_params: Environment parameters.
        eval_env: Evaluation environment (optional).
        eval_env_params: Evaluation environment parameters (optional).
        meta_kwargs: Meta-learning kwargs (optional, for meta scripts).
        eval_meta_kwargs: Eval meta-learning kwargs (optional, for meta scripts).

    Returns:
        Training configuration dictionary.
    """
    # Compute eval_seed with fallback
    eval_seed = args.eval_seed if args.eval_seed is not None else (args.seed + 10000)

    config = {
        # Model type
        "MODEL_TYPE": args.arch,
        # PPO hyperparameters
        "LR": args.lr,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "UPDATE_EPOCHS": args.update_epochs,
        "NUM_MINIBATCHES": args.num_minibatches,
        "GAMMA": 0.99,
        "GAE_LAMBDA": args.gae_lambda,
        "CLIP_EPS": 0.2,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 0.5,
        "ANNEAL_LR": (args.anneal_lr == 1),
        # Environment
        "ENV": env,
        "ENV_PARAMS": env_params,
        "EVAL_SEED": eval_seed,
        # Debug flags
        "DEBUG": True,
        "DEBUG_TRACE": (args.debug >= 2),
        # S5 architecture (included even when using other architectures)
        "S5_D_MODEL": 256,
        "S5_SSM_SIZE": 256,
        "S5_N_LAYERS": args.s5_n_layers,
        "S5_BLOCKS": 1,
        "S5_ACTIVATION": "full_glu",
        "S5_DO_NORM": (args.s5_do_norm == 1),
        "S5_PRENORM": (args.s5_prenorm == 1),
        "S5_DO_GTRXL_NORM": (args.s5_do_gtrxl_norm == 1),
        # AGaLiTe architecture (included even when using other architectures)
        "AGALITE_N_LAYERS": args.agalite_n_layers,
        "AGALITE_D_MODEL": args.agalite_d_model,
        "AGALITE_D_HEAD": args.agalite_d_head,
        "AGALITE_D_FFC": args.agalite_d_ffc,
        "AGALITE_N_HEADS": args.agalite_n_heads,
        "AGALITE_ETA": args.agalite_eta,
        "AGALITE_R": args.agalite_r,
    }

    # Add evaluation environment if provided
    if eval_env is not None:
        config["EVAL_ENV"] = eval_env
        config["EVAL_ENV_PARAMS"] = eval_env_params

    # Add meta-learning kwargs if provided
    if meta_kwargs is not None:
        config["META_KWARGS"] = meta_kwargs
        config["RESET_WORDS"] = (args.reset_words == 1)

    if eval_meta_kwargs is not None:
        config["EVAL_META_KWARGS"] = eval_meta_kwargs

    return config
