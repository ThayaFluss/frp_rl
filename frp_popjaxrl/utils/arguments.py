"""Argument definitions for training scripts.

This module centralizes argparse argument definitions to reduce code duplication
across training scripts. Arguments are organized into logical groups that can
be composed to create parsers for different training modes.

Example usage:
    from utils.arguments import create_meta_parser
    parser = create_meta_parser()
    args = parser.parse_args()
"""

from __future__ import annotations

import argparse


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by all training scripts.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    # Config file argument (must be first to allow YAML to set other args)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. CLI args override config values.",
    )

    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of training runs (default: %(default)s)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="s5",
        help="Architecture: gru, s5, or agalite (default: %(default)s)",
    )
    parser.add_argument(
        "--log_wandb",
        type=str,
        default="popgym",
        help="Wandb project name (default: %(default)s)",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="Debug mode: 0 or 1 (default: %(default)s)",
    )
    parser.add_argument(
        "--jax_profile",
        type=int,
        default=0,
        help="JAX profiling level: 0=disabled, 1=compile_log, 2=profiler+compile_log (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training (default: %(default)s)",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=None,
        help="Random seed for evaluation. If None, uses seed + 10000 (default: %(default)s)",
    )

    # Saving options
    parser.add_argument(
        "--save_results",
        type=int,
        default=0,
        help="Save results npy (default: %(default)s)",
    )
    parser.add_argument(
        "--save_model",
        type=int,
        default=0,
        help="Save model checkpoint (default: %(default)s)",
    )


def add_ppo_arguments(parser: argparse.ArgumentParser) -> None:
    """Add PPO hyperparameter arguments.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.0,
        help="Entropy coefficient (default: %(default)s)",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=1.0,
        help="GAE lambda (default: %(default)s)",
    )
    parser.add_argument(
        "--update_epochs",
        type=int,
        default=30,
        help="Number of update epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=64,
        help="Number of parallel environments (default: %(default)s)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1024,
        help="Number of steps per update (default: %(default)s)",
    )
    parser.add_argument(
        "--total_timesteps",
        type=float,
        default=15e6,
        help="Total timesteps (default: %(default)s)",
    )
    parser.add_argument(
        "--num_minibatches",
        type=int,
        default=8,
        help="Number of minibatches (default: %(default)s)",
    )
    parser.add_argument(
        "--anneal_lr",
        type=int,
        default=0,
        help="Anneal learning rate: 0 or 1 (default: %(default)s)",
    )


def add_s5_arguments(parser: argparse.ArgumentParser) -> None:
    """Add S5 architecture hyperparameter arguments.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--s5_n_layers",
        type=int,
        default=4,
        help="Number of S5 layers (default: %(default)s)",
    )
    parser.add_argument(
        "--s5_do_norm",
        type=int,
        default=0,
        help="S5 do normalization: 0 or 1 (default: %(default)s)",
    )
    parser.add_argument(
        "--s5_prenorm",
        type=int,
        default=0,
        help="S5 prenormalization: 0 or 1 (default: %(default)s)",
    )
    parser.add_argument(
        "--s5_do_gtrxl_norm",
        type=int,
        default=0,
        help="S5 GTrXL normalization: 0 or 1 (default: %(default)s)",
    )


def add_agalite_arguments(parser: argparse.ArgumentParser) -> None:
    """Add AGaLiTe architecture hyperparameter arguments.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--agalite_n_layers",
        type=int,
        default=4,
        help="Number of AGaLiTe layers (default: %(default)s)",
    )
    parser.add_argument(
        "--agalite_d_model",
        type=int,
        default=256,
        help="AGaLiTe model dimension (default: %(default)s)",
    )
    parser.add_argument(
        "--agalite_d_head",
        type=int,
        default=64,
        help="AGaLiTe head dimension (default: %(default)s)",
    )
    parser.add_argument(
        "--agalite_d_ffc",
        type=int,
        default=256,
        help="AGaLiTe feedforward dimension (default: %(default)s)",
    )
    parser.add_argument(
        "--agalite_n_heads",
        type=int,
        default=4,
        help="Number of AGaLiTe attention heads (default: %(default)s)",
    )
    parser.add_argument(
        "--agalite_eta",
        type=int,
        default=4,
        help="AGaLiTe eta parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--agalite_r",
        type=int,
        default=2,
        help="AGaLiTe r parameter (default: %(default)s)",
    )


def add_meta_arguments(parser: argparse.ArgumentParser) -> None:
    """Add meta-learning specific arguments.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--env",
        type=str,
        default="cartpole",
        help="Base env XXX of MetaXXX (default: %(default)s)",
    )

    # FRP/Meta environment configuration
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Output dim of metaaugnetwork (default: %(default)s)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Depth of MetaAugNetwork (default: %(default)s)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=8,
        help="Max depth metaaugnetwork, num parallel is 2**max_depth (default: %(default)s)",
    )
    parser.add_argument(
        "--with_adjoint",
        type=int,
        default=0,
        help="Use adjoint of orthogonal matrix in branch (default: %(default)s)",
    )
    parser.add_argument(
        "--reset_words",
        type=int,
        default=1,
        help="Reset words per epoch (default: %(default)s)",
    )

    # Evaluation configuration
    parser.add_argument(
        "--eval_method",
        type=str,
        default="tiling",
        help="Evaluation method: tiling / padding / identity (default: %(default)s)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=16,
        help="Number of trials per episode (default: %(default)s)",
    )
    parser.add_argument(
        "--eval_num_trials",
        type=int,
        default=16,
        help="Number of trials per episode for evaluation (default: %(default)s)",
    )

    # Normalization configuration (for gymnax environments)
    parser.add_argument(
        "--norm_strategy",
        type=str,
        default="fixed",
        help="Reward normalization strategy: dynamic/fixed/minmax/custom (default: %(default)s)",
    )
    parser.add_argument(
        "--norm_max_steps",
        type=int,
        default=200,
        help="Maximum steps for reward normalization scaling (default: %(default)s)",
    )

    # FRP input configuration
    parser.add_argument(
        "--frp_include_metadata",
        type=int,
        default=0,
        help="Include metadata (action, done, reset) in FRP input: 0 or 1 (default: %(default)s)",
    )
    parser.add_argument(
        "--frp_include_wrapper",
        type=int,
        default=0,
        help="Include wrapper data (AliasPrevActionV2) in FRP input: 0 or 1 (default: %(default)s)",
    )


def add_standard_env_argument(parser: argparse.ArgumentParser) -> None:
    """Add environment argument for standard (non-meta) training.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--env",
        type=str,
        default="StatelessCartPoleEasy",
        help="Environment name (default: %(default)s)",
    )


def create_meta_parser() -> argparse.ArgumentParser:
    """Create argument parser for meta-learning training scripts.

    Returns:
        Configured ArgumentParser for meta-learning training.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Training script for SEPARATED mode (FRP state externalized)",
    )

    add_common_arguments(parser)
    add_meta_arguments(parser)
    add_ppo_arguments(parser)
    add_s5_arguments(parser)
    add_agalite_arguments(parser)

    # Override default for log_wandb
    parser.set_defaults(log_wandb="popgym_separated")

    return parser


def create_standard_parser() -> argparse.ArgumentParser:
    """Create argument parser for standard (non-meta) training scripts.

    Returns:
        Configured ArgumentParser for standard training.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Training script for standard environments (no FRP/meta-learning)",
    )

    add_common_arguments(parser)
    add_standard_env_argument(parser)
    add_ppo_arguments(parser)
    add_s5_arguments(parser)
    add_agalite_arguments(parser)

    # Override defaults for standard training
    parser.set_defaults(
        log_wandb="popgym_standard",
        arch="gru",
        total_timesteps=1e6,
    )

    return parser
