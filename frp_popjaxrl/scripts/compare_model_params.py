#!/usr/bin/env python3
"""
Compare parameter counts across GRU, S5, and AGaLiTe architectures.

This script creates models with various configurations and counts their parameters,
outputting a formatted comparison table.

AGaLiTe reference: https://arxiv.org/abs/2504.06983
"""

import sys
from pathlib import Path

# Add the frp_popjaxrl directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a single model comparison."""

    arch: str
    layers: int
    dim: int
    # AGaLiTe-specific parameters
    n_heads: Optional[int] = None
    eta: Optional[int] = None
    r: Optional[int] = None

    def to_config_dict(self, obs_dim: int = 16, action_dim: int = 4) -> dict:
        """Convert to configuration dictionary for model creation."""
        config = {
            # Common settings
            "NUM_ENVS": 1,
            "NO_RESET": False,
            # S5 settings
            # Note: S5_SSM_SIZE must be 256 (fixed) because S5RepModel.rep_model_1 outputs 256
            # and the SSM internal dimensions depend on this. S5_D_MODEL affects the SSM's H parameter.
            "S5_D_MODEL": self.dim,
            "S5_SSM_SIZE": 256,  # Fixed: tied to rep_model_1 output (256)
            "S5_N_LAYERS": self.layers,
            "S5_BLOCKS": 1,
            "S5_ACTIVATION": "full_glu",
            "S5_DO_NORM": False,
            "S5_PRENORM": False,
            "S5_DO_GTRXL_NORM": False,
            # AGaLiTe settings
            "AGALITE_N_LAYERS": self.layers,
            "AGALITE_D_MODEL": self.dim,
            "AGALITE_D_HEAD": self.dim,
            "AGALITE_D_FFC": self.dim,
            "AGALITE_N_HEADS": self.n_heads if self.n_heads else 4,
            "AGALITE_ETA": self.eta if self.eta else 4,
            "AGALITE_R": self.r if self.r else 2,
        }
        return config

    def description(self) -> str:
        """Return a formatted description for the table."""
        n_heads_str = str(self.n_heads) if self.n_heads else "-"
        eta_str = str(self.eta) if self.eta else "-"
        r_str = str(self.r) if self.r else "-"
        return f"{self.arch:<12} {self.layers:<7} {self.dim:<5} {n_heads_str:<7} {eta_str:<5} {r_str:<5}"


def count_params(params) -> int:
    """Count total parameters in a PyTree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def count_core_params(
    arch: str, layers: int, dim: int, n_heads: int = 4, eta: int = 4, r: int = 2
) -> int:
    """Count parameters for a Core module only (excluding encoder/actor/critic heads).

    Args:
        arch: Architecture type ("GRU", "S5", "AGaLiTe")
        layers: Number of layers
        dim: Model dimension
        n_heads: Number of attention heads (AGaLiTe only)
        eta: Memory capacity parameter (AGaLiTe only)
        r: r parameter (AGaLiTe only)

    Returns:
        Number of parameters in the core
    """
    from frp_popjaxrl.algorithms.models import GRUCore, S5Core, AGaLiTeCore

    batch_size = 1
    seq_len = 1
    input_dim = 256  # PreCoreEncoder output dimension

    # Create config for each architecture
    config = {
        "S5_D_MODEL": dim,
        "S5_SSM_SIZE": 256,
        "S5_N_LAYERS": layers,
        "S5_BLOCKS": 1,
        "S5_ACTIVATION": "full_glu",
        "S5_DO_NORM": False,
        "S5_PRENORM": False,
        "S5_DO_GTRXL_NORM": False,
        "AGALITE_N_LAYERS": layers,
        "AGALITE_D_MODEL": dim,
        "AGALITE_D_HEAD": dim,
        "AGALITE_D_FFC": dim,
        "AGALITE_N_HEADS": n_heads,
        "AGALITE_ETA": eta,
        "AGALITE_R": r,
    }

    rng = jax.random.PRNGKey(0)

    if arch == "GRU":
        core = GRUCore()
        hidden = GRUCore.initialize_carry(batch_size, config)
    elif arch == "S5":
        core = S5Core(config=config)
        hidden = S5Core.initialize_carry(batch_size, config)
    elif arch == "AGaLiTe":
        core = AGaLiTeCore(config=config)
        hidden = AGaLiTeCore.initialize_carry(batch_size, config)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Initialize with dummy input
    embedding = jnp.zeros((seq_len, batch_size, input_dim))
    dones = jnp.zeros((seq_len, batch_size))

    params = core.init(rng, hidden, embedding, dones)
    return count_params(params)


def create_and_count_params(
    model_config: ModelConfig, obs_dim: int = 16, action_dim: int = 4
) -> int:
    """Create an ActorCritic model and count its parameters."""
    from frp_popjaxrl.algorithms.models import ActorCriticDiscrete, ActorCriticBase

    config = model_config.to_config_dict(obs_dim, action_dim)

    # Map architecture name to core_type
    core_type_map = {"GRU": "gru", "S5": "s5", "AGaLiTe": "agalite"}
    core_type = core_type_map.get(model_config.arch)
    if core_type is None:
        raise ValueError(f"Unknown architecture: {model_config.arch}")

    # Create ActorCritic network with unified interface
    network = ActorCriticDiscrete(
        core_type=core_type,
        action_dim=action_dim,
        config=config,
    )

    # Initialize and count parameters
    batch_size = 1
    seq_len = 1
    init_hidden = ActorCriticBase.initialize_carry(batch_size, core_type, config)
    init_x = (
        jnp.zeros((seq_len, batch_size, obs_dim)),
        jnp.zeros((seq_len, batch_size)),
    )

    rng = jax.random.PRNGKey(0)
    params = network.init(rng, init_hidden, init_x)

    return count_params(params)


def print_core_params_comparison():
    """Print Core-only parameter comparison table."""
    print()
    print("=" * 80)
    print("Core-Only Parameter Comparison (excluding encoder/actor/critic heads)")
    print("=" * 80)
    print(f"{'Core':<12} {'Layers':<7} {'Dim':<5} {'Heads':<7} {'Eta':<5} {'Params':>12} {'vs GRU':>10}")
    print("-" * 80)

    # GRU baseline (single layer, dim=256)
    gru_params = count_core_params("GRU", layers=1, dim=256)
    print(f"{'GRU':<12} {1:<7} {256:<5} {'-':<7} {'-':<5} {gru_params:>12,} {1.0:>9.2f}x")

    # S5 configurations
    for layers in [1, 2, 4]:
        params = count_core_params("S5", layers=layers, dim=256)
        ratio = params / gru_params
        print(f"{'S5':<12} {layers:<7} {256:<5} {'-':<7} {'-':<5} {params:>12,} {ratio:>9.2f}x")

    # AGaLiTe configurations
    for layers in [1, 2, 4]:
        params = count_core_params("AGaLiTe", layers=layers, dim=256, n_heads=4, eta=4)
        ratio = params / gru_params
        print(f"{'AGaLiTe':<12} {layers:<7} {256:<5} {4:<7} {4:<5} {params:>12,} {ratio:>9.2f}x")

    print("=" * 80)

    # Summary
    print()
    print("=" * 80)
    print("Core Parameter Summary")
    print("=" * 80)

    s5_1l = count_core_params("S5", layers=1, dim=256)
    s5_2l = count_core_params("S5", layers=2, dim=256)
    s5_4l = count_core_params("S5", layers=4, dim=256)

    agalite_1l = count_core_params("AGaLiTe", layers=1, dim=256, n_heads=4, eta=4)
    agalite_2l = count_core_params("AGaLiTe", layers=2, dim=256, n_heads=4, eta=4)
    agalite_4l = count_core_params("AGaLiTe", layers=4, dim=256, n_heads=4, eta=4)

    print(f"\n{'Core':<12} {'1 Layer':>12} {'2 Layers':>12} {'4 Layers':>12} {'vs GRU (1L)':>12}")
    print("-" * 60)
    print(f"{'GRU':<12} {gru_params:>12,} {'-':>12} {'-':>12} {'1.00x':>12}")
    print(f"{'S5':<12} {s5_1l:>12,} {s5_2l:>12,} {s5_4l:>12,} {s5_1l/gru_params:>11.2f}x")
    print(f"{'AGaLiTe':<12} {agalite_1l:>12,} {agalite_2l:>12,} {agalite_4l:>12,} {agalite_1l/gru_params:>11.2f}x")
    print("=" * 60)


def main():
    """Main function to compare model parameters."""
    # First, show Core-only comparison
    print_core_params_comparison()

    # Configuration
    obs_dim = 16
    action_dim = 4

    # Define configurations to compare
    configs = []

    # 1. GRU baseline (fixed configuration)
    configs.append(ModelConfig(arch="GRU", layers=1, dim=256))

    # 2. S5 configurations
    # Note: S5 has fixed dim=256 due to internal architecture constraints
    # (rep_model_1 outputs 256, and SSM dimensions are tied to this)
    # Layer comparison only
    for layers in [1, 2, 4]:
        configs.append(ModelConfig(arch="S5", layers=layers, dim=256))

    # 3. AGaLiTe configurations
    # Layer comparison with different dimensions
    for layers in [2, 4]:
        for dim in [64, 128, 256]:
            configs.append(
                ModelConfig(
                    arch="AGaLiTe",
                    layers=layers,
                    dim=dim,
                    n_heads=4,
                    eta=4,
                    r=2,
                )
            )

    # Eta comparison (layers=2, dim=64)
    for eta in [2, 4, 8]:
        configs.append(
            ModelConfig(
                arch="AGaLiTe",
                layers=2,
                dim=64,
                n_heads=4,
                eta=eta,
                r=2,
            )
        )

    # Count parameters for all configurations
    results = []
    gru_params = None

    print("Counting parameters for each configuration...")
    for config in configs:
        params = create_and_count_params(config, obs_dim, action_dim)
        results.append((config, params))
        if config.arch == "GRU":
            gru_params = params

    # Print results
    print()
    print("=" * 80)
    print("Model Parameter Comparison")
    print(f"(obs_dim={obs_dim}, action_dim={action_dim})")
    print("=" * 80)
    print(
        f"{'Arch':<12} {'Layers':<7} {'Dim':<5} {'Heads':<7} {'Eta':<5} {'R':<5} {'Params':>12} {'vs GRU':>10}"
    )
    print("-" * 80)

    for config, params in results:
        ratio = params / gru_params if gru_params else 1.0
        print(f"{config.description()} {params:>12,} {ratio:>9.1f}x")

    print("=" * 80)

    # Print summary by category
    print()
    print("=" * 80)
    print("Summary by Category")
    print("=" * 80)

    # Group by category
    gru_results = [(c, p) for c, p in results if c.arch == "GRU"]
    s5_results = [(c, p) for c, p in results if c.arch == "S5"]
    agalite_results = [(c, p) for c, p in results if c.arch == "AGaLiTe"]

    print(f"\nGRU (baseline): {gru_results[0][1]:,} params")

    print(f"\nS5 range: {min(p for _, p in s5_results):,} - {max(p for _, p in s5_results):,} params")
    print(f"  vs GRU: {min(p for _, p in s5_results) / gru_params:.1f}x - {max(p for _, p in s5_results) / gru_params:.1f}x")

    print(f"\nAGaLiTe range: {min(p for _, p in agalite_results):,} - {max(p for _, p in agalite_results):,} params")
    print(f"  vs GRU: {min(p for _, p in agalite_results) / gru_params:.1f}x - {max(p for _, p in agalite_results) / gru_params:.1f}x")

    print("=" * 80)


if __name__ == "__main__":
    main()
