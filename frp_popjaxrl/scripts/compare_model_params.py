#!/usr/bin/env python3
"""
Compare parameter counts across GRU, S5, and Transformer architectures.

This script creates models with various configurations and counts their parameters,
outputting a formatted comparison table.
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
    mem_len: Optional[int] = None
    heads: Optional[int] = None
    gating: Optional[bool] = None

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
            # Transformer settings
            "TRANSFORMER_D_MODEL": self.dim,
            "TRANSFORMER_NUM_HEADS": self.heads if self.heads else 4,
            "TRANSFORMER_N_LAYERS": self.layers,
            "TRANSFORMER_D_FF": self.dim * 4,
            "TRANSFORMER_MEM_LEN": self.mem_len if self.mem_len else 64,
            "TRANSFORMER_DROPOUT": 0.0,
            "TRANSFORMER_GATING": self.gating if self.gating is not None else True,
        }
        return config

    def description(self) -> str:
        """Return a formatted description for the table."""
        mem_str = str(self.mem_len) if self.mem_len else "-"
        heads_str = str(self.heads) if self.heads else "-"
        gating_str = "ON" if self.gating else ("OFF" if self.gating is False else "-")
        return f"{self.arch:<12} {self.layers:<7} {self.dim:<5} {mem_str:<7} {heads_str:<6} {gating_str:<11}"


def count_params(params) -> int:
    """Count total parameters in a PyTree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def create_and_count_params(
    model_config: ModelConfig, obs_dim: int = 16, action_dim: int = 4
) -> int:
    """Create a model and count its parameters."""
    from frp_popjaxrl.algorithms.models import (
        GRURepModel,
        S5RepModel,
        TransformerRepModel,
        ActorCriticDiscrete,
    )

    config = model_config.to_config_dict(obs_dim, action_dim)

    # Select RepModel based on architecture
    if model_config.arch == "GRU":
        rep_model = GRURepModel(config=config)
    elif model_config.arch == "S5":
        rep_model = S5RepModel(config=config)
    elif model_config.arch == "Transformer":
        rep_model = TransformerRepModel(config=config)
    else:
        raise ValueError(f"Unknown architecture: {model_config.arch}")

    # Create ActorCritic network
    network = ActorCriticDiscrete(
        rep_model=rep_model,
        action_dim=action_dim,
        config=config,
    )

    # Initialize and count parameters
    batch_size = 1
    seq_len = 1
    init_hidden = rep_model.initialize_carry(batch_size, config)
    init_x = (
        jnp.zeros((seq_len, batch_size, obs_dim)),
        jnp.zeros((seq_len, batch_size)),
    )

    rng = jax.random.PRNGKey(0)
    params = network.init(rng, init_hidden, init_x)

    return count_params(params)


def main():
    """Main function to compare model parameters."""
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

    # 3. Transformer configurations
    # Layer comparison with gating ON/OFF (dim=256, mem_len=64, heads=4)
    for layers in [1, 2, 4]:
        for gating in [True, False]:
            configs.append(
                ModelConfig(
                    arch="Transformer",
                    layers=layers,
                    dim=256,
                    mem_len=64,
                    heads=4,
                    gating=gating,
                )
            )

    # Dim comparison (layers=2, gating=True)
    for dim in [128, 512]:
        heads = max(2, dim // 64)  # Ensure divisibility
        configs.append(
            ModelConfig(
                arch="Transformer",
                layers=2,
                dim=dim,
                mem_len=64,
                heads=heads,
                gating=True,
            )
        )

    # Note: mem_len and heads do not affect parameter count
    # (they are architectural hyperparameters, not learned parameters)
    # So we don't include separate comparisons for them

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
        f"{'Arch':<12} {'Layers':<7} {'Dim':<5} {'MemLen':<7} {'Heads':<6} {'Gating':<11} {'Params':>12} {'vs GRU':>10}"
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
    transformer_results = [(c, p) for c, p in results if c.arch == "Transformer"]

    print(f"\nGRU (baseline): {gru_results[0][1]:,} params")

    print(f"\nS5 range: {min(p for _, p in s5_results):,} - {max(p for _, p in s5_results):,} params")
    print(f"  vs GRU: {min(p for _, p in s5_results) / gru_params:.1f}x - {max(p for _, p in s5_results) / gru_params:.1f}x")

    print(f"\nTransformer range: {min(p for _, p in transformer_results):,} - {max(p for _, p in transformer_results):,} params")
    print(f"  vs GRU: {min(p for _, p in transformer_results) / gru_params:.1f}x - {max(p for _, p in transformer_results) / gru_params:.1f}x")

    # Gating effect
    gating_on = [p for c, p in transformer_results if c.gating]
    gating_off = [p for c, p in transformer_results if not c.gating]
    if gating_on and gating_off:
        print(f"\nGating effect (average):")
        print(f"  Gating ON:  {sum(gating_on) / len(gating_on):,.0f} params")
        print(f"  Gating OFF: {sum(gating_off) / len(gating_off):,.0f} params")
        print(f"  Ratio: {sum(gating_on) / len(gating_on) / (sum(gating_off) / len(gating_off)):.2f}x")

    print("=" * 80)


if __name__ == "__main__":
    main()
