#!/usr/bin/env python3
"""
Verify Transformer parameter counts against theoretical calculations.

This script calculates the expected parameter counts based on the architecture
and compares them to the actual counts from model initialization.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp


def theoretical_transformer_params(
    obs_dim: int,
    action_dim: int,
    d_model: int,
    num_heads: int,
    n_layers: int,
    d_ff: int,
    use_gating: bool,
) -> dict:
    """Calculate theoretical parameter counts for TransformerRepModel + ActorCriticDiscrete."""

    # Input encoder (shared with GRU/S5)
    rep_model_0 = obs_dim * 128 + 128  # Dense(128)
    rep_model_1 = 128 * d_model + d_model  # Dense(d_model)
    encoder_params = rep_model_0 + rep_model_1

    # RelMultiHeadAttention
    head_dim = d_model // num_heads
    query_proj = d_model * d_model  # no bias
    key_proj = d_model * d_model
    value_proj = d_model * d_model
    pos_proj = d_model * d_model
    out_proj = d_model * d_model + d_model  # with bias
    r_w_bias = num_heads * head_dim
    r_r_bias = num_heads * head_dim
    attention_params = query_proj + key_proj + value_proj + pos_proj + out_proj + r_w_bias + r_r_bias

    # LayerNorm (ln1, ln2) - each has scale and bias
    layer_norm_params = d_model * 2 * 2  # 2 LayerNorms, each with scale + bias

    # FeedForward
    ff1 = d_model * d_ff + d_ff  # Dense(d_ff)
    ff2 = d_ff * d_model + d_model  # Dense(d_model)
    ff_params = ff1 + ff2

    # Gating (if enabled) - 2 Gating modules per block
    if use_gating:
        # Each Gating has:
        # - 6 Dense(d_model, use_bias=False): for reset gate, update gate, candidate
        # - 1 gating_bias parameter
        single_gating = 6 * d_model * d_model + d_model
        gating_params = single_gating * 2  # gate1 and gate2
    else:
        gating_params = 0

    # Per-layer total
    per_layer = attention_params + layer_norm_params + ff_params + gating_params

    # PositionalEmbedding has no learnable params (sinusoidal)

    # Actor head
    actor_dense1 = d_model * 128 + 128
    actor_dense2 = 128 * 128 + 128
    actor_dense3 = 128 * action_dim + action_dim
    actor_params = actor_dense1 + actor_dense2 + actor_dense3

    # Critic head
    critic_dense1 = d_model * 128 + 128
    critic_dense2 = 128 * 128 + 128
    critic_dense3 = 128 * 1 + 1
    critic_params = critic_dense1 + critic_dense2 + critic_dense3

    # Total
    total = encoder_params + per_layer * n_layers + actor_params + critic_params

    return {
        "encoder": encoder_params,
        "attention_per_layer": attention_params,
        "layer_norm_per_layer": layer_norm_params,
        "ff_per_layer": ff_params,
        "gating_per_layer": gating_params,
        "per_layer": per_layer,
        "actor": actor_params,
        "critic": critic_params,
        "total": total,
    }


def actual_transformer_params(
    obs_dim: int,
    action_dim: int,
    d_model: int,
    num_heads: int,
    n_layers: int,
    d_ff: int,
    use_gating: bool,
) -> int:
    """Get actual parameter count from model initialization."""
    from frp_popjaxrl.algorithms.models import TransformerRepModel, ActorCriticDiscrete

    config = {
        "NUM_ENVS": 1,
        "NO_RESET": False,
        "TRANSFORMER_D_MODEL": d_model,
        "TRANSFORMER_NUM_HEADS": num_heads,
        "TRANSFORMER_N_LAYERS": n_layers,
        "TRANSFORMER_D_FF": d_ff,
        "TRANSFORMER_MEM_LEN": 64,
        "TRANSFORMER_DROPOUT": 0.0,
        "TRANSFORMER_GATING": use_gating,
    }

    rep_model = TransformerRepModel(config=config)
    network = ActorCriticDiscrete(rep_model=rep_model, action_dim=action_dim, config=config)

    batch_size = 1
    seq_len = 1
    init_hidden = rep_model.initialize_carry(batch_size, config)
    init_x = (jnp.zeros((seq_len, batch_size, obs_dim)), jnp.zeros((seq_len, batch_size)))

    rng = jax.random.PRNGKey(0)
    params = network.init(rng, init_hidden, init_x)

    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def main():
    """Verify Transformer parameter counts."""
    obs_dim = 16
    action_dim = 4

    test_cases = [
        # (d_model, num_heads, n_layers, d_ff, use_gating)
        (256, 4, 1, 1024, True),
        (256, 4, 1, 1024, False),
        (256, 4, 2, 1024, True),
        (256, 4, 2, 1024, False),
        (256, 4, 4, 1024, True),
        (256, 4, 4, 1024, False),
        (128, 2, 2, 512, True),
        (512, 8, 2, 2048, True),
    ]

    print("=" * 100)
    print("Transformer Parameter Verification")
    print("=" * 100)
    print(f"{'d_model':<8} {'heads':<6} {'layers':<7} {'d_ff':<6} {'gating':<8} {'Theoretical':>14} {'Actual':>14} {'Match':<6}")
    print("-" * 100)

    all_match = True
    for d_model, num_heads, n_layers, d_ff, use_gating in test_cases:
        theory = theoretical_transformer_params(
            obs_dim, action_dim, d_model, num_heads, n_layers, d_ff, use_gating
        )
        actual = actual_transformer_params(
            obs_dim, action_dim, d_model, num_heads, n_layers, d_ff, use_gating
        )

        match = theory["total"] == actual
        match_str = "✓" if match else "✗"
        if not match:
            all_match = False

        gating_str = "ON" if use_gating else "OFF"
        print(
            f"{d_model:<8} {num_heads:<6} {n_layers:<7} {d_ff:<6} {gating_str:<8} "
            f"{theory['total']:>14,} {actual:>14,} {match_str:<6}"
        )

        if not match:
            print(f"  → Difference: {actual - theory['total']:+,}")

    print("=" * 100)

    if all_match:
        print("\n✓ All parameter counts match theoretical calculations!")
        print("\nTransformer implementation is CORRECT.")
    else:
        print("\n✗ Some parameter counts do not match!")
        print("\nTransformer implementation may have issues.")

    # Detailed breakdown for one case
    print("\n" + "=" * 100)
    print("Detailed Breakdown (d_model=256, heads=4, layers=2, d_ff=1024, gating=ON)")
    print("=" * 100)
    breakdown = theoretical_transformer_params(obs_dim, action_dim, 256, 4, 2, 1024, True)
    print(f"  Encoder (rep_model_0 + rep_model_1):     {breakdown['encoder']:>12,}")
    print(f"  Per-layer breakdown:")
    print(f"    - Attention:                          {breakdown['attention_per_layer']:>12,}")
    print(f"    - LayerNorm (2x):                     {breakdown['layer_norm_per_layer']:>12,}")
    print(f"    - FeedForward:                        {breakdown['ff_per_layer']:>12,}")
    print(f"    - Gating (2x):                        {breakdown['gating_per_layer']:>12,}")
    print(f"    - Per-layer total:                    {breakdown['per_layer']:>12,}")
    print(f"  Transformer layers (2x per-layer):      {breakdown['per_layer'] * 2:>12,}")
    print(f"  Actor head:                             {breakdown['actor']:>12,}")
    print(f"  Critic head:                            {breakdown['critic']:>12,}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  TOTAL:                                  {breakdown['total']:>12,}")
    print("=" * 100)

    # Gating analysis
    print("\n" + "=" * 100)
    print("Gating Effect Analysis")
    print("=" * 100)
    for n_layers in [1, 2, 4]:
        on = theoretical_transformer_params(obs_dim, action_dim, 256, 4, n_layers, 1024, True)
        off = theoretical_transformer_params(obs_dim, action_dim, 256, 4, n_layers, 1024, False)
        gating_total = on["gating_per_layer"] * n_layers
        ratio = on["total"] / off["total"]
        print(f"  {n_layers} layer(s): Gating adds {gating_total:,} params ({ratio:.2f}x total)")
    print("=" * 100)


if __name__ == "__main__":
    main()
