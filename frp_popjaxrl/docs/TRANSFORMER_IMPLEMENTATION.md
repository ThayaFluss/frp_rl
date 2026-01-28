# Transformer Implementation Documentation

**Last Updated:** 2026-01-28

This document describes the TransformerXL implementation in frp_popjaxrl, including parameter verification results and differences from the reference implementation (transformerXL_PPO_JAX).

---

## Overview

The TransformerXL implementation in `frp_popjaxrl/algorithms/transformer.py` provides a GTrXL-style (Gated TransformerXL) architecture for use in meta-RL settings. It integrates with the existing GRU/S5 architecture pattern via `TransformerRepModel` in `frp_popjaxrl/algorithms/models.py`.

---

## Architecture Components

### 1. PositionalEmbedding

Sinusoidal positional embeddings for relative position attention.

```
inv_freq = 1 / (10000^(2i/d_model))
pos_emb = [sin(pos * inv_freq), cos(pos * inv_freq)]
```

### 2. RelMultiHeadAttention

Relative position multi-head attention following TransformerXL.

**Parameters per layer:**

| Component | Shape | Count |
|-----------|-------|-------|
| query_proj | (d_model, d_model) | d_model² |
| key_proj | (d_model, d_model) | d_model² |
| value_proj | (d_model, d_model) | d_model² |
| pos_proj | (d_model, d_model) | d_model² |
| out_proj | (d_model, d_model + 1) | d_model² + d_model |
| r_w_bias | (num_heads, head_dim) | d_model |
| r_r_bias | (num_heads, head_dim) | d_model |

### 3. Gating (GTrXL)

GRU-style gating for residual connections.

**Parameters per Gating module:**

| Component | Shape | Count |
|-----------|-------|-------|
| 6 × Dense(d_model, use_bias=False) | (d_model, d_model) | 6 × d_model² |
| gating_bias | (d_model,) | d_model |

### 4. TransformerBlock

Single transformer layer with:

- Pre-LayerNorm architecture
- RelMultiHeadAttention
- FeedForward (d_model → d_ff → d_model)
- Optional GTrXL gating

### 5. StackedTransformer

Multi-layer stack with:

- Per-layer memory management
- Episode boundary handling (done mask)
- Sliding window memory updates

---

## Parameter Count Verification

All parameter counts have been verified to match theoretical calculations.

### Test Results

| d_model | heads | layers | d_ff | gating | Theoretical | Actual | Match |
|---------|-------|--------|------|--------|-------------|--------|-------|
| 256 | 4 | 1 | 256 | ON | 1,383,429 | 1,383,429 | ✓ |
| 256 | 4 | 1 | 256 | OFF | 596,485 | 596,485 | ✓ |
| 256 | 4 | 2 | 256 | ON | 2,632,197 | 2,632,197 | ✓ |
| 256 | 4 | 2 | 256 | OFF | 1,058,309 | 1,058,309 | ✓ |
| 256 | 4 | 4 | 256 | ON | 5,129,733 | 5,129,733 | ✓ |
| 256 | 4 | 4 | 256 | OFF | 1,981,957 | 1,981,957 | ✓ |
| 128 | 2 | 2 | 128 | ON | 711,557 | 711,557 | ✓ |
| 512 | 8 | 2 | 512 | ON | 10,209,029 | 10,209,029 | ✓ |

### Detailed Breakdown (d_model=256, 2 layers, gating ON)

```
Encoder (rep_model_0 + rep_model_1):           35,200 (1%)
Transformer layers (×2):                    2,497,536 (95%)
  Per-layer breakdown:
    - Attention:                              329,216
    - LayerNorm (2×):                           1,024
    - FeedForward:                            131,584
    - Gating (2×):                            786,944
    - Per-layer total:                      1,248,768
Actor head:                                    49,924 (2%)
Critic head:                                   49,537 (2%)
─────────────────────────────────────────────────────────
TOTAL:                                      2,632,197
```

### Gating Effect

| Layers | Gating Params | Total Ratio (ON/OFF) |
|--------|---------------|----------------------|
| 1 | 786,944 | 2.32× |
| 2 | 1,573,888 | 2.49× |
| 4 | 3,147,776 | 2.59× |

### Comparison with GRU/S5

| Architecture | Config | Params | vs GRU |
|--------------|--------|--------|--------|
| GRU | 1 layer, 256 dim | 528,901 | 1.0× |
| S5 | 1 layer, 256 dim | 397,957 | 0.8× |
| S5 | 2 layers, 256 dim | 661,253 | 1.3× |
| S5 | 4 layers, 256 dim | 1,187,845 | 2.2× |
| Transformer | 1 layer, gating OFF | 596,485 | 1.1× |
| Transformer | 2 layers, gating OFF | 1,058,309 | 2.0× |
| Transformer | 2 layers, gating ON | 2,632,197 | 5.0× |
| Transformer | 4 layers, gating ON | 6,702,597 | 12.7× |

**Note:** `mem_len` and `num_heads` do not affect parameter count (they are architectural hyperparameters).

---

## Differences from transformerXL_PPO_JAX

The reference implementation is [transformerXL_PPO_JAX](https://github.com/Reytuag/transformerXL_PPO_JAX) by Gautier Hamon.

### 1. Gating Bias Default

| Implementation | Default Value | Source |
|----------------|---------------|--------|
| transformerXL_PPO_JAX | `bg=0.0` | - |
| frp_popjaxrl | `bg=0.0` | Aligned with reference |

Both implementations use `bg=0.0` by default.

### 2. Gating Application

**transformerXL_PPO_JAX (line 79-80):**

```python
if(self.gating):
    out= self.gate2(out, jax.nn.relu(out_attention))
else:
    out = out + out_attention
```

**frp_popjaxrl:**

```python
if self.use_gating:
    x = self.gate2(x, jax.nn.relu(ff_out))
else:
    x = x + ff_out
```

The reference implementation uses swapped argument order for `gate2`. We tried matching that order but it did not improve results, so we use the standard gating convention: `gate(residual, new_input)`.

### 3. FeedForward Dimension

| Implementation | d_ff | Ratio |
|----------------|------|-------|
| transformerXL_PPO_JAX | d_model | 1× |
| frp_popjaxrl | d_model | 1× (aligned) |

Both implementations use `d_ff = d_model` by default.

### 4. Relative Position Shift

| Implementation | Method |
|----------------|--------|
| transformerXL_PPO_JAX | `roll_vmap` with per-query roll |
| frp_popjaxrl | Pad-reshape-slice method |

Both methods achieve the same result. The reference uses vectorized roll operations, while our implementation uses the standard pad-reshape approach from the original TransformerXL paper.

### 5. Episode Boundary Handling

| Implementation | Done Handling |
|----------------|---------------|
| transformerXL_PPO_JAX | No explicit done handling |
| frp_popjaxrl | Memory reset on episode boundaries |

Our implementation explicitly handles episode boundaries:

- Masks attention to prevent attending across done boundaries
- Zeros memory when done occurs in sequence
- Keeps only activations from the current episode in memory

### 6. Memory Management

| Implementation | Memory Format | Management |
|----------------|---------------|------------|
| transformerXL_PPO_JAX | External (batch, mem_len, layers, d_model) | Manual in trainer |
| frp_popjaxrl | List of (1, batch, mem_len × d_model) | Inside StackedTransformer |

Our implementation stores memory in a format compatible with GRU/S5 hidden state handling for minibatch shuffling.

### 7. Normalization Architecture

| Implementation | Architecture |
|----------------|--------------|
| transformerXL_PPO_JAX | Pre-norm (same) |
| frp_popjaxrl | Pre-norm (same) |

Both use pre-LayerNorm architecture.

### 8. Q/K/V Projection Bias

| Implementation | use_bias |
|----------------|----------|
| transformerXL_PPO_JAX | True (via DenseGeneral default) |
| frp_popjaxrl | True (aligned with reference) |

Both implementations use `use_bias=True` for Q/K/V projections, following the reference implementation. Position embedding projection uses `use_bias=False` in both.

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRANSFORMER_D_MODEL` | 256 | Model dimension |
| `TRANSFORMER_NUM_HEADS` | 4 | Number of attention heads |
| `TRANSFORMER_N_LAYERS` | 2 | Number of transformer layers |
| `TRANSFORMER_D_FF` | d_model | Feedforward hidden dimension (following transformerXL_PPO_JAX) |
| `TRANSFORMER_MEM_LEN` | 64 | Memory length per layer |
| `TRANSFORMER_DROPOUT` | 0.0 | Dropout rate |
| `TRANSFORMER_GATING` | True | Use GTrXL gating |

---

## Usage Example

```python
from frp_popjaxrl.algorithms.models import TransformerRepModel, ActorCriticDiscrete

config = {
    "TRANSFORMER_D_MODEL": 256,
    "TRANSFORMER_NUM_HEADS": 4,
    "TRANSFORMER_N_LAYERS": 2,
    "TRANSFORMER_D_FF": 256,  # d_ff = d_model (default)
    "TRANSFORMER_MEM_LEN": 64,
    "TRANSFORMER_DROPOUT": 0.0,
    "TRANSFORMER_GATING": True,
}

rep_model = TransformerRepModel(config=config)
network = ActorCriticDiscrete(
    rep_model=rep_model,
    action_dim=4,
    config=config,
)

# Initialize
hidden = rep_model.initialize_carry(batch_size, config)
params = network.init(rng, hidden, (obs, dones))

# Forward pass
new_hidden, pi, value = network.apply(params, hidden, (obs, dones))
```

---

## Verification Scripts

Two scripts are available for parameter verification:

1. **`scripts/compare_model_params.py`** - Compare parameter counts across GRU, S5, and Transformer
2. **`scripts/verify_transformer_params.py`** - Verify Transformer parameters against theoretical calculations

```bash
uv run python frp_popjaxrl/scripts/compare_model_params.py
uv run python frp_popjaxrl/scripts/verify_transformer_params.py
```

---

## References

- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
- [Stabilizing Transformers for Reinforcement Learning (GTrXL)](https://arxiv.org/abs/1910.06764)
- [transformerXL_PPO_JAX](https://github.com/Reytuag/transformerXL_PPO_JAX)
