"""
TransformerXL implementation for frp_popjaxrl.

This module implements TransformerXL with relative position embeddings
for use in meta-RL settings, following the GTrXL (Gated Transformer-XL)
architecture for improved training stability.

Components:
- PositionalEmbedding: Sinusoidal relative position embeddings
- RelMultiHeadAttention: Relative position multi-head attention
- Gating: GTrXL-style gating mechanism
- TransformerBlock: Single transformer layer with optional gating
- StackedTransformer: Multi-layer stack with memory management
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import zeros
from typing import Optional, List


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for relative position attention.

    Generates fixed sinusoidal embeddings based on position sequences,
    used for relative position encoding in TransformerXL attention.

    Attributes:
        d_model: Embedding dimension (must be even for sin/cos pairs)
    """

    d_model: int

    def setup(self):
        # Inverse frequency for sinusoidal encoding
        # Shape: (d_model // 2,)
        self.inv_freq = 1.0 / (
            10000 ** (jnp.arange(0.0, self.d_model, 2.0) / self.d_model)
        )

    def __call__(self, pos_seq: jnp.ndarray) -> jnp.ndarray:
        """
        Generate positional embeddings for given positions.

        Args:
            pos_seq: Position sequence, shape (seq_len,)

        Returns:
            Positional embeddings, shape (seq_len, d_model)
        """
        # Outer product: (seq_len,) x (d_model // 2,) -> (seq_len, d_model // 2)
        sinusoid_inp = jnp.outer(pos_seq, self.inv_freq)
        # Concatenate sin and cos: (seq_len, d_model)
        pos_emb = jnp.concatenate(
            [jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1
        )
        return pos_emb


class Gating(nn.Module):
    """
    GTrXL-style gating mechanism for residual connections.

    Implements gated residual connections as described in the
    Gated Transformer-XL paper for improved training stability.

    Attributes:
        d_model: Model dimension
        bg: Initial bias for gating (default 0.0, following transformerXL_PPO_JAX)
    """

    d_model: int
    bg: float = 0.0  # Following transformerXL_PPO_JAX reference implementation

    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Apply gating mechanism.

        Args:
            x: Residual input, shape (..., d_model)
            y: New input to gate, shape (..., d_model)

        Returns:
            Gated output, shape (..., d_model)
        """
        # Reset gate
        r = jax.nn.sigmoid(
            nn.Dense(self.d_model, use_bias=False)(y)
            + nn.Dense(self.d_model, use_bias=False)(x)
        )
        # Update gate with learnable bias
        gating_bias = self.param(
            "gating_bias", lambda rng, shape: jnp.full(shape, self.bg), (self.d_model,)
        )
        z = jax.nn.sigmoid(
            nn.Dense(self.d_model, use_bias=False)(y)
            + nn.Dense(self.d_model, use_bias=False)(x)
            - gating_bias
        )
        # Candidate hidden state
        h = jnp.tanh(
            nn.Dense(self.d_model, use_bias=False)(y)
            + nn.Dense(self.d_model, use_bias=False)(r * x)
        )
        # Gated output
        g = (1 - z) * x + z * h
        return g


class RelMultiHeadAttention(nn.Module):
    """
    Relative position multi-head attention (TransformerXL style).

    Implements attention with relative position biases, separating
    content-based and position-based attention components.

    Attributes:
        num_heads: Number of attention heads
        d_model: Model dimension
        dropout_rate: Dropout rate (default 0.0)
    """

    num_heads: int
    d_model: int
    dropout_rate: float = 0.0

    def setup(self):
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        )
        self.head_dim = self.d_model // self.num_heads

        # Q, K, V projections (with bias, following transformerXL_PPO_JAX)
        self.query_proj = nn.Dense(self.d_model, use_bias=True)
        self.key_proj = nn.Dense(self.d_model, use_bias=True)
        self.value_proj = nn.Dense(self.d_model, use_bias=True)

        # Position embedding projection (no bias, following transformerXL_PPO_JAX)
        self.pos_proj = nn.Dense(self.d_model, use_bias=False)

        # Output projection
        self.out_proj = nn.Dense(self.d_model, use_bias=True)

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key_value: jnp.ndarray,
        pos_emb: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Apply relative position multi-head attention.

        Args:
            query: Query tensor, shape (batch, q_len, d_model)
            key_value: Key/Value tensor, shape (batch, kv_len, d_model)
            pos_emb: Position embeddings, shape (kv_len, d_model)
            mask: Optional attention mask, shape (batch, 1, q_len, kv_len)
            deterministic: If True, disable dropout

        Returns:
            Attention output, shape (batch, q_len, d_model)
        """
        batch_size = query.shape[0]
        q_len = query.shape[1]
        kv_len = key_value.shape[1]

        # Project Q, K, V
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, head_dim)
        q = self.query_proj(query).reshape(
            batch_size, q_len, self.num_heads, self.head_dim
        )
        k = self.key_proj(key_value).reshape(
            batch_size, kv_len, self.num_heads, self.head_dim
        )
        v = self.value_proj(key_value).reshape(
            batch_size, kv_len, self.num_heads, self.head_dim
        )

        # Project position embeddings
        # (kv_len, d_model) -> (kv_len, num_heads, head_dim)
        r = self.pos_proj(pos_emb).reshape(kv_len, self.num_heads, self.head_dim)

        # Learnable biases for content and position attention
        # Shape: (num_heads, head_dim)
        r_w_bias = self.param("r_w_bias", zeros, (self.num_heads, self.head_dim))
        r_r_bias = self.param("r_r_bias", zeros, (self.num_heads, self.head_dim))

        # Content-based attention: (q + r_w_bias) @ k^T
        # q: (batch, q_len, heads, head_dim) + r_w_bias: (heads, head_dim)
        # k: (batch, kv_len, heads, head_dim)
        # -> attn_content: (batch, heads, q_len, kv_len)
        q_with_bias = q + r_w_bias  # (batch, q_len, heads, head_dim)
        attn_content = jnp.einsum("bqhd,bkhd->bhqk", q_with_bias, k)

        # Position-based attention: (q + r_r_bias) @ r^T
        # q: (batch, q_len, heads, head_dim) + r_r_bias: (heads, head_dim)
        # r: (kv_len, heads, head_dim)
        # -> attn_pos: (batch, heads, q_len, kv_len)
        q_with_pos_bias = q + r_r_bias  # (batch, q_len, heads, head_dim)
        attn_pos = jnp.einsum("bqhd,khd->bhqk", q_with_pos_bias, r)

        # Relative position shift for proper alignment
        attn_pos = self._rel_shift(attn_pos, q_len)

        # Combine content and position attention
        attn_score = (attn_content + attn_pos) / jnp.sqrt(self.head_dim).astype(
            query.dtype
        )

        # Apply mask if provided
        if mask is not None:
            attn_score = jnp.where(mask, attn_score, jnp.finfo(query.dtype).min)

        # Softmax and dropout
        attn_weights = jax.nn.softmax(attn_score, axis=-1)
        if not deterministic and self.dropout_rate > 0.0:
            attn_weights = nn.Dropout(rate=self.dropout_rate)(
                attn_weights, deterministic=deterministic
            )

        # Apply attention to values
        # attn_weights: (batch, heads, q_len, kv_len)
        # v: (batch, kv_len, heads, head_dim)
        # -> out: (batch, q_len, heads, head_dim)
        out = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)

        # Reshape and project output
        out = out.reshape(batch_size, q_len, self.d_model)
        out = self.out_proj(out)

        return out

    def _rel_shift(self, attn_pos: jnp.ndarray, q_len: int) -> jnp.ndarray:
        """
        Apply relative position shift for proper alignment.

        The position embeddings are ordered from newest to oldest position,
        so we need to shift to align them properly with the attention pattern.

        Args:
            attn_pos: Position attention scores, shape (batch, heads, q_len, kv_len)
            q_len: Query sequence length

        Returns:
            Shifted position attention, shape (batch, heads, q_len, kv_len)
        """
        batch, heads, q_len_actual, kv_len = attn_pos.shape

        # Pad with zeros on the left
        zero_pad = jnp.zeros((batch, heads, q_len_actual, 1), dtype=attn_pos.dtype)
        attn_pos_padded = jnp.concatenate([zero_pad, attn_pos], axis=-1)

        # Reshape and slice to shift positions
        attn_pos_padded = attn_pos_padded.reshape(
            batch, heads, kv_len + 1, q_len_actual
        )
        attn_pos_shifted = attn_pos_padded[:, :, 1:, :]  # Remove first row
        attn_pos_shifted = attn_pos_shifted.reshape(batch, heads, q_len_actual, kv_len)

        return attn_pos_shifted


class TransformerBlock(nn.Module):
    """
    Single TransformerXL layer with optional GTrXL gating.

    Implements pre-LayerNorm architecture with optional gated
    residual connections for improved training stability.

    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension (default: d_model, following transformerXL_PPO_JAX)
        dropout_rate: Dropout rate
        use_gating: Whether to use GTrXL gating
    """

    d_model: int
    num_heads: int
    d_ff: Optional[int] = None
    dropout_rate: float = 0.0
    use_gating: bool = True

    def setup(self):
        # Default d_ff = d_model, following transformerXL_PPO_JAX
        d_ff = self.d_ff if self.d_ff is not None else self.d_model

        self.attention = RelMultiHeadAttention(
            num_heads=self.num_heads,
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
        )
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

        # Feed-forward layers
        self.ff1 = nn.Dense(d_ff)
        self.ff2 = nn.Dense(self.d_model)

        # Optional gating
        if self.use_gating:
            self.gate1 = Gating(self.d_model)
            self.gate2 = Gating(self.d_model)

    def __call__(
        self,
        x: jnp.ndarray,
        memory: jnp.ndarray,
        pos_emb: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            memory: Memory tensor, shape (batch, mem_len, d_model)
            pos_emb: Position embeddings, shape (mem_len + seq_len, d_model)
            mask: Optional attention mask
            deterministic: If True, disable dropout

        Returns:
            Output tensor, shape (batch, seq_len, d_model)
        """
        # Concatenate memory and input for key/value
        # memory: (batch, mem_len, d_model), x: (batch, seq_len, d_model)
        kv = jnp.concatenate([memory, x], axis=1)  # (batch, mem_len + seq_len, d_model)

        # Pre-LayerNorm
        kv_normed = self.ln1(kv)
        x_normed = self.ln1(x)

        # Self-attention
        attn_out = self.attention(
            query=x_normed,
            key_value=kv_normed,
            pos_emb=pos_emb,
            mask=mask,
            deterministic=deterministic,
        )

        # Residual connection (with optional gating)
        if self.use_gating:
            x = self.gate1(x, jax.nn.relu(attn_out))
        else:
            x = x + attn_out

        # Feed-forward with pre-LayerNorm
        x_normed = self.ln2(x)
        ff_out = self.ff1(x_normed)
        ff_out = jax.nn.gelu(ff_out)
        ff_out = self.ff2(ff_out)

        # Residual connection (with optional gating)
        # NOTE: Argument order follows transformerXL_PPO_JAX reference implementation.
        # gate2(ff_out, x) differs from standard gating convention gate(residual, new_input).
        # This matches: out = self.gate2(out, jax.nn.relu(out_attention)) in the reference.
        if self.use_gating:
            x = self.gate2(ff_out, jax.nn.relu(x))
        else:
            x = x + ff_out

        return x


class StackedTransformer(nn.Module):
    """
    Stacked TransformerXL layers with memory management.

    Manages per-layer memory for sequence processing in RL,
    supporting sliding window memory updates.

    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward hidden dimension
        mem_len: Memory length per layer
        dropout_rate: Dropout rate
        use_gating: Whether to use GTrXL gating
    """

    d_model: int
    num_heads: int
    n_layers: int
    d_ff: Optional[int] = None
    mem_len: int = 64
    dropout_rate: float = 0.0
    use_gating: bool = True

    def setup(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.layers = [
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                use_gating=self.use_gating,
            )
            for _ in range(self.n_layers)
        ]

    @staticmethod
    def initialize_carry(batch_size: int, config: dict) -> List[jnp.ndarray]:
        """
        Initialize memory state for all layers.

        Args:
            batch_size: Number of environments
            config: Configuration dict with TRANSFORMER_D_MODEL, TRANSFORMER_N_LAYERS, TRANSFORMER_MEM_LEN

        Returns:
            List of memory tensors, one per layer, each shape (batch, mem_len, d_model)
        """
        d_model = config.get("TRANSFORMER_D_MODEL", 256)
        n_layers = config.get("TRANSFORMER_N_LAYERS", 3)
        mem_len = config.get("TRANSFORMER_MEM_LEN", 64)

        return [jnp.zeros((batch_size, mem_len, d_model)) for _ in range(n_layers)]

    def __call__(
        self,
        memories: List[jnp.ndarray],
        x: jnp.ndarray,
        dones: jnp.ndarray,
        deterministic: bool = True,
    ) -> tuple:
        """
        Forward pass through stacked transformer.

        Args:
            memories: List of memory tensors, each shape (batch, mem_len, d_model)
            x: Input sequence, shape (batch, seq_len, d_model)
            dones: Done flags, shape (batch, seq_len) - resets memory on done
            deterministic: If True, disable dropout

        Returns:
            (new_memories, output): Updated memories and output tensor
        """
        batch_size, seq_len, _ = x.shape
        mem_len = memories[0].shape[1]
        total_len = mem_len + seq_len

        # Generate position embeddings (from newest to oldest)
        pos_seq = jnp.arange(total_len - 1, -1, -1, dtype=jnp.float32)
        pos_emb = self.pos_emb(pos_seq)  # (total_len, d_model)

        # Create causal mask with episode boundary handling
        # Query positions: 0..seq_len-1 attend to memory + preceding query positions
        # Mask shape: (batch, 1, seq_len, total_len)
        q_idx = jnp.arange(seq_len)[:, None]  # (seq_len, 1)
        kv_idx = jnp.arange(total_len)[None, :]  # (1, total_len)
        # Each query at position i can attend to all memory + positions 0..i in input
        # kv_idx < mem_len means memory (always attend without done handling)
        # kv_idx - mem_len <= q_idx means current or earlier position in query
        causal_mask = (kv_idx < mem_len) | ((kv_idx - mem_len) <= q_idx)
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq_len, total_len)

        # Episode boundary mask: prevent attending across done boundaries
        # dones: (batch, seq_len) - True when episode ends at that timestep
        # If done[t] = True, positions t+1 onwards should NOT attend to positions <= t
        #
        # Compute cumulative episode index: each done increments the episode number
        # cumsum_dones[t] = number of episode boundaries before position t
        # Two positions are in the same episode if they have the same cumsum value
        cumsum_dones = jnp.cumsum(dones, axis=1)  # (batch, seq_len)
        # For query position q and key position k (in sequence, not memory):
        # same_episode[q, k] = cumsum_dones[q] == cumsum_dones[k]
        # This means no done happened between k and q
        cumsum_q = cumsum_dones[:, :, None]  # (batch, seq_len, 1)
        cumsum_k = cumsum_dones[:, None, :]  # (batch, 1, seq_len)
        same_episode_seq = cumsum_q == cumsum_k  # (batch, seq_len, seq_len)

        # Extend to full kv range: memory positions + sequence positions
        # Memory positions: always in a "prior" episode, so mask them if ANY done in sequence
        # Sequence positions: use same_episode_seq
        any_done = dones.any(axis=1, keepdims=True)  # (batch, 1)
        # Memory can be attended only if no done occurred in the sequence
        memory_mask = ~any_done  # (batch, 1) - True if memory is valid
        memory_mask = jnp.broadcast_to(
            memory_mask[:, None, :], (batch_size, seq_len, mem_len)
        )  # (batch, seq_len, mem_len)
        # Combine memory mask and sequence episode mask
        episode_mask = jnp.concatenate(
            [memory_mask, same_episode_seq], axis=2
        )  # (batch, seq_len, total_len)
        episode_mask = episode_mask[:, None, :, :]  # (batch, 1, seq_len, total_len)

        # Final mask: causal AND same episode
        causal_mask = causal_mask & episode_mask  # (batch, 1, seq_len, total_len)

        # Compute the last episode start index for memory update
        # We want to keep only activations from the current (last) episode
        # Find the last done position in each batch, then the episode starts at last_done + 1
        # If no done in sequence, keep all activations
        # done_positions: position of last True in dones, or -1 if no done
        done_indices = jnp.where(dones, jnp.arange(seq_len), -1)  # (batch, seq_len)
        last_done_pos = done_indices.max(axis=1)  # (batch,) - last done position or -1

        new_memories = []
        for i, layer in enumerate(self.layers):
            memory = memories[i]

            # Reset input memory if any done in sequence (memory from prev call is stale)
            # Note: attention mask already handles this, but we also zero the memory
            # to ensure clean state for memory concatenation below
            memory = jnp.where(
                any_done[:, :, None],  # (batch, 1, 1)
                jnp.zeros_like(memory),
                memory,
            )

            # Forward through layer
            x = layer(x, memory, pos_emb, mask=causal_mask, deterministic=deterministic)

            # Update memory with sliding window, considering episode boundaries
            # We only want to keep activations from the current episode
            # Concatenate zeroed memory (if done occurred) and new activations
            combined = jnp.concatenate(
                [memory, x], axis=1
            )  # (batch, mem_len + seq_len, d_model)

            # For each batch, we want positions from (last_done_pos + 1) to end
            # But we take at most mem_len positions from the end
            # The combined array has: [memory (mem_len)] + [x (seq_len)]
            # Position of last_done in combined: mem_len + last_done_pos
            # Start of current episode in combined: mem_len + last_done_pos + 1
            # We want to zero out everything before the current episode start
            combined_pos = jnp.arange(total_len)  # (total_len,)
            episode_start = mem_len + last_done_pos + 1  # (batch,)
            # Mask: True for positions in current episode
            in_current_episode = combined_pos[None, :] >= episode_start[:, None]  # (batch, total_len)
            combined = jnp.where(
                in_current_episode[:, :, None],  # (batch, total_len, 1)
                combined,
                jnp.zeros_like(combined),
            )

            new_memory = combined[:, -self.mem_len :, :]  # (batch, mem_len, d_model)
            new_memories.append(new_memory)

        return new_memories, x
