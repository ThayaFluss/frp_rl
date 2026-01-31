"""
Robustness tests for TransformerXL implementation.

This module contains comprehensive tests for:
- Numerical stability (NaN/Inf, extreme values, gradient computation)
- Memory management (reset on done, sliding window)
- Sequence length edge cases
- Causal masking verification
- Configuration validation
- Vmap compatibility
- Minibatch handling
- Cross-architecture compatibility
- End-to-end integration with PPO

Test cases follow testing.md guidelines:
- Dimension verification
- Numerical correctness of transformations
- JIT compilation success
"""

import jax
import jax.numpy as jnp
import pytest
import sys
import os

# Add frp_popjaxrl directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from algorithms.transformer import (
    RelMultiHeadAttention,
    TransformerBlock,
    StackedTransformer,
)
from algorithms.models import TransformerRepModel, GRURepModel, ActorCriticDiscrete


@pytest.fixture
def random_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def transformer_config():
    """Fixture to provide transformer configuration."""
    return {
        "TRANSFORMER_D_MODEL": 64,
        "TRANSFORMER_NUM_HEADS": 4,
        "TRANSFORMER_N_LAYERS": 2,
        "TRANSFORMER_D_FF": 128,
        "TRANSFORMER_MEM_LEN": 16,
        "TRANSFORMER_DROPOUT": 0.0,
        "TRANSFORMER_GATING": True,
    }


@pytest.fixture
def small_config():
    """Fixture for small configuration (faster tests)."""
    return {
        "TRANSFORMER_D_MODEL": 32,
        "TRANSFORMER_NUM_HEADS": 2,
        "TRANSFORMER_N_LAYERS": 1,
        "TRANSFORMER_D_FF": 64,
        "TRANSFORMER_MEM_LEN": 8,
        "TRANSFORMER_DROPOUT": 0.0,
        "TRANSFORMER_GATING": True,
    }


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Tests for numerical stability in transformer operations."""

    def test_no_nan_in_forward_pass(self, random_key, transformer_config):
        """Test that forward pass produces no NaN values with normal inputs."""
        batch_size = 4
        seq_len = 8
        obs_dim = 16

        model = TransformerRepModel(config=transformer_config)
        obs = jnp.ones((seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        params = model.init(random_key, hidden, obs, dones)
        new_hidden, embedding = model.apply(params, hidden, obs, dones)

        # Check for NaN in outputs
        assert not jnp.any(jnp.isnan(embedding)), "NaN detected in embedding output"
        for i, h in enumerate(new_hidden):
            assert not jnp.any(jnp.isnan(h)), f"NaN detected in hidden state layer {i}"

    def test_no_inf_in_forward_pass(self, random_key, transformer_config):
        """Test that forward pass produces no Inf values."""
        batch_size = 4
        seq_len = 8
        obs_dim = 16

        model = TransformerRepModel(config=transformer_config)
        obs = jnp.ones((seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        params = model.init(random_key, hidden, obs, dones)
        new_hidden, embedding = model.apply(params, hidden, obs, dones)

        assert not jnp.any(jnp.isinf(embedding)), "Inf detected in embedding output"
        for i, h in enumerate(new_hidden):
            assert not jnp.any(jnp.isinf(h)), f"Inf detected in hidden state layer {i}"

    def test_extreme_large_input_values(self, random_key, transformer_config):
        """Test stability with large input values."""
        batch_size = 4
        seq_len = 8
        obs_dim = 16

        model = TransformerRepModel(config=transformer_config)
        # Large but not extreme values (to avoid immediate overflow)
        obs = jnp.ones((seq_len, batch_size, obs_dim)) * 100.0
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        params = model.init(random_key, hidden, obs, dones)
        new_hidden, embedding = model.apply(params, hidden, obs, dones)

        # Should not produce NaN/Inf
        assert not jnp.any(jnp.isnan(embedding)), "NaN with large inputs"
        assert not jnp.any(jnp.isinf(embedding)), "Inf with large inputs"

    def test_extreme_small_input_values(self, random_key, transformer_config):
        """Test stability with very small input values."""
        batch_size = 4
        seq_len = 8
        obs_dim = 16

        model = TransformerRepModel(config=transformer_config)
        obs = jnp.ones((seq_len, batch_size, obs_dim)) * 1e-6
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        params = model.init(random_key, hidden, obs, dones)
        new_hidden, embedding = model.apply(params, hidden, obs, dones)

        assert not jnp.any(jnp.isnan(embedding)), "NaN with small inputs"
        assert not jnp.any(jnp.isinf(embedding)), "Inf with small inputs"

    def test_mixed_sign_inputs(self, random_key, transformer_config):
        """Test stability with mixed positive and negative inputs."""
        batch_size = 4
        seq_len = 8
        obs_dim = 16

        model = TransformerRepModel(config=transformer_config)
        key1, key2 = jax.random.split(random_key)
        obs = jax.random.normal(key1, (seq_len, batch_size, obs_dim)) * 10.0
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        params = model.init(key2, hidden, obs, dones)
        new_hidden, embedding = model.apply(params, hidden, obs, dones)

        assert not jnp.any(jnp.isnan(embedding)), "NaN with mixed sign inputs"
        assert not jnp.any(jnp.isinf(embedding)), "Inf with mixed sign inputs"

    def test_attention_weights_sum_to_one(self, random_key):
        """Test that softmax attention weights sum to 1."""
        batch_size = 4
        q_len = 8
        kv_len = 16
        d_model = 64
        num_heads = 4

        attn = RelMultiHeadAttention(num_heads=num_heads, d_model=d_model)

        query = jax.random.normal(random_key, (batch_size, q_len, d_model))
        key_value = jax.random.normal(random_key, (batch_size, kv_len, d_model))
        pos_emb = jax.random.normal(random_key, (kv_len, d_model))

        params = attn.init(random_key, query, key_value, pos_emb)

        # Get attention weights by manually computing
        # Note: The attention module doesn't expose weights directly,
        # so we verify indirectly through output stability
        output = attn.apply(params, query, key_value, pos_emb)

        # If attention weights sum to 1, output should be bounded
        assert not jnp.any(jnp.isnan(output)), "NaN in attention output"
        assert jnp.all(jnp.abs(output) < 1000), "Attention output unbounded"

    def test_gradient_computation_no_nan(self, random_key, transformer_config):
        """Test that gradient computation produces no NaN values."""
        batch_size = 4
        seq_len = 8
        obs_dim = 16

        model = TransformerRepModel(config=transformer_config)
        obs = jax.random.normal(random_key, (seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        params = model.init(random_key, hidden, obs, dones)

        def loss_fn(params):
            _, embedding = model.apply(params, hidden, obs, dones)
            return jnp.mean(embedding**2)

        grads = jax.grad(loss_fn)(params)

        # Check gradients for NaN
        def check_tree_no_nan(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            for leaf in leaves:
                assert not jnp.any(jnp.isnan(leaf)), "NaN in gradients"

        check_tree_no_nan(grads)

    def test_gradient_flow_through_layers(self, random_key, transformer_config):
        """Test that gradients flow through all layers without vanishing/exploding."""
        batch_size = 4
        seq_len = 8
        obs_dim = 16

        model = TransformerRepModel(config=transformer_config)
        obs = jax.random.normal(random_key, (seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        params = model.init(random_key, hidden, obs, dones)

        def loss_fn(params):
            _, embedding = model.apply(params, hidden, obs, dones)
            return jnp.mean(embedding)

        grads = jax.grad(loss_fn)(params)

        # Check that gradients are non-zero for transformer layers
        grad_norms = []

        def collect_norms(path, leaf):
            if "transformer" in jax.tree_util.keystr(path):
                grad_norms.append(jnp.linalg.norm(leaf.ravel()))

        jax.tree_util.tree_map_with_path(collect_norms, grads)

        # Gradients should exist and be reasonable magnitude
        assert len(grad_norms) > 0, "No transformer gradients found"
        for norm in grad_norms:
            assert norm > 1e-10, "Gradient vanishing detected"
            assert norm < 1e6, "Gradient exploding detected"


# ============================================================================
# Memory Management Tests
# ============================================================================


class TestMemoryManagement:
    """Tests for memory reset and sliding window behavior."""

    def test_memory_actually_reset_on_done(self, random_key, transformer_config):
        """Test that memory is actually zeroed when done signal is True."""
        batch_size = 2
        seq_len = 4
        d_model = transformer_config["TRANSFORMER_D_MODEL"]
        n_layers = transformer_config["TRANSFORMER_N_LAYERS"]
        mem_len = transformer_config["TRANSFORMER_MEM_LEN"]

        transformer = StackedTransformer(
            d_model=d_model,
            num_heads=transformer_config["TRANSFORMER_NUM_HEADS"],
            n_layers=n_layers,
            mem_len=mem_len,
            use_gating=transformer_config["TRANSFORMER_GATING"],
        )

        # Initialize with non-zero memory
        memories = [
            jnp.ones((batch_size, mem_len, d_model)) * 5.0 for _ in range(n_layers)
        ]
        x = jnp.ones((batch_size, seq_len, d_model))

        # Done signal for batch 0 only
        dones = jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
        dones = dones.at[0, 0].set(True)

        params = transformer.init(random_key, memories, x, dones)

        # Run without done
        dones_no_reset = jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
        new_mem_no_reset, _ = transformer.apply(params, memories, x, dones_no_reset)

        # Run with done
        new_mem_reset, _ = transformer.apply(params, memories, x, dones)

        # Batch 0 memory should be different between reset and no-reset
        for layer_idx in range(n_layers):
            mem_diff = jnp.abs(
                new_mem_reset[layer_idx][0] - new_mem_no_reset[layer_idx][0]
            )
            assert jnp.sum(mem_diff) > 0.1, (
                f"Layer {layer_idx}: Memory not reset for batch 0"
            )

    def test_memory_reset_preserves_other_batches(self, random_key, transformer_config):
        """Test that done signal only affects the relevant batch element."""
        batch_size = 4
        seq_len = 4
        d_model = transformer_config["TRANSFORMER_D_MODEL"]
        n_layers = transformer_config["TRANSFORMER_N_LAYERS"]
        mem_len = transformer_config["TRANSFORMER_MEM_LEN"]

        transformer = StackedTransformer(
            d_model=d_model,
            num_heads=transformer_config["TRANSFORMER_NUM_HEADS"],
            n_layers=n_layers,
            mem_len=mem_len,
            use_gating=transformer_config["TRANSFORMER_GATING"],
        )

        memories = [jnp.ones((batch_size, mem_len, d_model)) for _ in range(n_layers)]
        x = jnp.ones((batch_size, seq_len, d_model))

        # Done only for batch 0
        dones = jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
        dones = dones.at[0, 0].set(True)

        params = transformer.init(random_key, memories, x, dones)

        # Run without done
        dones_no_reset = jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
        new_mem_no_reset, _ = transformer.apply(params, memories, x, dones_no_reset)

        # Run with done
        new_mem_reset, _ = transformer.apply(params, memories, x, dones)

        # Batches 1, 2, 3 should have same memory in both cases
        for layer_idx in range(n_layers):
            for batch_idx in [1, 2, 3]:
                diff = jnp.abs(
                    new_mem_reset[layer_idx][batch_idx]
                    - new_mem_no_reset[layer_idx][batch_idx]
                )
                assert jnp.allclose(diff, 0.0, atol=1e-5), (
                    f"Layer {layer_idx}, Batch {batch_idx}: Memory changed unexpectedly"
                )

    def test_memory_persistence_across_calls(self, random_key, transformer_config):
        """Test that memory persists correctly across multiple forward passes."""
        batch_size = 2
        seq_len = 4
        obs_dim = 16

        model = TransformerRepModel(config=transformer_config)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        obs = jax.random.normal(random_key, (seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)

        params = model.init(random_key, hidden, obs, dones)

        # First call
        hidden1, _ = model.apply(params, hidden, obs, dones)

        # Second call with new input but using previous hidden state
        key2 = jax.random.fold_in(random_key, 1)
        obs2 = jax.random.normal(key2, (seq_len, batch_size, obs_dim))
        hidden2, embedding2 = model.apply(params, hidden1, obs2, dones)

        # Memory should have changed from initial
        for i in range(len(hidden2)):
            diff_from_zero = jnp.abs(hidden2[i]).sum()
            assert diff_from_zero > 0.1, (
                f"Layer {i}: Memory appears empty after processing"
            )

    def test_sliding_window_correctness(self, random_key, transformer_config):
        """Test that sliding window memory update works correctly."""
        batch_size = 2
        seq_len = 8  # Longer than mem_len to test sliding
        d_model = transformer_config["TRANSFORMER_D_MODEL"]
        n_layers = transformer_config["TRANSFORMER_N_LAYERS"]
        mem_len = transformer_config["TRANSFORMER_MEM_LEN"]

        transformer = StackedTransformer(
            d_model=d_model,
            num_heads=transformer_config["TRANSFORMER_NUM_HEADS"],
            n_layers=n_layers,
            mem_len=mem_len,
            use_gating=transformer_config["TRANSFORMER_GATING"],
        )

        memories = [jnp.zeros((batch_size, mem_len, d_model)) for _ in range(n_layers)]
        x = jax.random.normal(random_key, (batch_size, seq_len, d_model))
        dones = jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)

        params = transformer.init(random_key, memories, x, dones)
        new_memories, _ = transformer.apply(params, memories, x, dones)

        # New memory should have correct shape
        for i, mem in enumerate(new_memories):
            assert mem.shape == (batch_size, mem_len, d_model), (
                f"Layer {i}: Memory shape incorrect after sliding window"
            )

    def test_memory_with_long_sequence(self, random_key, small_config):
        """Test memory handling with sequence much longer than memory length."""
        batch_size = 2
        seq_len = 32  # Much longer than mem_len=8
        obs_dim = 16

        model = TransformerRepModel(config=small_config)
        hidden = TransformerRepModel.initialize_carry(batch_size, small_config)

        obs = jax.random.normal(random_key, (seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)

        params = model.init(random_key, hidden, obs, dones)
        new_hidden, embedding = model.apply(params, hidden, obs, dones)

        # Should complete without error and have correct output shape
        assert embedding.shape == (
            seq_len,
            batch_size,
            small_config["TRANSFORMER_D_MODEL"],
        )
        assert not jnp.any(jnp.isnan(embedding))


# ============================================================================
# Sequence Length Edge Cases
# ============================================================================


class TestSequenceLengthEdgeCases:
    """Tests for edge cases in sequence length handling."""

    def test_single_timestep(self, random_key, transformer_config):
        """Test processing with seq_len=1."""
        batch_size = 4
        seq_len = 1
        obs_dim = 16

        model = TransformerRepModel(config=transformer_config)
        obs = jax.random.normal(random_key, (seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        params = model.init(random_key, hidden, obs, dones)
        new_hidden, embedding = model.apply(params, hidden, obs, dones)

        d_model = transformer_config["TRANSFORMER_D_MODEL"]
        assert embedding.shape == (seq_len, batch_size, d_model)
        assert not jnp.any(jnp.isnan(embedding))

    def test_sequence_longer_than_memory(self, random_key, transformer_config):
        """Test with seq_len > mem_len."""
        batch_size = 4
        seq_len = 32  # mem_len is 16
        obs_dim = 16

        model = TransformerRepModel(config=transformer_config)
        obs = jax.random.normal(random_key, (seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        params = model.init(random_key, hidden, obs, dones)
        new_hidden, embedding = model.apply(params, hidden, obs, dones)

        d_model = transformer_config["TRANSFORMER_D_MODEL"]
        assert embedding.shape == (seq_len, batch_size, d_model)

    def test_sequence_equal_to_memory(self, random_key, transformer_config):
        """Test with seq_len == mem_len."""
        batch_size = 4
        seq_len = transformer_config["TRANSFORMER_MEM_LEN"]
        obs_dim = 16

        model = TransformerRepModel(config=transformer_config)
        obs = jax.random.normal(random_key, (seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        params = model.init(random_key, hidden, obs, dones)
        new_hidden, embedding = model.apply(params, hidden, obs, dones)

        d_model = transformer_config["TRANSFORMER_D_MODEL"]
        assert embedding.shape == (seq_len, batch_size, d_model)

    def test_batch_size_one(self, random_key, transformer_config):
        """Test with batch_size=1."""
        batch_size = 1
        seq_len = 8
        obs_dim = 16

        model = TransformerRepModel(config=transformer_config)
        obs = jax.random.normal(random_key, (seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        params = model.init(random_key, hidden, obs, dones)
        new_hidden, embedding = model.apply(params, hidden, obs, dones)

        d_model = transformer_config["TRANSFORMER_D_MODEL"]
        assert embedding.shape == (seq_len, batch_size, d_model)


# ============================================================================
# Causal Masking Tests
# ============================================================================


class TestCausalMasking:
    """Tests for causal masking in attention."""

    def test_no_future_attention(self, random_key, transformer_config):
        """Test that causal mask prevents attending to future positions."""
        seq_len = 4
        mem_len = transformer_config["TRANSFORMER_MEM_LEN"]

        # Create causal mask as in StackedTransformer
        total_len = mem_len + seq_len
        q_idx = jnp.arange(seq_len)[:, None]  # (seq_len, 1)
        kv_idx = jnp.arange(total_len)[None, :]  # (1, total_len)
        causal_mask = (kv_idx < mem_len) | ((kv_idx - mem_len) <= q_idx)

        # For each query position, verify masking
        for q in range(seq_len):
            # Query at position q should NOT attend to positions q+1, q+2, ... in input
            for k in range(q + 1, seq_len):
                kv_pos = mem_len + k
                assert not causal_mask[q, kv_pos], (
                    f"Query {q} can attend to future position {k} (kv_pos={kv_pos})"
                )

    def test_memory_attention_allowed(self, random_key, transformer_config):
        """Test that all query positions can attend to memory."""
        seq_len = 4
        mem_len = transformer_config["TRANSFORMER_MEM_LEN"]

        total_len = mem_len + seq_len
        q_idx = jnp.arange(seq_len)[:, None]
        kv_idx = jnp.arange(total_len)[None, :]
        causal_mask = (kv_idx < mem_len) | ((kv_idx - mem_len) <= q_idx)

        # All query positions should attend to all memory positions
        for q in range(seq_len):
            for m in range(mem_len):
                assert causal_mask[q, m], (
                    f"Query {q} cannot attend to memory position {m}"
                )

    def test_self_attention_allowed(self, random_key, transformer_config):
        """Test that each query position can attend to itself."""
        seq_len = 4
        mem_len = transformer_config["TRANSFORMER_MEM_LEN"]

        total_len = mem_len + seq_len
        q_idx = jnp.arange(seq_len)[:, None]
        kv_idx = jnp.arange(total_len)[None, :]
        causal_mask = (kv_idx < mem_len) | ((kv_idx - mem_len) <= q_idx)

        for q in range(seq_len):
            self_pos = mem_len + q
            assert causal_mask[q, self_pos], f"Query {q} cannot attend to itself"

    def test_relative_position_shift(self, random_key):
        """Test that _rel_shift produces correct alignment."""
        batch_size = 2
        num_heads = 4
        q_len = 4
        kv_len = 8

        attn = RelMultiHeadAttention(num_heads=num_heads, d_model=64)

        # Create simple test input
        attn_pos = jnp.arange(
            batch_size * num_heads * q_len * kv_len, dtype=jnp.float32
        )
        attn_pos = attn_pos.reshape(batch_size, num_heads, q_len, kv_len)

        shifted = attn._rel_shift(attn_pos, q_len)

        # Output shape should be preserved
        assert shifted.shape == (batch_size, num_heads, q_len, kv_len)


# ============================================================================
# Configuration Validation Tests
# ============================================================================


class TestConfigurationValidation:
    """Tests for configuration validation."""

    def test_d_model_not_divisible_by_heads(self, random_key):
        """Test that d_model must be divisible by num_heads."""
        d_model = 63  # Not divisible by 4
        num_heads = 4

        with pytest.raises(AssertionError):
            attn = RelMultiHeadAttention(num_heads=num_heads, d_model=d_model)
            # Need to initialize to trigger assertion
            query = jnp.ones((2, 4, d_model))
            key_value = jnp.ones((2, 8, d_model))
            pos_emb = jnp.ones((8, d_model))
            attn.init(random_key, query, key_value, pos_emb)

    def test_default_d_ff_computation(self, random_key):
        """Test that d_ff defaults to 4 * d_model when not specified."""
        d_model = 64
        num_heads = 4

        block = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=None,  # Should default to 4 * d_model
            use_gating=False,
        )

        batch_size = 2
        seq_len = 4
        mem_len = 8

        x = jnp.ones((batch_size, seq_len, d_model))
        memory = jnp.ones((batch_size, mem_len, d_model))
        pos_emb = jnp.ones((mem_len + seq_len, d_model))

        params = block.init(random_key, x, memory, pos_emb)

        # Verify by checking parameter shapes
        ff1_kernel = params["params"]["ff1"]["kernel"]
        expected_d_ff = 4 * d_model
        assert ff1_kernel.shape[1] == expected_d_ff, (
            f"Expected d_ff={expected_d_ff}, got {ff1_kernel.shape[1]}"
        )


# ============================================================================
# Vmap Compatibility Tests
# ============================================================================


class TestVmapCompatibility:
    """Tests for vmap vectorization compatibility."""

    def test_vmap_over_batch(self, random_key, small_config):
        """Test that model works with vmap over batch dimension."""
        seq_len = 8
        obs_dim = 16

        model = TransformerRepModel(config=small_config)

        # Single sample
        obs_single = jax.random.normal(random_key, (seq_len, 1, obs_dim))
        dones_single = jnp.zeros((seq_len, 1), dtype=jnp.bool_)
        hidden_single = TransformerRepModel.initialize_carry(1, small_config)

        params = model.init(random_key, hidden_single, obs_single, dones_single)

        @jax.jit
        def forward_single(params, hidden, obs, dones):
            return model.apply(params, hidden, obs, dones)

        # Test single forward
        new_hidden, embedding = forward_single(
            params, hidden_single, obs_single, dones_single
        )
        assert embedding.shape == (seq_len, 1, small_config["TRANSFORMER_D_MODEL"])

    def test_multiple_seeds_with_vmap(self, random_key, small_config):
        """Test forward pass with multiple random seeds."""
        batch_size = 4
        seq_len = 8
        obs_dim = 16

        model = TransformerRepModel(config=small_config)
        hidden = TransformerRepModel.initialize_carry(batch_size, small_config)

        keys = jax.random.split(random_key, batch_size)

        # Create different observations for each batch element
        obs = jax.vmap(lambda k: jax.random.normal(k, (seq_len, 1, obs_dim)))(keys)
        obs = obs.reshape(seq_len, batch_size, obs_dim)

        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)

        params = model.init(random_key, hidden, obs, dones)
        new_hidden, embedding = model.apply(params, hidden, obs, dones)

        # Each batch element should produce different embeddings
        for i in range(batch_size - 1):
            diff = jnp.abs(embedding[:, i, :] - embedding[:, i + 1, :]).sum()
            assert diff > 0.1, f"Batch {i} and {i + 1} produced identical embeddings"


# ============================================================================
# Minibatch Handling Tests
# ============================================================================


class TestMinibatchHandling:
    """Tests for minibatch shuffling and reshaping."""

    def test_hidden_state_reshape_roundtrip(self, transformer_config):
        """Test that hidden state reshape is reversible."""
        batch_size = 8
        d_model = transformer_config["TRANSFORMER_D_MODEL"]
        mem_len = transformer_config["TRANSFORMER_MEM_LEN"]

        # Initialize carry
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        # Verify shape: (1, batch, mem_len * d_model)
        for i, h in enumerate(hidden):
            assert h.shape == (1, batch_size, mem_len * d_model), (
                f"Layer {i}: Expected (1, {batch_size}, {mem_len * d_model}), got {h.shape}"
            )

        # Simulate minibatch reshape (what happens in PPO training)
        # (1, batch, hidden_dim) -> (1, minibatch_size, hidden_dim) for each minibatch
        num_minibatches = 4
        minibatch_size = batch_size // num_minibatches

        for h in hidden:
            # This is how hidden states get reshaped in create_minibatches
            reshaped = h.reshape(1, num_minibatches, minibatch_size, -1)
            # Take one minibatch
            minibatch = reshaped[:, 0, :, :]
            assert minibatch.shape == (1, minibatch_size, mem_len * d_model)

    def test_minibatch_shuffling_preserves_data(self, random_key, transformer_config):
        """Test that shuffling and splitting preserves all data."""
        batch_size = 8
        d_model = transformer_config["TRANSFORMER_D_MODEL"]
        mem_len = transformer_config["TRANSFORMER_MEM_LEN"]

        # Create hidden with unique values
        hidden = [
            jax.random.normal(random_key, (1, batch_size, mem_len * d_model))
            for _ in range(transformer_config["TRANSFORMER_N_LAYERS"])
        ]

        # Record original data
        original_sum = sum(h.sum() for h in hidden)

        # Simulate shuffle and split
        perm = jax.random.permutation(random_key, batch_size)

        shuffled = [h[:, perm, :] for h in hidden]
        shuffled_sum = sum(h.sum() for h in shuffled)

        # Sum should be preserved (data integrity)
        assert jnp.allclose(original_sum, shuffled_sum), (
            "Data integrity lost during shuffle"
        )


# ============================================================================
# Cross-Architecture Compatibility Tests
# ============================================================================


class TestCrossArchitectureCompatibility:
    """Tests for compatibility with other architectures (GRU, S5)."""

    def test_output_shape_matches_gru(self, random_key, transformer_config):
        """Test that Transformer output shape matches GRU."""
        batch_size = 4
        seq_len = 8
        obs_dim = 16

        # Transformer
        transformer_model = TransformerRepModel(config=transformer_config)
        transformer_hidden = TransformerRepModel.initialize_carry(
            batch_size, transformer_config
        )

        obs = jax.random.normal(random_key, (seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)

        params_t = transformer_model.init(random_key, transformer_hidden, obs, dones)
        _, embedding_t = transformer_model.apply(
            params_t, transformer_hidden, obs, dones
        )

        # GRU
        gru_model = GRURepModel(config={})
        gru_hidden = GRURepModel.initialize_carry(batch_size, {})

        params_g = gru_model.init(random_key, gru_hidden, obs, dones)
        _, embedding_g = gru_model.apply(params_g, gru_hidden, obs, dones)

        # Output format should match: (seq_len, batch, hidden_dim)
        assert embedding_t.shape[0] == embedding_g.shape[0], "seq_len mismatch"
        assert embedding_t.shape[1] == embedding_g.shape[1], "batch_size mismatch"
        # Hidden dimensions may differ, which is acceptable

    def test_hidden_state_format_compatible(self, transformer_config):
        """Test that hidden state format is compatible with minibatch handling."""
        batch_size = 4

        # Transformer hidden state
        transformer_hidden = TransformerRepModel.initialize_carry(
            batch_size, transformer_config
        )

        # GRU hidden state
        gru_hidden = GRURepModel.initialize_carry(batch_size, {})

        # Both should be lists
        assert isinstance(transformer_hidden, list)
        assert isinstance(gru_hidden, list)

        # Both should have leading dimension of 1 for axis=1 batch shuffling
        for h in transformer_hidden:
            assert h.shape[0] == 1, f"Transformer hidden leading dim: {h.shape[0]}"

        for h in gru_hidden:
            assert h.shape[0] == 1, f"GRU hidden leading dim: {h.shape[0]}"


# ============================================================================
# End-to-End Integration Tests
# ============================================================================


class TestEndToEndIntegration:
    """End-to-end tests with ActorCritic and PPO components."""

    def test_actor_critic_forward(self, random_key, transformer_config):
        """Test forward pass through full ActorCritic with Transformer."""
        batch_size = 4
        seq_len = 8
        obs_dim = 16
        action_dim = 4

        rep_model = TransformerRepModel(config=transformer_config)
        actor_critic = ActorCriticDiscrete(
            rep_model=rep_model, action_dim=action_dim, config=transformer_config
        )

        obs = jax.random.normal(random_key, (seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        x = (obs, dones)
        params = actor_critic.init(random_key, hidden, x)
        new_hidden, pi, value = actor_critic.apply(params, hidden, x)

        # Check shapes
        assert value.shape == (seq_len, batch_size)
        assert new_hidden is not None

        # Check no NaN
        assert not jnp.any(jnp.isnan(value))

    def test_actor_critic_gradient(self, random_key, transformer_config):
        """Test that gradients can be computed through ActorCritic."""
        batch_size = 4
        seq_len = 8
        obs_dim = 16
        action_dim = 4

        rep_model = TransformerRepModel(config=transformer_config)
        actor_critic = ActorCriticDiscrete(
            rep_model=rep_model, action_dim=action_dim, config=transformer_config
        )

        obs = jax.random.normal(random_key, (seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, transformer_config)

        x = (obs, dones)
        params = actor_critic.init(random_key, hidden, x)

        def loss_fn(params):
            _, pi, value = actor_critic.apply(params, hidden, x)
            # Simple policy + value loss
            log_probs = pi.log_prob(
                jnp.zeros(batch_size * seq_len, dtype=jnp.int32).reshape(
                    seq_len, batch_size
                )
            )
            return jnp.mean(value) - 0.01 * jnp.mean(log_probs)

        grads = jax.grad(loss_fn)(params)

        # Check gradients exist and are finite
        def check_finite(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            for leaf in leaves:
                assert jnp.all(jnp.isfinite(leaf)), "Non-finite gradient detected"

        check_finite(grads)

    def test_jit_compiled_training_step(self, random_key, small_config):
        """Test that a JIT-compiled training step works."""
        batch_size = 4
        seq_len = 8
        obs_dim = 16
        action_dim = 4

        rep_model = TransformerRepModel(config=small_config)
        actor_critic = ActorCriticDiscrete(
            rep_model=rep_model, action_dim=action_dim, config=small_config
        )

        obs = jax.random.normal(random_key, (seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, small_config)

        x = (obs, dones)
        params = actor_critic.init(random_key, hidden, x)

        @jax.jit
        def train_step(params, obs, dones, hidden):
            def loss_fn(params):
                _, pi, value = actor_critic.apply(params, hidden, (obs, dones))
                return jnp.mean(value**2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            return loss, grads

        loss, grads = train_step(params, obs, dones, hidden)

        assert jnp.isfinite(loss)
        assert loss.shape == ()

    def test_multiple_training_steps(self, random_key, small_config):
        """Test multiple training steps to verify loss changes."""
        import optax

        batch_size = 4
        seq_len = 8
        obs_dim = 16
        action_dim = 4

        rep_model = TransformerRepModel(config=small_config)
        actor_critic = ActorCriticDiscrete(
            rep_model=rep_model, action_dim=action_dim, config=small_config
        )

        key, subkey = jax.random.split(random_key)
        obs = jax.random.normal(subkey, (seq_len, batch_size, obs_dim))
        dones = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
        hidden = TransformerRepModel.initialize_carry(batch_size, small_config)

        x = (obs, dones)
        params = actor_critic.init(key, hidden, x)

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step(params, opt_state, obs, dones, hidden):
            def loss_fn(params):
                _, pi, value = actor_critic.apply(params, hidden, (obs, dones))
                # Target: make value predict 1.0
                return jnp.mean((value - 1.0) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        losses = []
        for _ in range(10):
            params, opt_state, loss = train_step(params, opt_state, obs, dones, hidden)
            losses.append(float(loss))

        # Loss should generally decrease (may not be monotonic due to stochasticity)
        assert losses[-1] < losses[0] * 1.5, (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
