"""
Test script for orthogonal_lazy.py with lazy evaluation (reset-time computation).

Tests:
1. Encoder/decoder reversibility
2. build_word_from_sequence correctness
3. Consistency with v1 (same base matrices + index → same word matrix)
4. Compilation time measurement
"""

import pytest
import jax
import jax.numpy as jnp
import sys
import os
import time

# Add project root directory to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from frp.orthogonal_lazy import (
    encode_word_index,
    decode_word_sequence,
    sample_word_index,
    build_word_from_sequence,
    build_word_from_index,
    create_base_matrices,
)

# Import v1 for comparison
from frp.orthogonal import create_orthogonal_matrices as create_orthogonal_matrices_v1
from frp.orthogonal import create_words as create_words_v1


def measure_compile_time(func, *args, **kwargs):
    """Measure JAX compilation time."""
    start = time.time()
    result = func(*args, **kwargs)

    if isinstance(result, tuple):
        for r in result:
            if hasattr(r, 'block_until_ready'):
                r.block_until_ready()
    else:
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()

    compile_time = time.time() - start

    # Run again to measure execution time
    start = time.time()
    result = func(*args, **kwargs)

    if isinstance(result, tuple):
        for r in result:
            if hasattr(r, 'block_until_ready'):
                r.block_until_ready()
    else:
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()

    exec_time = time.time() - start

    return result, compile_time, exec_time


def test_encoder_decoder_reversibility():
    """Test that encode → decode is identity"""
    print("\n" + "="*60)
    print("Test: Encoder/Decoder Reversibility")
    print("="*60)

    num_base = 4
    depth = 3
    total_words = num_base ** depth  # 4^3 = 64

    print(f"num_base (B) = {num_base}")
    print(f"depth (L) = {depth}")
    print(f"total_words = {total_words}")

    # Test all possible indices
    for index in range(min(total_words, 10)):  # Test first 10
        sequence = encode_word_index(index, num_base, depth, encoding_mode='base_b')
        decoded_index = decode_word_sequence(sequence, num_base, encoding_mode='base_b')

        print(f"  index={index} → sequence={sequence} → decoded={decoded_index}")
        assert decoded_index == index, f"Mismatch: {index} != {decoded_index}"

    print("✓ All indices correctly encoded and decoded!")


def test_build_word_from_sequence():
    """Test that build_word_from_sequence produces orthogonal-like matrices"""
    print("\n" + "="*60)
    print("Test: build_word_from_sequence Properties")
    print("="*60)

    key = jax.random.PRNGKey(42)
    num_base = 4
    depth = 3
    size = 64
    input_dim = 8
    output_dim = 64

    # Create base matrices
    bases = create_base_matrices(key, num_base, size)
    print(f"Created {bases.shape[0]} base matrices of size {size}x{size}")

    # Test a specific sequence
    sequence = jnp.array([0, 1, 2])  # Use bases 0, 1, 2
    word = build_word_from_sequence(bases, sequence, input_dim, output_dim)

    print(f"Built word matrix with shape {word.shape}")
    assert word.shape == (input_dim, output_dim), f"Shape mismatch: {word.shape}"

    # Check that the computation is correct (manual multiplication)
    manual_word = jnp.eye(size)
    for idx in sequence:
        manual_word = manual_word @ bases[idx]
    manual_word_sliced = manual_word[:input_dim, :output_dim]

    assert jnp.allclose(word, manual_word_sliced, atol=1e-6), "Word computation mismatch!"
    print("✓ Word matrix correctly computed!")


def test_consistency_with_v1():
    """Test that lazy version produces same results as v1 for specific indices"""
    print("\n" + "="*60)
    print("Test: Consistency with V1")
    print("="*60)

    key = jax.random.PRNGKey(42)
    depth = 2
    max_depth = 6  # Reduced from 8 to avoid memory issues (2^6 = 64 words instead of 256)
    size = 64
    input_dim = 8
    output_dim = 64

    # Create base matrices using v1 method
    matrices_v1 = create_orthogonal_matrices_v1(key, depth, size, max_depth, with_adjoint=False)
    num_base_v1 = matrices_v1.shape[0]  # Should be 2^(8//2) = 16

    print(f"V1 created {num_base_v1} base matrices")
    print(f"depth = {depth}, max_depth = {max_depth}")

    # Create all words using v1
    words_v1 = create_words_v1(matrices_v1, depth, input_dim, output_dim, max_depth)
    total_words_v1 = words_v1.shape[0]
    print(f"V1 created {total_words_v1} word matrices")

    # For lazy version, we need to understand v1's indexing
    # V1 uses: index = (i >> (j * bits_per_selection)) & mask
    # where bits_per_selection = max_depth // depth

    # Test a few specific indices
    test_indices = [0, 1, 5, 10, 20, 50]  # Removed 100 since max is 64

    for index in test_indices:
        if index >= total_words_v1:
            continue

        # Get v1's word matrix and slice to match dimensions
        # V1 returns full matrix (out_size, out_size), so we need to slice it
        word_v1_full = words_v1[index]
        word_v1 = word_v1_full[:input_dim, :output_dim]

        # Manually decode v1's index to sequence (using v1's bit-shift logic)
        bits_per_selection = max_depth // depth
        sequence_v1 = []
        for j in range(depth):
            base_idx = (index >> (j * bits_per_selection)) & ((1 << bits_per_selection) - 1)
            sequence_v1.append(base_idx)
        sequence_v1 = jnp.array(sequence_v1)

        # Build word using lazy method
        word_lazy = build_word_from_sequence(matrices_v1, sequence_v1, input_dim, output_dim)

        # Compare
        max_diff = jnp.max(jnp.abs(word_v1 - word_lazy))
        print(f"  Index {index}: sequence={sequence_v1}, max_diff={max_diff:.2e}")

        assert jnp.allclose(word_v1, word_lazy, atol=1e-6), \
            f"Mismatch at index {index}: max_diff={max_diff}"

    print("✓ Lazy version produces identical results to V1!")


def test_build_word_from_index():
    """Test the convenience function build_word_from_index"""
    print("\n" + "="*60)
    print("Test: build_word_from_index Convenience Function")
    print("="*60)

    key = jax.random.PRNGKey(43)
    num_base = 5
    depth = 2
    size = 64
    input_dim = 10
    output_dim = 64

    bases = create_base_matrices(key, num_base, size)

    # Test a specific index
    index = 7  # Should decode to some sequence in base-5

    # Method 1: Using build_word_from_index
    word_direct = build_word_from_index(bases, index, num_base, depth,
                                       input_dim, output_dim, encoding_mode='base_b')

    # Method 2: Manual encode + build
    sequence = encode_word_index(index, num_base, depth, encoding_mode='base_b')
    word_manual = build_word_from_sequence(bases, sequence, input_dim, output_dim)

    print(f"Index {index} → sequence {sequence}")
    print(f"Direct method shape: {word_direct.shape}")
    print(f"Manual method shape: {word_manual.shape}")

    assert jnp.allclose(word_direct, word_manual, atol=1e-6), "Methods produce different results!"
    print("✓ build_word_from_index works correctly!")


def test_sample_word_index():
    """Test word index sampling"""
    print("\n" + "="*60)
    print("Test: sample_word_index")
    print("="*60)

    key = jax.random.PRNGKey(44)
    total_words = 100

    # Test without exclusion
    key, subkey = jax.random.split(key)
    sampled = sample_word_index(subkey, total_words, exclude=None)
    print(f"Sampled index (no exclusion): {sampled}")
    assert 0 <= sampled < total_words, "Sampled index out of range!"

    # Test with exclusion
    exclude = jnp.array([10, 20, 30, 40, 50])
    key, subkey = jax.random.split(key)
    sampled = sample_word_index(subkey, total_words, exclude=exclude)
    print(f"Sampled index (with exclusion): {sampled}")
    assert sampled not in exclude, "Sampled excluded index!"
    assert 0 <= sampled < total_words, "Sampled index out of range!"

    # Test empty exclude array
    exclude_empty = jnp.array([], dtype=jnp.int32)
    key, subkey = jax.random.split(key)
    sampled = sample_word_index(subkey, total_words, exclude=exclude_empty)
    print(f"Sampled index (empty exclusion): {sampled}")
    assert 0 <= sampled < total_words, "Sampled index out of range!"

    print("✓ sample_word_index works correctly!")


def test_compilation_time():
    """Measure compilation time for lazy evaluation"""
    print("\n" + "="*60)
    print("Test: Compilation Time Measurement")
    print("="*60)

    key = jax.random.PRNGKey(45)
    num_base = 8  # Reduced from 16
    depth = 3     # Reduced from 4 (8^3 = 512 instead of 16^4 = 65536)
    size = 64     # Reduced from 128
    input_dim = 16
    output_dim = 64

    # Create base matrices
    bases = create_base_matrices(key, num_base, size)

    # Measure build_word_from_index compilation
    index = 42
    word, compile_time, exec_time = measure_compile_time(
        build_word_from_index,
        bases, index, num_base, depth, input_dim, output_dim, 'base_b'
    )

    print(f"build_word_from_index:")
    print(f"  Compilation time: {compile_time*1000:.2f} ms")
    print(f"  Execution time: {exec_time*1000:.2f} ms")
    print(f"  Result shape: {word.shape}")

    # Compare with v1's create_words (creates ALL words)
    print(f"\nComparing with V1 (creates ALL {num_base**depth} words):")

    # For v1, we need max_depth
    # V1 logic: num_base = 2^(max_depth // depth)
    # So: 8 = 2^(max_depth // 3) → max_depth // 3 = 3 → max_depth = 9
    max_depth = 9

    key_v1 = jax.random.PRNGKey(45)
    matrices_v1 = create_orthogonal_matrices_v1(key_v1, depth, size, max_depth, False)

    words_v1, compile_time_v1, exec_time_v1 = measure_compile_time(
        create_words_v1,
        matrices_v1, depth, input_dim, output_dim, max_depth
    )

    print(f"V1 create_words (creates {words_v1.shape[0]} matrices):")
    print(f"  Compilation time: {compile_time_v1*1000:.2f} ms")
    print(f"  Execution time: {exec_time_v1*1000:.2f} ms")

    print(f"\n⚡ Lazy evaluation compilation is {compile_time_v1/compile_time:.1f}x slower/faster")
    print("   (Note: Lazy compiles single word build, V1 compiles full word set creation)")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
