######
### The core code for Free Random Projection (Lazy Evaluation)
### Computes word matrices on-demand at reset time, avoiding precomputation of all words
######

import jax.numpy as jnp
import jax
import flax.linen as nn
from typing import Tuple, Optional
from flax.linen.initializers import constant


# ============================================
# Word Set Operations (separated from meta_env)
# ============================================

def encode_word_index(index: int, num_base: int, depth: int,
                      encoding_mode: str = 'base_b') -> jnp.ndarray:
    """
    Encode linear word index to base matrix indices sequence.

    Args:
        index: Linear word index (0 to total_words-1)
        num_base: Number of base matrices (B)
        depth: Depth of composition (L)
        encoding_mode: 'base_b' (base-B representation) or 'bitshift' (backward compat)

    Returns:
        jnp.ndarray of shape (depth,) containing base matrix indices

    Example:
        index=7, num_base=4, depth=2
        → 7 in base-4 is [3, 1] (7 = 3*4^0 + 1*4^1)
        → sequence = [3, 1]
    """
    if encoding_mode == 'bitshift':
        # Backward compatibility mode - requires max_depth
        raise NotImplementedError("bitshift mode requires max_depth parameter")

    # Base-B representation (default)
    # Extract digits: index = d0 + d1*B + d2*B^2 + ... + d(L-1)*B^(L-1)
    def extract_digit(carry, j):
        idx = carry
        digit = idx % num_base
        next_idx = idx // num_base
        return next_idx, digit

    _, digits = jax.lax.scan(extract_digit, index, jnp.arange(depth))
    return digits


def decode_word_sequence(sequence: jnp.ndarray, num_base: int,
                         encoding_mode: str = 'base_b') -> int:
    """
    Decode base matrix indices sequence back to linear word index.

    This is mainly for testing/debugging to verify encode/decode reversibility.

    Args:
        sequence: Base matrix indices, shape (depth,)
        num_base: Number of base matrices (B)
        encoding_mode: 'base_b' or 'bitshift'

    Returns:
        Linear word index
    """
    if encoding_mode == 'base_b':
        # Convert base-B digits to decimal
        # index = sequence[0]*B^0 + sequence[1]*B^1 + ... + sequence[L-1]*B^(L-1)
        powers = num_base ** jnp.arange(len(sequence))
        return jnp.sum(sequence * powers).astype(jnp.int32)
    else:
        raise NotImplementedError("bitshift decode not implemented")


def sample_word_index(key, total_words: int, exclude: Optional[jnp.ndarray] = None) -> int:
    """
    Sample a word index from the word set, optionally excluding certain indices.

    This is the successor to random_choice from v1, but separated from env logic.

    Args:
        key: JAX random key
        total_words: Total number of words (N or B^L)
        exclude: Optional array of indices to exclude

    Returns:
        Sampled word index (or -1 if no valid choices)
    """
    # If exclude is empty or None, just choose from all words
    if exclude is None or exclude.shape[0] == 0:
        return jax.random.randint(key, (), 0, total_words).astype(jnp.int32)

    # Same logic as v1's random_choice
    full_range = jnp.arange(total_words)
    mask = jnp.ones(total_words, dtype=bool)

    def update_mask(i, m):
        return m.at[exclude[i]].set(False)

    mask = jax.lax.fori_loop(0, exclude.shape[0], update_mask, mask)
    no_valid_choices = jnp.all(~mask)

    def error_case(_):
        return jnp.array(-1, dtype=jnp.int32)

    def normal_case(_):
        valid_range = jnp.where(mask, full_range, -1)
        return jax.random.choice(key, valid_range, p=mask / jnp.sum(mask)).astype(jnp.int32)

    return jax.lax.cond(no_valid_choices, error_case, normal_case, operand=None)


# ============================================
# Word Matrix Building (on-demand computation)
# ============================================

def build_word_from_sequence(base_matrices: jnp.ndarray,
                             sequence: jnp.ndarray,
                             input_dim: int,
                             output_dim: int) -> jnp.ndarray:
    """
    Build a single word matrix from base matrix indices sequence.

    This is the core lazy evaluation function: instead of precomputing all words,
    we compute one word at reset time using jax.lax.scan for efficient JIT compilation.

    Args:
        base_matrices: Base matrices with shape (B, D, D)
        sequence: Base matrix indices with shape (depth,)
        input_dim: Input dimension (D_i)
        output_dim: Output dimension (D_o)

    Returns:
        Word matrix with shape (input_dim, output_dim)

    Algorithm:
        word = I
        for j in range(depth):
            word = word @ base_matrices[sequence[j]]
        return word[: input_dim, :output_dim]
    """
    depth = sequence.shape[0]
    D = base_matrices.shape[1]  # Full matrix dimension

    # Matrix multiplication loop using jax.lax.scan
    def multiply_step(word, j):
        base_idx = sequence[j]
        base_matrix = base_matrices[base_idx]  # Shape: (D, D)
        new_word = word @ base_matrix
        return new_word, None

    # Start with identity matrix
    init_word = jnp.eye(D)
    final_word, _ = jax.lax.scan(multiply_step, init_word, jnp.arange(depth))

    # Slice to desired dimensions
    return final_word[:input_dim, :output_dim]


def build_word_from_index(base_matrices: jnp.ndarray,
                         index: int,
                         num_base: int,
                         depth: int,
                         input_dim: int,
                         output_dim: int,
                         encoding_mode: str = 'base_b') -> jnp.ndarray:
    """
    Convenience function: encode index → build word in one call.

    This is the main API for environments to use at reset time.

    Args:
        base_matrices: Base matrices (B, D, D)
        index: Linear word index
        num_base: Number of base matrices (B)
        depth: Depth of composition (L)
        input_dim: Input dimension
        output_dim: Output dimension
        encoding_mode: 'base_b' or 'bitshift'

    Returns:
        Word matrix (input_dim, output_dim)
    """
    sequence = encode_word_index(index, num_base, depth, encoding_mode)
    return build_word_from_sequence(base_matrices, sequence, input_dim, output_dim)


# ============================================
# Base Matrices Creation
# ============================================

def create_base_matrices(key, num_base: int, size: int = 64,
                        with_adjoint: bool = False):
    """
    Create orthogonal base matrices using QR decomposition.

    This is equivalent to create_orthogonal_matrices from v2, but renamed to
    emphasize that we're creating bases (not all words).

    Args:
        key: JAX random key
        num_base: Number of base matrices to create (B)
        size: Matrix dimension (D)
        with_adjoint: If True, include transpose of each matrix

    Returns:
        Stack of orthogonal matrices with shape (num_base, size, size)
    """
    if with_adjoint:
        # Create half the matrices and include their transposes
        num_matrices = num_base // 2
    else:
        num_matrices = num_base

    matrices = []
    for _ in range(num_matrices):
        key, subkey = jax.random.split(key)
        matrix = jax.random.normal(subkey, (size, size))
        q, _ = jnp.linalg.qr(matrix)
        matrices.append(q)
        if with_adjoint:
            matrices.append(q.T)

    return jnp.stack(matrices)


def detect_identity_matrices(array):
    """
    Detect identity matrices in an array of matrices.

    This function is kept for API compatibility with v1, but is NOT used
    in lazy evaluation (Option B: no identity detection).

    Args:
        array: Array of matrices with shape (N, D, D)

    Returns:
        Indices of matrices that are (approximately) identity
    """
    N, D, _ = array.shape
    identity = jnp.eye(D)
    is_identity = jnp.all(jnp.abs(array - identity[None, :, :]) < 1e-6, axis=(1, 2))
    identity_indices = jnp.where(is_identity)[0]

    return identity_indices


# ============================================
# Legacy compatibility functions
# ============================================

def get_weight_matrix_lazy(word: jnp.ndarray) -> jnp.ndarray:
    """
    Get weight matrix from a single word (lazy version).

    In lazy evaluation, we don't need indexing - the word is already computed.
    This function just applies the scaling factor.

    Args:
        word: The word matrix with shape (input_dim, output_dim)

    Returns:
        Scaled weight matrix
    """
    return jnp.sqrt(2) * word


# ============================================
# Flax Module (if needed for compatibility)
# ============================================

class MetaAugNetworkLazy(nn.Module):
    """
    Lazy version of MetaAugNetwork - uses pre-computed single word.

    Note: In lazy evaluation, the word is already selected and computed
    at reset time, so this module doesn't need to do indexing.
    """
    out_size: int = 64
    word: jnp.ndarray = None  # Single word matrix, not all words

    @nn.compact
    def __call__(self, x):
        # Use the pre-computed word directly
        weight = jnp.sqrt(2) * self.word
        return nn.Dense(self.out_size, kernel_init=lambda *_: weight, bias_init=constant(0.0))(x)
