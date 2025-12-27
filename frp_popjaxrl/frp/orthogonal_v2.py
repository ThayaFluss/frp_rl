######
### The core code for Free Random Projection (v2)
### Refactored version with B, N calculation logic consolidated in create_words
######

import jax.numpy as jnp
import jax
import flax.linen as nn
from typing import Tuple, Optional
from flax.linen.initializers import constant
import numpy as np


def create_orthogonal_matrices(key, num_base, size=64, with_adjoint=False):
    """
    Create orthogonal base matrices.

    Args:
        key: JAX random key
        num_base: Number of base matrices (B) to create
        size: Dimension of the matrices (D)
        with_adjoint: If True, also include adjoint (transpose) matrices

    Returns:
        Stack of orthogonal matrices
    """
    if with_adjoint:
        ### keep total number of words - create half the matrices and include their transposes
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


def create_words_ex(matrices, depth, in_size=2, out_size=64, max_depth=None, num_base=None, num_result_matrices=None, key=None):
    """
    Create result matrices and detect identity matrices.

    Args:
        matrices: Base matrices (B matrices)
        depth: Depth of composition (L)
        in_size: Input dimension (for compatibility)
        out_size: Output dimension
        max_depth: For backward compatibility (L_0)
        num_base: Number of base matrices (B), if specified
        num_result_matrices: Number of result matrices (N), if specified
        key: JAX random key for uniform sampling when L^B > N

    Returns:
        Tuple of (result matrices, indices of identity matrices)
    """
    result = create_words(matrices, depth, in_size, out_size, max_depth, num_base, num_result_matrices, key)
    exclude = detect_identity_matrices(result)

    return result, exclude


def create_words(matrices, depth, in_size=2, out_size=64, max_depth=None, num_base=None, num_result_matrices=None, key=None):
    """
    Create result matrices by composing base matrices.

    変数の対応:
        - depth = L: 掛け算の層数、語の長さ
        - num_base = B: base matrixの数
        - num_result_matrices = N: 最終的にできる行列の数
        - max_depth = L_0: 後方互換性のための変数

    Args:
        matrices: Base matrices (B matrices)
        depth: Depth of composition (L)
        in_size: Input dimension (for compatibility)
        out_size: Output dimension (D_o)
        max_depth: For backward compatibility (L_0)
        num_base: Number of base matrices (B), if specified
        num_result_matrices: Number of result matrices (N), if specified
        key: JAX random key for uniform sampling when L^B > N

    Returns:
        Stack of result matrices (N matrices)

    Logic for B and N calculation:
        1. (B, N specified): if L^B >= N, uniform sampling; if L^B < N, error
        2. (B specified only): N = L^B
        3. (N specified only): B = ceil(log N / log L)
        4. (B, N not specified): estimate from max_depth (backward compatibility)
            - B = max_depth // depth
            - N = 2^max_depth
    """
    # Get actual num_base (B) from matrices shape
    actual_num_base = matrices.shape[0]

    # Calculate B and N based on the specification
    if num_base is not None and num_result_matrices is not None:
        # Case 1: Both B and N specified
        B = num_base
        N = num_result_matrices
        max_possible = B ** depth
        if max_possible < N:
            raise ValueError(
                f"Cannot generate {N} matrices with depth {depth} and {B} base matrices. "
                f"Maximum possible: {max_possible} (B^L = {B}^{depth})"
            )
    elif num_base is not None and num_result_matrices is None:
        # Case 2: Only B specified
        B = num_base
        N = B ** depth
    elif num_base is None and num_result_matrices is not None:
        # Case 3: Only N specified
        # N = B^depth, so B = N^(1/depth)
        N = num_result_matrices
        B = int(np.ceil(N ** (1.0 / depth)))
    else:
        # Case 4: Neither specified - backward compatibility
        if max_depth is None:
            raise ValueError("Either (num_base, num_result_matrices), max_depth, or both must be specified")
        # Original logic: B = 2^(max_depth // depth), N = 2^max_depth
        B = 2 ** (max_depth // depth)
        N = 2 ** max_depth

    # Verify that actual_num_base matches B
    if actual_num_base != B:
        raise ValueError(
            f"Number of base matrices in input ({actual_num_base}) does not match calculated num_base ({B}). "
            f"Please create matrices with num_base={B}"
        )

    max_possible = B ** depth

    # Determine if we're in backward compatibility mode
    using_max_depth = (num_base is None and num_result_matrices is None and max_depth is not None)

    if using_max_depth:
        # Backward compatibility: use original indexing pattern
        # In the original implementation, indices go from 0 to 2^max_depth - 1
        # and the base matrix selection uses bit-shifting
        def create_word(i):
            word = jnp.eye(out_size)
            bits_per_selection = max_depth // depth  # Number of bits to extract each base matrix index
            for j in range(depth):
                # Original indexing: (i >> (j * bits_per_selection)) & ((1 << bits_per_selection) - 1)
                index = (i >> (j * bits_per_selection)) & ((1 << bits_per_selection) - 1)
                word = word @ matrices[index]
            return word

        indices = jnp.arange(N)
    else:
        # New logic: use base-B representation
        # Generate indices for N matrices
        if max_possible == N:
            # Use all possible combinations
            indices = jnp.arange(N)
        elif max_possible > N:
            # Uniform sampling from L^B possibilities
            if key is None:
                raise ValueError("Random key is required for uniform sampling when L^B > N")
            indices = jax.random.choice(key, max_possible, shape=(N,), replace=False)
            indices = jnp.sort(indices)
        else:
            # This should not happen due to the check above, but included for safety
            raise ValueError(f"Cannot generate {N} matrices. Maximum possible: {max_possible}")

        def create_word(i):
            word = jnp.eye(out_size)
            for j in range(depth):
                # Extract the j-th "digit" in base-B representation
                index = (i // (B ** j)) % B
                word = word @ matrices[index]
            return word

    result = jax.vmap(create_word)(indices)

    return result


def detect_identity_matrices(array):
    """
    Detect identity matrices in an array of matrices.

    Args:
        array: Array of matrices with shape (N, D, D)

    Returns:
        Indices of matrices that are (approximately) identity matrices
    """
    N, D, _ = array.shape
    identity = jnp.eye(D)
    is_identity = jnp.all(jnp.abs(array - identity[None, :, :]) < 1e-6, axis=(1, 2))
    identity_indices = jnp.where(is_identity)[0]

    return identity_indices


def get_weight_matrix(words, env_index, input_dim, output_dim):
    """
    Get the weight matrix for the current environment index in a JIT-compatible way.

    Args:
        words: The words array with shape (num_words, input_dim, output_dim)
        env_index: The index of the current environment
        input_dim: The input dimension
        output_dim: The output dimension

    Returns:
        The weight matrix for the current environment
    """
    return jnp.sqrt(2) * jax.lax.dynamic_slice(
                words,
                (env_index, 0, 0),
                (1, input_dim, output_dim)
            )[0]


def random_choice(key, total_words, exclude):
    """
    Randomly choose an index from total_words, excluding specified indices.

    Args:
        key: JAX random key
        total_words: Total number of words to choose from
        exclude: Indices to exclude from selection

    Returns:
        Randomly chosen index (or -1 if no valid choices)
    """
    # If exclude is empty, just choose from all words
    if exclude.shape[0] == 0:
        return jax.random.randint(key, (), 0, total_words)

    full_range = jnp.arange(total_words)

    mask = jnp.ones(total_words, dtype=bool)

    def update_mask(i, m):
        return m.at[exclude[i]].set(False)

    mask = jax.lax.fori_loop(0, exclude.shape[0], update_mask, mask)

    no_valid_choices = jnp.all(~mask)

    def error_case(_):
        return jnp.array(-1)

    def normal_case(_):
        valid_range = jnp.where(mask, full_range, -1)
        return jax.random.choice(key, valid_range, p=mask / jnp.sum(mask))

    return jax.lax.cond(
        no_valid_choices,
        error_case,
        normal_case,
        operand=None
    )



class MetaAugNetwork(nn.Module):
    """
    A neural network module that dynamically selects orthogonal matrices as weights.

    This network is designed for meta-augmentation tasks where different environments
    require different transformation matrices. It uses JAX's dynamic_slice operation
    to select a specific orthogonal matrix from a pre-computed set based on the
    environment index.

    Key features:
    - Uses orthogonal matrices as weights to preserve geometric properties
    - Dynamically selects matrices at runtime using environment index
    - Applies a scaling factor of sqrt(2) for stable gradient flow
    - Implements as a standard Dense layer with custom kernel initialization

    The dynamic_slice operation works by:
    1. Taking the words tensor with shape (num_words, input_dim, output_dim)
    2. Selecting a slice starting at (env_index, 0, 0) with size (1, input_dim, output_dim)
    3. Extracting the matrix and using it as the weight for a Dense layer

    This approach allows for efficient switching between different orthogonal
    transformations without needing separate network instances.
    """
    out_size: int = 64
    words: jnp.ndarray = None
    env_index : int = 1

    @nn.compact
    def __call__(self, x):
        # Select the orthogonal matrix for the current environment using dynamic_slice.
        # For dynamic sampling with jax, we avoid "weight = jnp.sqrt(2)*self.words[env_index]".
        weight = get_weight_matrix(self.words, self.env_index, self.words.shape[1], self.out_size)
        # For the compatibility with the orignal code:
        #   nn.Dense(self.out_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        # We avoid "return jnp.dot(x, weight)"
        return nn.Dense(self.out_size,  kernel_init=lambda *_: weight, bias_init=constant(0.0))(x)
