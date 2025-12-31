######
### The core code for Free Random Projection
######

import jax.numpy as jnp
import jax
import flax.linen as nn
from typing import Tuple, Optional
from flax.linen.initializers import constant
#from flax.linen.initializers import orthogonal


def create_orthogonal_matrices(key, depth, size=64, max_depth=8, with_adjoint=False):
    if with_adjoint:
        ### keep total number of words
        num_matrices = 2**(max_depth // depth -1 )
    else:
        num_matrices = 2**(max_depth // depth)

    matrices = []
    for _ in range(num_matrices):
        key, subkey = jax.random.split(key)
        matrix = jax.random.normal(subkey, (size, size))
        q, _ = jnp.linalg.qr(matrix)
        matrices.append(q)
        if with_adjoint:
            matrices.append(q.T)
    return jnp.stack(matrices)


def create_words_ex(matrices, depth, in_size=2, out_size=64, max_depth=8):
    result = create_words(matrices, depth, in_size, out_size, max_depth)
    exclude = detect_identity_matrices(result)
    
    return result, exclude

def create_words(matrices, depth, in_size=2, out_size=64, max_depth=8):
    def create_word(i):
        word = jnp.eye(out_size)
        for j in range(depth):
            index = (i >> (j * (max_depth // depth))) & (2**(max_depth // depth) - 1)
            word = word @ matrices[index]
        return word
    
    result = jax.vmap(create_word)(jnp.arange(2**max_depth))
    
    return result


def detect_identity_matrices(array):
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
    #return jnp.sqrt(2) * jnp.take(words, env_index, axis=0)
    return jnp.sqrt(2) * jax.lax.dynamic_slice(
                words, 
                (env_index, 0, 0), 
                (1, input_dim, output_dim)
            )[0]


def random_choice(key, total_words, exclude):
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
    [Duplicated]
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
