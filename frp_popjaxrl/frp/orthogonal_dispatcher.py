"""
Dispatcher module for choosing between orthogonal.py (v1) and orthogonal_v2.py

This module provides a simple way to switch between the original implementation
and the refactored version with consolidated B, N calculation logic.

Usage:
    # Set the version you want to use
    USE_V2 = True  # or False

    if USE_V2:
        from frp.orthogonal_v2 import (
            create_orthogonal_matrices,
            create_words,
            create_words_ex,
            detect_identity_matrices,
            get_weight_matrix,
            random_choice,
            MetaAugNetwork
        )
    else:
        from frp.orthogonal import (
            create_orthogonal_matrices,
            create_words,
            create_words_ex,
            detect_identity_matrices,
            get_weight_matrix,
            random_choice,
            MetaAugNetwork
        )

API Differences:

1. create_orthogonal_matrices:
   - V1: create_orthogonal_matrices(key, depth, size=64, max_depth=8, with_adjoint=False)
   - V2: create_orthogonal_matrices(key, num_base, size=64, with_adjoint=False)

   Migration:
   # V1
   matrices = create_orthogonal_matrices(key, depth=2, size=64, max_depth=8)
   # V2
   num_base = 2 ** (max_depth // depth)  # = 2^(8//2) = 16
   matrices = create_orthogonal_matrices(key, num_base=16, size=64)

2. create_words:
   - V1: create_words(matrices, depth, in_size=2, out_size=64, max_depth=8)
   - V2: create_words(matrices, depth, in_size=2, out_size=64, max_depth=None, num_base=None, num_result_matrices=None, key=None)

   Migration examples:

   a) Backward compatibility (use max_depth):
   # V1
   words = create_words(matrices, depth=2, out_size=64, max_depth=8)
   # V2 (same behavior)
   words = create_words(matrices, depth=2, out_size=64, max_depth=8)

   b) Specify both B and N (new feature in V2):
   # V2 only
   words = create_words(matrices, depth=3, out_size=64, num_base=4, num_result_matrices=50, key=subkey)
   # This generates 50 matrices by sampling from 3^4=81 possible combinations

   c) Specify only B (new feature in V2):
   # V2 only
   words = create_words(matrices, depth=2, out_size=64, num_base=5)
   # This generates 2^5=32 matrices

   d) Specify only N (new feature in V2):
   # V2 only
   words = create_words(matrices, depth=3, out_size=64, num_result_matrices=30, key=subkey)
   # This automatically calculates B=ceil(log 30 / log 3)=4 and generates 30 matrices

3. Other functions (identical API):
   - create_words_ex: Same API in both versions (V2 adds optional parameters)
   - detect_identity_matrices: Identical
   - get_weight_matrix: Identical
   - random_choice: Identical
   - MetaAugNetwork: Identical
"""

import os

# Environment variable to control which version to use
USE_V2 = os.environ.get('FRP_ORTHOGONAL_VERSION', 'v1') == 'v2'

if USE_V2:
    from frp.orthogonal_v2 import (
        create_orthogonal_matrices,
        create_words,
        create_words_ex,
        detect_identity_matrices,
        get_weight_matrix,
        random_choice,
        MetaAugNetwork
    )
else:
    from frp.orthogonal import (
        create_orthogonal_matrices,
        create_words,
        create_words_ex,
        detect_identity_matrices,
        get_weight_matrix,
        random_choice,
        MetaAugNetwork
    )

__all__ = [
    'create_orthogonal_matrices',
    'create_words',
    'create_words_ex',
    'detect_identity_matrices',
    'get_weight_matrix',
    'random_choice',
    'MetaAugNetwork',
    'USE_V2'
]
