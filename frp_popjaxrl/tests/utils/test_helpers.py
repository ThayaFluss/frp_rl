#!/usr/bin/env python3
"""
Test helper utilities for FRP RL unit tests.

Provides shared functions for creating test FRP words, managers, and assertions.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Union

from frp.frp_manager import FRPManager, FRPWords
from frp.orthogonal_legacy import (
    create_orthogonal_matrices as create_orthogonal_matrices_legacy,
    create_words as create_words_legacy,
    detect_identity_matrices as detect_identity_matrices_legacy,
)


def create_test_frp_words(
    seed=42,
    meta_depth=4,
    meta_dim=128,
    input_dim=4,
    meta_max_depth=8,
    with_adjoint=False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create FRP words for testing using legacy method for compatibility.

    Args:
        seed: Random seed for reproducibility
        meta_depth: Depth of meta augmentation network
        meta_dim: Dimension of meta augmentation network
        input_dim: Input dimension (will truncate words to this)
        meta_max_depth: Maximum depth for parallel words
        with_adjoint: Whether to use adjoint matrices

    Returns:
        Tuple of (words, exclude):
            - words: Array of shape [total_words, input_dim, meta_dim]
            - exclude: Array of indices to exclude (identity matrices)
    """
    rng = jax.random.PRNGKey(seed)

    # Create orthogonal matrices using legacy method
    matrices = create_orthogonal_matrices_legacy(
        rng,
        meta_depth,
        size=meta_dim,
        max_depth=meta_max_depth,
        with_adjoint=with_adjoint
    )

    # Create words from matrices
    words = create_words_legacy(
        matrices,
        meta_depth,
        out_size=meta_dim,
        max_depth=meta_max_depth
    )

    # Detect identity matrices BEFORE truncation (needs square matrices)
    exclude = detect_identity_matrices_legacy(words)

    # Truncate to input_dim for environments like cartpole
    words = words[:, :input_dim, :]

    return words, exclude


def create_test_frp_manager(
    input_dim=4,
    meta_dim=128,
    meta_depth=4,
    meta_max_depth=8,
    meta_with_adjoint=False,
    meta_truncate_aug=0,
) -> FRPManager:
    """Create FRPManager for testing.

    Args:
        input_dim: Input dimension
        meta_dim: Meta augmentation dimension (aug_output_dim)
        meta_depth: Depth of meta augmentation network
        meta_max_depth: Maximum depth for parallel words
        meta_with_adjoint: Whether to use adjoint matrices
        meta_truncate_aug: Truncation level for augmentation

    Returns:
        Configured FRPManager instance
    """
    return FRPManager(
        meta_depth=meta_depth,
        meta_dim=meta_dim,
        input_dim=input_dim,
        meta_max_depth=meta_max_depth,
        meta_with_adjoint=meta_with_adjoint,
        meta_truncate_aug=meta_truncate_aug,
    )


def assert_env_index_valid(env_index: Union[int, jnp.ndarray], total_words: int, exclude: jnp.ndarray):
    """Assert that env_index is valid.

    Args:
        env_index: Single index or array of indices
        total_words: Total number of words available
        exclude: Array of indices to exclude
    """
    # Handle both scalar and array inputs
    indices = jnp.atleast_1d(env_index)

    # Check all indices are in valid range
    assert jnp.all(indices >= 0), f"env_index contains negative values: {indices}"
    assert jnp.all(indices < total_words), f"env_index >= total_words ({total_words}): {indices}"

    # Check no excluded indices
    for idx in indices:
        assert idx not in exclude, f"env_index {idx} is in exclude list: {exclude}"


def assert_trajectory_pattern(
    trajectory: jnp.ndarray,
    expected_pattern: Union[str, jnp.ndarray]
):
    """Assert env_index trajectory matches expected pattern.

    Args:
        trajectory: Array of shape [num_episodes, num_envs] containing env_indices over time
        expected_pattern: Either:
            - "constant": all values same across episodes (per environment)
            - "changing": values change over episodes (at least once per environment)
            - custom boolean array of shape [num_episodes, num_envs] indicating where changes occur
    """
    num_episodes, num_envs = trajectory.shape

    if expected_pattern == "constant":
        # Each environment should have constant env_index across all episodes
        for env_idx in range(num_envs):
            env_trajectory = trajectory[:, env_idx]
            first_value = env_trajectory[0]
            assert jnp.all(env_trajectory == first_value), \
                f"Environment {env_idx} trajectory not constant: {env_trajectory}"

    elif expected_pattern == "changing":
        # Each environment should have at least one change
        for env_idx in range(num_envs):
            env_trajectory = trajectory[:, env_idx]
            changes = jnp.any(env_trajectory[1:] != env_trajectory[0])
            assert changes, \
                f"Environment {env_idx} trajectory has no changes: {env_trajectory}"

    elif isinstance(expected_pattern, jnp.ndarray):
        # Custom pattern: boolean array indicating expected changes
        assert expected_pattern.shape == trajectory.shape, \
            f"Pattern shape {expected_pattern.shape} != trajectory shape {trajectory.shape}"

        # Compare trajectory changes with expected pattern
        for episode in range(1, num_episodes):
            for env_idx in range(num_envs):
                expected_change = expected_pattern[episode, env_idx]
                actual_change = trajectory[episode, env_idx] != trajectory[episode-1, env_idx]

                if expected_change and not actual_change:
                    raise AssertionError(
                        f"Expected change at episode {episode}, env {env_idx} but found none. "
                        f"Value: {trajectory[episode, env_idx]}"
                    )
                elif not expected_change and actual_change:
                    raise AssertionError(
                        f"Unexpected change at episode {episode}, env {env_idx}. "
                        f"Values: {trajectory[episode-1, env_idx]} -> {trajectory[episode, env_idx]}"
                    )
    else:
        raise ValueError(f"Unknown pattern: {expected_pattern}. Use 'constant', 'changing', or boolean array")


def create_test_done_pattern(
    num_episodes: int,
    num_envs: int,
    pattern_type: str = "alternating"
) -> jnp.ndarray:
    """Create test done pattern for meta episode completion testing.

    Args:
        num_episodes: Number of episodes
        num_envs: Number of environments
        pattern_type: Type of pattern:
            - "alternating": environments complete in round-robin fashion
            - "all_done": all environments done every episode
            - "none_done": no environments done
            - "random": random done pattern (deterministic seed)

    Returns:
        Boolean array of shape [num_episodes, num_envs]
    """
    if pattern_type == "alternating":
        # Each environment completes in round-robin
        pattern = jnp.zeros((num_episodes, num_envs), dtype=bool)
        for ep in range(num_episodes):
            env_idx = ep % num_envs
            pattern = pattern.at[ep, env_idx].set(True)
        return pattern

    elif pattern_type == "all_done":
        # All environments done every episode
        return jnp.ones((num_episodes, num_envs), dtype=bool)

    elif pattern_type == "none_done":
        # No environments ever done
        return jnp.zeros((num_episodes, num_envs), dtype=bool)

    elif pattern_type == "random":
        # Random pattern with fixed seed
        rng = jax.random.PRNGKey(42)
        return jax.random.bernoulli(rng, p=0.3, shape=(num_episodes, num_envs))

    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")
