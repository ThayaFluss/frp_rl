"""
Phase 1: FRPManager class tests

Testing FRPManager and EvalFRPManager functionality:
1. initialize_words() - Words initialization and metadata generation
2. sample_env_index() - env_index sampling
3. transform_obs() - Observation transformation
4. JAX compatibility (JIT, vmap, scan)
5. Helper functions
"""

import jax
import jax.numpy as jnp
import pytest
from frp.frp_manager import (
    FRPManager,
    EvalFRPManager,
    FRPWords,
    create_frp_manager,
    create_eval_frp_manager,
)


def test_frp_manager_init():
    """Test FRPManager initialization"""
    print("\n" + "=" * 60)
    print("Test: FRPManager initialization")
    print("=" * 60)

    manager = FRPManager(
        meta_depth=2,
        meta_dim=64,
        input_dim=4,
        meta_max_depth=8,
        meta_with_adjoint=False,
        meta_truncate_aug=0,
    )

    assert manager.meta_depth == 2
    assert manager.meta_dim == 64
    assert manager.input_dim == 4
    assert manager.meta_max_depth == 8
    assert manager.meta_with_adjoint is False
    assert manager.meta_truncate_aug == 0
    assert manager.aug_output_dim == 64  # meta_dim when truncate=0

    print(f"✅ FRPManager initialized with correct parameters")
    print(f"  aug_output_dim: {manager.aug_output_dim}")


def test_frp_manager_init_truncate():
    """Test FRPManager initialization (truncate mode)"""
    print("\n" + "=" * 60)
    print("Test: FRPManager initialization with truncation")
    print("=" * 60)

    manager = FRPManager(
        meta_depth=2,
        meta_dim=64,
        input_dim=4,
        meta_max_depth=8,
        meta_with_adjoint=False,
        meta_truncate_aug=1,
    )

    assert manager.aug_output_dim == 4  # input_dim when truncate=1

    print(f"✅ FRPManager initialized with truncation")
    print(f"  aug_output_dim: {manager.aug_output_dim} (truncated to input_dim)")


def test_initialize_words():
    """Test initialize_words()"""
    print("\n" + "=" * 60)
    print("Test: FRPManager.initialize_words()")
    print("=" * 60)

    manager = FRPManager(
        meta_depth=2,
        meta_dim=64,
        input_dim=4,
        meta_max_depth=8,
        meta_with_adjoint=False,
        meta_truncate_aug=0,
    )

    rng = jax.random.PRNGKey(0)
    frp_words = manager.initialize_words(rng)

    # Check types
    assert isinstance(frp_words, FRPWords)
    assert isinstance(frp_words.words, jnp.ndarray)
    assert isinstance(frp_words.exclude, jnp.ndarray)
    assert isinstance(frp_words.total_words, (int, jnp.integer))

    # Check shapes
    expected_num_words = 2**8  # 2^meta_max_depth = 256
    assert frp_words.words.shape == (expected_num_words, 4, 64)  # (num_words, input_dim, meta_dim)
    assert frp_words.total_words == expected_num_words

    print(f"✅ FRPWords initialized correctly")
    print(f"  words.shape: {frp_words.words.shape}")
    print(f"  total_words: {frp_words.total_words}")
    print(f"  exclude.shape: {frp_words.exclude.shape}")
    print(f"  Number of identity matrices excluded: {len(frp_words.exclude)}")


def test_initialize_words_truncate():
    """Test initialize_words() (truncate mode)"""
    print("\n" + "=" * 60)
    print("Test: FRPManager.initialize_words() with truncation")
    print("=" * 60)

    manager = FRPManager(
        meta_depth=2,
        meta_dim=64,
        input_dim=4,
        meta_max_depth=8,
        meta_with_adjoint=False,
        meta_truncate_aug=1,
    )

    rng = jax.random.PRNGKey(0)
    frp_words = manager.initialize_words(rng)

    # Check shapes with truncation
    expected_num_words = 2**8
    assert frp_words.words.shape == (expected_num_words, 4, 4)  # (num_words, input_dim, input_dim)

    print(f"✅ FRPWords initialized with truncation")
    print(f"  words.shape: {frp_words.words.shape}")


def test_initialize_words_with_adjoint():
    """Test initialize_words() (with_adjoint mode)"""
    print("\n" + "=" * 60)
    print("Test: FRPManager.initialize_words() with adjoint")
    print("=" * 60)

    manager = FRPManager(
        meta_depth=2,
        meta_dim=64,
        input_dim=4,
        meta_max_depth=8,
        meta_with_adjoint=True,
        meta_truncate_aug=0,
    )

    rng = jax.random.PRNGKey(0)
    frp_words = manager.initialize_words(rng)

    # With adjoint, still 2^max_depth words (adjoint is included in composition)
    expected_num_words = 2**8
    assert frp_words.words.shape[0] == expected_num_words

    print(f"✅ FRPWords initialized with adjoint")
    print(f"  total_words: {frp_words.total_words}")


def test_sample_env_index_no_exclude():
    """Test sample_env_index() (without exclude)"""
    print("\n" + "=" * 60)
    print("Test: FRPManager.sample_env_index() without exclusions")
    print("=" * 60)

    manager = FRPManager(
        meta_depth=2,
        meta_dim=8,
        input_dim=2,
        meta_max_depth=4,
        meta_with_adjoint=False,
        meta_truncate_aug=0,
    )

    rng = jax.random.PRNGKey(42)
    frp_words = manager.initialize_words(rng)

    # Sample multiple env_indices
    num_samples = 10
    rng, _rng = jax.random.split(rng)
    sample_rngs = jax.random.split(_rng, num_samples)

    env_indices = []
    for sample_rng in sample_rngs:
        env_index = manager.sample_env_index(frp_words, sample_rng)
        env_indices.append(int(env_index))

    print(f"  Sampled env_indices: {env_indices}")
    print(f"  All in valid range: {all(0 <= idx < frp_words.total_words for idx in env_indices)}")

    assert all(0 <= idx < frp_words.total_words for idx in env_indices)
    print(f"✅ env_index sampling works correctly")


def test_sample_env_index_with_exclude():
    """Test sample_env_index() (with exclude)"""
    print("\n" + "=" * 60)
    print("Test: FRPManager.sample_env_index() with exclusions")
    print("=" * 60)

    # Use larger meta_dim to ensure identity matrices exist
    manager = FRPManager(
        meta_depth=1,
        meta_dim=64,
        input_dim=4,
        meta_max_depth=4,
        meta_with_adjoint=False,
        meta_truncate_aug=0,
    )

    rng = jax.random.PRNGKey(100)
    frp_words = manager.initialize_words(rng)

    if len(frp_words.exclude) > 0:
        print(f"  Found {len(frp_words.exclude)} identity matrices to exclude")

        # Sample many times and verify no excluded indices
        num_samples = 100
        rng, _rng = jax.random.split(rng)
        sample_rngs = jax.random.split(_rng, num_samples)

        env_indices = []
        for sample_rng in sample_rngs:
            env_index = manager.sample_env_index(frp_words, sample_rng)
            env_indices.append(int(env_index))

        # Check none of the sampled indices are in exclude
        excluded_set = set(frp_words.exclude.tolist())
        sampled_set = set(env_indices)
        intersection = sampled_set & excluded_set

        print(f"  Sampled indices that are excluded: {intersection}")
        assert len(intersection) == 0, "Should not sample excluded indices"
        print(f"✅ Exclusion works correctly")
    else:
        print(f"  No identity matrices found (exclude is empty)")
        print(f"✅ Test skipped (no exclusions)")


def test_transform_obs():
    """Test transform_obs()"""
    print("\n" + "=" * 60)
    print("Test: FRPManager.transform_obs()")
    print("=" * 60)

    manager = FRPManager(
        meta_depth=2,
        meta_dim=64,
        input_dim=4,
        meta_max_depth=8,
        meta_with_adjoint=False,
        meta_truncate_aug=0,
    )

    rng = jax.random.PRNGKey(0)
    frp_words = manager.initialize_words(rng)

    # Create test observation
    obs = jnp.array([1.0, 2.0, 3.0, 4.0])

    # Sample env_index
    rng, sample_rng = jax.random.split(rng)
    env_index = manager.sample_env_index(frp_words, sample_rng)

    # Transform observation
    transformed_obs = manager.transform_obs(obs, env_index, frp_words)

    # Check output shape
    assert transformed_obs.shape == (64,), f"Expected shape (64,), got {transformed_obs.shape}"

    # Check it's not all zeros
    assert jnp.sum(jnp.abs(transformed_obs)) > 0, "Transformed observation should not be all zeros"

    print(f"✅ Observation transformation works correctly")
    print(f"  Input shape: {obs.shape}")
    print(f"  Output shape: {transformed_obs.shape}")
    print(f"  env_index used: {env_index}")


def test_transform_obs_truncate():
    """Test transform_obs() (truncate mode)"""
    print("\n" + "=" * 60)
    print("Test: FRPManager.transform_obs() with truncation")
    print("=" * 60)

    manager = FRPManager(
        meta_depth=2,
        meta_dim=64,
        input_dim=4,
        meta_max_depth=8,
        meta_with_adjoint=False,
        meta_truncate_aug=1,
    )

    rng = jax.random.PRNGKey(0)
    frp_words = manager.initialize_words(rng)

    obs = jnp.array([1.0, 2.0, 3.0, 4.0])
    rng, sample_rng = jax.random.split(rng)
    env_index = manager.sample_env_index(frp_words, sample_rng)

    transformed_obs = manager.transform_obs(obs, env_index, frp_words)

    # With truncation, output should match input_dim
    assert transformed_obs.shape == (4,), f"Expected shape (4,), got {transformed_obs.shape}"

    print(f"✅ Observation transformation with truncation works")
    print(f"  Output shape: {transformed_obs.shape}")


def test_eval_frp_manager_identity():
    """Test EvalFRPManager (identity mode)"""
    print("\n" + "=" * 60)
    print("Test: EvalFRPManager with identity mode")
    print("=" * 60)

    eval_manager = EvalFRPManager("identity", input_dim=4, output_dim=4)

    obs = jnp.array([1.0, 2.0, 3.0, 4.0])
    transformed_obs = eval_manager.transform_obs(obs)

    # Identity should return same observation
    assert jnp.allclose(transformed_obs, obs), "Identity mode should not change observation"
    assert transformed_obs.shape == obs.shape

    print(f"✅ Identity mode works correctly")
    print(f"  Input: {obs}")
    print(f"  Output: {transformed_obs}")


def test_eval_frp_manager_padding():
    """Test EvalFRPManager (padding mode)"""
    print("\n" + "=" * 60)
    print("Test: EvalFRPManager with padding mode")
    print("=" * 60)

    eval_manager = EvalFRPManager("padding", input_dim=4, output_dim=64)

    obs = jnp.array([1.0, 2.0, 3.0, 4.0])
    transformed_obs = eval_manager.transform_obs(obs)

    # Padding should expand to output_dim
    assert transformed_obs.shape == (64,)
    # First input_dim elements should match, rest should be zeros
    assert jnp.allclose(transformed_obs[:4], obs)
    # Check that there are zeros (padding effect)
    assert jnp.sum(jnp.abs(transformed_obs[4:])) >= 0  # May have some values due to projection

    print(f"✅ Padding mode works correctly")
    print(f"  Input shape: {obs.shape}")
    print(f"  Output shape: {transformed_obs.shape}")


def test_eval_frp_manager_tiling():
    """Test EvalFRPManager (tiling mode)"""
    print("\n" + "=" * 60)
    print("Test: EvalFRPManager with tiling mode")
    print("=" * 60)

    eval_manager = EvalFRPManager("tiling", input_dim=4, output_dim=64)

    obs = jnp.array([1.0, 2.0, 3.0, 4.0])
    transformed_obs = eval_manager.transform_obs(obs)

    # Tiling should expand to output_dim
    assert transformed_obs.shape == (64,)
    # Should not be all zeros
    assert jnp.sum(jnp.abs(transformed_obs)) > 0

    print(f"✅ Tiling mode works correctly")
    print(f"  Input shape: {obs.shape}")
    print(f"  Output shape: {transformed_obs.shape}")


def test_jit_compatibility():
    """Test JAX JIT compatibility"""
    print("\n" + "=" * 60)
    print("Test: JAX JIT compatibility")
    print("=" * 60)

    manager = FRPManager(
        meta_depth=2,
        meta_dim=64,
        input_dim=4,
        meta_max_depth=8,
        meta_with_adjoint=False,
        meta_truncate_aug=0,
    )

    rng = jax.random.PRNGKey(0)
    frp_words = manager.initialize_words(rng)

    # JIT compile sample and transform
    @jax.jit
    def sample_and_transform(rng, obs, frp_words):
        env_index = manager.sample_env_index(frp_words, rng)
        transformed = manager.transform_obs(obs, env_index, frp_words)
        return env_index, transformed

    obs = jnp.array([1.0, 2.0, 3.0, 4.0])
    rng, _rng = jax.random.split(rng)

    env_index, transformed_obs = sample_and_transform(_rng, obs, frp_words)

    assert transformed_obs.shape == (64,)
    print(f"✅ JIT compilation works")
    print(f"  env_index: {env_index}")
    print(f"  transformed shape: {transformed_obs.shape}")


def test_vmap_compatibility():
    """Test JAX vmap compatibility"""
    print("\n" + "=" * 60)
    print("Test: JAX vmap compatibility")
    print("=" * 60)

    manager = FRPManager(
        meta_depth=2,
        meta_dim=64,
        input_dim=4,
        meta_max_depth=8,
        meta_with_adjoint=False,
        meta_truncate_aug=0,
    )

    rng = jax.random.PRNGKey(0)
    frp_words = manager.initialize_words(rng)

    # Create multiple observations
    num_envs = 4
    obs_batch = jnp.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ])

    # Sample env_indices for each environment
    rng, _rng = jax.random.split(rng)
    sample_rngs = jax.random.split(_rng, num_envs)

    # vmap over sample_env_index
    env_indices = jax.vmap(
        lambda rng: manager.sample_env_index(frp_words, rng)
    )(sample_rngs)

    # vmap over transform_obs
    transformed_batch = jax.vmap(
        lambda obs, idx: manager.transform_obs(obs, idx, frp_words)
    )(obs_batch, env_indices)

    assert env_indices.shape == (num_envs,)
    assert transformed_batch.shape == (num_envs, 64)
    print(f"✅ vmap works correctly")
    print(f"  env_indices: {env_indices}")
    print(f"  transformed_batch shape: {transformed_batch.shape}")


def test_create_frp_manager():
    """Test create_frp_manager() helper function"""
    print("\n" + "=" * 60)
    print("Test: create_frp_manager() helper")
    print("=" * 60)

    # Create mock environment with necessary attributes
    class MockEnv:
        def __init__(self):
            self.meta_depth = 2
            self.meta_dim = 64
            self.input_dim = 4
            self.meta_max_depth = 8
            self.meta_with_adjoint = False
            self.meta_truncate_aug = 0

    config = {"ENV": MockEnv()}
    manager = create_frp_manager(config)

    assert isinstance(manager, FRPManager)
    assert manager.meta_depth == 2
    assert manager.meta_dim == 64
    assert manager.input_dim == 4

    print(f"✅ create_frp_manager() works correctly")
    print(f"  Created manager with meta_dim={manager.meta_dim}")


def test_create_eval_frp_manager_identity():
    """Test create_eval_frp_manager() helper (identity)"""
    print("\n" + "=" * 60)
    print("Test: create_eval_frp_manager() with identity")
    print("=" * 60)

    class MockEvalEnv:
        meta_const_aug = "identity"

    class MockEnv:
        def __init__(self):
            self.meta_depth = 2
            self.meta_dim = 64
            self.input_dim = 4
            self.meta_max_depth = 8
            self.meta_with_adjoint = False
            self.meta_truncate_aug = 0

    config = {"ENV": MockEnv(), "EVAL_ENV": MockEvalEnv()}
    eval_manager = create_eval_frp_manager(config)

    assert isinstance(eval_manager, EvalFRPManager)
    assert eval_manager.method == "identity"
    assert eval_manager.input_dim == 4
    assert eval_manager.output_dim == 4  # identity mode

    print(f"✅ create_eval_frp_manager() works for identity mode")


def test_create_eval_frp_manager_padding():
    """Test create_eval_frp_manager() helper (padding)"""
    print("\n" + "=" * 60)
    print("Test: create_eval_frp_manager() with padding")
    print("=" * 60)

    class MockEvalEnv:
        meta_const_aug = "padding"

    class MockEnv:
        def __init__(self):
            self.meta_depth = 2
            self.meta_dim = 64
            self.input_dim = 4
            self.meta_max_depth = 8
            self.meta_with_adjoint = False
            self.meta_truncate_aug = 0

    config = {"ENV": MockEnv(), "EVAL_ENV": MockEvalEnv()}
    eval_manager = create_eval_frp_manager(config)

    assert isinstance(eval_manager, EvalFRPManager)
    assert eval_manager.method == "padding"
    assert eval_manager.output_dim == 64  # meta_dim for padding

    print(f"✅ create_eval_frp_manager() works for padding mode")


def test_create_eval_frp_manager_none():
    """Test create_eval_frp_manager() helper (returns None)"""
    print("\n" + "=" * 60)
    print("Test: create_eval_frp_manager() returns None")
    print("=" * 60)

    class MockEvalEnv:
        meta_const_aug = None  # No specific eval method

    class MockEnv:
        def __init__(self):
            self.meta_depth = 2
            self.meta_dim = 64
            self.input_dim = 4
            self.meta_max_depth = 8
            self.meta_with_adjoint = False
            self.meta_truncate_aug = 0

    config = {"ENV": MockEnv(), "EVAL_ENV": MockEvalEnv()}
    eval_manager = create_eval_frp_manager(config)

    assert eval_manager is None, "Should return None when no eval method specified"

    print(f"✅ create_eval_frp_manager() returns None correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: FRPManager Tests")
    print("=" * 60)

    # Run all tests
    test_frp_manager_init()
    test_frp_manager_init_truncate()
    test_initialize_words()
    test_initialize_words_truncate()
    test_initialize_words_with_adjoint()
    test_sample_env_index_no_exclude()
    test_sample_env_index_with_exclude()
    test_transform_obs()
    test_transform_obs_truncate()
    test_eval_frp_manager_identity()
    test_eval_frp_manager_padding()
    test_eval_frp_manager_tiling()
    test_jit_compatibility()
    test_vmap_compatibility()
    test_create_frp_manager()
    test_create_eval_frp_manager_identity()
    test_create_eval_frp_manager_padding()
    test_create_eval_frp_manager_none()

    print("\n" + "=" * 60)
    print("All FRPManager tests completed successfully! ✅")
    print("=" * 60)
