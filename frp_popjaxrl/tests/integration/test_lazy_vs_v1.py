"""
Integration test: v1 vs lazy environment behavior consistency test
Tests that v1 (all words pre-computed) and lazy (on-reset computation) environments
produce identical observations, rewards, and dones given the same seed.
"""
import jax
import jax.numpy as jnp
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from envs.meta_environment import create_meta_environment as create_meta_environment_v1
from envs.meta_environment_lazy import create_meta_environment as create_meta_environment_lazy


def test_single_reset_consistency():
    """Test that a single reset produces the same observation"""
    seed = 42
    key = jax.random.PRNGKey(seed)

    # Small configuration
    env_kwargs = {
        'max_steps_in_episode': 200,
        'noise_sigma': 0.0
    }

    meta_kwargs = {
        'meta_depth': 2,
        'meta_dim': 32,
        'meta_max_depth': 4,
        'meta_with_adjoint': False,
        'rng': jax.random.PRNGKey(seed)
    }

    # Create v1 environment
    key1, key2 = jax.random.split(key)
    env_v1 = create_meta_environment_v1("cartpole", env_kwargs, meta_kwargs)
    env_params_v1 = env_v1.default_params
    obs_v1, state_v1 = env_v1.reset(key1, env_params_v1)

    # Create lazy environment
    env_lazy = create_meta_environment_lazy("cartpole", env_kwargs, meta_kwargs)
    env_params_lazy = env_lazy.default_params
    obs_lazy, state_lazy = env_lazy.reset(key1, env_params_lazy)

    # Check observation shapes match
    assert obs_v1.shape == obs_lazy.shape, \
        f"Observation shape mismatch: v1={obs_v1.shape}, lazy={obs_lazy.shape}"

    # Check observations are close (should be identical for same seed)
    max_diff = jnp.max(jnp.abs(obs_v1 - obs_lazy))
    print(f"\nSingle reset max difference: {max_diff}")
    assert max_diff < 1e-5, f"Observations differ too much: max_diff={max_diff}"


def test_multiple_resets_consistency():
    """Test that multiple resets produce consistent observations"""
    seed = 42
    key = jax.random.PRNGKey(seed)
    num_resets = 5

    env_kwargs = {
        'max_steps_in_episode': 200,
        'noise_sigma': 0.0
    }

    meta_kwargs = {
        'meta_depth': 2,
        'meta_dim': 32,
        'meta_max_depth': 4,
        'meta_with_adjoint': False,
        'rng': jax.random.PRNGKey(seed)
    }

    # Create environments
    env_v1 = create_meta_environment_v1("cartpole", env_kwargs, meta_kwargs)
    env_lazy = create_meta_environment_lazy("cartpole", env_kwargs, meta_kwargs)
    env_params_v1 = env_v1.default_params
    env_params_lazy = env_lazy.default_params

    max_diffs = []

    for i in range(num_resets):
        key, subkey = jax.random.split(key)

        obs_v1, state_v1 = env_v1.reset(subkey, env_params_v1)
        obs_lazy, state_lazy = env_lazy.reset(subkey, env_params_lazy)

        max_diff = jnp.max(jnp.abs(obs_v1 - obs_lazy))
        max_diffs.append(float(max_diff))

        assert max_diff < 1e-5, \
            f"Reset {i}: observations differ too much: max_diff={max_diff}"

    print(f"\nMultiple resets max differences: {max_diffs}")
    print(f"Average: {jnp.mean(jnp.array(max_diffs)):.2e}")


def test_step_consistency():
    """Test that step function produces consistent results"""
    seed = 42
    key = jax.random.PRNGKey(seed)

    env_kwargs = {
        'max_steps_in_episode': 200,
        'noise_sigma': 0.0
    }

    meta_kwargs = {
        'meta_depth': 2,
        'meta_dim': 32,
        'meta_max_depth': 4,
        'meta_with_adjoint': False,
        'rng': jax.random.PRNGKey(seed)
    }

    # Create environments
    env_v1 = create_meta_environment_v1("cartpole", env_kwargs, meta_kwargs)
    env_lazy = create_meta_environment_lazy("cartpole", env_kwargs, meta_kwargs)
    env_params_v1 = env_v1.default_params
    env_params_lazy = env_lazy.default_params

    # Reset both
    key, reset_key = jax.random.split(key)
    obs_v1, state_v1 = env_v1.reset(reset_key, env_params_v1)
    obs_lazy, state_lazy = env_lazy.reset(reset_key, env_params_lazy)

    # Take same action
    action = 1
    key, step_key = jax.random.split(key)

    obs_next_v1, state_next_v1, reward_v1, done_v1, _ = env_v1.step(
        step_key, state_v1, action, env_params_v1
    )
    obs_next_lazy, state_next_lazy, reward_lazy, done_lazy, _ = env_lazy.step(
        step_key, state_lazy, action, env_params_lazy
    )

    # Check observations
    obs_diff = jnp.max(jnp.abs(obs_next_v1 - obs_next_lazy))
    print(f"\nStep observation max difference: {obs_diff}")
    assert obs_diff < 1e-5, f"Step observations differ: max_diff={obs_diff}"

    # Check rewards (should be identical for deterministic CartPole)
    reward_diff = jnp.abs(reward_v1 - reward_lazy)
    print(f"Step reward difference: {reward_diff}")
    assert reward_diff < 1e-6, f"Step rewards differ: diff={reward_diff}"

    # Check done flags
    assert done_v1 == done_lazy, f"Done flags differ: v1={done_v1}, lazy={done_lazy}"


def test_episode_consistency():
    """Test that a full episode produces consistent results"""
    seed = 42
    key = jax.random.PRNGKey(seed)
    num_steps = 20

    env_kwargs = {
        'max_steps_in_episode': 200,
        'noise_sigma': 0.0
    }

    meta_kwargs = {
        'meta_depth': 2,
        'meta_dim': 32,
        'meta_max_depth': 4,
        'meta_with_adjoint': False,
        'rng': jax.random.PRNGKey(seed)
    }

    # Create environments
    env_v1 = create_meta_environment_v1("cartpole", env_kwargs, meta_kwargs)
    env_lazy = create_meta_environment_lazy("cartpole", env_kwargs, meta_kwargs)
    env_params_v1 = env_v1.default_params
    env_params_lazy = env_lazy.default_params

    # Reset both
    key, reset_key = jax.random.split(key)
    obs_v1, state_v1 = env_v1.reset(reset_key, env_params_v1)
    obs_lazy, state_lazy = env_lazy.reset(reset_key, env_params_lazy)

    obs_diffs = []
    reward_diffs = []

    for step in range(num_steps):
        key, action_key, step_key = jax.random.split(key, 3)

        # Sample random action
        action = jax.random.randint(action_key, (), 0, env_v1.num_actions)

        # Step both environments
        obs_next_v1, state_next_v1, reward_v1, done_v1, _ = env_v1.step(
            step_key, state_v1, action, env_params_v1
        )
        obs_next_lazy, state_next_lazy, reward_lazy, done_lazy, _ = env_lazy.step(
            step_key, state_lazy, action, env_params_lazy
        )

        # Track differences
        obs_diff = jnp.max(jnp.abs(obs_next_v1 - obs_next_lazy))
        reward_diff = jnp.abs(reward_v1 - reward_lazy)
        obs_diffs.append(float(obs_diff))
        reward_diffs.append(float(reward_diff))

        # Check consistency
        assert obs_diff < 1e-5, \
            f"Step {step}: observations differ: max_diff={obs_diff}"
        assert reward_diff < 1e-6, \
            f"Step {step}: rewards differ: diff={reward_diff}"
        assert done_v1 == done_lazy, \
            f"Step {step}: done flags differ: v1={done_v1}, lazy={done_lazy}"

        # Update states
        state_v1 = state_next_v1
        state_lazy = state_next_lazy

        # Break if done
        if done_v1:
            print(f"\nEpisode ended at step {step}")
            break

    print(f"\nEpisode consistency:")
    print(f"  Max obs difference: {max(obs_diffs):.2e}")
    print(f"  Avg obs difference: {jnp.mean(jnp.array(obs_diffs)):.2e}")
    print(f"  Max reward difference: {max(reward_diffs):.2e}")
    print(f"  Avg reward difference: {jnp.mean(jnp.array(reward_diffs)):.2e}")


def test_multiple_episodes_statistics():
    """Test that statistics over multiple episodes are consistent"""
    seed = 42
    key = jax.random.PRNGKey(seed)
    num_episodes = 10
    max_steps_per_episode = 50

    env_kwargs = {
        'max_steps_in_episode': 200,
        'noise_sigma': 0.0
    }

    meta_kwargs = {
        'meta_depth': 2,
        'meta_dim': 32,
        'meta_max_depth': 4,
        'meta_with_adjoint': False,
        'rng': jax.random.PRNGKey(seed)
    }

    # Create environments
    env_v1 = create_meta_environment_v1("cartpole", env_kwargs, meta_kwargs)
    env_lazy = create_meta_environment_lazy("cartpole", env_kwargs, meta_kwargs)
    env_params_v1 = env_v1.default_params
    env_params_lazy = env_lazy.default_params

    episode_returns_v1 = []
    episode_returns_lazy = []
    episode_lengths_v1 = []
    episode_lengths_lazy = []

    for episode in range(num_episodes):
        # Reset with different keys for each episode
        key, reset_key = jax.random.split(key)
        obs_v1, state_v1 = env_v1.reset(reset_key, env_params_v1)
        obs_lazy, state_lazy = env_lazy.reset(reset_key, env_params_lazy)

        total_reward_v1 = 0.0
        total_reward_lazy = 0.0
        steps = 0

        for step in range(max_steps_per_episode):
            key, action_key, step_key = jax.random.split(key, 3)
            action = jax.random.randint(action_key, (), 0, env_v1.num_actions)

            obs_next_v1, state_next_v1, reward_v1, done_v1, _ = env_v1.step(
                step_key, state_v1, action, env_params_v1
            )
            obs_next_lazy, state_next_lazy, reward_lazy, done_lazy, _ = env_lazy.step(
                step_key, state_lazy, action, env_params_lazy
            )

            total_reward_v1 += float(reward_v1)
            total_reward_lazy += float(reward_lazy)
            steps += 1

            state_v1 = state_next_v1
            state_lazy = state_next_lazy

            if done_v1:
                break

        episode_returns_v1.append(total_reward_v1)
        episode_returns_lazy.append(total_reward_lazy)
        episode_lengths_v1.append(steps)
        episode_lengths_lazy.append(steps)

    # Compare statistics
    returns_v1 = jnp.array(episode_returns_v1)
    returns_lazy = jnp.array(episode_returns_lazy)
    lengths_v1 = jnp.array(episode_lengths_v1)
    lengths_lazy = jnp.array(episode_lengths_lazy)

    print(f"\nMulti-episode statistics:")
    print(f"  V1   - Mean return: {returns_v1.mean():.2f}, Mean length: {lengths_v1.mean():.2f}")
    print(f"  Lazy - Mean return: {returns_lazy.mean():.2f}, Mean length: {lengths_lazy.mean():.2f}")
    print(f"  Return difference: {jnp.abs(returns_v1 - returns_lazy).max():.2e}")
    print(f"  Length difference: {jnp.abs(lengths_v1 - lengths_lazy).max():.2e}")

    # Check consistency
    assert jnp.allclose(returns_v1, returns_lazy, atol=1e-5), \
        "Episode returns differ between v1 and lazy"
    assert jnp.array_equal(lengths_v1, lengths_lazy), \
        "Episode lengths differ between v1 and lazy"


if __name__ == "__main__":
    print("Running integration tests: v1 vs lazy")
    print("=" * 60)

    test_single_reset_consistency()
    print("\n" + "=" * 60)

    test_multiple_resets_consistency()
    print("\n" + "=" * 60)

    test_step_consistency()
    print("\n" + "=" * 60)

    test_episode_consistency()
    print("\n" + "=" * 60)

    test_multiple_episodes_statistics()
    print("\n" + "=" * 60)

    print("\n✅ All integration tests passed!")
