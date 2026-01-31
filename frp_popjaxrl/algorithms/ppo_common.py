"""
Common utilities for PPO training implementations.

This module contains functions and data structures that are shared across
all PPO implementations (GRU/S5, lazy/eager).
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple
from gymnax.environments import spaces


class Transition(NamedTuple):
    """Single timestep transition for PPO training.

    Attributes:
        done: Episode done flags
        action: Actions taken
        value: Value estimates
        reward: Rewards received
        log_prob: Log probabilities of actions
        obs: Observations
        info: Environment info
    """
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_linear_schedule(config):
    """
    Create learning rate schedule function.

    Args:
        config: Configuration dict with NUM_MINIBATCHES, UPDATE_EPOCHS, NUM_UPDATES, LR

    Returns:
        Function that takes count and returns learning rate
    """
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac
    return linear_schedule


def calculate_gae(traj_batch, last_val, last_done, gamma, gae_lambda):
    """
    Calculate Generalized Advantage Estimation (GAE).

    Args:
        traj_batch: Trajectory batch (Transition NamedTuple)
        last_val: Value estimate for the last state
        last_done: Done flag for the last state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        Tuple of (advantages, targets)
    """
    def _get_advantages(carry, transition):
        gae, next_value, next_done = carry
        done, value, reward = transition.done, transition.value, transition.reward
        delta = reward + gamma * next_value * (1 - next_done) - value
        gae = delta + gamma * gae_lambda * (1 - next_done) * gae
        return (gae, value, done), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val, last_done),
        traj_batch,
        reverse=True,
        unroll=16
    )
    return advantages, advantages + traj_batch.value


def safe_mean(info):
    """
    Safely calculate mean of episode returns, handling zero episodes.

    Args:
        info: Info dict with 'returned_episode' and 'return_info' fields

    Returns:
        Mean return, or 0.0 if no episodes completed
    """
    returned_episodes = info["returned_episode"].sum()
    returns_sum = (info["return_info"][..., 1] * info["returned_episode"]).sum()
    return jnp.where(returned_episodes > 0, returns_sum / returned_episodes, 0.0)


def create_minibatches(batch, num_envs, num_minibatches, rng):
    """
    Shuffle and create minibatches from a batch.

    Args:
        batch: Tuple of (init_hstate, traj_batch, advantages, targets)
        num_envs: Number of environments
        num_minibatches: Number of minibatches to create
        rng: JAX random key

    Returns:
        Minibatches with shape [num_minibatches, ...]
    """
    rng, _rng = jax.random.split(rng)
    permutation = jax.random.permutation(_rng, num_envs)

    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=1), batch
    )

    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(
            jnp.reshape(
                x, [x.shape[0], num_minibatches, -1] + list(x.shape[2:])
            ), 1, 0
        ),
        shuffled_batch,
    )

    return minibatches, rng


def make_training_callback(max_train_metric, max_eval_metric):
    """
    Create a callback function for logging metrics.

    Args:
        max_train_metric: Reference to max training metric (will be updated)
        max_eval_metric: Reference to max eval metric (will be updated)

    Returns:
        Callback function for jax.debug.callback
    """
    def callback(train_metric, in_context_metric, train_done, eval_done):
        import wandb
        # Update max values (using nonlocal would work in original context)
        new_max_train = max(float(max_train_metric), float(train_metric))
        new_max_eval = max(float(max_eval_metric), float(in_context_metric))

        print(f"Train metric: {train_metric}, In-context: {in_context_metric}")
        print(f"Train episode done: {train_done}, Eval episode done: {eval_done}")
        wandb.log({
            "metric": train_metric,
            "eval_metric": in_context_metric,
            "max_metric": new_max_train,
            "max_eval_metric": new_max_eval,
            "train_episode_done_count": train_done,
            "eval_episode_done_count": eval_done,
        })

        return new_max_train, new_max_eval

    return callback


def setup_config(config):
    """
    Setup standard config parameters for PPO training.

    Args:
        config: Configuration dictionary

    Returns:
        Updated config dictionary
    """
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    return config


def create_network(encoder_type: str, action_space, config):
    """
    Create ActorCritic network with specified encoder and action space.

    This function:
    1. Selects encoder (GRU or S5) based on encoder_type
    2. Detects action space type (continuous or discrete)
    3. Assembles the appropriate ActorCritic network

    All selection happens at Python time (before jax.jit), ensuring
    no conditional branches in compiled code.

    Args:
        encoder_type: 'gru' or 's5'
        action_space: Environment action space (spaces.Box or spaces.Discrete)
        config: Configuration dictionary

    Returns:
        Instantiated ActorCritic network (with GRU/S5 encoder and Continuous/Discrete head)

    Example:
        >>> # GRU encoder with automatic action space detection
        >>> network = create_network('gru', env.action_space(env_params), config)
        >>>
        >>> # S5 encoder with automatic action space detection
        >>> network = create_network('s5', env.action_space(env_params), config)
    """
    from algorithms.models import (
        GRURepModel, S5RepModel,
        ActorCriticContinuous, ActorCriticDiscrete
    )

    # Select RepModel based on type
    if encoder_type.lower() == 'gru':
        rep_model = GRURepModel(config=config)
    elif encoder_type.lower() == 's5':
        rep_model = S5RepModel(config=config)
    else:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. "
            f"Valid values are 'gru' or 's5'."
        )

    # Detect action space type and create appropriate ActorCritic
    if isinstance(action_space, spaces.Box):
        # Continuous action space
        action_dim = action_space.shape[0]
        config["CONTINUOUS"] = True
        return ActorCriticContinuous(
            rep_model=rep_model,
            action_dim=action_dim,
            config=config
        )
    else:  # Discrete
        # Discrete action space
        action_dim = action_space.n
        config["CONTINUOUS"] = False
        return ActorCriticDiscrete(
            rep_model=rep_model,
            action_dim=action_dim,
            config=config
        )


def create_gru_network(action_space, config):
    """
    Create GRU-based ActorCritic network.

    Convenience function for create_network('gru', ...).
    """
    return create_network('gru', action_space, config)


def create_s5_network(action_space, config):
    """
    Create S5-based ActorCritic network.

    Convenience function for create_network('s5', ...).
    """
    return create_network('s5', action_space, config)
