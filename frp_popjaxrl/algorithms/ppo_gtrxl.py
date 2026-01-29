"""
GTrXL PPO Implementation - Self-contained PPO for GTrXL architecture.

This module implements PPO training specifically optimized for GTrXL (Gated Transformer-XL).
It follows the reference implementation pattern from:
    https://github.com/Reytuag/transformerXL_PPO_JAX

Key differences from GRU/S5 PPO:
- GTrXL-specific Transition with memory caching
- FIFO memory management for sliding window attention
- WINDOW_GRAD chunked training with Truncated BPTT
- Memory reset on episode boundaries

This file is self-contained and does not depend on ppo_common.py for Transition.
"""
import logging
from typing import NamedTuple, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from gymnax.environments import spaces

from envs.wrappers import LogWrapper

from .ppo_common import (
    make_linear_schedule,
    calculate_gae,
    safe_mean,
    setup_config,
    create_network,
)

logger = logging.getLogger(__name__)


# ===== GTrXL-specific Data Structures =====

class Transition(NamedTuple):
    """GTrXL-specific transition with memory caching.

    Attributes:
        done: Episode done flags, shape (num_envs,)
        action: Actions taken, shape (num_envs,) or (num_envs, action_dim)
        value: Value estimates, shape (num_envs,)
        reward: Rewards received, shape (num_envs,)
        log_prob: Log probabilities of actions, shape (num_envs,)
        obs: Observations, shape (num_envs, obs_dim)
        info: Environment info
        memory: Cached memory at this step for WINDOW_GRAD training
                List of arrays, each shape (num_envs, mem_len, d_model)
    """
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    memory: List[jnp.ndarray]


# ===== Memory Management Functions =====

def reset_memory_on_done(memory: List[jnp.ndarray], done: jnp.ndarray) -> List[jnp.ndarray]:
    """Reset memory to zeros for environments where episode ended.

    Args:
        memory: List of memory tensors, each shape (1, num_envs, mem_len * d_model)
                This format is compatible with GTrXLRepModel's hidden state format.
        done: Done flags, shape (num_envs,)

    Returns:
        Updated memory with zeros for done environments
    """
    # Memory shape: (1, num_envs, mem_len * d_model)
    # Expand done for broadcasting: (num_envs,) -> (1, num_envs, 1)
    done_expanded = done[None, :, None]

    return [
        jnp.where(done_expanded, jnp.zeros_like(mem), mem)
        for mem in memory
    ]


def make_train(config):
    """Create GTrXL-specific PPO training function.

    Args:
        config: Configuration dictionary with GTrXL and PPO parameters

    Returns:
        Training function that takes RNG and returns (runner_state, metrics)
    """
    config = setup_config(config)

    env, env_params = config["ENV"], config["ENV_PARAMS"]
    env = LogWrapper(env)

    # For standard training, eval uses the same environment type
    eval_env, eval_env_params = config.get("EVAL_ENV", env), config.get("EVAL_ENV_PARAMS", env_params)
    if config.get("EVAL_ENV") is not None:
        eval_env = LogWrapper(eval_env)
    else:
        eval_env = env
        eval_env_params = env_params

    config["CONTINUOUS"] = type(env.action_space(env_params)) == spaces.Box

    linear_schedule = make_linear_schedule(config)

    # Create GTrXL network
    network = create_network("gtrxl", env.action_space(env_params), config)

    # Observation size is directly from environment
    obs_size = env.observation_space(env_params).shape[0]

    init_x = (jnp.zeros((1, config["NUM_ENVS"], obs_size)),
              jnp.zeros((1, config["NUM_ENVS"])))

    # GTrXL specific config
    window_grad = config.get("GTRXL_WINDOW_GRAD", 16)

    def train(rng):
        # Initialize MER (Mean Episodic Return) tracking
        max_train_mer = float('-inf')
        max_eval_mer = float('-inf')

        # INIT NETWORK PARAMETERS
        rng, _rng = jax.random.split(rng)
        init_hstate = network.initialize_core_hidden_state(config["NUM_ENVS"])
        network_params = network.init(_rng, init_hstate, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5)
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # Initialize GTrXL memory
        memory = network.initialize_core_hidden_state(config["NUM_ENVS"])

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            # Unpack state
            train_state, env_state, obsv, last_done, memory, rng = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, step_idx):
                train_state, env_state, last_obs, last_done, memory, rng = runner_state

                # Reset memory for environments where episode ended
                memory = reset_memory_on_done(memory, last_done)

                rng, _rng = jax.random.split(rng)

                # SELECT ACTION (single timestep forward)
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                new_memory, pi, value = network.apply(train_state.params, memory, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

                # Cache memory BEFORE update for training
                # This memory corresponds to the state at the start of this step
                cached_memory = memory

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info, cached_memory
                )
                runner_state = (train_state, env_state, obsv, done, new_memory, rng)
                return runner_state, transition

            # Run trajectory collection
            runner_state_inner = (train_state, env_state, obsv, last_done, memory, rng)
            runner_state_inner, traj_batch = jax.lax.scan(
                _env_step, runner_state_inner, jnp.arange(config["NUM_STEPS"])
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, memory, rng = runner_state_inner

            # Reset memory before final value computation
            memory_for_value = reset_memory_on_done(memory, last_done)
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, memory_for_value, ac_in)
            last_val = last_val.squeeze(0)

            # Calculate advantages using common GAE function
            advantages, targets = calculate_gae(
                traj_batch, last_val, last_done,
                config["GAMMA"], config["GAE_LAMBDA"]
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn_gtrxl(params, traj_batch, gae, targets):
                        """GTrXL loss function with WINDOW_GRAD chunked processing.

                        Implements Truncated BPTT:
                        - Splits trajectory into WINDOW_GRAD-sized chunks
                        - Uses cached memory from rollout for each chunk start
                        - Computes loss with gradients only within each chunk
                        """
                        num_steps = traj_batch.obs.shape[0]  # (seq_len, batch, obs_dim)
                        batch_size = traj_batch.obs.shape[1]
                        num_chunks = num_steps // window_grad

                        def process_chunk(carry, chunk_idx):
                            """Process a single WINDOW_GRAD chunk with Truncated BPTT."""
                            _ = carry  # Not used - we use cached memory instead
                            start = chunk_idx * window_grad

                            # Extract chunk data using dynamic_slice
                            chunk_obs = jax.lax.dynamic_slice(
                                traj_batch.obs, (start, 0, 0),
                                (window_grad, batch_size, traj_batch.obs.shape[2])
                            )
                            chunk_done = jax.lax.dynamic_slice(
                                traj_batch.done, (start, 0),
                                (window_grad, batch_size)
                            )
                            chunk_action = jax.lax.dynamic_slice(
                                traj_batch.action,
                                (start, 0) if traj_batch.action.ndim == 2 else (start, 0, 0),
                                (window_grad,) + traj_batch.action.shape[1:]
                            )
                            chunk_log_prob_old = jax.lax.dynamic_slice(
                                traj_batch.log_prob, (start, 0),
                                (window_grad, batch_size)
                            )
                            chunk_value_old = jax.lax.dynamic_slice(
                                traj_batch.value, (start, 0),
                                (window_grad, batch_size)
                            )
                            chunk_gae = jax.lax.dynamic_slice(
                                gae, (start, 0),
                                (window_grad, batch_size)
                            )
                            chunk_targets = jax.lax.dynamic_slice(
                                targets, (start, 0),
                                (window_grad, batch_size)
                            )

                            # Get cached memory from the start of this chunk
                            # traj_batch.memory is a list of arrays, each with shape:
                            # (num_steps, 1, num_envs, mem_len * d_model)
                            # After scan, the first axis is num_steps.
                            # We extract memory at chunk start and keep (1, num_envs, mem_len * d_model)
                            chunk_memory = [
                                jax.lax.dynamic_slice(
                                    mem, (start, 0, 0, 0),
                                    (1, 1, batch_size, mem.shape[3])
                                ).squeeze(0)  # Remove the first dim (was chunk index), keep (1, batch, mem)
                                for mem in traj_batch.memory
                            ]

                            # Stop gradient on memory for Truncated BPTT
                            chunk_memory = jax.tree_util.tree_map(
                                jax.lax.stop_gradient, chunk_memory
                            )

                            # Forward pass through chunk
                            new_memory, pi, value = network.apply(
                                params, chunk_memory, (chunk_obs, chunk_done)
                            )
                            log_prob = pi.log_prob(chunk_action)

                            # Compute losses for this chunk
                            value_pred_clipped = chunk_value_old + (value - chunk_value_old).clip(
                                -config["CLIP_EPS"], config["CLIP_EPS"]
                            )
                            value_losses = jnp.square(value - chunk_targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - chunk_targets)
                            chunk_value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped)

                            # Normalize GAE within chunk
                            chunk_gae_norm = (chunk_gae - chunk_gae.mean()) / (chunk_gae.std() + 1e-8)

                            ratio = jnp.exp(log_prob - chunk_log_prob_old)
                            loss_actor1 = ratio * chunk_gae_norm
                            loss_actor2 = jnp.clip(
                                ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
                            ) * chunk_gae_norm
                            chunk_actor_loss = -jnp.minimum(loss_actor1, loss_actor2)
                            chunk_entropy = pi.entropy()

                            # Return chunk statistics (sum for later averaging)
                            chunk_stats = (
                                chunk_value_loss.sum(),
                                chunk_actor_loss.sum(),
                                chunk_entropy.sum(),
                                jnp.array(window_grad * batch_size, dtype=jnp.float32)
                            )
                            return None, chunk_stats

                        # Process all chunks
                        _, chunk_results = jax.lax.scan(
                            process_chunk, None, jnp.arange(num_chunks)
                        )

                        # Aggregate losses across chunks
                        total_value_loss, total_actor_loss, total_entropy, total_count = chunk_results
                        total_count_sum = total_count.sum()

                        value_loss = total_value_loss.sum() / total_count_sum
                        loss_actor = total_actor_loss.sum() / total_count_sum
                        entropy = total_entropy.sum() / total_count_sum

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn_gtrxl, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state

                # GTrXL-specific: Shuffle only along NUM_ENVS dimension to preserve temporal order
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                # GTrXL memory has shape (steps, 1, batch, mem_dim) - need special handling
                # Separate memory from other fields for custom reshape
                traj_no_mem = Transition(
                    done=traj_batch.done,
                    action=traj_batch.action,
                    value=traj_batch.value,
                    reward=traj_batch.reward,
                    log_prob=traj_batch.log_prob,
                    obs=traj_batch.obs,
                    info=traj_batch.info,
                    memory=None  # Handle separately
                )

                # Shuffle non-memory fields along env dimension (axis=1)
                shuffled_no_mem = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1) if x is not None else None,
                    traj_no_mem
                )
                shuffled_adv = jnp.take(advantages, permutation, axis=1)
                shuffled_tgt = jnp.take(targets, permutation, axis=1)

                # Shuffle memory along batch dimension (axis=2 for shape (steps, 1, batch, mem))
                shuffled_memory = [jnp.take(mem, permutation, axis=2) for mem in traj_batch.memory]

                # Reshape for minibatches
                def reshape_field_to_minibatches(x):
                    """Reshape (steps, envs, ...) -> (num_mb, steps, envs_per_mb, ...)"""
                    if x is None:
                        return None
                    shape = x.shape
                    new_shape = (shape[0], config["NUM_MINIBATCHES"], -1) + shape[2:]
                    return jnp.reshape(x, new_shape).swapaxes(0, 1)

                def reshape_memory_to_minibatches(mem_list):
                    """Reshape memory (steps, 1, batch, mem) -> list of (num_mb, steps, 1, batch/num_mb, mem)"""
                    result = []
                    for mem in mem_list:
                        # mem shape: (steps, 1, batch, mem_dim)
                        steps, one, batch, mem_dim = mem.shape
                        # Reshape batch dim to (num_mb, batch/num_mb)
                        reshaped = mem.reshape(steps, one, config["NUM_MINIBATCHES"], -1, mem_dim)
                        # Swap to (num_mb, steps, 1, batch/num_mb, mem_dim)
                        reshaped = reshaped.swapaxes(0, 2)  # (num_mb, 1, steps, batch/num_mb, mem_dim)
                        reshaped = reshaped.swapaxes(1, 2)  # (num_mb, steps, 1, batch/num_mb, mem_dim)
                        result.append(reshaped)
                    return result

                # Reshape non-memory fields
                mb_no_mem = jax.tree_util.tree_map(reshape_field_to_minibatches, shuffled_no_mem)
                mb_adv = reshape_field_to_minibatches(shuffled_adv)
                mb_tgt = reshape_field_to_minibatches(shuffled_tgt)
                mb_memory = reshape_memory_to_minibatches(shuffled_memory)

                # Reconstruct minibatches as list of (traj, adv, tgt) for each minibatch
                num_minibatches = config["NUM_MINIBATCHES"]

                def process_single_minibatch(train_state, mb_idx):
                    # Extract this minibatch's data
                    mb_traj = Transition(
                        done=mb_no_mem.done[mb_idx],
                        action=mb_no_mem.action[mb_idx],
                        value=mb_no_mem.value[mb_idx],
                        reward=mb_no_mem.reward[mb_idx],
                        log_prob=mb_no_mem.log_prob[mb_idx],
                        obs=mb_no_mem.obs[mb_idx],
                        info=jax.tree_util.tree_map(lambda x: x[mb_idx] if x is not None else None, mb_no_mem.info),
                        memory=[m[mb_idx] for m in mb_memory]
                    )
                    mb_batch_info = (mb_traj, mb_adv[mb_idx], mb_tgt[mb_idx])
                    return _update_minbatch(train_state, mb_batch_info)

                train_state, total_loss = jax.lax.scan(
                    process_single_minibatch, train_state, jnp.arange(num_minibatches)
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # Extract loss metrics
            total_loss, (value_loss, actor_loss, entropy) = jax.tree_util.tree_map(
                lambda x: x.mean(), loss_info
            )

            # Prepare next runner_state
            next_runner_state = (train_state, env_state, last_obs, last_done, memory, rng)

            # Calculate train MER and log
            train_mer = safe_mean(traj_batch.info)
            train_num_done_episodes = traj_batch.done.sum()

            def train_env_callback(train_mer, train_done, step):
                nonlocal max_train_mer
                max_train_mer = max(max_train_mer, float(train_mer))
                logger.info(f"[Step {int(step)}]")
                logger.info(f"Train MER: {train_mer:.6f}, MMER: {max_train_mer:.6f}, #Done: {train_done}")
                wandb.log({
                    "train/env/mer": train_mer,
                    "train/env/mmer": max_train_mer,
                    "train/env/num_done_episodes": train_done,
                }, step=int(step))

            def train_loss_callback(total_loss, value_loss, actor_loss, entropy, step):
                logger.info(f"Total Loss: {total_loss:.6f}")
                logger.info(f"Val: {value_loss:.6f}, Act: {actor_loss:.6f}, Ent: {entropy:.6f}")
                wandb.log({
                    "train/loss/total": total_loss,
                    "train/loss/value": value_loss,
                    "train/loss/actor": actor_loss,
                    "train/loss/entropy": entropy,
                }, step=int(step))

            jax.debug.callback(train_env_callback, train_mer, train_num_done_episodes, update_idx)
            jax.debug.callback(train_loss_callback, total_loss, value_loss, actor_loss, entropy, update_idx)

            # EVALUATION
            def _eval_env_step(runner_state, unused):
                train_state, eval_env_state, eval_last_obs, eval_last_done, eval_memory, eval_rng = runner_state
                eval_rng, _eval_rng = jax.random.split(eval_rng)

                # Reset memory on done
                eval_memory = reset_memory_on_done(eval_memory, eval_last_done)

                # SELECT ACTION
                ac_in = (eval_last_obs[np.newaxis, :], eval_last_done[np.newaxis, :])
                eval_memory, pi, value = network.apply(train_state.params, eval_memory, ac_in)
                action = pi.sample(seed=_eval_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

                # STEP EVAL ENV
                eval_rng, _eval_rng = jax.random.split(eval_rng)
                eval_rng_step = jax.random.split(_eval_rng, config["NUM_ENVS"])
                eval_obsv, eval_env_state, reward, done, info = jax.vmap(
                    eval_env.step, in_axes=(0, 0, 0, None)
                )(eval_rng_step, eval_env_state, action, eval_env_params)

                # Create transition without memory (eval doesn't need it)
                from .ppo_common import Transition as CommonTransition
                transition = CommonTransition(eval_last_done, action, value, reward, log_prob, eval_last_obs, info)
                runner_state = (train_state, eval_env_state, eval_obsv, done, eval_memory, eval_rng)
                return runner_state, transition

            # Use fixed eval_seed for deterministic evaluation
            eval_rng = jax.random.key(config.get("EVAL_SEED", 12345))

            # Reset eval env
            reset_rng = jax.random.split(eval_rng, config["NUM_ENVS"])
            eval_obsv, eval_env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(reset_rng, eval_env_params)
            eval_last_done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)

            # Initialize eval memory
            eval_memory = network.initialize_core_hidden_state(config["NUM_ENVS"])

            eval_step_rng = jax.random.fold_in(eval_rng, 0)
            eval_runner_state = (train_state, eval_env_state, eval_obsv, eval_last_done, eval_memory, eval_step_rng)
            eval_runner_state, eval_traj_batch = jax.lax.scan(
                _eval_env_step, eval_runner_state, None, config["NUM_STEPS"]
            )

            # Calculate eval MER and log
            eval_mer = safe_mean(eval_traj_batch.info)
            eval_num_done_episodes = eval_traj_batch.done.sum()

            def eval_callback(eval_mer, eval_done, step):
                nonlocal max_eval_mer
                max_eval_mer = max(max_eval_mer, float(eval_mer))
                logger.info(f"Eval  MER: {eval_mer:.6f}, MMER: {max_eval_mer:.6f}, #Done: {eval_done}")
                wandb.log({
                    "eval/env/mer": eval_mer,
                    "eval/env/mmer": max_eval_mer,
                    "eval/env/num_done_episodes": eval_done,
                }, step=int(step))

            jax.debug.callback(eval_callback, eval_mer, eval_num_done_episodes, update_idx)

            # Create metrics dictionary
            metrics_dict = {
                "train_mer": train_mer,
                "eval_mer": eval_mer,
                "train_num_done_episodes": train_num_done_episodes,
                "eval_num_done_episodes": eval_num_done_episodes,
            }

            return next_runner_state, metrics_dict

        rng, _rng = jax.random.split(rng)
        last_done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)

        runner_state = (train_state, env_state, obsv, last_done, memory, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )

        # Get the final metrics from the last update
        final_metrics = jax.tree_util.tree_map(lambda x: x[-1], metrics)

        # Add MMER values tracked in callback
        final_metrics["max_train_mer"] = max_train_mer
        final_metrics["max_eval_mer"] = max_eval_mer

        return runner_state, final_metrics

    return train
