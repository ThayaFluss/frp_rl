"""
GTrXL PPO with FRP (Free Random Projection) for Meta-Learning - Memory-efficient version.

This module implements PPO training for GTrXL with FRP transformations,
combining GTrXL's memory-based sequence modeling with FRP's task diversity
through orthogonal transformations.

Key design decisions for memory efficiency:
- Uses ppo_common.Transition (no memory field) to avoid storing memory at each step
- Memory is passed sequentially through chunks during loss computation
- Memory is reset to zero at the start of each epoch
- This reduces GPU memory from ~29GB to ~1-2GB

Reference implementation:
    https://github.com/Reytuag/transformerXL_PPO_JAX
"""
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from gymnax.environments import spaces

from envs.wrappers import LogWrapper
from frp.frp_manager import (
    FRPState,
    create_frp_manager,
    create_eval_frp_manager,
)

from .ppo_common import (
    Transition,
    make_linear_schedule,
    calculate_gae,
    safe_mean,
    setup_config,
    create_network,
)

logger = logging.getLogger(__name__)


# ===== Memory Management Functions =====

def reset_memory_on_done(memory: list, done: jnp.ndarray) -> list:
    """Reset memory to zeros for environments where episode ended.

    Args:
        memory: List of memory tensors, each shape (1, num_envs, mem_len * d_model)
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
    """Create GTrXL+FRP PPO training function.

    Args:
        config: Configuration dictionary with GTrXL, PPO, and FRP parameters

    Returns:
        Training function that takes RNG and returns (runner_state, metrics)
    """
    config = setup_config(config)

    env, env_params = config["ENV"], config["ENV_PARAMS"]
    env = LogWrapper(env)

    eval_env, eval_env_params = config["EVAL_ENV"], config["EVAL_ENV_PARAMS"]
    eval_env = LogWrapper(eval_env)

    config["CONTINUOUS"] = type(env.action_space(env_params)) == spaces.Box

    linear_schedule = make_linear_schedule(config)

    # Create GTrXL network
    network = create_network("gtrxl", env.action_space(env_params), config)

    # Create FRP managers
    frp_manager = create_frp_manager(config)
    eval_frp_manager = create_eval_frp_manager(config)

    # Calculate FRP-transformed observation size
    base_env_obs_size = env.observation_space(env_params).shape[0]
    frp_input_dim = frp_manager.frp_input_dim
    non_transformed_size = base_env_obs_size - frp_input_dim
    frp_transformed_obs_size = frp_manager.aug_output_dim + non_transformed_size

    init_x = (jnp.zeros((1, config["NUM_ENVS"], frp_transformed_obs_size)),
              jnp.zeros((1, config["NUM_ENVS"])))

    # GTrXL specific config
    window_grad = config.get("GTRXL_WINDOW_GRAD", 16)

    def train(rng):
        # Initialize MMER tracking
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

        # Create initial FRP words
        rng, _rng = jax.random.split(rng)  # Match RNG consumption with legacy mode
        frp_word_rng = config["META_KWARGS"]["meta_rng"]
        frp_words = frp_manager.initialize_words(frp_word_rng)

        # Initialize FRP for each environment
        def initialize_frp_single_env(obs, reset_rng_i):
            _, _, index_key = jax.random.split(reset_rng_i, 3)
            env_index = frp_manager.sample_env_index(frp_words, index_key)

            input_dim = frp_manager.input_dim
            metadata_dim = frp_manager.metadata_dim

            raw_obs = obs[:input_dim]
            metadata = obs[input_dim:input_dim + metadata_dim]
            wrapper = obs[input_dim + metadata_dim:]

            frp_input_parts = [raw_obs]
            if frp_manager.include_metadata:
                frp_input_parts.append(metadata)
            if frp_manager.include_wrapper:
                frp_input_parts.append(wrapper)

            frp_input = jnp.concatenate(frp_input_parts)
            transformed_obs = frp_manager.transform_obs(frp_input, env_index, frp_words)

            remaining_parts = []
            if not frp_manager.include_metadata:
                remaining_parts.append(metadata)
            if not frp_manager.include_wrapper:
                remaining_parts.append(wrapper)

            if remaining_parts:
                new_obs = jnp.concatenate([transformed_obs] + remaining_parts)
            else:
                new_obs = transformed_obs

            return env_index, new_obs

        env_indices, obsv = jax.vmap(initialize_frp_single_env)(obsv, reset_rng)
        frp_state = FRPState(env_indices=env_indices, frp_words=frp_words)

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            train_state, env_state, obsv, last_done, memory, rng, frp_state = runner_state

            # Update frp_words if RESET_WORDS is enabled
            if config.get("RESET_WORDS", False):
                rng, _rng = jax.random.split(rng)
                new_frp_words = frp_manager.initialize_words(_rng)
                frp_state = frp_state.replace(frp_words=new_frp_words)

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, step_idx):
                train_state, env_state, last_obs, last_done, memory, rng, env_indices = runner_state

                # Reset memory on done
                memory = reset_memory_on_done(memory, last_done)

                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                new_memory, pi, value = network.apply(train_state.params, memory, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                # Resample env_indices for done environments
                def maybe_resample_env_index(rng_step_i, env_idx, is_done):
                    def resample_branch(_):
                        key, key_reset = jax.random.split(rng_step_i)
                        _, _, index_key = jax.random.split(key_reset, 3)
                        return frp_manager.sample_env_index(frp_state.frp_words, index_key)

                    def keep_branch(_):
                        return env_idx

                    return jax.lax.cond(is_done, resample_branch, keep_branch, operand=None)

                env_indices = jax.vmap(maybe_resample_env_index)(rng_step, env_indices, done)

                # Apply FRP transformation
                def apply_frp_transform(obs, env_idx):
                    input_dim = frp_manager.input_dim
                    metadata_dim = frp_manager.metadata_dim

                    raw_obs = obs[:input_dim]
                    metadata = obs[input_dim:input_dim + metadata_dim]
                    wrapper = obs[input_dim + metadata_dim:]

                    frp_input_parts = [raw_obs]
                    if frp_manager.include_metadata:
                        frp_input_parts.append(metadata)
                    if frp_manager.include_wrapper:
                        frp_input_parts.append(wrapper)

                    frp_input = jnp.concatenate(frp_input_parts)
                    transformed_obs = frp_manager.transform_obs(frp_input, env_idx, frp_state.frp_words)

                    remaining_parts = []
                    if not frp_manager.include_metadata:
                        remaining_parts.append(metadata)
                    if not frp_manager.include_wrapper:
                        remaining_parts.append(wrapper)

                    if remaining_parts:
                        return jnp.concatenate([transformed_obs] + remaining_parts)
                    else:
                        return transformed_obs

                obsv = jax.vmap(apply_frp_transform)(obsv, env_indices)

                # Memory-efficient: don't cache memory in Transition
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, done, new_memory, rng, env_indices)
                return runner_state, transition

            runner_state_inner = (train_state, env_state, obsv, last_done, memory, rng, frp_state.env_indices)
            runner_state_inner, traj_batch = jax.lax.scan(
                _env_step, runner_state_inner, jnp.arange(config["NUM_STEPS"])
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, memory, rng, final_env_indices = runner_state_inner

            memory_for_value = reset_memory_on_done(memory, last_done)
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, memory_for_value, ac_in)
            last_val = last_val.squeeze(0)

            advantages, targets = calculate_gae(
                traj_batch, last_val, last_done,
                config["GAMMA"], config["GAE_LAMBDA"]
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn_gtrxl(params, traj_batch, gae, targets):
                        """GTrXL loss function with memory-efficient chunked processing.

                        Key changes for memory efficiency:
                        - Memory is initialized to zero at epoch start
                        - Memory is passed through chunks via carry (not cached in Transition)
                        - stop_gradient on memory at each chunk boundary (Truncated BPTT)
                        - GAE is normalized across the entire batch for stable gradients
                        """
                        num_steps = traj_batch.obs.shape[0]
                        batch_size = traj_batch.obs.shape[1]
                        num_chunks = num_steps // window_grad

                        # Normalize GAE across entire batch (like GRU PPO)
                        gae_normalized = (gae - gae.mean()) / (gae.std() + 1e-8)

                        def process_chunk(carry, chunk_idx):
                            memory = carry  # Memory from previous chunk
                            start = chunk_idx * window_grad

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
                                gae_normalized, (start, 0),
                                (window_grad, batch_size)
                            )
                            chunk_targets = jax.lax.dynamic_slice(
                                targets, (start, 0),
                                (window_grad, batch_size)
                            )

                            # Stop gradient on memory for Truncated BPTT
                            memory = jax.tree_util.tree_map(
                                jax.lax.stop_gradient, memory
                            )

                            new_memory, pi, value = network.apply(
                                params, memory, (chunk_obs, chunk_done)
                            )
                            log_prob = pi.log_prob(chunk_action)

                            value_pred_clipped = chunk_value_old + (value - chunk_value_old).clip(
                                -config["CLIP_EPS"], config["CLIP_EPS"]
                            )
                            value_losses = jnp.square(value - chunk_targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - chunk_targets)
                            chunk_value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped)

                            ratio = jnp.exp(log_prob - chunk_log_prob_old)
                            loss_actor1 = ratio * chunk_gae
                            loss_actor2 = jnp.clip(
                                ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
                            ) * chunk_gae
                            chunk_actor_loss = -jnp.minimum(loss_actor1, loss_actor2)
                            chunk_entropy = pi.entropy()

                            chunk_stats = (
                                chunk_value_loss.mean(),
                                chunk_actor_loss.mean(),
                                chunk_entropy.mean(),
                            )
                            return new_memory, chunk_stats

                        # Initialize memory to zero at epoch start
                        init_memory = network.initialize_core_hidden_state(batch_size)

                        # Process all chunks, passing memory through carry
                        _, chunk_results = jax.lax.scan(
                            process_chunk, init_memory, jnp.arange(num_chunks)
                        )

                        # Aggregate losses across chunks (mean of means)
                        total_value_loss, total_actor_loss, total_entropy = chunk_results

                        value_loss = total_value_loss.mean()
                        loss_actor = total_actor_loss.mean()
                        entropy = total_entropy.mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn_gtrxl, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state

                # Shuffle only along NUM_ENVS dimension to preserve temporal order
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                # Shuffle along env dimension (axis=1)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), traj_batch
                )
                shuffled_adv = jnp.take(advantages, permutation, axis=1)
                shuffled_tgt = jnp.take(targets, permutation, axis=1)

                # Reshape for minibatches: (steps, envs, ...) -> (num_mb, steps, envs_per_mb, ...)
                def reshape_to_minibatches(x):
                    shape = x.shape
                    new_shape = (shape[0], config["NUM_MINIBATCHES"], -1) + shape[2:]
                    return jnp.reshape(x, new_shape).swapaxes(0, 1)

                mb_batch = jax.tree_util.tree_map(reshape_to_minibatches, shuffled_batch)
                mb_adv = reshape_to_minibatches(shuffled_adv)
                mb_tgt = reshape_to_minibatches(shuffled_tgt)

                # Process minibatches
                def process_single_minibatch(train_state, mb_idx):
                    mb_traj = jax.tree_util.tree_map(lambda x: x[mb_idx], mb_batch)
                    mb_batch_info = (mb_traj, mb_adv[mb_idx], mb_tgt[mb_idx])
                    return _update_minbatch(train_state, mb_batch_info)

                train_state, total_loss = jax.lax.scan(
                    process_single_minibatch, train_state, jnp.arange(config["NUM_MINIBATCHES"])
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            total_loss, (value_loss, actor_loss, entropy) = jax.tree_util.tree_map(
                lambda x: x.mean(), loss_info
            )

            frp_state = frp_state.replace(env_indices=final_env_indices)
            next_runner_state = (train_state, env_state, last_obs, last_done, memory, rng, frp_state)

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

                eval_memory = reset_memory_on_done(eval_memory, eval_last_done)

                ac_in = (eval_last_obs[np.newaxis, :], eval_last_done[np.newaxis, :])
                eval_memory, pi, value = network.apply(train_state.params, eval_memory, ac_in)
                action = pi.sample(seed=_eval_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

                eval_rng, _eval_rng = jax.random.split(eval_rng)
                eval_rng_step = jax.random.split(_eval_rng, config["NUM_ENVS"])
                eval_obsv, eval_env_state, reward, done, info = jax.vmap(
                    eval_env.step, in_axes=(0, 0, 0, None)
                )(eval_rng_step, eval_env_state, action, eval_env_params)

                # Apply evaluation FRP transformation
                def apply_eval_frp_transform(obs):
                    input_dim = frp_manager.input_dim
                    metadata_dim = frp_manager.metadata_dim

                    raw_obs = obs[:input_dim]
                    metadata = obs[input_dim:input_dim + metadata_dim]
                    wrapper = obs[input_dim + metadata_dim:]

                    frp_input_parts = [raw_obs]
                    if frp_manager.include_metadata:
                        frp_input_parts.append(metadata)
                    if frp_manager.include_wrapper:
                        frp_input_parts.append(wrapper)

                    frp_input = jnp.concatenate(frp_input_parts)

                    if eval_frp_manager is not None:
                        transformed_obs = eval_frp_manager.transform_obs(frp_input)
                    else:
                        transformed_obs = frp_input

                    remaining_parts = []
                    if not frp_manager.include_metadata:
                        remaining_parts.append(metadata)
                    if not frp_manager.include_wrapper:
                        remaining_parts.append(wrapper)

                    if remaining_parts:
                        return jnp.concatenate([transformed_obs] + remaining_parts)
                    else:
                        return transformed_obs

                eval_obsv = jax.vmap(apply_eval_frp_transform)(eval_obsv)

                transition = Transition(eval_last_done, action, value, reward, log_prob, eval_last_obs, info)
                runner_state = (train_state, eval_env_state, eval_obsv, done, eval_memory, eval_rng)
                return runner_state, transition

            eval_rng = jax.random.key(config["EVAL_SEED"])

            reset_rng = jax.random.split(eval_rng, config["NUM_ENVS"])
            def eval_partial_reset(x):
                return env.reset(x, eval_env_params)
            eval_obsv, eval_env_state = jax.vmap(eval_partial_reset)(reset_rng)

            # Apply evaluation FRP transformation to initial observations
            def apply_eval_frp_transform_init(obs):
                input_dim = frp_manager.input_dim
                metadata_dim = frp_manager.metadata_dim

                raw_obs = obs[:input_dim]
                metadata = obs[input_dim:input_dim + metadata_dim]
                wrapper = obs[input_dim + metadata_dim:]

                frp_input_parts = [raw_obs]
                if frp_manager.include_metadata:
                    frp_input_parts.append(metadata)
                if frp_manager.include_wrapper:
                    frp_input_parts.append(wrapper)

                frp_input = jnp.concatenate(frp_input_parts)

                if eval_frp_manager is not None:
                    transformed_obs = eval_frp_manager.transform_obs(frp_input)
                else:
                    transformed_obs = frp_input

                remaining_parts = []
                if not frp_manager.include_metadata:
                    remaining_parts.append(metadata)
                if not frp_manager.include_wrapper:
                    remaining_parts.append(wrapper)

                if remaining_parts:
                    return jnp.concatenate([transformed_obs] + remaining_parts)
                else:
                    return transformed_obs

            eval_obsv = jax.vmap(apply_eval_frp_transform_init)(eval_obsv)

            eval_last_done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
            eval_memory = network.initialize_core_hidden_state(config["NUM_ENVS"])

            eval_step_rng = jax.random.fold_in(eval_rng, 0)
            eval_runner_state = (train_state, eval_env_state, eval_obsv, eval_last_done, eval_memory, eval_step_rng)
            eval_runner_state, eval_traj_batch = jax.lax.scan(
                _eval_env_step, eval_runner_state, None, config["NUM_STEPS"]
            )

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

            metrics_dict = {
                "train_mer": train_mer,
                "eval_mer": eval_mer,
                "train_num_done_episodes": train_num_done_episodes,
                "eval_num_done_episodes": eval_num_done_episodes,
            }

            return next_runner_state, metrics_dict

        rng, _rng = jax.random.split(rng)
        last_done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)

        runner_state = (train_state, env_state, obsv, last_done, memory, _rng, frp_state)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )

        final_metrics = jax.tree_util.tree_map(lambda x: x[-1], metrics)
        final_metrics["max_train_mer"] = max_train_mer
        final_metrics["max_eval_mer"] = max_eval_mer

        return runner_state, final_metrics

    return train
