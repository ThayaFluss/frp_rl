"""
PPO training with FRP (Function-space Representation Probe) - SEPARATED mode.

This module implements PPO training where FRP state management is externalized from
the environment. Key features:

- FRP state (env_indices, frp_words) managed separately from MetaEnvironment
- Independent RNG for training and evaluation (completely decoupled)
- Evaluation uses fixed eval_seed for deterministic, reproducible results
- Supports both GRU and S5 encoders via MODEL_TYPE configuration
- FRP words created eagerly at initialization and optionally reset each epoch

For legacy/lazy modes (FRP integrated into environment), use ppo_in_context_legacy.py
"""
import logging
from typing import NamedTuple, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from gymnax.environments import spaces

from envs.wrappers import LogWrapper
from frp.frp_manager import (
    FRPManager,
    EvalFRPManager,
    FRPWords,
    FRPState,
    create_frp_manager,
    create_eval_frp_manager,
)

from .ppo_common import (
    Transition,
    make_linear_schedule,
    calculate_gae,
    safe_mean,
    create_minibatches,
    setup_config,
    create_network,
)

logger = logging.getLogger(__name__)


def make_train(config):
    config = setup_config(config)

    env, env_params = config["ENV"], config["ENV_PARAMS"]
    env = LogWrapper(env)

    eval_env, eval_env_params = config["EVAL_ENV"], config["EVAL_ENV_PARAMS"]
    eval_env = LogWrapper(eval_env)

    config["CONTINUOUS"] = type(env.action_space(env_params)) == spaces.Box

    linear_schedule = make_linear_schedule(config)

    # Create network with encoder type from config (defaults to 'gru')
    model_type = config.get("MODEL_TYPE", "gru").lower()
    network = create_network(model_type, env.action_space(env_params), config)

    frp_manager = create_frp_manager(config)
    eval_frp_manager = create_eval_frp_manager(config)

    # Original: Calculate FRP-transformed observation size explicitly
    # Original obs from env: input_dim + 3 (MetaEnv metadata)
    # After FRP transform: aug_output_dim + 3 (MetaEnv metadata)
    # After AliasPrevActionV2 wrapper: + (n+1) where n=num_actions for Discrete
    base_env_obs_size = env.observation_space(env_params).shape[0]
    # base_env_obs_size = input_dim + 3 + (n+1)
    # We need to replace input_dim with aug_output_dim
    metadata_and_wrapper_size = base_env_obs_size - frp_manager.input_dim
    frp_transformed_obs_size = frp_manager.aug_output_dim + metadata_and_wrapper_size

    init_x = (jnp.zeros((1, config["NUM_ENVS"], frp_transformed_obs_size)),
                jnp.zeros((1, config["NUM_ENVS"])))

    def train(rng):
        # Initialize max metric tracking
        max_train_metric = float('-inf')
        max_eval_metric = float('-inf')


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
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        # env.reset  is defined by gymnax.environments.environment
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = network.initialize_core_hidden_state(config["NUM_ENVS"])

        # Create initial FRP words
        # CRITICAL FIX: To match RNG consumption with legacy mode, we must split the main RNG here
        # even though we use meta_kwargs['meta_rng'] for FRP initialization.
        #
        # Legacy mode does this (ppo_in_context_legacy.py:114-115):
        #   rng, _rng = jax.random.split(rng)
        #   train_words = _create_words(_rng)
        #
        # To maintain RNG consumption parity, Separated mode must also split rng:
        rng, _rng = jax.random.split(rng)  # Split to match legacy RNG consumption
        # But we still use meta_rng for FRP words (not _rng) for deterministic FRP initialization
        frp_word_rng = config["META_KWARGS"]["meta_rng"]
        frp_words = frp_manager.initialize_words(frp_word_rng)

        # IMPORTANT: To match RNG consumption with legacy mode at initialization,
        # we derive sample_rng from reset_rng (not from a new RNG split).
        # Legacy mode: env.reset(reset_rng) → reset_env() → sample env_index
        # Separated mode: env.reset(reset_rng) → then sample env_index using derived RNG
        def initialize_frp_single_env(obs, reset_rng_i):
            # Derive sample RNG from reset_rng to match legacy RNG flow
            # This replicates the split that happens in reset_env()
            _, _, index_key = jax.random.split(reset_rng_i, 3)

            # Sample env_index using the same RNG that legacy mode would use
            env_index = frp_manager.sample_env_index(frp_words, index_key)

            # Observation structure (from MetaEnvironment + wrappers):
            # [raw_obs (input_dim)] + [MetaEnv metadata (3)] + [AliasPrevActionV2 (n+1)]
            #
            # After FRP transformation:
            # [transformed_obs (aug_output_dim)] + [MetaEnv metadata (3)] + [AliasPrevActionV2 (n+1)]
            #
            # We need to:
            # 1. Extract raw_obs[:input_dim]
            # 2. Apply FRP transform → transformed_obs (aug_output_dim)
            # 3. Keep rest of observation unchanged (metadata + wrapper data)
            input_dim = frp_manager.input_dim

            raw_obs = obs[:input_dim]  # Original observation from environment
            rest = obs[input_dim:]  # Everything after raw obs (metadata + wrapper data)

            # Apply FRP transformation
            transformed_obs = frp_manager.transform_obs(raw_obs, env_index, frp_words)

            # Reconstruct: [transformed_obs (aug_output_dim)] + [rest]
            new_obs = jnp.concatenate([transformed_obs, rest])

            return env_index, new_obs

        env_indices, obsv = jax.vmap(initialize_frp_single_env)(obsv, reset_rng)

        # Create FRP state
        frp_state = FRPState(env_indices=env_indices, frp_words=frp_words)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # Unpack state including frp_state
            train_state, env_state, obsv, last_done, hstate, rng, frp_state = runner_state

            # Update frp_words if RESET_WORDS is enabled (static config check)
            if config.get("RESET_WORDS", False):
                rng, _rng = jax.random.split(rng)
                new_frp_words = frp_manager.initialize_words(_rng)
                frp_state = frp_state.replace(frp_words=new_frp_words)

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, env_indices = runner_state

                # DEBUG_TRACE: Trace RNG at start of step
                if config.get("DEBUG_TRACE", False):
                    jax.debug.callback(lambda r, e: print(f"[SEPARATED] Step start RNG: {r}, env_indices: {e}"), rng, env_indices)

                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

                # DEBUG: Trace action
                if config.get("DEBUG_TRACE", False):
                    jax.debug.callback(lambda a, d, e: print(f"[SEPARATED] action={a}, last_done={d}, env_indices={e}"),
                                     action, last_done, env_indices)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )

                # Resample env_indices for environments that are done (meta episode ended)
                # IMPORTANT: To match RNG consumption with legacy mode, we derive resample RNG
                # from rng_step (not from main rng). This ensures deterministic equivalence.
                # Legacy mode: env.step(rng_step) → auto-reset → reset_env(key_reset) → sample env_index
                # Separated mode: env.step(rng_step) → then manually resample using derived key
                def maybe_resample_env_index(rng_step_i, env_idx, is_done):
                    """Resample env_index if episode is done, using RNG derived from rng_step.

                    This matches the RNG consumption pattern in legacy mode:
                    1. rng_step_i is split inside env.step() → (key, key_reset)
                    2. key_reset is split in reset_env() → (env_key, obs_key, index_key)
                    3. index_key is used to sample env_index

                    CRITICAL: We only consume RNG when is_done=True to match legacy behavior.
                    Legacy mode only calls reset_env() (which consumes RNG) when done=True.
                    """
                    def resample_branch(_):
                        # Replicate gymnax env.step() RNG split
                        key, key_reset = jax.random.split(rng_step_i)

                        # Replicate reset_env() RNG split (matches meta_environment_legacy.py:173)
                        _, _, index_key = jax.random.split(key_reset, 3)

                        # Sample env_index using the same RNG that legacy mode would use
                        return frp_manager.sample_env_index(frp_state.frp_words, index_key)

                    def keep_branch(_):
                        # Don't consume RNG, just return current index
                        return env_idx

                    # Only consume RNG and resample when episode is done
                    return jax.lax.cond(is_done, resample_branch, keep_branch, operand=None)

                env_indices = jax.vmap(maybe_resample_env_index)(rng_step, env_indices, done)

                # DEBUG: Trace done, reward, and env_indices after resample
                if config.get("DEBUG_TRACE", False):
                    jax.debug.callback(lambda d, r, e: print(f"[SEPARATED] done={d}, reward={r}, new_env_indices={e}"),
                                     done, reward, env_indices)

                # Apply FRP transformation to observations
                def apply_frp_transform(obs, env_idx):
                    input_dim = frp_manager.input_dim

                    raw_obs = obs[:input_dim]  # Original observation from environment
                    rest = obs[input_dim:]  # Everything after raw obs (metadata + wrapper data)

                    # Apply FRP transformation using env_index
                    transformed_obs = frp_manager.transform_obs(raw_obs, env_idx, frp_state.frp_words)

                    # Reconstruct full observation
                    return jnp.concatenate([transformed_obs, rest])

                obsv = jax.vmap(apply_frp_transform)(obsv, env_indices)

                transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, done, hstate, rng, env_indices)
                return runner_state, transition

            # Minimal runner state for inner loop (include env_indices)
            runner_state_inner = (train_state, env_state, obsv, last_done, hstate, rng, frp_state.env_indices)
            initial_hstate = runner_state_inner[4]  # hstate is at index 4
            runner_state_inner, traj_batch = jax.lax.scan(_env_step, runner_state_inner, None, config["NUM_STEPS"])

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng, final_env_indices = runner_state_inner
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            # Calculate advantages using common GAE function
            advantages, targets = calculate_gae(
                traj_batch, last_val, last_done,
                config["GAMMA"], config["GAE_LAMBDA"]
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch,  advantages, targets = batch_info
                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(params, init_hstate, (traj_batch.obs, traj_batch.done))
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state

                # Create minibatches using common function
                batch = (init_hstate, traj_batch, advantages, targets)
                minibatches, rng = create_minibatches(
                    batch, config["NUM_ENVS"], config["NUM_MINIBATCHES"], rng
                )

                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, initial_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            # Update frp_state with final env_indices from trajectory collection
            frp_state = frp_state.replace(env_indices=final_env_indices)

            # Prepare next runner_state for training (independent of evaluation)
            next_runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, frp_state)

            # EVALUATION (side-effect only, doesn't affect training state)
            def _eval_env_step(runner_state, unused):
                train_state, eval_env_state, eval_last_obs, eval_last_done, eval_hstate, eval_rng = runner_state
                eval_rng, _eval_rng = jax.random.split(eval_rng)

                # SELECT ACTION (using trained params, but eval hidden state)
                ac_in = (eval_last_obs[np.newaxis, :], eval_last_done[np.newaxis, :])
                eval_hstate, pi, value = network.apply(train_state.params, eval_hstate, ac_in)
                action = pi.sample(seed=_eval_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

                # STEP EVAL ENV (completely independent from training env)
                eval_rng, _eval_rng = jax.random.split(eval_rng)
                eval_rng_step = jax.random.split(_eval_rng, config["NUM_ENVS"])
                eval_obsv, eval_env_state, reward, done, info = jax.vmap(eval_env.step, in_axes=(0,0,0,None))(
                    eval_rng_step, eval_env_state, action, eval_env_params
                )

                # Apply evaluation FRP transformation
                def apply_eval_frp_transform(obs):
                    input_dim = frp_manager.input_dim

                    raw_obs = obs[:input_dim]  # Original observation from environment
                    rest = obs[input_dim:]  # Everything after raw obs (metadata + wrapper data)

                    # Apply evaluation FRP transformation
                    if eval_frp_manager is not None:
                        transformed_obs = eval_frp_manager.transform_obs(raw_obs)
                    else:
                        # If no eval manager, use raw observation as-is
                        transformed_obs = raw_obs

                    # Reconstruct full observation
                    return jnp.concatenate([transformed_obs, rest])

                eval_obsv = jax.vmap(apply_eval_frp_transform)(eval_obsv)

                transition = Transition(eval_last_done, action, value, reward, log_prob, eval_last_obs, info)
                runner_state = (train_state, eval_env_state, eval_obsv, done, eval_hstate, eval_rng)
                return runner_state, transition

            # In-Context evaluation
            # CRITICAL: Use fixed eval_seed for deterministic evaluation
            # Generate fresh RNG from eval_seed every evaluation (xland-minigrid pattern)
            eval_rng = jax.random.key(config["EVAL_SEED"])

            # DEBUG_TRACE: Trace eval loop start RNG
            if config.get("DEBUG_TRACE", False):
                jax.debug.callback(lambda r: print(f"[SEPARATED] Eval start RNG: {r}"), eval_rng)

            # Reset eval env before collecting trajectory
            reset_rng = jax.random.split(eval_rng, config["NUM_ENVS"])
            # env.reset  is defined by gymnax.environments.environment
            eval_partial_reset = lambda x: env.reset(x, eval_env_params)
            eval_obsv, eval_env_state = jax.vmap(eval_partial_reset)(reset_rng)

            # Apply evaluation FRP transformation to initial observations
            def apply_eval_frp_transform_init(obs):
                input_dim = frp_manager.input_dim

                raw_obs = obs[:input_dim]  # Original observation from environment
                rest = obs[input_dim:]  # Everything after raw obs (metadata + wrapper data)

                if eval_frp_manager is not None:
                    transformed_obs = eval_frp_manager.transform_obs(raw_obs)
                else:
                    # If no eval manager, use raw observation as-is
                    transformed_obs = raw_obs

                return jnp.concatenate([transformed_obs, rest])

            eval_obsv = jax.vmap(apply_eval_frp_transform_init)(eval_obsv)

            eval_last_done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)

            # Initialize eval hidden state (deterministic zero initialization)
            # Always starts from the same initial state for reproducible evaluation
            eval_hstate = network.initialize_core_hidden_state(config["NUM_ENVS"])

            # Use eval_rng for evaluation step loop (not training rng)
            # This ensures evaluation doesn't affect training RNG consumption
            eval_step_rng = jax.random.fold_in(eval_rng, 0)
            eval_runner_state = (train_state, eval_env_state, eval_obsv, eval_last_done, eval_hstate, eval_step_rng)
            eval_runner_state, eval_traj_batch = jax.lax.scan(_eval_env_step, eval_runner_state, None, config["NUM_STEPS"])

            # DEBUG_TRACE: Trace eval loop end RNG
            if config.get("DEBUG_TRACE", False):
                jax.debug.callback(lambda r: print(f"[SEPARATED] Eval end RNG: {r}"), eval_runner_state[-1])

            # Calculate metrics using common safe_mean function
            train_metric = safe_mean(traj_batch.info)
            eval_metric = safe_mean(eval_traj_batch.info)

            # Count episode done events (episode end = reset_env will be used next step)
            train_episode_done_count = traj_batch.done.sum()
            eval_episode_done_count = eval_traj_batch.done.sum()

            def callback(train_metric, eval_metric, train_done, eval_done):
                nonlocal max_train_metric, max_eval_metric

                # Update max values
                max_train_metric = max(max_train_metric, float(train_metric))
                max_eval_metric = max(max_eval_metric, float(eval_metric))

                logger.info(f"Train metric: {train_metric}, Eval metric: {eval_metric}")
                logger.info(f"Train episode done: {train_done}, Eval episode done: {eval_done}")
                wandb.log({
                        "train/metric": train_metric,
                        "train/max_metric": max_train_metric,
                        "train/episode_done_count": train_done,
                        "eval/metric": eval_metric,
                        "eval/max_metric": max_eval_metric,
                        "eval/episode_done_count": eval_done,
                })
            jax.debug.callback(callback, train_metric, eval_metric, train_episode_done_count, eval_episode_done_count)

            # Create metrics dictionary
            metrics_dict = {
                "train_metric": train_metric,
                "eval_metric": eval_metric,
                "train_episode_done_count": train_episode_done_count,
                "eval_episode_done_count": eval_episode_done_count,
            }

            # DEBUG: Track RNG state at end of update
            if config.get("DEBUG_TRACE", False):
                jax.debug.callback(lambda r, e: print(f"[SEPARATED] End-of-update RNG: {r}, env_indices: {e}"),
                                 rng, final_env_indices)

            return next_runner_state, metrics_dict

        rng, _rng = jax.random.split(rng)
        last_done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)

        runner_state = (train_state, env_state, obsv, last_done, init_hstate, _rng,
                       frp_state)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])

        # Get the final metrics from the last update
        final_metrics = jax.tree_util.tree_map(lambda x: x[-1], metrics)

        # Add max values tracked in callback
        final_metrics["max_train_metric"] = max_train_metric
        final_metrics["max_eval_metric"] = max_eval_metric

        return runner_state, final_metrics
    
    return train

if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 1,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e5,
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 1,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "MemoryChain-bsuite",
        "ANNEAL_LR": True,
        "DEBUG": True,
    }

    jit_train = jax.jit(make_train(config))

    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
