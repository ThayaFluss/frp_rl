"""
Unified PPO training with in-context learning (eager word creation).

This module supports both GRU and S5 encoders via the MODEL_TYPE configuration parameter.
Words are created eagerly at initialization and optionally reset each epoch.
"""
from typing import NamedTuple, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from gymnax.environments import spaces

from envs.wrappers import LogWrapper
from frp.orthogonal import create_words, create_orthogonal_matrices

from .ppo_common import (
    Transition,
    make_linear_schedule,
    calculate_gae,
    safe_mean,
    create_minibatches,
    setup_config,
    create_network,
)

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

    def train(rng):
        # Initialize max metric tracking
        max_train_metric = float('-inf')
        max_eval_metric = float('-inf')

        # INIT NETWORK PARAMETERS
        rng, _rng = jax.random.split(rng)
        init_x = (jnp.zeros((1, config["NUM_ENVS"], *env.observation_space(env_params).shape)), jnp.zeros((1, config["NUM_ENVS"])))
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
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = network.initialize_core_hidden_state(config["NUM_ENVS"])

        # INIT EVAL ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        eval_obsv, eval_env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(reset_rng, eval_env_params)
        eval_init_hstate = network.initialize_core_hidden_state(config["NUM_ENVS"])

        # Add function to create words that will be called periodically
        def _create_words(key):
            matrices = create_orthogonal_matrices(
                key,
                config["ENV"].meta_depth,
                size=config["ENV"].meta_dim,
                max_depth=config["ENV"].meta_max_depth,
                with_adjoint=config["ENV"].meta_with_adjoint
            )
            words = create_words(
                matrices,
                config["ENV"].meta_depth,
                out_size=config["ENV"].meta_dim,
                max_depth=config["ENV"].meta_max_depth
            )
            input_dim = config["ENV"].obs_shape[0]
            # For identity eval method, we need to ensure the output of words match the observation dimension
            if hasattr(config["EVAL_ENV"], "meta_const_aug") and config["EVAL_ENV"].meta_const_aug == "identity":
                # For identity, we need to ensure the output dimension matches the input dimension
                # We'll slice the words to match the input dimension for both input and output
                return words[:, :input_dim, :input_dim]
            else:
                # For other evaluation methods, keep the original behavior
                # (truncate input dimension but keep output dimension as meta_dim)
                return words[:, :input_dim, :]

        # Create initial words for both training and eval
        rng, _rng = jax.random.split(rng)
        train_words = _create_words(_rng)

        # Set the words in environments
        env.words = train_words

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # Unpack state including words
            train_state, env_state, obsv, last_done, hstate, rng, words = runner_state

            if config.get("RESET_WORDS"):
                if config["RESET_WORDS"]:
                    # Update words after each epoch
                    rng, _rng = jax.random.split(rng)
                    words = _create_words(_rng)
                    
                    # Update environment words
                    env.words = words
                    env_state.env_state.replace(
                        obs_words=words)

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, done, hstate, rng)
                return runner_state, transition

            # Minimal runner state for inner loop
            runner_state_inner = (train_state, env_state, obsv, last_done, hstate, rng)
            initial_hstate = runner_state_inner[-2]  # hstate is at index 4
            runner_state_inner, traj_batch = jax.lax.scan(_env_step, runner_state_inner, None, config["NUM_STEPS"])

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state_inner
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

            # EVALUATION
            def _eval_env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(eval_env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, eval_env_params
                )
                transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, done, hstate, rng)
                return runner_state, transition

            # In-Context evaluation 
            rng, _rng = jax.random.split(rng)
            # Reset eval env before collecting trajectly
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            eval_partial_reset = lambda x: env.reset(x, eval_env_params)
            eval_obsv, eval_env_state = jax.vmap(eval_partial_reset)(reset_rng)
            eval_last_done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
            eval_hstate=initial_hstate
            
            rng, _rng = jax.random.split(rng)
            eval_runner_state = (train_state, eval_env_state, eval_obsv, eval_last_done, eval_hstate, _rng)
            eval_runner_state, eval_traj_batch = jax.lax.scan(_eval_env_step, eval_runner_state, None, config["NUM_STEPS"])

            # Calculate metrics using common safe_mean function
            train_metric = safe_mean(traj_batch.info)
            in_context_metric = safe_mean(eval_traj_batch.info)

            # Count episode done events (episode end = reset_env will be used next step)
            train_episode_done_count = traj_batch.done.sum()
            eval_episode_done_count = eval_traj_batch.done.sum()

            def callback(train_metric, in_context_metric, train_done, eval_done):
                nonlocal max_train_metric, max_eval_metric

                # Update max values
                max_train_metric = max(max_train_metric, float(train_metric))
                max_eval_metric = max(max_eval_metric, float(in_context_metric))

                print(f"Train metric: {train_metric}, In-context: {in_context_metric}")
                print(f"Train episode done: {train_done}, Eval episode done: {eval_done}")
                wandb.log({
                        "metric": train_metric,
                        "eval_metric": in_context_metric,
                        "max_metric": max_train_metric,
                        "max_eval_metric": max_eval_metric,
                        "train_episode_done_count": train_done,
                        "eval_episode_done_count": eval_done,
                })
            jax.debug.callback(callback, train_metric, in_context_metric, train_episode_done_count, eval_episode_done_count)

            # Create metrics dictionary
            metrics_dict = {
                "train_metric": train_metric,
                "in_context_metric": safe_mean(eval_traj_batch.info),
                "train_episode_done_count": train_episode_done_count,
                "eval_episode_done_count": eval_episode_done_count,
            }

            return (train_state, env_state, last_obs, last_done, hstate, rng, words), metrics_dict

        rng, _rng = jax.random.split(rng)
        last_done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                              
        runner_state = (train_state, env_state, obsv, last_done, init_hstate, _rng, 
                       train_words)
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
