"""
AGaLiTe PPO Implementation - based on agalite/src_pure/purejaxrl/ppo_rnn.py

This module implements PPO training specifically for AGaLiTe (Approximate Gated Linear Transformer).
It follows the agalite reference implementation pattern with:
- BatchedAGaLiTe as rnn_module
- Dictionary-based memory state
- reset_on_terminate=True for automatic episode reset (built into AGaLiTe)

Key features:
- O(N) linear attention (vs O(N^2) for GTrXL)
- Memory efficient (~1-2GB)
- Built-in reset_hidden_on_terminate support
"""
import logging
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.environments import spaces

from envs.wrappers import LogWrapper
from .agalite import BatchedAGaLiTe

from .ppo_common import (
    Transition,
    make_linear_schedule,
    calculate_gae,
    safe_mean,
    setup_config,
)

logger = logging.getLogger(__name__)


class ActorCriticAGaLiTe(nn.Module):
    """Actor-Critic network using AGaLiTe as the recurrent backbone.

    This follows the agalite ppo_rnn.py pattern where rnn_module is called as:
        hidden, embedding = self.rnn_module()(hidden, rnn_in)
    """
    action_dim: int
    config: Dict

    def setup(self):
        """Setup AGaLiTe module."""
        self.rnn_module = BatchedAGaLiTe(
            n_layers=self.config.get("AGALITE_N_LAYERS", 4),
            d_model=self.config.get("AGALITE_D_MODEL", 64),
            d_head=self.config.get("AGALITE_D_HEAD", 64),
            d_ffc=self.config.get("AGALITE_D_FFC", 64),
            n_heads=self.config.get("AGALITE_N_HEADS", 4),
            eta=self.config.get("AGALITE_ETA", 4),
            r=self.config.get("AGALITE_R", 2),
            reset_on_terminate=True  # Built-in episode reset
        )

    @nn.compact
    def __call__(self, hidden, x):
        """Forward pass through ActorCritic.

        Args:
            hidden: Memory state (dict with layer memories)
            x: Tuple of (obs, dones)
                obs: (T, B, obs_dim)
                dones: (T, B)

        Returns:
            hidden: Updated memory state
            pi: Action distribution
            value: Value estimates (T, B)
        """
        obs, dones = x

        # Input embedding to d_model dimension
        # Note: AGaLiTe's first layer has use_dense=True which handles embedding,
        # but we add a pre-embedding here for flexibility with different obs sizes
        d_model = self.config.get("AGALITE_D_MODEL", 64)
        embedding = nn.Dense(
            d_model,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        # AGaLiTe processing (agalite style)
        rnn_in = (embedding, dones)
        hidden, embedding = self.rnn_module(hidden, rnn_in)

        # Actor head
        hidden_size = self.config.get("HIDDEN", 256)
        actor_mean = nn.Dense(
            hidden_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(
            hidden_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    def initialize_carry(self, batch_size: int):
        """Initialize AGaLiTe memory state.

        Args:
            batch_size: Number of parallel environments

        Returns:
            Initial memory state (dict)
        """
        return BatchedAGaLiTe.initialize_carry(
            batch_size=batch_size,
            n_layers=self.config.get("AGALITE_N_LAYERS", 4),
            n_heads=self.config.get("AGALITE_N_HEADS", 4),
            d_head=self.config.get("AGALITE_D_HEAD", 64),
            eta=self.config.get("AGALITE_ETA", 4),
            r=self.config.get("AGALITE_R", 2)
        )


def make_train(config):
    """Create AGaLiTe PPO training function.

    Args:
        config: Configuration dictionary with AGaLiTe and PPO parameters

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

    # Create AGaLiTe network
    network = ActorCriticAGaLiTe(
        action_dim=env.action_space(env_params).n,
        config=config
    )

    # Observation size is directly from environment
    obs_size = env.observation_space(env_params).shape[0]

    init_x = (
        jnp.zeros((1, config["NUM_ENVS"], obs_size)),
        jnp.zeros((1, config["NUM_ENVS"]))
    )

    def train(rng):
        # Initialize MER (Mean Episodic Return) tracking
        max_train_mer = float('-inf')
        max_eval_mer = float('-inf')

        # INIT NETWORK PARAMETERS
        rng, _rng = jax.random.split(rng)
        init_hstate = network.initialize_carry(config["NUM_ENVS"])
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

        # Initialize AGaLiTe memory
        init_hstate = network.initialize_carry(config["NUM_ENVS"])

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            # Unpack state
            train_state, env_state, obsv, last_done, hstate, rng = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                rng, _rng = jax.random.split(rng)

                # SELECT ACTION (single timestep forward)
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, done, hstate, rng)
                return runner_state, transition

            # Run trajectory collection
            initial_hstate = hstate
            runner_state_inner = (train_state, env_state, obsv, last_done, hstate, rng)
            runner_state_inner, traj_batch = jax.lax.scan(
                _env_step, runner_state_inner, None, config["NUM_STEPS"]
            )

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
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn_agalite(params, init_hstate, traj_batch, gae, targets):
                        """AGaLiTe loss function following agalite ppo_rnn.py pattern."""
                        # RERUN NETWORK (agalite style: first_state from init_hstate)
                        first_state = jax.tree_map(lambda x: x[0], init_hstate)
                        _, pi, value = network.apply(
                            params, first_state, (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn_agalite, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            # Prepare init_hstate for minibatch processing (agalite style: TBH format)
            init_hstate = jax.tree_map(lambda x: x[None, :], initial_hstate)

            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
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
            next_runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)

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
                train_state, eval_env_state, eval_last_obs, eval_last_done, eval_hstate, eval_rng = runner_state
                eval_rng, _eval_rng = jax.random.split(eval_rng)

                # SELECT ACTION
                ac_in = (eval_last_obs[np.newaxis, :], eval_last_done[np.newaxis, :])
                eval_hstate, pi, value = network.apply(train_state.params, eval_hstate, ac_in)
                action = pi.sample(seed=_eval_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

                # STEP EVAL ENV
                eval_rng, _eval_rng = jax.random.split(eval_rng)
                eval_rng_step = jax.random.split(_eval_rng, config["NUM_ENVS"])
                eval_obsv, eval_env_state, reward, done, info = jax.vmap(
                    eval_env.step, in_axes=(0, 0, 0, None)
                )(eval_rng_step, eval_env_state, action, eval_env_params)

                transition = Transition(eval_last_done, action, value, reward, log_prob, eval_last_obs, info)
                runner_state = (train_state, eval_env_state, eval_obsv, done, eval_hstate, eval_rng)
                return runner_state, transition

            # Use fixed eval_seed for deterministic evaluation
            eval_rng = jax.random.key(config.get("EVAL_SEED", 12345))

            # Reset eval env
            reset_rng = jax.random.split(eval_rng, config["NUM_ENVS"])
            eval_obsv, eval_env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(reset_rng, eval_env_params)
            eval_last_done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)

            # Initialize eval hidden state
            eval_hstate = network.initialize_carry(config["NUM_ENVS"])

            eval_step_rng = jax.random.fold_in(eval_rng, 0)
            eval_runner_state = (train_state, eval_env_state, eval_obsv, eval_last_done, eval_hstate, eval_step_rng)
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

        runner_state = (train_state, env_state, obsv, last_done, init_hstate, _rng)
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
