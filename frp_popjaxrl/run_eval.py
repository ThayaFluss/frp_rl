"""
Evaluation script for saved model checkpoints.

This script evaluates trained models and tracks per-trial statistics (returns, steps,
success rates) to analyze in-context learning performance over multiple trials.

Example usage:
    python run_eval.py --checkpoint=checkpoints/cartpole_gru_seed42.pkl \
                       --eval_num_trials=32 \
                       --eval_method=tiling \
                       --num_episodes=10 \
                       --log_wandb=popgym_eval

Outputs:
    - Console: Per-episode and per-trial statistics (returns, steps, success rates)
    - PNG: Plots of mean return/steps/success rate ± std across trials
    - CSV: Trial-wise statistics data (returns, steps, success rates)
    - Wandb: Per-trial mean/std for all metrics logged with trial number as x-axis
"""

import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from flax.core import freeze
from gymnax.environments import spaces

from utils.checkpoint import load_checkpoint
from algorithms.ppo_gru_in_context import ActorCriticRNN, ScannedRNN
from algorithms.ppo_s5_in_context import ActorCriticS5
from algorithms.ppo_s5_in_context import init_S5SSM, make_DPLR_HiPPO, StackedEncoderModel
from envs.meta_environment import create_meta_environment
from envs.wrappers import AliasPrevActionV2, LogWrapper


class Transition:
    """Simple transition class for evaluation"""
    def __init__(self, done, action, value, reward, log_prob, obs, info):
        self.done = done
        self.action = action
        self.value = value
        self.reward = reward
        self.log_prob = log_prob
        self.obs = obs
        self.info = info


def evaluate_model(checkpoint_path, eval_num_trials=16, num_episodes=10, seed=0, eval_method="tiling", log_wandb=None):
    """
    Evaluate a saved model checkpoint and track per-trial statistics.

    This function:
    1. Loads a trained model from checkpoint
    2. Evaluates it over multiple episodes
    3. Tracks returns, step counts, and success rates for each trial within episodes
    4. Computes statistics (mean ± std) across episodes for each trial
    5. Saves results as plots (PNG) and data (CSV)
    6. Optionally logs results to wandb

    Args:
        checkpoint_path: Path to checkpoint (.pkl file)
        eval_num_trials: Number of trials per episode (default: 16)
        num_episodes: Number of episodes to evaluate (default: 10)
        seed: Random seed for evaluation (default: 0)
        eval_method: Evaluation method - tiling/padding/identity (default: "tiling")
        log_wandb: Wandb project name for logging (default: None, no logging)

    Returns:
        dict: Evaluation results including:
            - trial_returns, trial_return_means, trial_return_stds: Trial return statistics
            - trial_steps, trial_step_means, trial_step_stds: Trial step count statistics
            - trial_successes, trial_success_rates, trial_success_stds: Trial success statistics

    Example:
        results = evaluate_model(
            "checkpoints/cartpole_gru_seed42.pkl",
            eval_num_trials=32,
            num_episodes=10,
            eval_method="tiling",
            log_wandb="popgym_eval"
        )
    """
    # --- Initialize wandb if requested ---
    if log_wandb is not None:
        wandb.init(
            project=log_wandb,
            config={
                "checkpoint_path": checkpoint_path,
                "eval_num_trials": eval_num_trials,
                "num_episodes": num_episodes,
                "seed": seed,
                "eval_method": eval_method,
            }
        )

    # --- Load checkpoint and setup environment ---
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint, metadata = load_checkpoint(checkpoint_path)

    # Extract checkpoint information
    params_dict = checkpoint["params"]
    config = metadata["config"]
    arch = metadata["arch"]
    env_name = metadata["env_name"]
    env_kwargs = metadata["env_kwargs"]
    meta_kwargs = metadata["meta_kwargs"].copy()
    norm_kwargs = metadata["norm_kwargs"]

    # Set num_trials for evaluation
    meta_kwargs["num_trials_per_episode"] = eval_num_trials
    print(f"Using num_trials_per_episode={eval_num_trials} for evaluation")

    # Create evaluation environment
    print(f"Creating evaluation environment: {env_name}")
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    meta_kwargs["meta_rng"] = _rng
    meta_kwargs["meta_eval"] = True

    # Set up eval environment augmentation method
    if eval_method == "padding":
        meta_kwargs["meta_const_aug"] = "padding"
    elif eval_method == "tiling":
        meta_kwargs["meta_const_aug"] = "tiling"
    elif eval_method == "identity":
        meta_kwargs["meta_const_aug"] = "identity"
    print(f"Using eval_method={eval_method}")

    env = create_meta_environment(env_name, env_kwargs, meta_kwargs, norm_kwargs)
    env = AliasPrevActionV2(env)
    env = LogWrapper(env)
    env_params = env.default_params

    # Update config with environment
    config["ENV"] = env
    config["ENV_PARAMS"] = env_params
    config["CONTINUOUS"] = isinstance(env.action_space(env_params), spaces.Box)

    # --- Initialize network (GRU or S5) ---
    print(f"Initializing {arch} network")
    if arch == "gru":
        if config["CONTINUOUS"]:
            network = ActorCriticRNN(env.action_space(env_params).shape[0], config=config)
        else:
            network = ActorCriticRNN(env.action_space(env_params).n, config=config)

        # Initialize network params structure
        rng, _rng = jax.random.split(rng)
        init_x = (jnp.zeros((1, 1, *env.observation_space(env_params).shape)), jnp.zeros((1, 1)))
        init_hstate = ScannedRNN.initialize_carry(1, 256)
        network_params = network.init(_rng, init_hstate, init_x)

    elif arch == "s5":
        d_model = config["S5_D_MODEL"]
        ssm_size = config["S5_SSM_SIZE"]
        n_layers = config["S5_N_LAYERS"]
        blocks = config["S5_BLOCKS"]

        Lambda, _, _, V, _ = make_DPLR_HiPPO(ssm_size)
        block_size = (ssm_size // blocks) // 2
        ssm_size = ssm_size // 2
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vinv = V.conj().T

        ssm_init_fn = init_S5SSM(
            H=d_model,
            P=ssm_size,
            Lambda_re_init=Lambda.real,
            Lambda_im_init=Lambda.imag,
            V=V,
            Vinv=Vinv,
            C_init="lecun_normal",
            discretization="zoh",
            dt_min=0.001,
            dt_max=0.1,
            conj_sym=True,
            clip_eigs=False,
            bidirectional=False
        )

        if config["CONTINUOUS"]:
            network = ActorCriticS5(env.action_space(env_params).shape[0], config=config, ssm_init_fn=ssm_init_fn)
        else:
            network = ActorCriticS5(env.action_space(env_params).n, config=config, ssm_init_fn=ssm_init_fn)

        # Initialize network params structure
        rng, _rng = jax.random.split(rng)
        obs_shape = env.observation_space(env_params).shape
        init_x = (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1)))
        init_hstate = StackedEncoderModel.initialize_carry(1, ssm_size, n_layers)
        network_params = network.init(_rng, init_hstate, init_x)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # --- Load saved model parameters ---
    print("Loading saved parameters")
    # params_dict has structure {'params': {...}}, so we need to use it directly
    network_params = freeze(params_dict)

    # --- Run evaluation episodes ---
    print(f"\nRunning {num_episodes} evaluation episodes...")
    episode_returns = []
    episode_lengths = []

    def evaluate_episode(rng, env_params):
        """
        Evaluate a single episode and return per-trial returns.

        Runs one episode and tracks rewards for each trial separately.
        Returns total episode return, length, and per-trial returns array.
        """
        # Reset environment (with batch size 1)
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, 1)

        def partial_reset(x):
            return env.reset(x, env_params)

        obs, env_state = jax.vmap(partial_reset)(reset_rngs)

        if arch == "gru":
            hstate = ScannedRNN.initialize_carry(1, 256)
        else:
            hstate = StackedEncoderModel.initialize_carry(1, ssm_size, n_layers)

        done = jnp.zeros((1,), dtype=bool)
        episode_return = jnp.zeros((1,), dtype=jnp.float32)
        episode_length = jnp.zeros((1,), dtype=jnp.int32)

        # Track rewards and trial numbers for each step
        num_trials = meta_kwargs.get('num_trials_per_episode', 16)
        max_steps = num_trials * 200  # Assume max 200 steps per trial

        def step_fn(carry, _):
            """Single environment step with trial tracking."""
            rng, obs, env_state, hstate, done, episode_return, episode_length = carry

            # Select action
            ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
            hstate, pi, _ = network.apply(network_params, hstate, ac_in)

            rng, action_rng = jax.random.split(rng)
            action = pi.sample(seed=action_rng)
            action = action.squeeze(0)

            # Step environment (only if not already done)
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, 1)
            obs_new, env_state_new, reward, done_new, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_rngs, env_state, action, env_params
            )

            # Get current trial number from environment state
            current_trial_num = env_state_new.env_state.trial_num[0]

            # Update return and length only if not done before this step
            episode_return = episode_return + reward * (1 - done.astype(jnp.float32))
            episode_length = episode_length + (1 - done.astype(jnp.int32))

            # Keep environment state and observations the same if already done
            obs_new = jnp.where(done[:, None], obs, obs_new)
            done_new = jnp.logical_or(done, done_new)

            carry = (rng, obs_new, env_state_new, hstate, done_new, episode_return, episode_length)
            # Return reward and trial_num for post-processing
            # Note: current_trial_num is AFTER auto-increment, so if env_done=True,
            # trial_num has already been incremented. We need to use the trial_num
            # BEFORE the step to correctly attribute the termination reward.
            trial_num_before_step = env_state.env_state.trial_num[0]
            return carry, (reward[0], trial_num_before_step, done[0])

        # Run for maximum steps (this should be enough for the episode to finish)
        carry = (rng, obs, env_state, hstate, done, episode_return, episode_length)
        (rng, obs, env_state, hstate, done, episode_return, episode_length), step_data = jax.lax.scan(
            step_fn, carry, None, max_steps
        )

        # --- Calculate per-trial statistics from step data ---
        rewards, trial_nums, dones = step_data

        # Compute trial returns using vectorized operations
        # For each trial, sum rewards collected during that trial
        def compute_trial_return(trial_idx):
            # Sum rewards where trial_num equals trial_idx AND not done before this step
            # We use ~dones to exclude steps after the episode has ended
            # Note: The termination reward (-1.0) is included because it occurs when done[0]=False
            # (the step that CAUSES termination). Steps after done[0]=True are excluded.
            mask = (trial_nums == trial_idx) & (~dones)
            return jnp.sum(rewards * mask)

        # Compute trial step counts
        def compute_trial_steps(trial_idx):
            # Count steps where trial_num equals trial_idx AND not done before this step
            # We use ~dones to exclude steps after the episode has ended
            # (lax.scan continues for max_steps even after episode ends)
            mask = (trial_nums == trial_idx) & (~dones)
            return jnp.sum(mask)

        # Compute trial success (1.0 if trial completed successfully, 0.0 otherwise)
        def compute_trial_success(trial_idx):
            # A trial is successful if its return is close to 1.0 (max possible)
            # This happens when the agent survives max_steps_in_episode
            trial_return = compute_trial_return(trial_idx)
            # Success threshold: return >= 0.99 (accounting for numerical precision)
            return jnp.where(trial_return >= 0.99, 1.0, 0.0)

        trial_returns = jax.vmap(compute_trial_return)(jnp.arange(num_trials))
        trial_steps = jax.vmap(compute_trial_steps)(jnp.arange(num_trials))
        trial_successes = jax.vmap(compute_trial_success)(jnp.arange(num_trials))

        return episode_return[0], episode_length[0], trial_returns, trial_steps, trial_successes

    all_trial_returns = []  # Store trial returns for all episodes
    all_trial_steps = []  # Store trial step counts for all episodes
    all_trial_successes = []  # Store trial success flags for all episodes

    # Evaluate over multiple episodes
    for episode in range(num_episodes):
        rng, _rng = jax.random.split(rng)
        episode_return, episode_length, trial_returns, trial_steps, trial_successes = evaluate_episode(_rng, env_params)

        episode_returns.append(float(episode_return))
        episode_lengths.append(int(episode_length))
        all_trial_returns.append(np.array(trial_returns))
        all_trial_steps.append(np.array(trial_steps))
        all_trial_successes.append(np.array(trial_successes))

        print(f"Episode {episode+1}: Return = {episode_return:.2f}, Length = {episode_length}")

    # --- Compute statistics across episodes ---
    all_trial_returns = np.array(all_trial_returns)  # Shape: (num_episodes, num_trials)
    all_trial_steps = np.array(all_trial_steps)  # Shape: (num_episodes, num_trials)
    all_trial_successes = np.array(all_trial_successes)  # Shape: (num_episodes, num_trials)
    num_trials = all_trial_returns.shape[1]

    # Calculate mean and std across episodes for each trial
    trial_return_means = np.mean(all_trial_returns, axis=0)
    trial_return_stds = np.std(all_trial_returns, axis=0)

    trial_step_means = np.mean(all_trial_steps, axis=0)
    trial_step_stds = np.std(all_trial_steps, axis=0)

    # Success rate: mean of success flags (0 or 1) gives success rate
    trial_success_rates = np.mean(all_trial_successes, axis=0)
    trial_success_stds = np.std(all_trial_successes, axis=0)

    # Print summary statistics
    print("\n" + "="*70)
    print("Evaluation Summary:")
    print(f"Mean Episode Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print("="*70)

    # Print per-trial statistics
    print("\nPer-Trial Statistics (averaged across episodes):")
    print("Trial | Mean Return | Std Return | Mean Steps | Std Steps | Success Rate")
    print("-" * 78)
    for trial_idx in range(num_trials):
        print(f"{trial_idx:5d} | {trial_return_means[trial_idx]:11.2f} | "
              f"{trial_return_stds[trial_idx]:10.2f} | "
              f"{trial_step_means[trial_idx]:10.1f} | "
              f"{trial_step_stds[trial_idx]:9.1f} | "
              f"{trial_success_rates[trial_idx]:12.2%}")

    # --- Log to wandb if requested ---
    if log_wandb is not None:
        # Log overall statistics
        wandb.log({
            "overall/mean_return": np.mean(episode_returns),
            "overall/std_return": np.std(episode_returns),
            "overall/mean_length": np.mean(episode_lengths),
            "overall/std_length": np.std(episode_lengths),
        })

        # Log per-trial statistics with trial number as x-axis
        for trial_idx in range(num_trials):
            wandb.log({
                "trial/mean_return": trial_return_means[trial_idx],
                "trial/std_return": trial_return_stds[trial_idx],
                "trial/mean_steps": trial_step_means[trial_idx],
                "trial/std_steps": trial_step_stds[trial_idx],
                "trial/success_rate": trial_success_rates[trial_idx],
                "trial/success_std": trial_success_stds[trial_idx],
                "trial/trial_number": trial_idx,
            })

    # --- Generate and save visualization ---
    trial_numbers = np.arange(num_trials)

    # Create a figure with 3 subplots (returns, steps, success rate)
    _, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Plot 1: Returns
    axes[0].plot(trial_numbers, trial_return_means, 'b-', linewidth=2, label='Mean Return')
    axes[0].fill_between(trial_numbers,
                         trial_return_means - trial_return_stds,
                         trial_return_means + trial_return_stds,
                         alpha=0.3,
                         label='±1 Std')
    axes[0].set_xlabel('Trial Number', fontsize=12)
    axes[0].set_ylabel('Return', fontsize=12)
    axes[0].set_title(f'Return vs Trial Number (N={num_episodes} episodes)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Plot 2: Steps
    axes[1].plot(trial_numbers, trial_step_means, 'g-', linewidth=2, label='Mean Steps')
    axes[1].fill_between(trial_numbers,
                         trial_step_means - trial_step_stds,
                         trial_step_means + trial_step_stds,
                         alpha=0.3,
                         label='±1 Std')
    axes[1].set_xlabel('Trial Number', fontsize=12)
    axes[1].set_ylabel('Steps', fontsize=12)
    axes[1].set_title(f'Steps vs Trial Number (N={num_episodes} episodes)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    # Plot 3: Success Rate
    axes[2].plot(trial_numbers, trial_success_rates, 'r-', linewidth=2, label='Success Rate')
    axes[2].fill_between(trial_numbers,
                         trial_success_rates - trial_success_stds,
                         trial_success_rates + trial_success_stds,
                         alpha=0.3,
                         label='±1 Std')
    axes[2].set_xlabel('Trial Number', fontsize=12)
    axes[2].set_ylabel('Success Rate', fontsize=12)
    axes[2].set_title(f'Success Rate vs Trial Number (N={num_episodes} episodes)', fontsize=14)
    axes[2].set_ylim([-0.05, 1.05])  # Set y-axis range for success rate
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)

    plt.tight_layout()

    # Save plot as PNG
    plot_filename = checkpoint_path.replace('.pkl', f'_trial_stats_n{num_trials}.png')
    plt.savefig(plot_filename, dpi=150)
    print(f"\nPlot saved to: {plot_filename}")
    plt.close()

    # --- Save data to CSV ---
    csv_filename = checkpoint_path.replace('.pkl', f'_trial_stats_n{num_trials}.csv')
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write("Trial,Mean_Return,Std_Return,Mean_Steps,Std_Steps,Success_Rate,Success_Std\n")
        for trial_idx in range(num_trials):
            f.write(f"{trial_idx},"
                   f"{trial_return_means[trial_idx]:.6f},"
                   f"{trial_return_stds[trial_idx]:.6f},"
                   f"{trial_step_means[trial_idx]:.6f},"
                   f"{trial_step_stds[trial_idx]:.6f},"
                   f"{trial_success_rates[trial_idx]:.6f},"
                   f"{trial_success_stds[trial_idx]:.6f}\n")
    print(f"Trial statistics data saved to: {csv_filename}")

    return {
        "returns": episode_returns,
        "lengths": episode_lengths,
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "trial_returns": all_trial_returns,
        "trial_return_means": trial_return_means,
        "trial_return_stds": trial_return_stds,
        "trial_steps": all_trial_steps,
        "trial_step_means": trial_step_means,
        "trial_step_stds": trial_step_stds,
        "trial_successes": all_trial_successes,
        "trial_success_rates": trial_success_rates,
        "trial_success_stds": trial_success_stds,
    }


if __name__ == "__main__":
    # --- Command-line interface ---
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file (e.g., checkpoints/cartpole_gru_seed42.pkl)")
    parser.add_argument("--eval_num_trials", type=int, default=16,
                        help="Number of trials per episode for evaluation (default: %(default)s)")
    parser.add_argument("--eval_method", type=str, default="tiling",
                        help="Evaluation method: tiling / padding / identity (default: %(default)s)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes to evaluate (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for evaluation (default: %(default)s)")
    parser.add_argument("--log_wandb", type=str, default="popgym_eval",
                        help="Wandb project name for logging (default: %(default)s). Set to empty string to disable.")

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        eval_num_trials=args.eval_num_trials,
        num_episodes=args.num_episodes,
        seed=args.seed,
        eval_method=args.eval_method,
        log_wandb=args.log_wandb if args.log_wandb else None
    )
