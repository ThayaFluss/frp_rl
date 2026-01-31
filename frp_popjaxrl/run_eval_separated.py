"""
Evaluation script for SEPARATED mode checkpoints.

This script evaluates trained models from separated mode where FRP state is managed
externally from MetaEnvironment. Key features:
- Independent eval_seed for deterministic evaluation
- External FRP manager reconstruction
- Manual observation transformation

For legacy/lazy mode checkpoints, use run_eval.py instead.

Example usage:
    python run_eval_separated.py --checkpoint=checkpoints/cartpole_gru_seed42.pkl \
                                 --eval_num_trials=32 \
                                 --eval_method=tiling \
                                 --num_episodes=10 \
                                 --log_wandb=popgym_eval_separated

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
from algorithms.ppo_common import create_network
from algorithms.models import GRURepModel, S5RepModel
from algorithms.s5 import StackedEncoderModel
from envs.meta_environment_separated import create_meta_environment
from envs.wrappers import AliasPrevActionV2, LogWrapper
from frp.frp_manager import FRPManager, create_eval_frp_manager


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


def reconstruct_frp_manager(metadata, env, env_params):
    """
    Reconstruct FRP manager from checkpoint metadata.

    Args:
        metadata: Checkpoint metadata dictionary
        env: Wrapped environment (LogWrapper(AliasPrevActionV2(...)))
        env_params: Environment parameters

    Returns:
        FRPManager instance reconstructed from checkpoint config
    """
    meta_kwargs = metadata["meta_kwargs"]

    # Get base environment observation dimension by unwrapping
    # Structure: LogWrapper -> AliasPrevActionV2 -> MetaEnvironment -> base_env
    base_env = env
    unwrapped_params = env_params

    # Unwrap LogWrapper
    if hasattr(base_env, 'env'):
        base_env = base_env.env
        if hasattr(unwrapped_params, 'env_params'):
            unwrapped_params = unwrapped_params.env_params

    # Unwrap AliasPrevActionV2
    if hasattr(base_env, 'env'):
        base_env = base_env.env
        if hasattr(unwrapped_params, 'env_params'):
            unwrapped_params = unwrapped_params.env_params

    # Unwrap MetaEnvironment
    if hasattr(base_env, 'env'):
        base_env = base_env.env
        if hasattr(unwrapped_params, 'env_params'):
            unwrapped_params = unwrapped_params.env_params

    base_env_obs_dim = base_env.observation_space(unwrapped_params).shape[0]

    # Get action space dimension for wrapper
    action_space = env.action_space(env_params)
    if isinstance(action_space, spaces.Box):
        action_dim = action_space.shape[0]
    else:
        action_dim = action_space.n

    # Create FRP manager with same configuration as training
    manager = FRPManager(
        meta_depth=meta_kwargs["meta_depth"],
        meta_dim=meta_kwargs["meta_dim"],
        input_dim=base_env_obs_dim,
        meta_max_depth=meta_kwargs["meta_max_depth"],
        meta_with_adjoint=meta_kwargs["meta_with_adjoint"],
        meta_truncate_aug=meta_kwargs.get("meta_truncate_aug", 0),
        include_metadata=meta_kwargs.get("frp_include_metadata", False),
        include_wrapper=meta_kwargs.get("frp_include_wrapper", False),
        metadata_dim=3,
        wrapper_dim=action_dim
    )

    return manager


def evaluate_model(checkpoint_path, eval_num_trials=32, num_episodes=10, seed=None, eval_method="tiling", log_wandb=None):
    """
    Evaluate a saved model checkpoint (separated mode) and track per-trial statistics.

    This function:
    1. Loads a trained model from checkpoint
    2. Reconstructs FRP manager from checkpoint metadata
    3. Evaluates it over multiple episodes with deterministic eval_seed
    4. Applies FRP transformations manually after each env.step()
    5. Tracks returns, step counts, and success rates for each trial within episodes
    6. Computes statistics (mean ± std) across episodes for each trial
    7. Saves results as plots (PNG) and data (CSV)
    8. Optionally logs results to wandb

    Args:
        checkpoint_path: Path to checkpoint (.pkl file)
        eval_num_trials: Number of trials per episode (default: 32)
        num_episodes: Number of episodes to evaluate (default: 10)
        seed: Random seed for evaluation (default: None, uses checkpoint's eval_seed)
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
            log_wandb="popgym_eval_separated"
        )
    """
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

    # Use checkpoint's eval_seed if not specified
    if seed is None:
        seed = config.get("EVAL_SEED", config.get("SEED", 0) + 10000)
    print(f"Using evaluation seed: {seed}")

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

    # Reconstruct FRP manager from checkpoint
    print("Reconstructing FRP manager from checkpoint")
    frp_manager = reconstruct_frp_manager(metadata, env, env_params)
    print(f"FRP Manager: input_dim={frp_manager.input_dim}, output_dim={frp_manager.aug_output_dim}, "
          f"include_metadata={frp_manager.include_metadata}, include_wrapper={frp_manager.include_wrapper}")

    # Create evaluation FRP manager based on eval_method
    eval_meta_kwargs = meta_kwargs.copy()
    eval_config = {
        "ENV": env,
        "EVAL_ENV": env,  # Same env for evaluation
        "META_KWARGS": meta_kwargs,
        "EVAL_META_KWARGS": eval_meta_kwargs
    }
    eval_frp_manager = create_eval_frp_manager(eval_config)
    if eval_frp_manager is not None:
        print(f"Using eval FRP manager with method: {eval_method}")
    else:
        print("No eval FRP transformation (using training FRP)")

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

    # --- Initialize network (GRU or S5) ---
    print(f"Initializing {arch} network")
    network = create_network(arch, env.action_space(env_params), config)

    # Initialize network params structure
    rng, _rng = jax.random.split(rng)
    obs_shape = env.observation_space(env_params).shape
    init_x = (jnp.zeros((1, 1, *obs_shape)), jnp.zeros((1, 1)))
    init_hstate = network.initialize_core_hidden_state(1)
    network_params = network.init(_rng, init_hstate, init_x)

    # --- Load saved model parameters ---
    print("Loading saved parameters")
    network_params = freeze(params_dict)

    # Define FRP transformation function
    def apply_frp_transform(obs):
        """Apply FRP transformation to a single observation."""
        input_dim = frp_manager.input_dim
        metadata_dim = frp_manager.metadata_dim
        wrapper_dim = frp_manager.wrapper_dim

        # Split observation into components
        raw_obs = obs[:input_dim]
        metadata = obs[input_dim:input_dim + metadata_dim]
        wrapper = obs[input_dim + metadata_dim:]

        # Build FRP input based on configuration
        frp_input_parts = [raw_obs]
        if frp_manager.include_metadata:
            frp_input_parts.append(metadata)
        if frp_manager.include_wrapper:
            frp_input_parts.append(wrapper)

        frp_input = jnp.concatenate(frp_input_parts)

        # Apply transformation
        if eval_frp_manager is not None:
            transformed_obs = eval_frp_manager.transform_obs(frp_input)
        else:
            transformed_obs = frp_input

        # Reconstruct observation with remaining parts
        remaining_parts = []
        if not frp_manager.include_metadata:
            remaining_parts.append(metadata)
        if not frp_manager.include_wrapper:
            remaining_parts.append(wrapper)

        if remaining_parts:
            return jnp.concatenate([transformed_obs] + remaining_parts)
        return transformed_obs

    # --- Run evaluation episodes ---
    print(f"\nRunning {num_episodes} evaluation episodes...")
    episode_returns = []
    episode_lengths = []

    def evaluate_episode(rng, env_params):
        """
        Evaluate a single episode and return per-trial returns.

        Runs one episode and tracks rewards for each trial separately.
        Applies FRP transformations manually after env.step().
        Returns total episode return, length, and per-trial returns array.
        """
        # Reset environment (with batch size 1)
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, 1)

        def partial_reset(x):
            return env.reset(x, env_params)

        obs, env_state = jax.vmap(partial_reset)(reset_rngs)

        # Apply FRP transformation to initial observation
        obs = jax.vmap(apply_frp_transform)(obs)

        # Initialize hidden state using network's method
        hstate = network.initialize_core_hidden_state(1)

        done = jnp.zeros((1,), dtype=bool)
        episode_return = jnp.zeros((1,), dtype=jnp.float32)
        episode_length = jnp.zeros((1,), dtype=jnp.int32)

        # Track rewards and trial numbers for each step
        num_trials = meta_kwargs.get('num_trials_per_episode', 32)
        max_steps = num_trials * 200  # Assume max 200 steps per trial

        def step_fn(carry, _):
            """Single environment step with trial tracking and FRP transformation."""
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

            # Apply FRP transformation to new observation
            obs_new = jax.vmap(apply_frp_transform)(obs_new)

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
            trial_num_before_step = env_state.env_state.trial_num[0]
            return carry, (reward[0], trial_num_before_step, done[0])

        # Run for maximum steps
        carry = (rng, obs, env_state, hstate, done, episode_return, episode_length)
        (rng, obs, env_state, hstate, done, episode_return, episode_length), step_data = jax.lax.scan(
            step_fn, carry, None, max_steps
        )

        # --- Calculate per-trial statistics from step data ---
        rewards, trial_nums, dones = step_data

        # Compute trial returns using vectorized operations
        def compute_trial_return(trial_idx):
            mask = (trial_nums == trial_idx) & (~dones)
            return jnp.sum(rewards * mask)

        # Compute trial step counts
        def compute_trial_steps(trial_idx):
            mask = (trial_nums == trial_idx) & (~dones)
            return jnp.sum(mask)

        # Compute trial success
        def compute_trial_success(trial_idx):
            trial_return = compute_trial_return(trial_idx)
            return jnp.where(trial_return >= 0.99, 1.0, 0.0)

        trial_returns = jax.vmap(compute_trial_return)(jnp.arange(num_trials))
        trial_steps = jax.vmap(compute_trial_steps)(jnp.arange(num_trials))
        trial_successes = jax.vmap(compute_trial_success)(jnp.arange(num_trials))

        return episode_return[0], episode_length[0], trial_returns, trial_steps, trial_successes

    all_trial_returns = []
    all_trial_steps = []
    all_trial_successes = []

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
    all_trial_returns = np.array(all_trial_returns)
    all_trial_steps = np.array(all_trial_steps)
    all_trial_successes = np.array(all_trial_successes)
    num_trials = all_trial_returns.shape[1]

    trial_return_means = np.mean(all_trial_returns, axis=0)
    trial_return_stds = np.std(all_trial_returns, axis=0)

    trial_step_means = np.mean(all_trial_steps, axis=0)
    trial_step_stds = np.std(all_trial_steps, axis=0)

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
        wandb.log({
            "overall/mean_return": np.mean(episode_returns),
            "overall/std_return": np.std(episode_returns),
            "overall/mean_length": np.mean(episode_lengths),
            "overall/std_length": np.std(episode_lengths),
        })

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
    axes[2].set_ylim([-0.05, 1.05])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)

    plt.tight_layout()

    # Save plot as PNG
    plot_filename = checkpoint_path.replace('.pkl', f'_trial_stats_n{num_trials}_separated.png')
    plt.savefig(plot_filename, dpi=150)
    print(f"\nPlot saved to: {plot_filename}")
    plt.close()

    # --- Save data to CSV ---
    csv_filename = checkpoint_path.replace('.pkl', f'_trial_stats_n{num_trials}_separated.csv')
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
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model checkpoint (separated mode)",
        epilog="Note: For legacy/lazy mode checkpoints, use run_eval.py instead."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file (e.g., checkpoints/cartpole_gru_seed42.pkl)")
    parser.add_argument("--eval_num_trials", type=int, default=32,
                        help="Number of trials per episode for evaluation (default: %(default)s)")
    parser.add_argument("--eval_method", type=str, default="tiling",
                        help="Evaluation method: tiling / padding / identity (default: %(default)s)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes to evaluate (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for evaluation (default: None, uses checkpoint's eval_seed)")
    parser.add_argument("--log_wandb", type=str, default="popgym_eval_separated",
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
