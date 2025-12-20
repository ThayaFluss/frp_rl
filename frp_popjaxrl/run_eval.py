"""
Evaluation script for saved model checkpoints.

This script evaluates trained models and tracks per-trial returns to analyze
in-context learning performance over multiple trials.

Example usage:
    python run_eval.py --checkpoint=checkpoints/cartpole_gru_seed42.pkl \
                       --num_trials_eval=32 \
                       --num_episodes=10

Outputs:
    - Console: Per-episode and per-trial statistics
    - PNG: Plot of mean return ± std across trials
    - CSV: Trial-wise return data
"""

import argparse
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax.core import freeze
from gymnax.environments import spaces

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


def evaluate_model(checkpoint_path, num_trials_eval=None, num_episodes=10, seed=0):
    """
    Evaluate a saved model checkpoint and track per-trial returns.

    This function:
    1. Loads a trained model from checkpoint
    2. Evaluates it over multiple episodes
    3. Tracks returns for each trial within episodes
    4. Computes statistics (mean ± std) across episodes
    5. Saves results as plot (PNG) and data (CSV)

    Args:
        checkpoint_path: Path to checkpoint (.pkl file)
        num_trials_eval: Number of trials per episode (default: use checkpoint value)
        num_episodes: Number of episodes to evaluate (default: 10)
        seed: Random seed for evaluation (default: 0)

    Returns:
        dict: Evaluation results including trial_returns, trial_means, trial_stds

    Example:
        results = evaluate_model(
            "checkpoints/cartpole_gru_seed42.pkl",
            num_trials_eval=32,
            num_episodes=10
        )
    """
    # --- Load checkpoint and setup environment ---
    print(f"Loading checkpoint from {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    # Extract checkpoint information
    params_dict = checkpoint["params"]
    config = checkpoint["config"]
    arch = checkpoint["arch"]
    env_name = checkpoint["env_name"]
    env_kwargs = checkpoint["env_kwargs"]
    meta_kwargs = checkpoint["meta_kwargs"].copy()
    norm_kwargs = checkpoint["norm_kwargs"]

    # Override num_trials if specified
    if num_trials_eval is not None:
        meta_kwargs["num_trials_per_episode"] = num_trials_eval
        print(f"Using num_trials_per_episode={num_trials_eval} for evaluation")
    else:
        print(f"Using num_trials_per_episode={meta_kwargs.get('num_trials_per_episode', 16)} from checkpoint")

    # Create evaluation environment
    print(f"Creating evaluation environment: {env_name}")
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    meta_kwargs["meta_rng"] = _rng
    meta_kwargs["meta_eval"] = True

    # Use tiling augmentation for evaluation by default
    meta_kwargs["meta_const_aug"] = "tiling"

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
        max_steps = num_trials * 500  # Assume max 500 steps per trial

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
            return carry, (reward[0], current_trial_num, done[0])

        # Run for maximum steps (this should be enough for the episode to finish)
        carry = (rng, obs, env_state, hstate, done, episode_return, episode_length)
        (rng, obs, env_state, hstate, done, episode_return, episode_length), step_data = jax.lax.scan(
            step_fn, carry, None, max_steps
        )

        # --- Calculate per-trial returns from step data ---
        rewards, trial_nums, dones = step_data

        # Compute trial returns using vectorized operations
        # For each trial, sum rewards collected during that trial
        def compute_trial_return(trial_idx):
            # Sum rewards where trial_num equals trial_idx and not done before
            mask = (trial_nums == trial_idx) & (~dones)
            return jnp.sum(rewards * mask)

        trial_returns = jax.vmap(compute_trial_return)(jnp.arange(num_trials))

        return episode_return[0], episode_length[0], trial_returns

    all_trial_returns = []  # Store trial returns for all episodes

    # Evaluate over multiple episodes
    for episode in range(num_episodes):
        rng, _rng = jax.random.split(rng)
        episode_return, episode_length, trial_returns = evaluate_episode(_rng, env_params)

        episode_returns.append(float(episode_return))
        episode_lengths.append(int(episode_length))
        all_trial_returns.append(np.array(trial_returns))

        print(f"Episode {episode+1}: Return = {episode_return:.2f}, Length = {episode_length}")

    # --- Compute statistics across episodes ---
    all_trial_returns = np.array(all_trial_returns)  # Shape: (num_episodes, num_trials)
    num_trials = all_trial_returns.shape[1]

    # Calculate mean and std across episodes for each trial
    trial_means = np.mean(all_trial_returns, axis=0)
    trial_stds = np.std(all_trial_returns, axis=0)

    # Print summary statistics
    print("\n" + "="*50)
    print("Evaluation Summary:")
    print(f"Mean Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print("="*50)

    # Print per-trial statistics
    print("\nPer-Trial Statistics:")
    print("Trial | Mean Return | Std Return")
    print("-" * 40)
    for trial_idx in range(num_trials):
        print(f"{trial_idx:5d} | {trial_means[trial_idx]:11.2f} | {trial_stds[trial_idx]:10.2f}")

    # --- Generate and save visualization ---
    # Plot: Trial number (x-axis) vs Return (y-axis) with mean ± std
    plt.figure(figsize=(10, 6))
    trial_numbers = np.arange(num_trials)

    plt.plot(trial_numbers, trial_means, 'b-', linewidth=2, label='Mean Return')
    plt.fill_between(trial_numbers,
                     trial_means - trial_stds,
                     trial_means + trial_stds,
                     alpha=0.3,
                     label='±1 Std')

    plt.xlabel('Trial Number', fontsize=12)
    plt.ylabel('Return', fontsize=12)
    plt.title(f'Return vs Trial Number (N={num_episodes} episodes)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save plot as PNG
    plot_filename = checkpoint_path.replace('.pkl', f'_trial_returns_n{num_trials}.png')
    plt.savefig(plot_filename, dpi=150)
    print(f"\nPlot saved to: {plot_filename}")
    plt.close()

    # --- Save data to CSV ---
    csv_filename = checkpoint_path.replace('.pkl', f'_trial_returns_n{num_trials}.csv')
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write("Trial,Mean_Return,Std_Return\n")
        for trial_idx in range(num_trials):
            f.write(f"{trial_idx},{trial_means[trial_idx]:.6f},{trial_stds[trial_idx]:.6f}\n")
    print(f"Trial return data saved to: {csv_filename}")

    return {
        "returns": episode_returns,
        "lengths": episode_lengths,
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "trial_returns": all_trial_returns,
        "trial_means": trial_means,
        "trial_stds": trial_stds,
    }


if __name__ == "__main__":
    # --- Command-line interface ---
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file (e.g., checkpoints/cartpole_gru_seed42.pkl)")
    parser.add_argument("--num_trials_eval", type=int, default=None,
                        help="Number of trials per episode for evaluation (default: use checkpoint value)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes to evaluate (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for evaluation (default: %(default)s)")

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        num_trials_eval=args.num_trials_eval,
        num_episodes=args.num_episodes,
        seed=args.seed
    )
