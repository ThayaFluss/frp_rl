"""
Training script for SEPARATED mode ONLY.

IMPORTANT: This file is exclusively for SEPARATED mode.
- FRP state management is externalized from MetaEnvironment
- env_index is managed by FRPManager, not stored in environment state
- Training and evaluation use INDEPENDENT RNG seeds
- For legacy/lazy modes, use run_meta_popgym.py

Key differences from legacy/lazy:
- --eval_seed: Separate RNG seed for evaluation (default: seed + 10000)
- Evaluation FRP words are created independently from training FRP words
- No RNG dependencies between training and evaluation paths
"""
import jax
import jax.numpy as jnp
import jax.profiler
import time
import logging
from envs.wrappers import AliasPrevActionV2
from utils.checkpoint import create_experiment_directory, save_config_yaml, save_checkpoint, save_run_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# SEPARATED mode imports
from envs.meta_environment_separated import create_meta_environment


def get_make_train(arch: str):
    """Get the appropriate make_train function based on architecture.

    GRU/S5/AGaLiTe use the standard FRP PPO module (ppo_frp_separated.py).

    Args:
        arch: Architecture name ('gru', 's5', or 'agalite')

    Returns:
        make_train function for the specified architecture
    """
    from algorithms.ppo_frp_separated import make_train
    return make_train


def run(args, num_runs, env_name, arch="gru", file_tag="", env_kwargs={}, meta_kwargs={}, norm_kwargs={}, wandb_run_id=None):
    """
    Run training with SEPARATED mode implementation.

    Key features:
    - Independent RNG for training and evaluation
    - FRP state managed externally from environment
    """
    # Configure JAX profiling options
    jax_enable_compile_log = (args.jax_profile >= 1)
    jax_enable_profiler = (args.jax_profile >= 2)

    jax.config.update("jax_log_compiles", jax_enable_compile_log)

    if args.jax_profile == 0:
        logger.info("JAX profiling: DISABLED")
    elif args.jax_profile == 1:
        logger.info("JAX profiling: COMPILE_LOG only")
    else:  # >= 2
        logger.info("JAX profiling: COMPILE_LOG + PROFILER")

    logger.info("=" * 50)
    logger.info("Running in mode: SEPARATED")
    logger.info("=" * 50)

    # Setup training RNG
    train_seed = args.seed
    train_rng = jax.random.PRNGKey(train_seed)
    train_rng, _train_rng = jax.random.split(train_rng)
    meta_kwargs["meta_rng"] = _train_rng

    logger.info(f"Training seed: {train_seed}")

    if args.eval_method == "identity":
        meta_kwargs["meta_truncate_aug"] = 1
    env = create_meta_environment(env_name, env_kwargs, meta_kwargs, norm_kwargs)
    env_params = env.default_params

    # Setup evaluation RNG (completely independent from training)
    eval_seed = args.eval_seed if args.eval_seed is not None else (args.seed + 10000)
    eval_rng = jax.random.PRNGKey(eval_seed)

    logger.info(f"Evaluation seed: {eval_seed} (independent from training)")

    eval_env_kwargs = env_kwargs.copy()
    eval_meta_kwargs = meta_kwargs.copy()
    eval_norm_kwargs = norm_kwargs.copy() if norm_kwargs else None

    # Use independent eval RNG (not derived from training RNG)
    eval_rng, _eval_rng = jax.random.split(eval_rng)
    eval_meta_kwargs["meta_rng"] = _eval_rng
    eval_meta_kwargs["meta_eval"] = True
    eval_meta_kwargs["num_trials_per_episode"] = args.eval_num_trials

    # Set up eval environment augmentation method
    if args.eval_method == "padding":
        eval_meta_kwargs["meta_const_aug"] = "padding"
    elif args.eval_method == "tiling":
        eval_meta_kwargs["meta_const_aug"] = "tiling"
    elif args.eval_method == "identity":
        eval_meta_kwargs["meta_const_aug"] = "identity"

    eval_env = create_meta_environment(env_name, eval_env_kwargs, eval_meta_kwargs, eval_norm_kwargs)
    eval_env_params = eval_env.default_params

    # Build unified training config (no debug branching needed)
    from utils.config import build_training_config
    config = build_training_config(
        args=args,
        env=AliasPrevActionV2(env),
        env_params=env_params,
        eval_env=AliasPrevActionV2(eval_env),
        eval_env_params=eval_env_params,
        meta_kwargs=meta_kwargs,
        eval_meta_kwargs=eval_meta_kwargs,
    )

    rngs = jax.random.split(train_rng, num_runs)
    info_dict = {}

    # Get the appropriate make_train function for this architecture
    make_train = get_make_train(arch)

    if arch == "s5":
        logger.info("Starting S5 compilation...")
        train_vjit_s5 = jax.jit(jax.vmap(make_train(config)))

        # Start JAX profiler for compilation analysis (if enabled)
        if jax_enable_profiler:
            logger.info("Starting JAX profiler...")
            jax.profiler.start_trace("/tmp/jax-trace")

        t0 = time.time()
        compiled_s5 = train_vjit_s5.lower(rngs).compile()
        compile_s5_time = time.time() - t0

        # Stop JAX profiler after compilation (if enabled)
        if jax_enable_profiler:
            jax.profiler.stop_trace()
            logger.info("JAX profiler trace saved to /tmp/jax-trace")

        logger.info(f"S5 compilation completed in {compile_s5_time:.2f}s")

        logger.info("Starting S5 training execution...")
        t0 = time.time()
        out_s5 = jax.block_until_ready(compiled_s5(rngs))
        run_s5_time = time.time() - t0
        logger.info(f"S5 training completed in {run_s5_time:.2f}s")

        # Calculate total time
        total_s5_time = compile_s5_time + run_s5_time

        # Display summary
        logger.info("=" * 50)
        logger.info("S5 Training Summary:")
        logger.info(f"  Compile time:  {compile_s5_time:>8.2f}s")
        logger.info(f"  Training time: {run_s5_time:>8.2f}s")
        logger.info(f"  Total time:    {total_s5_time:>8.2f}s")
        logger.info("=" * 50)

        # Keep arrays as arrays, only convert scalars
        metrics = jax.tree_util.tree_map(
            lambda x: x.item() if (hasattr(x, 'item') and (not hasattr(x, 'shape') or x.shape == ())) else x,
            out_s5[1]
        )

        # Create base info dictionary with common metrics
        info_dict["s5"] = {
            "compile_s5_time": compile_s5_time,
            "run_s5_time": run_s5_time,
            "total_s5_time": total_s5_time,
            "train_mer": metrics["train_mer"],
            "eval_mer": metrics["eval_mer"],
        }

        if "few_shot_metric" in metrics:
            info_dict["s5"]["few_shot_metrics"] = metrics["few_shot_metric"]

        # Log timing metrics to wandb with unified names
        wandb.log({
            "time/compile_time": compile_s5_time,
            "time/run_time": run_s5_time,
            "time/total_time": total_s5_time,
        })

    elif arch == "gru":
        logger.info("Starting GRU compilation...")
        train_vjit_rnn = jax.jit(jax.vmap(make_train(config)))

        # Start JAX profiler for compilation analysis (if enabled)
        if jax_enable_profiler:
            logger.info("Starting JAX profiler...")
            jax.profiler.start_trace("/tmp/jax-trace")

        t0 = time.time()
        compiled_rnn = train_vjit_rnn.lower(rngs).compile()
        compile_rnn_time = time.time() - t0

        # Stop JAX profiler after compilation (if enabled)
        if jax_enable_profiler:
            jax.profiler.stop_trace()
            logger.info("JAX profiler trace saved to /tmp/jax-trace")

        logger.info(f"GRU compilation completed in {compile_rnn_time:.2f}s")

        logger.info("Starting GRU training execution...")
        t0 = time.time()
        out_rnn = jax.block_until_ready(compiled_rnn(rngs))
        run_rnn_time = time.time() - t0
        logger.info(f"GRU training completed in {run_rnn_time:.2f}s")

        # Calculate total time
        total_rnn_time = compile_rnn_time + run_rnn_time

        # Display summary
        logger.info("=" * 50)
        logger.info("GRU Training Summary:")
        logger.info(f"  Compile time:  {compile_rnn_time:>8.2f}s")
        logger.info(f"  Training time: {run_rnn_time:>8.2f}s")
        logger.info(f"  Total time:    {total_rnn_time:>8.2f}s")
        logger.info("=" * 50)

        # Keep arrays as arrays, only convert scalars
        metrics = jax.tree_util.tree_map(
            lambda x: x.item() if (hasattr(x, 'item') and (not hasattr(x, 'shape') or x.shape == ())) else x,
            out_rnn[1]
        )

        # Create base info dictionary with common metrics
        info_dict["gru"] = {
            "compile_rnn_time": compile_rnn_time,
            "run_rnn_time": run_rnn_time,
            "total_rnn_time": total_rnn_time,
            "train_mer": metrics["train_mer"],
            "eval_mer": metrics["eval_mer"],
        }

        if "few_shot_metric" in metrics:
            info_dict["gru"]["few_shot_metrics"] = metrics["few_shot_metric"]

        # Log timing metrics to wandb with unified names
        wandb.log({
            "time/compile_time": compile_rnn_time,
            "time/run_time": run_rnn_time,
            "time/total_time": total_rnn_time,
        })

    elif arch == "agalite":
        logger.info("Starting AGaLiTe compilation...")
        train_vjit_agalite = jax.jit(jax.vmap(make_train(config)))

        # Start JAX profiler for compilation analysis (if enabled)
        if jax_enable_profiler:
            logger.info("Starting JAX profiler...")
            jax.profiler.start_trace("/tmp/jax-trace")

        t0 = time.time()
        compiled_agalite = train_vjit_agalite.lower(rngs).compile()
        compile_agalite_time = time.time() - t0

        # Stop JAX profiler after compilation (if enabled)
        if jax_enable_profiler:
            jax.profiler.stop_trace()
            logger.info("JAX profiler trace saved to /tmp/jax-trace")

        logger.info(f"AGaLiTe compilation completed in {compile_agalite_time:.2f}s")

        logger.info("Starting AGaLiTe training execution...")
        t0 = time.time()
        out_agalite = jax.block_until_ready(compiled_agalite(rngs))
        run_agalite_time = time.time() - t0
        logger.info(f"AGaLiTe training completed in {run_agalite_time:.2f}s")

        # Calculate total time
        total_agalite_time = compile_agalite_time + run_agalite_time

        # Display summary
        logger.info("=" * 50)
        logger.info("AGaLiTe Training Summary:")
        logger.info(f"  Compile time:  {compile_agalite_time:>8.2f}s")
        logger.info(f"  Training time: {run_agalite_time:>8.2f}s")
        logger.info(f"  Total time:    {total_agalite_time:>8.2f}s")
        logger.info("=" * 50)

        # Keep arrays as arrays, only convert scalars
        metrics = jax.tree_util.tree_map(
            lambda x: x.item() if (hasattr(x, 'item') and (not hasattr(x, 'shape') or x.shape == ())) else x,
            out_agalite[1]
        )

        # Create base info dictionary with common metrics
        info_dict["agalite"] = {
            "compile_agalite_time": compile_agalite_time,
            "run_agalite_time": run_agalite_time,
            "total_agalite_time": total_agalite_time,
            "train_mer": metrics["train_mer"],
            "eval_mer": metrics["eval_mer"],
        }

        if "few_shot_metric" in metrics:
            info_dict["agalite"]["few_shot_metrics"] = metrics["few_shot_metric"]

        # Log timing metrics to wandb with unified names
        wandb.log({
            "time/compile_time": compile_agalite_time,
            "time/run_time": run_agalite_time,
            "time/total_time": total_agalite_time,
        })

    else:
        raise NotImplementedError(f"Unknown architecture: {arch}. Valid values are 'gru', 's5', or 'agalite'.")

    if args.save_results == 1:
        jnp.save(f"results/{num_runs}_{env_name}_{arch}_{file_tag}.npy", info_dict)

    # Save model checkpoint if requested
    if args.save_model == 1:
        # Create experiment directory with timestamp
        exp_dir = create_experiment_directory("exp")

        # Extract params from the output
        if arch == "s5":
            runner_state = out_s5[0]
        elif arch == "gru":
            runner_state = out_rnn[0]
        elif arch == "agalite":
            runner_state = out_agalite[0]

        # Get the first run's state (in case of multiple runs)
        train_state = jax.tree_util.tree_map(lambda x: x[0] if len(x.shape) > 0 else x, runner_state[0])

        # Filter meta_kwargs (remove meta_rng)
        filtered_meta_kwargs = {k: v for k, v in meta_kwargs.items() if k != "meta_rng"}

        # Save config.yaml
        save_config_yaml(
            config=config,
            arch=arch,
            env_name=env_name,
            env_kwargs=env_kwargs,
            meta_kwargs=filtered_meta_kwargs,
            norm_kwargs=norm_kwargs,
            seed=args.seed,
            exp_dir=exp_dir
        )

        # Calculate number of updates
        num_updates = int(config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"]))

        # Get the final eval MER (Mean Episodic Return)
        # Handle both array and scalar cases (fallback for safety)
        eval_mer = metrics["eval_mer"]
        try:
            # Try to get the last element if it's an array
            current_eval_mer = float(eval_mer[-1])
        except (TypeError, IndexError):
            # If it's already a scalar, use it directly
            current_eval_mer = float(eval_mer)

        # Save checkpoint
        save_checkpoint(
            params=train_state.params,
            config=config,
            arch=arch,
            env_name=env_name,
            env_kwargs=env_kwargs,
            meta_kwargs=filtered_meta_kwargs,
            norm_kwargs=norm_kwargs,
            eval_metric=current_eval_mer,
            num_updates=num_updates,
            exp_dir=exp_dir
        )

        # Extract metrics for run_info.yaml
        # Handle both array and scalar cases for all metrics
        def extract_final_value(metric_value):
            """Extract final value from metric (handles both arrays and scalars)."""
            try:
                # Try to get the last element if it's an array
                return float(metric_value[-1])
            except (TypeError, IndexError):
                # If it's already a scalar, use it directly
                return float(metric_value)

        train_mer_final = extract_final_value(metrics.get("train_mer", 0.0))
        eval_mer_final = extract_final_value(metrics.get("eval_mer", 0.0))
        train_mmer = extract_final_value(metrics.get("max_train_mer", train_mer_final))
        eval_mmer = extract_final_value(metrics.get("max_eval_mer", eval_mer_final))

        # Save run info (wandb run ID and metrics) to YAML
        save_run_info(
            exp_dir=exp_dir,
            wandb_run_id=wandb_run_id,
            train_metric_final=train_mer_final,
            train_metric_max=train_mmer,
            eval_metric_final=eval_mer_final,
            eval_metric_max=eval_mmer
        )


if __name__ == "__main__":
    import wandb
    from utils.arguments import create_meta_parser
    from utils.config import load_config_with_debug_support

    parser = create_meta_parser()
    args = parser.parse_args()
    args = load_config_with_debug_support(parser, args)

    # Meta environment specific kwargs
    meta_kwargs = {
        "meta_depth": args.depth,
        "meta_max_depth": args.max_depth,
        "meta_dim": args.dim,
        "meta_with_adjoint": (args.with_adjoint == 1),
        "num_trials_per_episode": args.num_trials,
        "frp_include_metadata": (args.frp_include_metadata == 1),
        "frp_include_wrapper": (args.frp_include_wrapper == 1),
    }

    # Environment specific kwargs
    env_kwargs = {}

    # Normalization specific kwargs
    norm_kwargs = {
        "strategy": args.norm_strategy,
        "max_steps": args.norm_max_steps
    }

    wandb.init(project=args.log_wandb, config=args)
    wandb_run_id = wandb.run.id if wandb.run else None
    run(args, args.num_runs, args.env, args.arch, env_kwargs=env_kwargs, meta_kwargs=meta_kwargs, norm_kwargs=norm_kwargs, wandb_run_id=wandb_run_id)
