"""
Training script for standard (non-meta) environments with new models.

This script trains GRU/S5/AGaLiTe models on standard popgym environments
WITHOUT FRP transformations or MetaEnvironment wrappers.

Key differences from run_meta_popgym_separated.py:
- Uses base environments directly (no MetaEnvironment)
- No FRP transformations applied to observations
- Simpler configuration (no meta_kwargs, frp_kwargs)
- Same model architectures as meta-learning implementation

Use this script to:
- Test new model architectures on standard RL tasks
- Establish baselines without FRP/meta-learning
- Debug model implementations in a simpler setting
"""
import jax
import jax.numpy as jnp
import jax.profiler
import time
import logging
from envs import make
from envs.wrappers import AliasPrevActionV2
from utils.checkpoint import create_experiment_directory, save_config_yaml, save_checkpoint, save_run_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_make_train(arch: str):
    """Get the appropriate make_train function based on architecture.

    All architectures (GRU, S5, AGaLiTe) use the unified ppo_standard.py module.
    AGaLiTeRepModel has been added to models.py with the same interface as
    GRURepModel/S5RepModel.

    Args:
        arch: Architecture name ('gru', 's5', or 'agalite')

    Returns:
        make_train function for the specified architecture
    """
    from algorithms.ppo_standard import make_train
    return make_train


def run(args, num_runs, env_name, arch="gru", file_tag="", env_kwargs={}, wandb_run_id=None):
    """
    Run training with standard PPO implementation (no FRP/meta-learning).

    Key features:
    - Uses base environments directly
    - No FRP transformations
    - Same model architectures as meta-learning
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
    logger.info("Running in mode: STANDARD (no FRP/meta-learning)")
    logger.info("=" * 50)

    # Setup training RNG
    train_seed = args.seed
    train_rng = jax.random.PRNGKey(train_seed)

    logger.info(f"Training seed: {train_seed}")

    # Create environment directly (no MetaEnvironment wrapper)
    env, env_params = make(env_name, **env_kwargs)

    # Setup evaluation seed
    eval_seed = args.eval_seed if args.eval_seed is not None else (args.seed + 10000)
    logger.info(f"Evaluation seed: {eval_seed}")

    # For standard training, eval uses the same environment
    eval_env, eval_env_params = make(env_name, **env_kwargs)

    # Build unified training config (no debug branching needed)
    from utils.config import build_training_config
    config = build_training_config(
        args=args,
        env=AliasPrevActionV2(env),
        env_params=env_params,
        eval_env=AliasPrevActionV2(eval_env),
        eval_env_params=eval_env_params,
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
        info_dict[arch] = {
            "compile_agalite_time": compile_agalite_time,
            "run_agalite_time": run_agalite_time,
            "total_agalite_time": total_agalite_time,
            "train_mer": metrics["train_mer"],
            "eval_mer": metrics["eval_mer"],
        }

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

        # Save config.yaml
        save_config_yaml(
            config=config,
            arch=arch,
            env_name=env_name,
            env_kwargs=env_kwargs,
            meta_kwargs={},  # No meta_kwargs for standard training
            norm_kwargs={},
            seed=args.seed,
            exp_dir=exp_dir
        )

        # Calculate number of updates
        num_updates = int(config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"]))

        # Get the final eval MER (Mean Episodic Return)
        eval_mer = metrics["eval_mer"]
        try:
            current_eval_mer = float(eval_mer[-1])
        except (TypeError, IndexError):
            current_eval_mer = float(eval_mer)

        # Save checkpoint
        save_checkpoint(
            params=train_state.params,
            config=config,
            arch=arch,
            env_name=env_name,
            env_kwargs=env_kwargs,
            meta_kwargs={},
            norm_kwargs={},
            eval_metric=current_eval_mer,
            num_updates=num_updates,
            exp_dir=exp_dir
        )

        # Extract metrics for run_info.yaml
        def extract_final_value(metric_value):
            """Extract final value from metric (handles both arrays and scalars)."""
            try:
                return float(metric_value[-1])
            except (TypeError, IndexError):
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
    from utils.arguments import create_standard_parser
    from utils.config import load_config_with_debug_support

    parser = create_standard_parser()
    args = parser.parse_args()
    args = load_config_with_debug_support(parser, args)

    # Environment specific kwargs (empty for standard environments)
    env_kwargs = {}

    wandb.init(project=args.log_wandb, config=args)
    wandb_run_id = wandb.run.id if wandb.run else None
    run(args, args.num_runs, args.env, args.arch, env_kwargs=env_kwargs, wandb_run_id=wandb_run_id)
