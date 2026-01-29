"""
Training script for standard (non-meta) environments with new models.

This script trains GRU/S5/Transformer models on standard popgym environments
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
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import standard PPO (no FRP)
from algorithms.ppo_standard import make_train


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

    if args.debug == 1:
        config = {
            "MODEL_TYPE": arch,  # 'gru', 's5', or 'transformer'
            "LR": 2.5e-4,
            "NUM_ENVS": 2,
            "NUM_STEPS": 16,
            "TOTAL_TIMESTEPS": 1e3,
            "UPDATE_EPOCHS": 2,
            "NUM_MINIBATCHES": 2,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 1.0,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.0,
            "VF_COEF": 1.0,
            "MAX_GRAD_NORM": 0.5,
            "ENV": AliasPrevActionV2(env),
            "ENV_PARAMS": env_params,
            "EVAL_ENV": AliasPrevActionV2(eval_env),
            "EVAL_ENV_PARAMS": eval_env_params,
            "EVAL_SEED": eval_seed,
            "ANNEAL_LR": False,
            "DEBUG": True,
            "S5_D_MODEL": 256,
            "S5_SSM_SIZE": 256,
            "S5_N_LAYERS": 1,
            "S5_BLOCKS": 1,
            "S5_ACTIVATION": "full_glu",
            "S5_DO_NORM": False,
            "S5_PRENORM": False,
            "S5_DO_GTRXL_NORM": False,
            # Transformer config (debug mode)
            "TRANSFORMER_D_MODEL": 64,
            "TRANSFORMER_NUM_HEADS": 2,
            "TRANSFORMER_N_LAYERS": 1,
            "TRANSFORMER_D_FF": 128,
            "TRANSFORMER_MEM_LEN": 16,
            "TRANSFORMER_DROPOUT": 0.0,
            "TRANSFORMER_GATING": True,
        }
    else:
        config = {
            "MODEL_TYPE": arch,  # 'gru', 's5', or 'transformer'
            "LR": args.lr,
            "NUM_ENVS": args.num_envs,
            "NUM_STEPS": args.num_steps,
            "TOTAL_TIMESTEPS": args.total_timesteps,
            "UPDATE_EPOCHS": args.update_epochs,
            "NUM_MINIBATCHES": args.num_minibatches,
            "GAMMA": 0.99,
            "GAE_LAMBDA": args.gae_lambda,
            "CLIP_EPS": 0.2,
            "ENT_COEF": args.ent_coef,
            "VF_COEF": 1.0,
            "MAX_GRAD_NORM": 0.5,
            "ENV": AliasPrevActionV2(env),
            "ENV_PARAMS": env_params,
            "EVAL_ENV": AliasPrevActionV2(eval_env),
            "EVAL_ENV_PARAMS": eval_env_params,
            "EVAL_SEED": eval_seed,
            "ANNEAL_LR": (args.anneal_lr == 1),
            "DEBUG": True,
            "S5_D_MODEL": 256,
            "S5_SSM_SIZE": 256,
            "S5_N_LAYERS": args.s5_n_layers,
            "S5_BLOCKS": 1,
            "S5_ACTIVATION": "full_glu",
            "S5_DO_NORM": (args.s5_do_norm == 1),
            "S5_PRENORM": (args.s5_prenorm == 1),
            "S5_DO_GTRXL_NORM": (args.s5_do_gtrxl_norm == 1),
            # Transformer config
            "TRANSFORMER_D_MODEL": args.transformer_d_model,
            "TRANSFORMER_NUM_HEADS": args.transformer_num_heads,
            "TRANSFORMER_N_LAYERS": args.transformer_n_layers,
            "TRANSFORMER_D_FF": args.transformer_d_model * 4,
            "TRANSFORMER_MEM_LEN": args.transformer_mem_len,
            "TRANSFORMER_DROPOUT": 0.0,
            "TRANSFORMER_GATING": (args.transformer_gating == 1),
        }

    rngs = jax.random.split(train_rng, num_runs)
    info_dict = {}

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

    elif arch == "transformer":
        logger.info("Starting Transformer compilation...")
        train_vjit_tf = jax.jit(jax.vmap(make_train(config)))

        # Start JAX profiler for compilation analysis (if enabled)
        if jax_enable_profiler:
            logger.info("Starting JAX profiler...")
            jax.profiler.start_trace("/tmp/jax-trace")

        t0 = time.time()
        compiled_tf = train_vjit_tf.lower(rngs).compile()
        compile_tf_time = time.time() - t0

        # Stop JAX profiler after compilation (if enabled)
        if jax_enable_profiler:
            jax.profiler.stop_trace()
            logger.info("JAX profiler trace saved to /tmp/jax-trace")

        logger.info(f"Transformer compilation completed in {compile_tf_time:.2f}s")

        logger.info("Starting Transformer training execution...")
        t0 = time.time()
        out_tf = jax.block_until_ready(compiled_tf(rngs))
        run_tf_time = time.time() - t0
        logger.info(f"Transformer training completed in {run_tf_time:.2f}s")

        # Calculate total time
        total_tf_time = compile_tf_time + run_tf_time

        # Display summary
        logger.info("=" * 50)
        logger.info("Transformer Training Summary:")
        logger.info(f"  Compile time:  {compile_tf_time:>8.2f}s")
        logger.info(f"  Training time: {run_tf_time:>8.2f}s")
        logger.info(f"  Total time:    {total_tf_time:>8.2f}s")
        logger.info("=" * 50)

        # Keep arrays as arrays, only convert scalars
        metrics = jax.tree_util.tree_map(
            lambda x: x.item() if (hasattr(x, 'item') and (not hasattr(x, 'shape') or x.shape == ())) else x,
            out_tf[1]
        )

        # Create base info dictionary with common metrics
        info_dict["transformer"] = {
            "compile_tf_time": compile_tf_time,
            "run_tf_time": run_tf_time,
            "total_tf_time": total_tf_time,
            "train_mer": metrics["train_mer"],
            "eval_mer": metrics["eval_mer"],
        }

        # Log timing metrics to wandb with unified names
        wandb.log({
            "time/compile_time": compile_tf_time,
            "time/run_time": run_tf_time,
            "time/total_time": total_tf_time,
        })

    else:
        raise NotImplementedError(f"Unknown architecture: {arch}. Valid values are 'gru', 's5', or 'transformer'.")

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
        elif arch == "transformer":
            runner_state = out_tf[0]

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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Training script for standard environments (no FRP/meta-learning)"
    )
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of training runs (default: %(default)s)")
    parser.add_argument("--env", type=str, default="StatelessCartPoleEasy",
                        help="Environment name (default: %(default)s)")
    parser.add_argument("--arch", type=str, default="gru",
                        help="Architecture: gru, s5, or transformer (default: %(default)s)")
    parser.add_argument("--log_wandb", type=str, default="popgym_standard",
                        help="Wandb project name (default: %(default)s)")
    parser.add_argument("--debug", type=int, default=0,
                        help="Debug mode: 0 or 1 (default: %(default)s)")
    parser.add_argument("--jax_profile", type=int, default=0,
                        help="JAX profiling level: 0=disabled, 1=compile_log, 2=profiler+compile_log (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for training (default: %(default)s)")
    parser.add_argument("--eval_seed", type=int, default=None,
                        help="Random seed for evaluation. If None, uses seed + 10000 (default: %(default)s)")

    ### For saving results and models
    parser.add_argument("--save_results", type=int, default=0,
                        help="Save results npy (default: %(default)s)")
    parser.add_argument("--save_model", type=int, default=0,
                        help="Save model checkpoint (default: %(default)s)")

    ### For PPO hyperparameters (only used when debug=0)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: %(default)s)")
    parser.add_argument("--ent_coef", type=float, default=0.0,
                        help="Entropy coefficient (default: %(default)s)")
    parser.add_argument("--gae_lambda", type=float, default=1.0,
                        help="GAE lambda (default: %(default)s)")
    parser.add_argument("--update_epochs", type=int, default=30,
                        help="Number of update epochs (default: %(default)s)")
    parser.add_argument("--num_envs", type=int, default=64,
                        help="Number of parallel environments (default: %(default)s)")
    parser.add_argument("--num_steps", type=int, default=1024,
                        help="Number of steps per update (default: %(default)s)")
    parser.add_argument("--total_timesteps", type=float, default=15e6,
                        help="Total timesteps (default: %(default)s)")
    parser.add_argument("--num_minibatches", type=int, default=8,
                        help="Number of minibatches (default: %(default)s)")
    parser.add_argument("--anneal_lr", type=int, default=0,
                        help="Anneal learning rate: 0 or 1 (default: %(default)s)")

    ### For S5 architecture hyperparameters (only used when debug=0)
    parser.add_argument("--s5_n_layers", type=int, default=4,
                        help="Number of S5 layers (default: %(default)s)")
    parser.add_argument("--s5_do_norm", type=int, default=0,
                        help="S5 do normalization: 0 or 1 (default: %(default)s)")
    parser.add_argument("--s5_prenorm", type=int, default=0,
                        help="S5 prenormalization: 0 or 1 (default: %(default)s)")
    parser.add_argument("--s5_do_gtrxl_norm", type=int, default=0,
                        help="S5 GTrXL normalization: 0 or 1 (default: %(default)s)")

    ### For Transformer architecture hyperparameters (only used when arch=transformer)
    parser.add_argument("--transformer_d_model", type=int, default=256,
                        help="Transformer model dimension (default: %(default)s)")
    parser.add_argument("--transformer_num_heads", type=int, default=2,
                        help="Transformer number of attention heads (default: %(default)s)")
    parser.add_argument("--transformer_n_layers", type=int, default=3,
                        help="Number of Transformer layers (default: %(default)s)")
    parser.add_argument("--transformer_mem_len", type=int, default=64,
                        help="Transformer memory length (default: %(default)s)")
    parser.add_argument("--transformer_gating", type=int, default=1,
                        help="Use GTrXL gating: 0 or 1 (default: %(default)s)")

    args = parser.parse_args()

    # Environment specific kwargs (empty for standard environments)
    env_kwargs = {}

    wandb.init(project=args.log_wandb, config=args)
    wandb_run_id = wandb.run.id if wandb.run else None
    run(args, args.num_runs, args.env, args.arch, env_kwargs=env_kwargs, wandb_run_id=wandb_run_id)
