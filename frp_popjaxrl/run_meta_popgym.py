import jax
import jax.numpy as jnp
import jax.profiler
import time
import logging
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

# Dispatcher: imports will be selected based on --mode flag

def run(args, num_runs, env_name, arch="gru", file_tag="", env_kwargs={}, meta_kwargs={}, norm_kwargs={}, mode="v1", wandb_run_id=None):
    """
    Run training with specified implementation mode.

    Args:
        mode: "v1" (original) or "lazy" (lazy evaluation)
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

    logger.info("="*50)
    logger.info(f"Running in mode: {mode}")
    logger.info("="*50)

    # Dispatcher: import appropriate modules based on mode
    if mode == "lazy":
        from envs.meta_environment_lazy import create_meta_environment
        from algorithms.ppo_in_context_lazy import make_train
    elif mode == "legacy":
        # Legacy implementation (before FRP state separation)
        from envs.meta_environment_legacy import create_meta_environment
        from algorithms.ppo_in_context_legacy import make_train
    elif mode == "separated":
        # New implementation with FRP state separation (default)
        from envs.meta_environment import create_meta_environment
        from algorithms.ppo_in_context import make_train
    else:
        raise ValueError(f"Unknown mode: {mode}. Valid modes are: 'separated', 'legacy', 'lazy'")

    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = jax.random.split(rng)
    meta_kwargs["meta_rng"] = _rng
    if args.eval_method == "identity":
        meta_kwargs["meta_truncate_aug"] = 1
    env = create_meta_environment(env_name, env_kwargs, meta_kwargs, norm_kwargs)
    env_params = env.default_params

    eval_env_kwargs = env_kwargs.copy()
    eval_meta_kwargs = meta_kwargs.copy()
    eval_norm_kwargs = norm_kwargs.copy() if norm_kwargs else None
    # use indep random vars for eval
    rng, _rng = jax.random.split(rng)
    eval_meta_kwargs["meta_rng"] = _rng
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

    if args.debug==1:
        config = {
        "MODEL_TYPE": arch,  # 'gru' or 's5'
        "LR": 2.5e-4,
        "NUM_ENVS": 2,
        "NUM_STEPS": 16,  # Reduced from 128
        "TOTAL_TIMESTEPS": 1e3,  # Reduced from 1e4
        "UPDATE_EPOCHS": 2,
        "NUM_MINIBATCHES":2,
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
        "RESET_WORDS": (args.reset_words==1),
        }
    else:
        config = {
        "MODEL_TYPE": arch,  # 'gru' or 's5'
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
        "ANNEAL_LR": (args.anneal_lr==1),
        "DEBUG": True,
        "S5_D_MODEL": 256,
        "S5_SSM_SIZE": 256,
        "S5_N_LAYERS": args.s5_n_layers,
        "S5_BLOCKS": 1,
        "S5_ACTIVATION": "full_glu",
        "S5_DO_NORM": (args.s5_do_norm==1),
        "S5_PRENORM": (args.s5_prenorm==1),
        "S5_DO_GTRXL_NORM": (args.s5_do_gtrxl_norm==1),
        "RESET_WORDS": (args.reset_words==1)
        }

    rngs = jax.random.split(rng, num_runs)
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
        logger.info("="*50)
        logger.info("S5 Training Summary:")
        logger.info(f"  Compile time:  {compile_s5_time:>8.2f}s")
        logger.info(f"  Training time: {run_s5_time:>8.2f}s")
        logger.info(f"  Total time:    {total_s5_time:>8.2f}s")
        logger.info("="*50)

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
            "train_metrics": metrics["train_metric"],
            "in_context_metrics": metrics["in_context_metric"],
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
        logger.info("="*50)
        logger.info("GRU Training Summary:")
        logger.info(f"  Compile time:  {compile_rnn_time:>8.2f}s")
        logger.info(f"  Training time: {run_rnn_time:>8.2f}s")
        logger.info(f"  Total time:    {total_rnn_time:>8.2f}s")
        logger.info("="*50)

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
            "train_metrics": metrics["train_metric"],
            "in_context_metrics": metrics["in_context_metric"],
        }

        if "few_shot_metric" in metrics:
            info_dict["gru"]["few_shot_metrics"] = metrics["few_shot_metric"]

        # Log timing metrics to wandb with unified names
        wandb.log({
            "time/compile_time": compile_rnn_time,
            "time/run_time": run_rnn_time,
            "time/total_time": total_rnn_time,
        })
    
    else:
        raise NotImplementedError

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

        # Get the final eval metric (in_context_metric)
        # Handle both array and scalar cases (fallback for safety)
        in_context_metric = metrics["in_context_metric"]
        try:
            # Try to get the last element if it's an array
            current_eval_metric = float(in_context_metric[-1])
        except (TypeError, IndexError):
            # If it's already a scalar, use it directly
            current_eval_metric = float(in_context_metric)

        # Save checkpoint
        save_checkpoint(
            params=train_state.params,
            config=config,
            arch=arch,
            env_name=env_name,
            env_kwargs=env_kwargs,
            meta_kwargs=filtered_meta_kwargs,
            norm_kwargs=norm_kwargs,
            eval_metric=current_eval_metric,
            num_updates=num_updates,
            exp_dir=exp_dir
        )

        # Extract metrics for run_info.yaml
        train_metric_final = float(metrics.get("train_metric", 0.0))
        eval_metric_final = float(metrics.get("in_context_metric", 0.0))
        train_metric_max = float(metrics.get("max_train_metric", train_metric_final))
        eval_metric_max = float(metrics.get("max_eval_metric", eval_metric_final))

        # Save run info (wandb run ID and metrics) to YAML
        save_run_info(
            exp_dir=exp_dir,
            wandb_run_id=wandb_run_id,
            train_metric_final=train_metric_final,
            train_metric_max=train_metric_max,
            eval_metric_final=eval_metric_final,
            eval_metric_max=eval_metric_max
        )


if __name__ == "__main__":
    import wandb
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of training runs (default: %(default)s)")
    parser.add_argument("--env", type=str, default="cartpole",
                        help="Base env XXX of MetaXXX (default: %(default)s)")
    parser.add_argument("--arch", type=str, default="s5",
                        help="Architecture: gru or s5 (default: %(default)s)")
    parser.add_argument("--log_wandb", type=str, default="popgym",
                        help="Wandb project name (default: %(default)s)")
    parser.add_argument("--debug", type=int, default=0,
                        help="Debug mode: 0 or 1 (default: %(default)s)")
    parser.add_argument("--jax_profile", type=int, default=0,
                        help="JAX profiling level: 0=disabled, 1=compile_log, 2=profiler+compile_log (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: %(default)s)")

    ### For meta envs
    parser.add_argument("--dim", type=int, default=128,
                        help="Output dim of metaaugnetwork (default: %(default)s)")
    parser.add_argument("--depth", type=int, default=4,
                        help="Depth of MetaAugNetwork (default: %(default)s)")
    parser.add_argument("--max_depth", type=int, default=8,
                        help="Max depth metaaugnetwork, num parallel is 2**max_depth (default: %(default)s)")
    parser.add_argument("--with_adjoint", type=int, default=0,
                        help="Use adjoint of orthogonal matrix in branch (default: %(default)s)")
    parser.add_argument("--reset_words", type=int, default=1,
                        help="Reset words per epoch (default: %(default)s)")

    ### For evaluation
    parser.add_argument("--eval_method", type=str, default="tiling",
                        help="Evaluation method: tiling / padding / identity (default: %(default)s)")
    parser.add_argument("--num_trials", type=int, default=16,
                        help="Number of trials per episode (default: %(default)s)")
    parser.add_argument("--eval_num_trials", type=int, default=16,
                        help="Number of trials per episode for evaluation (default: %(default)s)")

    ### For gymnax enviroments. Unnecessary  for popgym.
    parser.add_argument("--norm_strategy", type=str, default="fixed",
                        help="Reward normalization strategy: dynamic/fixed/minmax/custom (default: %(default)s)")
    parser.add_argument("--norm_max_steps", type=int, default=200,
                        help="Maximum steps for reward normalization scaling (default: %(default)s)")
    
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

    ### Dispatcher: select implementation version
    parser.add_argument("--mode", type=str, default="legacy",
                        help="Implementation mode: separated (FRP state separation), legacy (before separation), or lazy (lazy evaluation) (default: %(default)s)")

    args = parser.parse_args()
    
    # Meta environment specific kwargs
    meta_kwargs = {
        "meta_depth": args.depth,
        "meta_max_depth": args.max_depth,
        "meta_dim": args.dim,
        "meta_with_adjoint": (args.with_adjoint==1),
        "num_trials_per_episode": args.num_trials,
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
    run(args, args.num_runs, args.env, args.arch, env_kwargs=env_kwargs, meta_kwargs=meta_kwargs, norm_kwargs=norm_kwargs, mode=args.mode, wandb_run_id=wandb_run_id)
