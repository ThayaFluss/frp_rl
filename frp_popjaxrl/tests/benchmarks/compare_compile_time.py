"""
Compare compilation time between v1 and lazy implementations.

This script measures ONLY the compilation time of the training function,
not the execution time. The hypothesis is that state size affects compile time.
"""
import jax
import jax.numpy as jnp
import time
import logging
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def measure_compile_time_v1(args):
    """Measure compilation time for v1 implementation"""
    import wandb
    from envs.meta_environment import create_meta_environment
    from algorithms.ppo_gru_in_context import make_train as make_train_gru
    from envs.wrappers import AliasPrevActionV2

    logger.info("="*60)
    logger.info("V1 IMPLEMENTATION")
    logger.info("="*60)

    # Initialize wandb (required by PPO callback)
    wandb.init(mode="disabled")

    # Setup
    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = jax.random.split(rng)

    env_kwargs = {}
    meta_kwargs = {
        "meta_depth": args.depth,
        "meta_max_depth": args.max_depth,
        "meta_dim": args.dim,
        "meta_with_adjoint": False,
        "num_trials_per_episode": 16,
        "meta_rng": _rng,
    }

    logger.info("Creating environment...")
    env = create_meta_environment(args.env, env_kwargs, meta_kwargs, None)
    env_params = env.default_params

    # Evaluation environment
    rng, _rng = jax.random.split(rng)
    eval_meta_kwargs = meta_kwargs.copy()
    eval_meta_kwargs["meta_rng"] = _rng
    eval_meta_kwargs["meta_eval"] = True
    eval_meta_kwargs["meta_const_aug"] = "tiling"
    eval_env = create_meta_environment(args.env, env_kwargs, eval_meta_kwargs, None)
    eval_env_params = eval_env.default_params

    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 1,
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
        "RESET_WORDS": False,
    }

    logger.info(f"Configuration:")
    logger.info(f"  NUM_ENVS: {config['NUM_ENVS']}")
    logger.info(f"  NUM_STEPS: {config['NUM_STEPS']}")
    logger.info(f"  Meta depth: {args.depth}")
    logger.info(f"  Meta max_depth: {args.max_depth}")
    logger.info(f"  Meta dim: {args.dim}")
    logger.info(f"  Num words: {2 ** args.max_depth}")
    logger.info(f"  Num base: {2 ** (args.max_depth // args.depth)}")

    # Clear JAX cache
    logger.info("Clearing JAX cache...")
    jax.clear_caches()

    # Measure compilation time
    rngs = jax.random.split(rng, args.num_runs)
    train_vjit = jax.jit(jax.vmap(make_train_gru(config)))

    logger.info("Starting compilation...")
    t0 = time.time()
    compiled_fn = train_vjit.lower(rngs).compile()
    compile_time = time.time() - t0
    logger.info(f"Compilation completed in {compile_time:.2f}s")

    # Get state size information
    logger.info("Analyzing state structure...")
    dummy_out = compiled_fn(rngs[:1])
    runner_state = dummy_out[0]

    # Print state structure
    def count_params(pytree):
        """Count total parameters in a pytree"""
        total = 0
        def counter(x):
            nonlocal total
            if isinstance(x, jnp.ndarray):
                total += x.size
        jax.tree_util.tree_map(counter, pytree)
        return total

    total_size = count_params(runner_state)
    logger.info(f"Total state size: {total_size:,} elements")

    return {
        "mode": "v1",
        "compile_time": compile_time,
        "state_size": total_size,
        "config": {
            "depth": args.depth,
            "max_depth": args.max_depth,
            "dim": args.dim,
            "num_words": 2 ** args.max_depth,
            "num_base": 2 ** (args.max_depth // args.depth),
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
        }
    }


def measure_compile_time_lazy(args):
    """Measure compilation time for lazy implementation"""
    import wandb
    from envs.meta_environment_lazy import create_meta_environment
    from algorithms.ppo_gru_in_context_lazy import make_train as make_train_gru
    from envs.wrappers import AliasPrevActionV2

    logger.info("="*60)
    logger.info("LAZY IMPLEMENTATION")
    logger.info("="*60)

    # Initialize wandb (required by PPO callback)
    wandb.init(mode="disabled", reinit=True)

    # Setup
    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = jax.random.split(rng)

    env_kwargs = {}
    meta_kwargs = {
        "meta_depth": args.depth,
        "meta_max_depth": args.max_depth,
        "meta_dim": args.dim,
        "meta_with_adjoint": False,
        "num_trials_per_episode": 16,
        "meta_rng": _rng,
    }

    logger.info("Creating environment...")
    env = create_meta_environment(args.env, env_kwargs, meta_kwargs, None)
    env_params = env.default_params

    # Evaluation environment
    rng, _rng = jax.random.split(rng)
    eval_meta_kwargs = meta_kwargs.copy()
    eval_meta_kwargs["meta_rng"] = _rng
    eval_meta_kwargs["meta_eval"] = True
    eval_meta_kwargs["meta_const_aug"] = "tiling"
    eval_env = create_meta_environment(args.env, env_kwargs, eval_meta_kwargs, None)
    eval_env_params = eval_env.default_params

    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 1,
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
        "RESET_WORDS": False,
    }

    logger.info(f"Configuration:")
    logger.info(f"  NUM_ENVS: {config['NUM_ENVS']}")
    logger.info(f"  NUM_STEPS: {config['NUM_STEPS']}")
    logger.info(f"  Meta depth: {args.depth}")
    logger.info(f"  Meta max_depth: {args.max_depth}")
    logger.info(f"  Meta dim: {args.dim}")
    logger.info(f"  Num words: {2 ** args.max_depth}")
    logger.info(f"  Num base: {2 ** (args.max_depth // args.depth)}")

    # Clear JAX cache
    logger.info("Clearing JAX cache...")
    jax.clear_caches()

    # Measure compilation time
    rngs = jax.random.split(rng, args.num_runs)
    train_vjit = jax.jit(jax.vmap(make_train_gru(config)))

    logger.info("Starting compilation...")
    t0 = time.time()
    compiled_fn = train_vjit.lower(rngs).compile()
    compile_time = time.time() - t0
    logger.info(f"Compilation completed in {compile_time:.2f}s")

    # Get state size information
    logger.info("Analyzing state structure...")
    dummy_out = compiled_fn(rngs[:1])
    runner_state = dummy_out[0]

    # Print state structure
    def count_params(pytree):
        """Count total parameters in a pytree"""
        total = 0
        def counter(x):
            nonlocal total
            if isinstance(x, jnp.ndarray):
                total += x.size
        jax.tree_util.tree_map(counter, pytree)
        return total

    total_size = count_params(runner_state)
    logger.info(f"Total state size: {total_size:,} elements")

    return {
        "mode": "lazy",
        "compile_time": compile_time,
        "state_size": total_size,
        "config": {
            "depth": args.depth,
            "max_depth": args.max_depth,
            "dim": args.dim,
            "num_words": 2 ** args.max_depth,
            "num_base": 2 ** (args.max_depth // args.depth),
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
        }
    }


def compare_results(v1_result, lazy_result):
    """Compare and display results"""
    logger.info("\n" + "="*60)
    logger.info("COMPILATION TIME COMPARISON")
    logger.info("="*60)

    v1_compile = v1_result["compile_time"]
    lazy_compile = lazy_result["compile_time"]
    v1_size = v1_result["state_size"]
    lazy_size = lazy_result["state_size"]

    logger.info(f"\nCompile Time:")
    logger.info(f"  V1:   {v1_compile:>8.2f}s")
    logger.info(f"  Lazy: {lazy_compile:>8.2f}s")
    logger.info(f"  Ratio: {lazy_compile/v1_compile:.2f}x")
    if lazy_compile < v1_compile:
        improvement = (1 - lazy_compile / v1_compile) * 100
        logger.info(f"  Improvement: {improvement:.1f}% faster")
    else:
        degradation = (lazy_compile / v1_compile - 1) * 100
        logger.info(f"  Degradation: {degradation:.1f}% slower")

    logger.info(f"\nState Size:")
    logger.info(f"  V1:   {v1_size:>12,} elements")
    logger.info(f"  Lazy: {lazy_size:>12,} elements")
    logger.info(f"  Ratio: {lazy_size/v1_size:.2f}x")
    if lazy_size < v1_size:
        reduction = (1 - lazy_size / v1_size) * 100
        logger.info(f"  Reduction: {reduction:.1f}% smaller")
    else:
        increase = (lazy_size / v1_size - 1) * 100
        logger.info(f"  Increase: {increase:.1f}% larger")

    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Compare compilation time between v1 and lazy')
    parser.add_argument('--env', type=str, default='cartpole', help='Environment name')
    parser.add_argument('--depth', type=int, default=4, help='Meta depth')
    parser.add_argument('--max_depth', type=int, default=8, help='Meta max depth')
    parser.add_argument('--dim', type=int, default=128, help='Meta dimension')
    parser.add_argument('--num_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--num_steps', type=int, default=16, help='Number of steps per rollout')
    parser.add_argument('--total_timesteps', type=float, default=1e3, help='Total timesteps')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of training runs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("COMPILE TIME BENCHMARK")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  Environment: {args.env}")
    logger.info(f"  Depth: {args.depth}")
    logger.info(f"  Max Depth: {args.max_depth}")
    logger.info(f"  Dimension: {args.dim}")
    logger.info(f"  Num Words (N): {2 ** args.max_depth}")
    logger.info(f"  Num Base (B): {2 ** (args.max_depth // args.depth)}")
    logger.info("")

    # Measure v1
    v1_result = measure_compile_time_v1(args)

    logger.info("\n" + "="*60)
    logger.info("Waiting 2 seconds before next test...")
    logger.info("="*60)
    import time
    time.sleep(2)

    # Measure lazy
    lazy_result = measure_compile_time_lazy(args)

    # Compare
    compare_results(v1_result, lazy_result)


if __name__ == "__main__":
    main()
