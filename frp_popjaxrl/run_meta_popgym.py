import jax
import jax.numpy as jnp
import time
from envs.meta_environment import create_meta_environment
from envs.wrappers import AliasPrevActionV2
from algorithms.ppo_gru_in_context import make_train as make_train_gru
from algorithms.ppo_s5_in_context import make_train as make_train_s5
import pickle
import os
from flax.core import unfreeze

import argparse

def run(args, num_runs, env_name, arch="gru", file_tag="", env_kwargs={}, meta_kwargs={}, norm_kwargs={}, use_few_shot=False):
    print("*"*10)
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
        "LR": 2.5e-4,
        "NUM_ENVS": 1,
        "NUM_STEPS": 16,  # Reduced from 128
        "TOTAL_TIMESTEPS": 1e3,  # Reduced from 1e4
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES":1,
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
        "LR": 5e-5,
        "NUM_ENVS": 64,
        "NUM_STEPS": 1024,
        "TOTAL_TIMESTEPS": 15e6, 
        "UPDATE_EPOCHS": 30,
        "NUM_MINIBATCHES": 8,
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
        "S5_N_LAYERS": 4,
        "S5_BLOCKS": 1,
        "S5_ACTIVATION": "full_glu",
        "S5_DO_NORM": False,
        "S5_PRENORM": False,
        "S5_DO_GTRXL_NORM": False,
        "RESET_WORDS": (args.reset_words==1)
        }

    rngs = jax.random.split(rng, num_runs)
    info_dict = {}

    if arch == "s5":
        train_vjit_s5 = jax.jit(jax.vmap(make_train_s5(config)))
        t0 = time.time()
        compiled_s5 = train_vjit_s5.lower(rngs).compile()
        compile_s5_time = time.time() - t0
        print(f"s5 compile time: {compile_s5_time}")

        t0 = time.time()
        out_s5 = jax.block_until_ready(compiled_s5(rngs))
        run_s5_time = time.time() - t0
        print(f"s5 time: {run_s5_time}")
        metrics = jax.tree_util.tree_map(lambda x: x.item() if hasattr(x, 'item') else x, out_s5[1])
        
        # Create base info dictionary with common metrics
        info_dict["s5"] = {
            "compile_s5_time": compile_s5_time,
            "run_s5_time": run_s5_time,
            "train_metrics": metrics["train_metric"],
            "in_context_metrics": metrics["in_context_metric"],
        }
        
        # Add few_shot_metrics only if few-shot learning is enabled
        if "few_shot_metric" in metrics:
            info_dict["s5"]["few_shot_metrics"] = metrics["few_shot_metric"]
    
    elif arch == "gru":
        train_vjit_rnn = jax.jit(jax.vmap(make_train_gru(config)))
        t0 = time.time()
        compiled_rnn = train_vjit_rnn.lower(rngs).compile()
        compile_rnn_time = time.time() - t0
        print(f"gru compile time: {compile_rnn_time}")

        t0 = time.time()
        out_rnn = jax.block_until_ready(compiled_rnn(rngs))
        run_rnn_time = time.time() - t0
        print(f"gru time: {run_rnn_time}")
        metrics = jax.tree_util.tree_map(lambda x: x.item() if hasattr(x, 'item') else x, out_rnn[1])
        
        # Create base info dictionary with common metrics
        info_dict["gru"] = {
            "compile_rnn_time": compile_rnn_time,
            "run_rnn_time": run_rnn_time,
            "train_metrics": metrics["train_metric"],
            "in_context_metrics": metrics["in_context_metric"],
        }
        
        # Add few_shot_metrics only if few-shot learning is enabled
        if "few_shot_metric" in metrics:
            info_dict["gru"]["few_shot_metrics"] = metrics["few_shot_metric"]
    
    else:
        raise NotImplementedError

    if args.save_results == 1:
        jnp.save(f"results/{num_runs}_{env_name}_{arch}_{file_tag}.npy", info_dict)

    # Save model checkpoint if requested
    if args.save_model == 1:
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_name = f"checkpoints/{env_name}_{arch}_{file_tag}_seed{args.seed}.pkl"

        # Extract params from the output
        if arch == "s5":
            runner_state = out_s5[0]
        elif arch == "gru":
            runner_state = out_rnn[0]

        # Get the first run's state (in case of multiple runs)
        train_state = jax.tree_util.tree_map(lambda x: x[0] if len(x.shape) > 0 else x, runner_state[0])

        # Convert params to a serializable format
        params_dict = unfreeze(train_state.params)

        # Create a copy of config without environment objects (which can't be pickled)
        config_to_save = {k: v for k, v in config.items() if k not in ["ENV", "ENV_PARAMS", "EVAL_ENV", "EVAL_ENV_PARAMS"]}

        # Save checkpoint with all necessary information
        checkpoint = {
            "params": params_dict,
            "config": config_to_save,
            "arch": arch,
            "env_name": env_name,
            "env_kwargs": env_kwargs,
            "meta_kwargs": meta_kwargs,
            "norm_kwargs": norm_kwargs,
            "seed": args.seed,
        }

        with open(checkpoint_name, "wb") as f:
            pickle.dump(checkpoint, f)

        print(f"Model saved to {checkpoint_name}")

if __name__ == "__main__":
    import wandb
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--env", type=str, default="cartpole", help="Base env XXX of MetaXXX.")
    parser.add_argument("--arch", type=str, default="s5")
    parser.add_argument("--log_wandb", type=str, default="popgym")
    parser.add_argument("--debug", type=int, default=0, help="debug")
    parser.add_argument("--seed", type=int, default=42,  help="Random seed")

    ### For meta envs
    parser.add_argument("--dim", type=int, default=64, help="output dim of metaaugnetwork")
    parser.add_argument("--depth", type=int, default=2, help="depth of MetaAugNetwork")
    parser.add_argument("--max_depth", type=int, default=8, help="max depth metaaugnetwork (num paralell is 2**max_depth)")
    parser.add_argument("--beta", type=float, default=1, help="beta for nelf")    
    parser.add_argument("--with_adjoint", type=int, default=0, help="use adjoint of orthogonal matrix in branch")
    parser.add_argument("--reset_words", type=int, default=1, help="reset words per epoch")

    ### For evaluation
    parser.add_argument("--eval_method", type=str, default="tiling", help="tiling / padding / identity")
    parser.add_argument("--use_few_shot", type=int, default=0, help="use few-shot learning (1) or in-context only (0)")
    parser.add_argument("--num_trials", type=int, default=16, help="number of trials per episode")

    ### For gymnax enviroments. Unnecessary  for popgym.
    parser.add_argument("--norm_strategy", type=str, default="fixed", help="reward normalization strategy: 'dynamic', 'fixed', 'minmax', or 'custom'")
    parser.add_argument("--norm_max_steps", type=int, default=200, help="maximum steps for reward normalization scaling")
    
    ### For saving results and models
    parser.add_argument("--save_results", type=int, default=0, help="save results npy (default:%(default)s)")
    parser.add_argument("--save_model", type=int, default=0, help="save model checkpoint (default:%(default)s)")

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
    # Pass use_few_shot directly to ensure consistency throughout the code
    use_few_shot = args.use_few_shot == 1
    run(args, args.num_runs, args.env, args.arch, env_kwargs=env_kwargs, meta_kwargs=meta_kwargs, norm_kwargs=norm_kwargs, use_few_shot=use_few_shot)
