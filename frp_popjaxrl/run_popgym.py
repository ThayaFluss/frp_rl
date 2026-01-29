import jax
import jax.numpy as jnp
import time
from envs import make
from envs.wrappers import AliasPrevActionV2
from algorithms.ppo_gru_origin import make_train as make_train_gru
from algorithms.ppo_s5_origin import make_train as make_train_s5
import argparse
#from envs.environments.meta_cartpole import NoisyStatelessMetaCartPole, MetaEnvParams

def run(args, num_runs, env_name,arch="gru", file_tag="", env_kwargs={}):
    print("*"*10)
    #env, env_params = make(env_name)       
    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = jax.random.split(rng)
    env_kwargs["meta_rng"] = _rng
    env, env_params = make(env_name, **env_kwargs) 


    eval_env_kwargs = env_kwargs
    # use indep random vars for eval
    rng, _rng = jax.random.split(rng)
    eval_env_kwargs["meta_rng"] = _rng

    
    if args.eval_method in ["const", "padding"]:      
        # use constant meta_aug for eval      
        eval_env_kwargs["meta_const_aug"] = "padding"
    elif args.eval_method in ["tiling"]:      
        # use constant meta_aug for eval      
        eval_env_kwargs["meta_const_aug"] = "tiling"
        
    elif args.eval_method == "uniform":
        # use uniform random aug for eval
        eval_env_kwargs["meta_const_aug"] = False
        eval_env_kwargs["meta_depth"] = 1
        eval_env_kwargs["meta_with_adjoint"] = False
    else:
        raise ValueError()


    eval_env, eval_env_params = make(env_name, **eval_env_kwargs)

    if args.debug==1:
        config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 1,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e4, 
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
        "RESET_WORDS": (args.reset_words==1)
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

    
    #train_vjit_s5 = jax.jit(jax.vmap(make_train_s5(config)))
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
        info_dict["s5"] = {
            "compile_s5_time": compile_s5_time,
            "run_s5_time": run_s5_time,
            "out": out_s5[1],
        }

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
        info_dict["gru"] = {
            "compile_rnn_time": compile_rnn_time,
            "run_rnn_time": run_rnn_time,
            "out": out_rnn[1],
        }
    else:
        raise NotImplementedError

    jnp.save(f"results/{num_runs}_{env_name}_{arch}_{file_tag}.npy", info_dict)

if __name__ == "__main__":
    import wandb
    parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--env", type=str, default="NoisyStatelessMetaCartPole")
    parser.add_argument("--arch", type=str, default="s5")
    parser.add_argument("--log_wandb", type=str, default="popgym")
    parser.add_argument("--depth", type=int,  default=2, help="depth of MetaAugNetwork (1 or 2)")
    parser.add_argument("--max_depth", type=int, default=4, help="max depth metaaugnetwork (num paralell is 2**max_depth)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dim", type=int, default=4, help="output dim of metaaugnetwork")
    parser.add_argument("--with_adjoint", type=int, default=0, help="use adjoint of orthogonal matrix in branch (keep toral num branch)")
    parser.add_argument("--debug", type=int, default=0, help="debug")
    parser.add_argument("--eval_method", type=str, default="tiling", help="tiling or padding")
    parser.add_argument("--reset_words", type=int, default=0, help="reset words per epoch")


    #parser.add_argument("-ek", "--env-kwargs", type=str, default="{'meta_depth':2}")
    args = parser.parse_args()
    args.env_kwargs = {
        "meta_depth": args.depth,
        "meta_max_depth": args.max_depth,
        "meta_dim": args.dim,
        "meta_with_adjoint": (args.with_adjoint==1),
    }
    #args.env_kwargs = eval(args.env_kwargs)

    wandb.init(project=args.log_wandb, config=args)
    run(args, args.num_runs, args.env, args.arch, env_kwargs = args.env_kwargs)
