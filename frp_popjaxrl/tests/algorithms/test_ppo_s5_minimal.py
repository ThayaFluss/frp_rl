import unittest
import jax
from algorithms.ppo_in_context_lazy import make_train
from envs.wrappers import AliasPrevActionV2
from envs.meta_environment_lazy import create_meta_environment
import wandb


class TestPPOS5(unittest.TestCase):
    def test_on_MetaCartPole(self):
        wandb.init()
        rng = jax.random.PRNGKey(30)

        meta_env_kwargs = {
            "meta_dim": 2,
            "rng": rng,
            "meta_max_depth": 8,
            "meta_depth": 2,
            "meta_with_adjoint": 1,
        }
        env = create_meta_environment("cartpole", {}, meta_env_kwargs)

        eval_env_kwargs = meta_env_kwargs.copy()
        eval_env_kwargs["meta_const_aug"] = "padding"
        eval_env = create_meta_environment("cartpole", {}, eval_env_kwargs)

        config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 1,
            "NUM_STEPS": 128,
            "TOTAL_TIMESTEPS": 1e4,
            "UPDATE_EPOCHS": 1,
            "NUM_MINIBATCHES": 1,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 1.0,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.0,
            "VF_COEF": 1.0,
            "MAX_GRAD_NORM": 0.5,
            "ENV": AliasPrevActionV2(env),
            "ENV_PARAMS": env.default_params,
            "EVAL_ENV": AliasPrevActionV2(eval_env),
            "EVAL_ENV_PARAMS": eval_env.default_params,
            "ANNEAL_LR": False,
            "DEBUG": True,
            "MODEL_TYPE": "s5",
            "S5_D_MODEL": 256,
            "S5_SSM_SIZE": 256,
            "S5_N_LAYERS": 1,
            "S5_BLOCKS": 1,
            "S5_ACTIVATION": "full_glu",
            "S5_DO_NORM": False,
            "S5_PRENORM": False,
            "S5_DO_GTRXL_NORM": False,
        }

        train_jit = jax.jit(make_train(config))
        out = train_jit(rng)
