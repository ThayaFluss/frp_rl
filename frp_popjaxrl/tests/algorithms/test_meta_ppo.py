import unittest
import jax
from algorithms.ppo_in_context_lazy import make_train
from envs.wrappers import AliasPrevActionV2
from envs.meta_environment_lazy import create_meta_environment
import wandb

class TestMetaPPO(unittest.TestCase):
    def setUp(self):
        self.base_config = {
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
            "ANNEAL_LR": False,
            "DEBUG": True,
        }

        self.s5_specific = {
            "S5_D_MODEL": 256,
            "S5_SSM_SIZE": 256,
            "S5_N_LAYERS": 1,
            "S5_BLOCKS": 1,
            "S5_ACTIVATION": "full_glu",
            "S5_DO_NORM": False,
            "S5_PRENORM": False,
            "S5_DO_GTRXL_NORM": False,
        }

    def _create_env(self, env_name, rng):
        meta_env_kwargs = {
            "meta_dim": 2,
            "rng": rng,
            "meta_max_depth": 8,
            "meta_depth": 2,
            "meta_with_adjoint": 1
        }
        env = create_meta_environment(env_name, {}, meta_env_kwargs)
        
        eval_env_kwargs = meta_env_kwargs.copy()
        eval_env_kwargs["meta_const_aug"] = "tiling"
        eval_env = create_meta_environment(env_name, {}, eval_env_kwargs)
        
        return env, eval_env

    def test_cartpole_s5(self):
        """Test CartPole with PPO-S5"""
        wandb.init(project="test-meta-ppo", name="cartpole-s5")
        rng = jax.random.PRNGKey(0)

        env, eval_env = self._create_env("cartpole", rng)
        config = {**self.base_config, **self.s5_specific}
        config["MODEL_TYPE"] = "s5"
        config["ENV"] = AliasPrevActionV2(env)
        config["ENV_PARAMS"] = env.default_params
        config["EVAL_ENV"] = AliasPrevActionV2(eval_env)
        config["EVAL_ENV_PARAMS"] = eval_env.default_params

        train_fn = make_train(config)
        runner_state, metrics_dict = jax.jit(train_fn)(rng)

        # Verify training progress
        final_returns = metrics_dict["train_metric"]
        self.assertGreater(final_returns, -0.01, "CartPole metric must be positive")
        wandb.finish()

    def test_cartpole_gru(self):
        """Test CartPole with PPO-GRU"""
        wandb.init(project="test-meta-ppo", name="cartpole-gru")
        rng = jax.random.PRNGKey(1)

        env, eval_env = self._create_env("cartpole", rng)
        config = self.base_config.copy()
        config["MODEL_TYPE"] = "gru"
        config["ENV"] = AliasPrevActionV2(env)
        config["ENV_PARAMS"] = env.default_params
        config["EVAL_ENV"] = AliasPrevActionV2(eval_env)
        config["EVAL_ENV_PARAMS"] = eval_env.default_params

        train_fn = make_train(config)
        runner_state, metrics_dict = jax.jit(train_fn)(rng)

        # Verify training progress
        final_returns = metrics_dict["train_metric"]
        self.assertGreater(final_returns, -0.01, "CartPole-GRU training did not achieve sufficient performance")
        wandb.finish()

    def test_pendulum_s5(self):
        """Test Pendulum with PPO-S5"""
        wandb.init(project="test-meta-ppo", name="pendulum-s5")
        rng = jax.random.PRNGKey(2)

        env, eval_env = self._create_env("pendulum", rng)
        config = {**self.base_config, **self.s5_specific}
        config["MODEL_TYPE"] = "s5"
        config["ENV"] = AliasPrevActionV2(env)
        config["ENV_PARAMS"] = env.default_params
        config["EVAL_ENV"] = AliasPrevActionV2(eval_env)
        config["EVAL_ENV_PARAMS"] = eval_env.default_params

        train_fn = make_train(config)
        runner_state, metrics_dict = jax.jit(train_fn)(rng)

        # Verify training progress (Pendulum returns are negative, better performance = higher/closer to 0)
        final_returns = metrics_dict["train_metric"]
        self.assertGreater(final_returns, -1000.0, "Pendulum-S5 training did not achieve sufficient performance")
        wandb.finish()

    def test_pendulum_gru(self):
        """Test Pendulum with PPO-GRU"""
        wandb.init(project="test-meta-ppo", name="pendulum-gru")
        rng = jax.random.PRNGKey(3)

        env, eval_env = self._create_env("pendulum", rng)
        config = self.base_config.copy()
        config["MODEL_TYPE"] = "gru"
        config["ENV"] = AliasPrevActionV2(env)
        config["ENV_PARAMS"] = env.default_params
        config["EVAL_ENV"] = AliasPrevActionV2(eval_env)
        config["EVAL_ENV_PARAMS"] = eval_env.default_params

        train_fn = make_train(config)
        runner_state, metrics_dict = jax.jit(train_fn)(rng)

        # Verify training progress (Pendulum returns are negative, better performance = higher/closer to 0)
        final_returns = metrics_dict["train_metric"]
        self.assertGreater(final_returns, -1000.0, "Pendulum-GRU training did not achieve sufficient performance")
        wandb.finish()

if __name__ == "__main__":
    unittest.main()
