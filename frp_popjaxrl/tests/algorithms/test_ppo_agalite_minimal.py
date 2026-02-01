"""
Minimal tests for AGaLiTe-based PPO training.

Tests:
1. AGaLiTeCore initialization and hidden state format
2. AGaLiTe training with ppo_standard.py
"""

import unittest
import jax
import jax.numpy as jnp

from algorithms.models import AGaLiTeCore
from algorithms.agalite import BatchedAGaLiTe


class TestAGaLiTeCore(unittest.TestCase):
    """Tests for AGaLiTeCore initialization and interface."""

    def test_initialize_carry_shape(self):
        """Test that initialize_carry returns (1, batch, ...) format."""
        batch_size = 4
        n_layers = 2
        config = {
            "AGALITE_N_LAYERS": n_layers,
            "AGALITE_D_MODEL": 64,
            "AGALITE_D_HEAD": 64,
            "AGALITE_D_FFC": 64,
            "AGALITE_N_HEADS": 4,
            "AGALITE_ETA": 4,
            "AGALITE_R": 2,
        }

        carry = AGaLiTeCore.initialize_carry(batch_size, config)

        # Verify carry is a dict with 'layer_1', 'layer_2', ... keys
        for layer_idx in range(1, n_layers + 1):
            layer_key = f"layer_{layer_idx}"
            self.assertIn(layer_key, carry, f"Expected {layer_key} in carry")

            # Each layer has a tuple: (tilde_k, tilde_v, s, tick)
            layer_memory = carry[layer_key]
            self.assertEqual(len(layer_memory), 4, "Layer memory should have 4 elements")

            # Check that first dimension is 1 (leading 1 format)
            tilde_k, tilde_v, s, tick = layer_memory
            self.assertEqual(tilde_k.shape[0], 1, "tilde_k first dimension should be 1")
            self.assertEqual(tilde_k.shape[1], batch_size, "tilde_k second dimension should be batch_size")
            self.assertEqual(tilde_v.shape[0], 1, "tilde_v first dimension should be 1")
            self.assertEqual(s.shape[0], 1, "s first dimension should be 1")
            self.assertEqual(tick.shape[0], 1, "tick first dimension should be 1")

    def test_initialize_carry_default_config(self):
        """Test initialize_carry with default config values."""
        batch_size = 2
        config = {}  # Use all defaults

        carry = AGaLiTeCore.initialize_carry(batch_size, config)

        # Should work with defaults (n_layers defaults to 4)
        self.assertIn("layer_1", carry)
        self.assertIn("layer_4", carry)  # Default n_layers is 4

    def test_core_forward_pass(self):
        """Test AGaLiTeCore forward pass."""
        batch_size = 2
        seq_len = 8
        embed_dim = 64  # This should match d_model for AGaLiTe

        config = {
            "AGALITE_N_LAYERS": 2,
            "AGALITE_D_MODEL": 64,
            "AGALITE_D_HEAD": 64,
            "AGALITE_D_FFC": 64,
            "AGALITE_N_HEADS": 4,
            "AGALITE_ETA": 4,
            "AGALITE_R": 2,
            "NO_RESET": False,
        }

        model = AGaLiTeCore(config=config)
        carry = AGaLiTeCore.initialize_carry(batch_size, config)

        # Create dummy inputs (embedding, not raw obs - Core expects encoded input)
        embedding = jnp.zeros((seq_len, batch_size, embed_dim))
        dones = jnp.zeros((seq_len, batch_size))

        # Initialize and run forward pass
        rng = jax.random.PRNGKey(0)
        variables = model.init(rng, carry, embedding, dones)
        new_carry, output = model.apply(variables, carry, embedding, dones)

        # Verify output shapes
        self.assertEqual(output.shape, (seq_len, batch_size, config["AGALITE_D_MODEL"]))

        # Verify carry has same structure and leading 1 dimension
        for layer_idx in range(1, config["AGALITE_N_LAYERS"] + 1):
            layer_key = f"layer_{layer_idx}"
            self.assertIn(layer_key, new_carry)
            tilde_k = new_carry[layer_key][0]
            self.assertEqual(tilde_k.shape[0], 1, "Output carry should have leading 1 dimension")


class TestBatchedAGaLiTe(unittest.TestCase):
    """Tests for BatchedAGaLiTe module."""

    def test_initialize_carry(self):
        """Test BatchedAGaLiTe.initialize_carry produces correct shapes."""
        batch_size = 4
        n_layers = 2
        n_heads = 4
        d_head = 64
        eta = 4
        r = 2

        carry = BatchedAGaLiTe.initialize_carry(
            batch_size=batch_size,
            n_layers=n_layers,
            n_heads=n_heads,
            d_head=d_head,
            eta=eta,
            r=r
        )

        # Verify structure: dict with layer_1, layer_2, ... keys
        for layer_idx in range(1, n_layers + 1):
            layer_key = f"layer_{layer_idx}"
            self.assertIn(layer_key, carry)

            # Each layer has: (tilde_k, tilde_v, s, tick)
            tilde_k, tilde_v, s, tick = carry[layer_key]

            # tilde_k: (batch, r, n_heads, eta * d_head)
            self.assertEqual(tilde_k.shape[0], batch_size)
            self.assertEqual(tilde_k.shape[1], r)
            self.assertEqual(tilde_k.shape[2], n_heads)
            self.assertEqual(tilde_k.shape[3], eta * d_head)

            # tilde_v: (batch, r, n_heads, d_head)
            self.assertEqual(tilde_v.shape[0], batch_size)
            self.assertEqual(tilde_v.shape[3], d_head)

            # s: (batch, n_heads, eta * d_head)
            self.assertEqual(s.shape[0], batch_size)
            self.assertEqual(s.shape[1], n_heads)

            # tick: (batch, 1)
            self.assertEqual(tick.shape[0], batch_size)


class TestAGaLiTeTraining(unittest.TestCase):
    """Integration test for AGaLiTe training with ppo_standard.py."""

    def test_agalite_training_cartpole(self):
        """Test AGaLiTe training on StatelessCartPoleEasy environment."""
        import wandb
        from algorithms.ppo_standard import make_train
        from envs.wrappers import AliasPrevActionV2
        from envs import make

        wandb.init(mode="disabled")  # Disable wandb for testing

        rng = jax.random.PRNGKey(42)

        # Create environment (use registered name)
        env_name = "StatelessCartPoleEasy"
        env, env_params = make(env_name)
        env = AliasPrevActionV2(env)

        config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 2,
            "NUM_STEPS": 64,
            "TOTAL_TIMESTEPS": 2e3,  # Very short for testing
            "UPDATE_EPOCHS": 1,
            "NUM_MINIBATCHES": 1,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ENV": env,
            "ENV_PARAMS": env_params,
            "ANNEAL_LR": False,
            "DEBUG": True,
            "MODEL_TYPE": "agalite",  # Use AGaLiTe
            # AGaLiTe config
            "AGALITE_N_LAYERS": 2,
            "AGALITE_D_MODEL": 64,
            "AGALITE_D_HEAD": 64,
            "AGALITE_D_FFC": 64,
            "AGALITE_N_HEADS": 4,
            "AGALITE_ETA": 4,
            "AGALITE_R": 2,
        }

        # JIT compile and run training
        train_fn = jax.jit(make_train(config))
        out = train_fn(rng)

        # Verify training completed without errors
        self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
