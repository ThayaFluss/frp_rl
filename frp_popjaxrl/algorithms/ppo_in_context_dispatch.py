"""
Unified PPO training dispatcher.

This module provides a simple dispatcher that selects the appropriate PPO
implementation based on the LAZY configuration flag.

The dispatcher uses existing implementations without any runtime overhead:
- No conditional branches inside jax.jit
- No performance impact on compile time or execution time
- Simple import-based selection at Python time

Note: MODEL_TYPE ('gru' or 's5') is now a config parameter handled within
the implementation files, not in the dispatcher.

Usage:
    from algorithms.ppo_in_context_dispatch import make_train

    config = {
        "MODEL_TYPE": "gru",  # "gru" or "s5" (handled in implementation)
        "LAZY": False,        # False (eager) or True (lazy word creation)
        # ... other config parameters
    }

    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    runner_state, metrics = train_jit(rng)
"""

import logging

logger = logging.getLogger(__name__)


def make_train(config):
    """
    Dispatch to appropriate PPO implementation based on config.

    This dispatcher now only selects between eager and lazy implementations.
    The MODEL_TYPE ('gru' or 's5') is a config parameter read by the
    implementation files.

    Args:
        config: Configuration dictionary with keys:
            - MODEL_TYPE: "gru" or "s5" (default: "gru") - handled in implementation
            - LAZY: True for lazy word creation, False for eager (default: False)
            - Plus all standard PPO and environment configs

    Returns:
        train function that takes an RNG key and returns (runner_state, metrics)

    Raises:
        ValueError: If LAZY flag is invalid
    """
    lazy = config.get("LAZY", False)
    model_type = config.get("MODEL_TYPE", "gru").lower()

    logger.info(f"Dispatching to PPO implementation: MODEL_TYPE={model_type}, LAZY={lazy}")

    # Simple dispatch based on lazy flag
    # MODEL_TYPE is now handled as a config parameter within the implementation
    # No conditional logic enters jax.jit - dispatch happens at Python time
    if not lazy:
        from algorithms.ppo_in_context import make_train
        logger.debug(f"Selected: ppo_in_context (eager, {model_type} encoder)")
    elif lazy:
        from algorithms.ppo_in_context_lazy import make_train
        logger.debug(f"Selected: ppo_in_context_lazy (lazy, {model_type} encoder)")
    else:
        raise ValueError(
            f"Invalid LAZY flag: {lazy}. LAZY must be True or False."
        )

    # Return the selected make_train function applied to config
    # The returned function is exactly the same as if we imported directly
    return make_train(config)
