######
### FRP Manager - Centralized management of Free Random Projection
######

import jax
import jax.numpy as jnp
from flax import struct
import chex
from typing import Optional
from .orthogonal import (
    create_orthogonal_matrices,
    create_words,
    detect_identity_matrices,
    get_weight_matrix,
    random_choice,
)


@struct.dataclass
class FRPWords:
    """Dataclass for storing FRP words and metadata.

    This dataclass holds all the information needed for FRP transformations:
    - words: The actual transformation matrices
    - exclude: Indices of identity matrices to exclude from sampling
    - total_words: Total number of available words (2^meta_max_depth)

    All fields are JAX arrays or primitives for JIT compatibility.
    """
    words: chex.Array       # [num_words, input_dim, output_dim]
    exclude: chex.Array     # identity matrix indices
    total_words: int        # 2^meta_max_depth


@struct.dataclass
class FRPState:
    """Dataclass for storing per-environment FRP state.

    This dataclass holds the FRP state that needs to be tracked per environment:
    - env_indices: Index of the FRP word to use for each environment [num_envs]
    - frp_words: Shared FRP words used across all environments

    This is managed externally from the environment state, providing complete
    separation between environment dynamics and FRP transformations.
    """
    env_indices: chex.Array  # [num_envs] - each env's current FRP word index
    frp_words: FRPWords      # shared transformation matrices


class FRPManager:
    """Manager for Free Random Projection transformations during training.

    This class encapsulates all FRP-related functionality that was previously
    scattered across MetaEnvironment. It handles:
    - Initialization of orthogonal transformation matrices (words)
    - Sampling of environment indices
    - Observation transformations using the selected word

    The FRPManager is designed to work with JAX's JIT compilation and vmap.
    All stateful data is stored in the FRPWords dataclass.

    Example usage:
        >>> manager = FRPManager(meta_depth=1, meta_dim=64, input_dim=4,
        ...                      meta_max_depth=8, meta_with_adjoint=False,
        ...                      meta_truncate_aug=0)
        >>> rng = jax.random.PRNGKey(0)
        >>> frp_words = manager.initialize_words(rng)
        >>> env_index = manager.sample_env_index(frp_words, rng)
        >>> transformed_obs = manager.transform_obs(obs, env_index, frp_words)
    """

    def __init__(
        self,
        meta_depth: int,
        meta_dim: int,
        input_dim: int,
        meta_max_depth: int,
        meta_with_adjoint: bool,
        meta_truncate_aug: int,
    ):
        """Initialize FRP Manager.

        Args:
            meta_depth: Depth of word tree (controls granularity)
            meta_dim: Dimension of meta augmentation (output size)
            input_dim: Dimension of input observations
            meta_max_depth: Maximum depth for parallel words (controls # of words)
            meta_with_adjoint: Whether to include adjoint matrices
            meta_truncate_aug: Whether to truncate augmentation output (0 or 1)
        """
        self.meta_depth = meta_depth
        self.meta_dim = meta_dim
        self.input_dim = input_dim
        self.meta_max_depth = meta_max_depth
        self.meta_with_adjoint = meta_with_adjoint
        self.meta_truncate_aug = meta_truncate_aug

        # Calculate output dimension based on truncation setting
        if meta_truncate_aug == 1:
            self.aug_output_dim = input_dim
        else:
            self.aug_output_dim = meta_dim

    def initialize_words(self, rng_key: chex.PRNGKey) -> FRPWords:
        """Initialize FRP words and metadata.

        This method creates the orthogonal transformation matrices (words) that
        will be used for observation augmentation. The process:
        1. Create base orthogonal matrices using QR decomposition
        2. Compose matrices to create 2^meta_max_depth unique words
        3. Detect identity matrices to exclude from sampling
        4. Truncate words to appropriate dimensions

        Args:
            rng_key: JAX random key for generating orthogonal matrices

        Returns:
            FRPWords dataclass containing words, exclude indices, and total_words
        """
        # Create orthogonal matrices
        matrices = create_orthogonal_matrices(
            rng_key,
            self.meta_depth,
            size=self.meta_dim,
            max_depth=self.meta_max_depth,
            with_adjoint=self.meta_with_adjoint
        )

        # Create words by composing matrices
        words = create_words(
            matrices,
            self.meta_depth,
            out_size=self.meta_dim,
            max_depth=self.meta_max_depth
        )

        # Detect identity matrices BEFORE truncation
        # This is important because we want to exclude identities from sampling
        exclude = detect_identity_matrices(words)

        # Truncate words based on configuration
        if self.meta_truncate_aug == 1:
            # Truncate both input and output dimensions
            words = words[:, :self.input_dim, :self.aug_output_dim]
        else:
            # Only truncate input dimension, keep full output
            words = words[:, :self.input_dim, :]

        total_words = words.shape[0]

        return FRPWords(words=words, exclude=exclude, total_words=total_words)

    def sample_env_index(self, frp_words: FRPWords, rng_key: chex.PRNGKey) -> int:
        """Sample an environment index for FRP transformation.

        Samples a random word index, excluding identity matrices if any exist.
        This function is JIT-compatible and uses JAX's random sampling.

        Args:
            frp_words: FRPWords dataclass with words and metadata
            rng_key: JAX random key for sampling

        Returns:
            Integer index in range [0, total_words)
        """
        # Use words.shape[0] instead of total_words for JAX tracer compatibility
        num_words = frp_words.words.shape[0]

        if frp_words.exclude.shape[0] == 0:
            # No exclusions, sample uniformly
            return jax.random.randint(
                rng_key, (), 0, num_words
            ).astype(jnp.int32)
        else:
            # Exclude identity matrices
            return random_choice(
                rng_key,
                total_words=num_words,
                exclude=frp_words.exclude
            ).astype(jnp.int32)

    def transform_obs(
        self,
        obs: chex.Array,
        env_index: int,
        frp_words: FRPWords
    ) -> chex.Array:
        """Transform observation using FRP.

        Applies the orthogonal transformation corresponding to env_index to
        the input observation. This is the core FRP operation.

        Args:
            obs: Input observation vector [input_dim]
            env_index: Index of the word to use for transformation
            frp_words: FRPWords dataclass with transformation matrices

        Returns:
            Transformed observation [aug_output_dim]
        """
        weight = get_weight_matrix(
            frp_words.words,
            env_index,
            self.input_dim,
            self.aug_output_dim
        )
        # Apply transformation: obs @ weight
        return (obs[None, :] @ weight)[0]


class EvalFRPManager:
    """Manager for FRP transformations during evaluation.

    Unlike FRPManager which uses random orthogonal matrices, EvalFRPManager
    uses deterministic transformations for consistent evaluation. Supported
    methods:
    - "identity": No transformation, obs stays unchanged
    - "padding": Pad observation with zeros to meta_dim
    - "tiling": Tile observation periodically to meta_dim

    Example usage:
        >>> eval_manager = EvalFRPManager("padding", input_dim=4, output_dim=64)
        >>> transformed_obs = eval_manager.transform_obs(obs)
    """

    def __init__(self, method: str, input_dim: int, output_dim: int):
        """Initialize evaluation FRP manager.

        Args:
            method: Evaluation method - "identity", "padding", or "tiling"
            input_dim: Dimension of input observations
            output_dim: Dimension of output (meta_dim or input_dim)
        """
        self.method = method
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Prepare transformation matrix based on method
        if method == "padding":
            # Create padding matrix: eye(output_dim)[:input_dim, :]
            self.eval_weight = jnp.eye(output_dim)[:input_dim, :]
        elif method == "tiling":
            # Create periodic tiling matrix
            from envs.environments.metaaug.padding import create_periodic_weight
            self.eval_weight = create_periodic_weight(
                input_dim=input_dim,
                output_dim=output_dim,
                period=round(output_dim / 2)
            )
        elif method == "identity":
            # No transformation matrix needed
            self.eval_weight = None
        else:
            raise ValueError(f"Unknown eval method: {method}")

    def transform_obs(self, obs: chex.Array) -> chex.Array:
        """Transform observation for evaluation.

        Applies deterministic transformation based on the evaluation method.

        Args:
            obs: Input observation vector [input_dim]

        Returns:
            Transformed observation [output_dim] or [input_dim] for identity
        """
        if self.method == "identity":
            # No transformation
            return obs
        else:
            # Apply transformation matrix
            return (obs[None, :] @ self.eval_weight)[0]


def create_frp_manager(config) -> FRPManager:
    """Create FRPManager from configuration dict.

    This is a convenience function that extracts the necessary parameters
    from a config dict and creates an FRPManager instance.

    Args:
        config: Configuration dict with ENV settings
            - config["ENV"].meta_depth
            - config["ENV"].meta_dim
            - config["ENV"].input_dim
            - config["ENV"].meta_max_depth
            - config["ENV"].meta_with_adjoint
            - config["ENV"].meta_truncate_aug

    Returns:
        Initialized FRPManager instance

    Example:
        >>> manager = create_frp_manager(config)
        >>> frp_words = manager.initialize_words(rng)
    """
    # Unwrap environment to get base MetaEnvironment
    env = config["ENV"]
    while hasattr(env, '_env'):
        env = env._env

    return FRPManager(
        meta_depth=env.meta_depth,
        meta_dim=env.meta_dim,
        input_dim=env.input_dim,
        meta_max_depth=env.meta_max_depth,
        meta_with_adjoint=env.meta_with_adjoint,
        meta_truncate_aug=env.meta_truncate_aug,
    )


def create_eval_frp_manager(config) -> Optional[EvalFRPManager]:
    """Create EvalFRPManager from configuration dict.

    This function checks if evaluation FRP is needed based on the config
    and creates an appropriate EvalFRPManager if required.

    Args:
        config: Configuration dict with ENV and EVAL_ENV settings
            - config["ENV"]: Base environment config
            - config.get("EVAL_ENV").meta_const_aug: Evaluation method

    Returns:
        EvalFRPManager instance or None if no eval transformation needed

    Example:
        >>> eval_manager = create_eval_frp_manager(config)
        >>> if eval_manager:
        ...     transformed_obs = eval_manager.transform_obs(obs)
    """
    # Unwrap environment to get base MetaEnvironment
    env = config["ENV"]
    while hasattr(env, '_env'):
        env = env._env

    # Get evaluation environment config
    eval_env = config.get("EVAL_ENV")
    method = getattr(eval_env, "meta_const_aug", None) if eval_env else None

    # Only create eval manager for known methods
    if method in ["padding", "tiling", "identity"]:
        # Determine output dimension based on method and truncation
        if method == "identity":
            output_dim = env.input_dim
        elif env.meta_truncate_aug == 1:
            output_dim = env.input_dim
        else:
            output_dim = env.meta_dim

        return EvalFRPManager(
            method=method,
            input_dim=env.input_dim,
            output_dim=output_dim
        )
    else:
        # No eval manager needed (will use training FRP)
        return None
