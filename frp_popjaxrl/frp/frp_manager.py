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
        include_metadata: bool = False,
        include_wrapper: bool = False,
        metadata_dim: int = 3,
        wrapper_dim: int = 0
    ):
        """Initialize FRP Manager.

        Args:
            meta_depth: Depth of word tree (controls granularity)
            meta_dim: Dimension of meta augmentation (output size)
            input_dim: Dimension of raw environment observations (before metadata/wrapper)
            meta_max_depth: Maximum depth for parallel words (controls # of words)
            meta_with_adjoint: Whether to include adjoint matrices
            meta_truncate_aug: Whether to truncate augmentation output (0 or 1)
            include_metadata: Whether to include metadata (3D: action, done, reset) in FRP input
            include_wrapper: Whether to include wrapper data in FRP input
            metadata_dim: Dimension of metadata (default: 3)
            wrapper_dim: Dimension of wrapper data (set at runtime based on action space)
        """
        self.meta_depth = meta_depth
        self.meta_dim = meta_dim
        self.input_dim = input_dim  # Raw environment observation dimension
        self.meta_max_depth = meta_max_depth
        self.meta_with_adjoint = meta_with_adjoint
        self.meta_truncate_aug = meta_truncate_aug

        # FRP input configuration
        self.include_metadata = include_metadata
        self.include_wrapper = include_wrapper
        self.metadata_dim = metadata_dim
        self.wrapper_dim = wrapper_dim

        # Calculate FRP input dimension based on what's included
        # Always includes input_dim (raw env obs)
        self.frp_input_dim = input_dim
        if include_metadata:
            self.frp_input_dim += metadata_dim
        if include_wrapper:
            self.frp_input_dim += wrapper_dim

        # Calculate output dimension based on truncation setting
        if meta_truncate_aug == 1:
            self.aug_output_dim = input_dim
        else:
            self.aug_output_dim = meta_dim

    def initialize_words(self, rng_key: chex.PRNGKey) -> FRPWords:
        """Initialize FRP words and metadata (SEPARATED mode - no identity exclusion).

        This method creates the orthogonal transformation matrices (words) that
        will be used for observation augmentation. The process:
        1. Create base orthogonal matrices using QR decomposition
        2. Compose matrices to create 2^meta_max_depth unique words
        3. Truncate words to appropriate dimensions

        NOTE: Separated mode does NOT exclude identity matrices (simplified implementation).

        Args:
            rng_key: JAX random key for generating orthogonal matrices

        Returns:
            FRPWords dataclass containing words, empty exclude array, and total_words
        """
        matrices = create_orthogonal_matrices(
            rng_key,
            self.meta_depth,
            size=self.meta_dim,
            max_depth=self.meta_max_depth,
            with_adjoint=self.meta_with_adjoint
        )

        words = create_words(
            matrices,
            self.meta_depth,
            out_size=self.meta_dim,
            max_depth=self.meta_max_depth
        )

        # SEPARATED MODE: No identity detection (simplified)
        exclude = jnp.array([], dtype=jnp.int32)

        # Truncate words based on configuration
        if self.meta_truncate_aug == 1:
            # Truncate both input and output dimensions
            words = words[:, :self.frp_input_dim, :self.aug_output_dim]
        else:
            # Only truncate input dimension, keep full output
            words = words[:, :self.frp_input_dim, :]

        total_words = words.shape[0]

        return FRPWords(words=words, exclude=exclude, total_words=total_words)

    def sample_env_index(self, frp_words: FRPWords, rng_key: chex.PRNGKey) -> int:
        """Sample an environment index for FRP transformation (SEPARATED mode - no exclusion).

        Samples a random word index using simple uniform sampling.
        This function is JIT-compatible and uses JAX's random sampling.

        NOTE: Separated mode does NOT exclude identity matrices (simplified implementation).

        Args:
            frp_words: FRPWords dataclass with words and metadata
            rng_key: JAX random key for sampling

        Returns:
            Integer index in range [0, total_words)
        """
        num_words = frp_words.words.shape[0]
        return jax.random.randint(rng_key, (), 0, num_words).astype(jnp.int32)

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
            obs: Input observation vector [frp_input_dim]
                 Should contain raw_obs + (metadata if included) + (wrapper if included)
            env_index: Index of the word to use for transformation
            frp_words: FRPWords dataclass with transformation matrices

        Returns:
            Transformed observation [aug_output_dim]
        """
        weight = get_weight_matrix(
            frp_words.words,
            env_index,
            self.frp_input_dim,
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

    def __init__(self, method: str, input_dim: int, output_dim: int,
                 include_metadata: bool = False, include_wrapper: bool = False,
                 metadata_dim: int = 3, wrapper_dim: int = 0):
        """Initialize evaluation FRP manager.

        Args:
            method: Evaluation method - "identity", "padding", or "tiling"
            input_dim: Dimension of raw input observations
            output_dim: Dimension of output (meta_dim or input_dim)
            include_metadata: Whether to include metadata in FRP input
            include_wrapper: Whether to include wrapper data in FRP input
            metadata_dim: Dimension of metadata (default: 3)
            wrapper_dim: Dimension of wrapper data
        """
        self.method = method
        self.input_dim = input_dim
        self.output_dim = output_dim

        # FRP input configuration
        self.include_metadata = include_metadata
        self.include_wrapper = include_wrapper
        self.metadata_dim = metadata_dim
        self.wrapper_dim = wrapper_dim

        # Calculate FRP input dimension
        self.frp_input_dim = input_dim
        if include_metadata:
            self.frp_input_dim += metadata_dim
        if include_wrapper:
            self.frp_input_dim += wrapper_dim

        # Prepare transformation matrix based on method
        if method == "padding":
            # Create padding matrix: eye(output_dim)[:frp_input_dim, :]
            self.eval_weight = jnp.eye(output_dim)[:self.frp_input_dim, :]
        elif method == "tiling":
            # Create periodic tiling matrix
            from envs.environments.metaaug.padding import create_periodic_weight
            self.eval_weight = create_periodic_weight(
                input_dim=self.frp_input_dim,
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
            obs: Input observation vector [frp_input_dim]
                 Should contain raw_obs + (metadata if included) + (wrapper if included)

        Returns:
            Transformed observation [output_dim] or [frp_input_dim] for identity
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
        config: Configuration dict with META_KWARGS and ENV settings
            - config["META_KWARGS"]: meta learning parameters
            - config["ENV"]: environment (for input_dim)

    Returns:
        Initialized FRPManager instance

    Example:
        >>> manager = create_frp_manager(config)
        >>> frp_words = manager.initialize_words(rng)
    """
    # Unwrap environment to get base MetaEnvironment for input_dim
    env = config["ENV"]
    while hasattr(env, '_env'):
        env = env._env

    meta_kwargs = config["META_KWARGS"]

    # Get wrapper configuration
    wrapper_env = config["ENV"]
    wrapper_dim = 0
    if hasattr(wrapper_env, '__class__') and 'AliasPrevActionV2' in wrapper_env.__class__.__name__:
        # Calculate wrapper dimension based on action space
        from gymnax.environments import spaces
        action_space = wrapper_env.action_space(config["ENV_PARAMS"])
        if isinstance(action_space, spaces.Discrete):
            wrapper_dim = action_space.n + 1  # one-hot + reset flag
        elif isinstance(action_space, spaces.Box):
            wrapper_dim = 2  # action + reset flag

    return FRPManager(
        meta_depth=meta_kwargs.get('meta_depth', 1),
        meta_dim=meta_kwargs.get('meta_dim', 4),
        input_dim=env.input_dim,
        meta_max_depth=meta_kwargs.get('meta_max_depth', 2),
        meta_with_adjoint=meta_kwargs.get('meta_with_adjoint', False),
        meta_truncate_aug=meta_kwargs.get('meta_truncate_aug', 0),
        include_metadata=meta_kwargs.get('frp_include_metadata', False),
        include_wrapper=meta_kwargs.get('frp_include_wrapper', False),
        metadata_dim=3,
        wrapper_dim=wrapper_dim,
    )


def create_eval_frp_manager(config) -> Optional[EvalFRPManager]:
    """Create EvalFRPManager from configuration dict.

    This function checks if evaluation FRP is needed based on the config
    and creates an appropriate EvalFRPManager if required.

    Args:
        config: Configuration dict with META_KWARGS, EVAL_META_KWARGS and ENV settings
            - config["EVAL_META_KWARGS"]: evaluation meta learning parameters
            - config["META_KWARGS"]: training meta learning parameters
            - config["ENV"]: environment (for input_dim)

    Returns:
        EvalFRPManager instance or None if no eval transformation needed

    Example:
        >>> eval_manager = create_eval_frp_manager(config)
        >>> if eval_manager:
        ...     transformed_obs = eval_manager.transform_obs(obs)
    """
    # Unwrap environment to get base MetaEnvironment for input_dim
    env = config["ENV"]
    while hasattr(env, '_env'):
        env = env._env

    # Get evaluation method from EVAL_META_KWARGS
    eval_meta_kwargs = config.get("EVAL_META_KWARGS", {})
    method = eval_meta_kwargs.get("meta_const_aug", None)

    # Only create eval manager for known methods
    if method in ["padding", "tiling", "identity"]:
        # Get meta_kwargs for dimension parameters
        meta_kwargs = config["META_KWARGS"]
        meta_truncate_aug = meta_kwargs.get('meta_truncate_aug', 0)
        meta_dim = meta_kwargs.get('meta_dim', 4)

        # Get wrapper configuration
        wrapper_env = config["EVAL_ENV"]
        wrapper_dim = 0
        if hasattr(wrapper_env, '__class__') and 'AliasPrevActionV2' in wrapper_env.__class__.__name__:
            # Calculate wrapper dimension based on action space
            from gymnax.environments import spaces
            action_space = wrapper_env.action_space(config["EVAL_ENV_PARAMS"])
            if isinstance(action_space, spaces.Discrete):
                wrapper_dim = action_space.n + 1  # one-hot + reset flag
            elif isinstance(action_space, spaces.Box):
                wrapper_dim = 2  # action + reset flag

        # Determine output dimension based on method and truncation
        if method == "identity":
            output_dim = env.input_dim
        elif meta_truncate_aug == 1:
            output_dim = env.input_dim
        else:
            output_dim = meta_dim

        return EvalFRPManager(
            method=method,
            input_dim=env.input_dim,
            output_dim=output_dim,
            include_metadata=meta_kwargs.get('frp_include_metadata', False),
            include_wrapper=meta_kwargs.get('frp_include_wrapper', False),
            metadata_dim=3,
            wrapper_dim=wrapper_dim,
        )
    else:
        # No eval manager needed (will use training FRP)
        return None
