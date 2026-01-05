# Architecture Guidelines for frp_popjaxrl

**Last Updated:** 2026-01-05

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Loop (PPO)                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. Sample FRP words (orthogonal matrices)                │  │
│  │  2. Initialize environments with FRP indices              │  │
│  │  3. Collect trajectories                                  │  │
│  │  4. Apply FRP transformations to observations             │  │
│  │  5. Update policy using transformed observations          │  │
│  │  6. Periodically regenerate FRP words (if reset_words=1) │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │              FRP Manager (SEPARATED)                │
    │  ┌───────────────────────────────────────────────┐  │
    │  │  - Generate orthogonal matrices (QR decomp)   │  │
    │  │  - Compose words (matrix multiplication)      │  │
    │  │  - Sample environment indices                 │  │
    │  │  - Transform observations (matrix mult)       │  │
    │  └───────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │            Meta Environment (SEPARATED)             │
    │  ┌───────────────────────────────────────────────┐  │
    │  │  - Wrap base environment (POPopGym)           │  │
    │  │  - Add metadata (action, done, reset)         │  │
    │  │  - NO FRP logic (externalized)                │  │
    │  └───────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │         Observation Wrappers (Gymnax)               │
    │  ┌───────────────────────────────────────────────┐  │
    │  │  - AliasPrevActionV2: Add previous action     │  │
    │  │  - LogWrapper: Record episode statistics      │  │
    │  └───────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │              Neural Network (S5/GRU)                │
    │  ┌───────────────────────────────────────────────┐  │
    │  │  Input: Transformed observation               │  │
    │  │  ├─ Encoder (Dense layers)                    │  │
    │  │  ├─ Recurrent Core (S5 or GRU)                │  │
    │  │  ├─ Actor Head (policy)                       │  │
    │  │  └─ Critic Head (value)                       │  │
    │  └───────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

### 1. FRP Manager (`frp/frp_manager.py`)

**Purpose:** Centralized management of Free Random Projection transformations

**Responsibilities:**
- Generate orthogonal transformation matrices (via `orthogonal.py`)
- Compose word matrices from base orthogonal matrices
- Sample environment indices for FRP word selection
- Transform observations using selected FRP word
- Manage configurable FRP scope (metadata/wrapper inclusion)

**Key Classes:**
- `FRPManager`: Training-time FRP management
- `EvalFRPManager`: Evaluation-time FRP management (identity/padding/tiling)
- `FRPWords`: Dataclass holding transformation matrices
- `FRPState`: Dataclass holding per-environment FRP state

**Does NOT:**
- Manage environment dynamics
- Handle episode resets
- Store environment state

**Design Principle:** Stateless transformer - all state in dataclasses

---

### 2. Meta Environment (`envs/meta_environment_separated.py`)

**Purpose:** Wrap base environments with meta-learning metadata

**Responsibilities:**
- Wrap base POPopGym environments
- Add 3D metadata to observations:
  - `action_value`: Previous action value
  - `env_done`: Individual trial completion (0.0 or 1.0)
  - `reset_flag`: Episode reset indicator (1.0 on reset, 0.0 otherwise)
- Manage trial counting and meta-episode structure
- Implement Gymnax environment interface

**Key Classes:**
- `MetaEnvironment`: Main environment class
- `MetaEnvState`: Environment state (trial_num, env_state, etc.)
- `MetaEnvParams`: Environment parameters

**Does NOT:**
- Apply FRP transformations (delegated to FRPManager)
- Store FRP state (managed externally)
- Handle observation wrappers (Gymnax handles this)

**Design Principle:** Pure environment dynamics, no FRP coupling

---

### 3. PPO Algorithm (`algorithms/ppo_frp_separated.py`)

**Purpose:** Implement PPO training with external FRP state management

**Responsibilities:**
- Initialize FRPManager and environments
- Manage FRPState (env_indices, frp_words) externally
- Apply FRP transformations to observations at:
  - Environment initialization
  - Training steps
  - Evaluation steps
- Calculate observation dimensions accounting for FRP configuration
- Implement PPO update logic
- Periodically regenerate FRP words (if configured)

**Key Functions:**
- `make_train()`: Setup training configuration and networks
- `initialize_frp_single_env()`: Initialize FRP for single environment
- `apply_frp_transform()`: Apply FRP during training
- `apply_eval_frp_transform()`: Apply FRP during evaluation
- `_env_step()`: Single environment step with FRP
- `_update_step()`: Full PPO update with optional word regeneration

**Does NOT:**
- Generate orthogonal matrices directly (uses FRPManager)
- Embed FRP logic in environment (uses external state)

**Design Principle:** Orchestrator that coordinates FRP + Environment + Model

---

### 4. Neural Networks (`algorithms/models.py`)

**Purpose:** S5/GRU architectures for in-context learning

**Responsibilities:**
- Encode transformed observations
- Maintain recurrent hidden state for in-context adaptation
- Output policy (actor) and value (critic) predictions
- Support both discrete and continuous action spaces

**Key Classes:**
- `ActorCriticDiscrete`: Actor-critic for discrete actions
- `ActorCriticContinuous`: Actor-critic for continuous actions
- `S5RepModel`: S5-based representation model
- `GRURepModel`: GRU-based representation model

**Design Principle:** Model-agnostic to FRP (just sees transformed observations)

---

### 5. Orthogonal Matrix Generation (`frp/orthogonal.py`)

**Purpose:** Core FRP mathematics - orthogonal matrix generation and composition

**Responsibilities:**
- Generate random orthogonal matrices via QR decomposition
- Compose word matrices via matrix multiplication
- Detect identity matrices for exclusion
- Extract weight matrices for transformation
- Sample environment indices

**Key Functions:**
- `create_orthogonal_matrices()`: Generate base orthogonal matrices
- `create_words()`: Compose word matrices from base matrices
- `get_weight_matrix()`: Extract transformation matrix for given index
- `detect_identity_matrices()`: Find identity matrices to exclude
- `random_choice()`: Sample environment index

**Design Principle:** Pure mathematical functions, highly optimized for JAX

---

## Data Flow Diagrams

### Training Step Data Flow

```
1. Environment Reset
   ├─ Sample env_index from FRPWords
   ├─ Reset base environment → raw_obs
   ├─ Add metadata → [raw_obs, metadata]
   └─ Apply wrapper → [raw_obs, metadata, wrapper]

2. FRP Transformation
   ├─ Split: [raw_obs, metadata, wrapper]
   ├─ Build FRP input (configurable):
   │  └─ [raw_obs] + [metadata?] + [wrapper?]
   ├─ Transform: frp_input @ weight_matrix → transformed
   └─ Reconstruct: [transformed] + [remaining]

3. Network Forward
   ├─ Input: transformed_obs + dones
   ├─ Encoder: transformed_obs → embedding
   ├─ Recurrent: (hidden, embedding, dones) → new_hidden
   ├─ Actor: new_hidden → action_logits
   └─ Critic: new_hidden → value

4. Environment Step
   ├─ Execute action in base environment
   ├─ Add metadata to next observation
   ├─ Apply same FRP transformation (same env_index)
   └─ Return to step 3 (until episode done)

5. PPO Update
   ├─ Collect full trajectory
   ├─ Calculate advantages and returns
   ├─ Update policy parameters
   └─ Optionally regenerate FRP words
```

### Evaluation Data Flow

```
1. Evaluation Reset
   ├─ Use evaluation FRP (identity/padding/tiling)
   ├─ Reset environment
   └─ Initialize hidden state

2. Evaluation Step
   ├─ Apply eval FRP transformation
   ├─ Network forward (frozen parameters)
   ├─ Execute action
   └─ Update hidden state only (no parameter updates)

3. In-Context Adaptation
   ├─ Hidden state evolves with new FRP transformation
   ├─ Agent adapts through recurrent dynamics
   └─ No gradient updates to parameters
```

---

## Mode Comparison: SEPARATED vs LAZY vs LEGACY

### SEPARATED Mode (Recommended)

**Philosophy:** Complete separation of FRP logic from environment

**Architecture:**
```
PPO Algorithm
    ├─ Manages FRPState externally
    ├─ Applies FRP transformations
    └─ Calls MetaEnvironment (no FRP awareness)

FRPManager
    ├─ Generates orthogonal matrices
    └─ Provides transformation functions

MetaEnvironment
    └─ Pure environment dynamics (no FRP)
```

**Advantages:**
- Clear separation of concerns
- Easy to test components independently
- Flexible FRP configuration
- No hidden state in environment

**Files:**
- `ppo_frp_separated.py`
- `meta_environment_separated.py`
- `frp_manager.py`
- `orthogonal.py`

---

### LAZY Mode

**Philosophy:** FRP state in environment, regenerated on-demand

**Architecture:**
```
PPO Algorithm
    └─ Calls MetaEnvironment with FRP

MetaEnvironment (LAZY)
    ├─ Stores FRP words in MetaEnvState
    ├─ Regenerates words when needed
    └─ Applies transformations internally
```

**Advantages:**
- Simpler algorithm code
- FRP state co-located with env state

**Disadvantages:**
- Tighter coupling
- Harder to test FRP independently

**Files:**
- `ppo_in_context_lazy.py`
- `meta_environment_lazy.py`
- `orthogonal_lazy.py`

---

### LEGACY Mode (Deprecated)

**Philosophy:** Original implementation with embedded FRP

**Status:** Do not use for new features

**Files:**
- `ppo_in_context_legacy.py`
- `meta_environment_legacy.py`
- `orthogonal_legacy.py`

---

## Design Patterns

### 1. Immutable State with Dataclasses

**Pattern:** Use `flax.struct.dataclass` for all stateful data

```python
from flax import struct
import chex

@struct.dataclass
class FRPState:
    """Immutable FRP state."""
    env_indices: chex.Array  # [num_envs]
    frp_words: FRPWords

# Update with replace()
new_state = old_state.replace(env_indices=new_indices)
```

**Why:** JAX transformations require immutable data

---

### 2. Factory Functions for Configuration

**Pattern:** Use factory functions to construct managers with config

```python
def create_frp_manager(config: Dict) -> FRPManager:
    """Create FRPManager from configuration dictionary."""
    env = config["ENV"]

    # Auto-detect wrapper dimension
    if isinstance(env.action_space(env.default_params), spaces.Discrete):
        n_actions = env.action_space(env.default_params).n
        wrapper_dim = n_actions + 1  # One-hot + reset flag
    else:
        wrapper_dim = 2  # Action + reset flag

    return FRPManager(
        meta_depth=config["META_DEPTH"],
        meta_dim=config["META_DIM"],
        input_dim=env.input_dim,
        meta_max_depth=config["META_MAX_DEPTH"],
        meta_with_adjoint=config["META_WITH_ADJOINT"],
        meta_truncate_aug=config.get("META_TRUNCATE_AUG", 0),
        include_metadata=config.get("FRP_INCLUDE_METADATA", False),
        include_wrapper=config.get("FRP_INCLUDE_WRAPPER", False),
        metadata_dim=3,
        wrapper_dim=wrapper_dim
    )
```

**Why:** Centralizes configuration logic, easier to extend

---

### 3. Vectorization with vmap

**Pattern:** Use `jax.vmap` for batch operations

```python
# Single observation transformation
def transform_single(obs, env_index, frp_words):
    weight = get_weight_matrix(
        frp_words.words, env_index,
        input_dim, output_dim
    )
    return obs @ weight

# Vectorized transformation
vmap_transform = jax.vmap(
    transform_single,
    in_axes=(0, 0, None)  # batch obs, batch indices, shared words
)

# Apply to batch
transformed_batch = vmap_transform(obs_batch, env_indices, frp_words)
```

**Why:** Parallelizes computation, leverages hardware accelerators

---

### 4. Static Arguments for JIT

**Pattern:** Mark non-array arguments as static in JIT

```python
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def step(self, key, state, action, params=None):
    """Environment step with self as static argument."""
    # self is static, won't trigger recompilation
    pass
```

**Why:** Prevents recompilation, improves performance

---

### 5. Configuration-Driven Behavior

**Pattern:** Use configuration flags instead of subclasses

```python
# ✅ GOOD: Configuration-driven
class FRPManager:
    def __init__(self, include_metadata: bool, include_wrapper: bool):
        self.include_metadata = include_metadata
        self.include_wrapper = include_wrapper

    def build_frp_input(self, raw_obs, metadata, wrapper):
        parts = [raw_obs]
        if self.include_metadata:
            parts.append(metadata)
        if self.include_wrapper:
            parts.append(wrapper)
        return jnp.concatenate(parts)

# ❌ BAD: Subclass proliferation
class FRPManagerWithMetadata(FRPManager):
    ...
class FRPManagerWithWrapper(FRPManager):
    ...
class FRPManagerWithBoth(FRPManager):
    ...
```

**Why:** Reduces code duplication, easier to test all configurations

---

## Dimension Management Strategy

### Dimension Vocabulary

**Clear naming prevents confusion:**

| Term | Meaning | Example (CartPole) |
|------|---------|-------------------|
| `input_dim` | Raw environment observation | 4 |
| `metadata_dim` | Metadata size (action, done, reset) | 3 |
| `wrapper_dim` | Wrapper data size | 3 (discrete: 2+1) |
| `base_obs_size` | Total before FRP | 10 (4+3+3) |
| `frp_input_dim` | What goes into FRP | 4 or 7 or 10 |
| `aug_output_dim` | FRP output size | 64 (meta_dim) |
| `non_transformed_size` | What doesn't go into FRP | 6 or 3 or 0 |
| `frp_transformed_obs_size` | Final observation size | 70 or 67 or 64 |

### Dimension Calculation Checkpoints

**At environment creation:**
```python
# MetaEnvironment
self.input_dim = self.env.observation_space(params).shape[0]  # Raw obs
self.obs_shape = (self.input_dim + 3,)  # + metadata
```

**After wrapper application:**
```python
# AliasPrevActionV2 wrapper adds wrapper_dim
base_obs_size = input_dim + metadata_dim + wrapper_dim
```

**In FRPManager:**
```python
# Calculate FRP input dimension
self.frp_input_dim = input_dim
if include_metadata:
    self.frp_input_dim += metadata_dim
if include_wrapper:
    self.frp_input_dim += wrapper_dim
```

**In PPO algorithm:**
```python
# Calculate final observation size after FRP
base_env_obs_size = env.observation_space(env_params).shape[0]
frp_input_dim = frp_manager.frp_input_dim
non_transformed_size = base_env_obs_size - frp_input_dim
frp_transformed_obs_size = frp_manager.aug_output_dim + non_transformed_size
```

**Invariant:** `base_obs_size == frp_input_dim + non_transformed_size`

---

## Extension Points

### Adding New FRP Algorithms

1. Create new file in `frp/`: `orthogonal_<variant>.py`
2. Implement core functions:
   - `create_orthogonal_matrices()`
   - `create_words()`
   - `get_weight_matrix()`
3. Update `frp_manager.py` to support new variant
4. Add tests in `tests/test_orthogonal_<variant>.py`

### Adding New Environments

1. Create environment in `envs/environments/<env_name>.py`
2. Implement Gymnax environment interface
3. Register in `meta_environment_factory.py`
4. Add dimension test for new environment
5. Test with both S5 and GRU architectures

### Adding New Model Architectures

1. Add model class in `algorithms/models.py`
2. Implement `__call__()` method with standard signature:
   ```python
   def __call__(self, hidden, x):
       obs, dones = x
       # ... process obs and dones
       return new_hidden, pi, value
   ```
3. Update `create_network()` in `ppo_frp_separated.py`
4. Test with existing environments

---

## Performance Considerations

### JIT Compilation

**Compilation happens once per unique shape:**
- Avoid dynamic shapes in training loop
- Use consistent batch sizes
- Mark static arguments appropriately

**Trigger recompilation if:**
- Input shapes change
- Static arguments change
- Control flow paths change

### Memory Management

**JAX uses functional arrays:**
- Arrays are immutable (copy-on-write)
- Use `.at[].set()` for updates
- Avoid unnecessary concatenations in hot loops

### Vectorization

**Always prefer vmap over Python loops:**
- 10-100x faster on GPU/TPU
- Automatic parallelization
- Better device utilization

---

## Testing Architecture

### Unit Tests

- **FRPManager:** Transformation correctness, dimension calculations
- **MetaEnvironment:** Episode structure, metadata addition
- **Models:** Forward pass shapes, hidden state handling

### Integration Tests

- **PPO + FRP + Env:** Full training loop
- **Dimension flow:** End-to-end dimension verification
- **All configurations:** All FRP scope combinations

### Regression Tests

- **Historical bugs:** Prevent reoccurrence
- **Edge cases:** Documented in test names

---

## Documentation Standards

### Code Documentation

**Docstrings for:**
- All public functions
- All classes
- Complex algorithms

**Inline comments for:**
- Non-obvious dimension calculations
- JAX-specific patterns
- Gymnax auto-reset behavior
- Performance optimizations

### Architecture Documentation

**Update when:**
- Adding new components
- Changing data flows
- Modifying dimension calculations
- Adding new modes or variants

**Location:**
- This file (`architecture.md`)
- CLAUDE.md for high-level overview
- Inline comments for implementation details
