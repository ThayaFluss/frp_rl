# Code Style Guidelines for frp_popjaxrl

**Last Updated:** 2026-01-05

---

## JAX/Flax Programming Principles

### 1. Pure Functional Programming

**All JAX functions must be pure (no side effects):**

```python
# ✅ GOOD: Pure function
def transform_obs(obs: chex.Array, weight: chex.Array) -> chex.Array:
    """Transform observation using weight matrix."""
    return obs @ weight

# ❌ BAD: Side effects
global_counter = 0
def transform_obs_bad(obs: chex.Array, weight: chex.Array) -> chex.Array:
    global global_counter
    global_counter += 1  # Side effect!
    return obs @ weight
```

**Why:** JAX's JIT compilation assumes pure functions. Side effects cause undefined behavior.

---

### 2. Immutable Data Structures

**Use `flax.struct.dataclass` for all stateful data:**

```python
# ✅ GOOD: Immutable dataclass
from flax import struct
import chex

@struct.dataclass
class FRPState:
    """Dataclass for storing per-environment FRP state."""
    env_indices: chex.Array  # [num_envs]
    frp_words: FRPWords

# Update using replace()
new_state = old_state.replace(env_indices=new_indices)

# ❌ BAD: Mutable class
class FRPStateBad:
    def __init__(self):
        self.env_indices = None  # Mutable!

    def update(self, new_indices):
        self.env_indices = new_indices  # In-place mutation!
```

**Why:** JAX transformations (vmap, jit) require immutable data.

---

### 3. Type Hints and Documentation

**All public functions require:**
- Type hints for parameters and return values
- Docstring with Args/Returns sections
- Shape annotations for array parameters

```python
# ✅ GOOD: Complete documentation
def transform_obs(
    obs: chex.Array,           # [batch, input_dim]
    env_index: int,
    frp_words: FRPWords
) -> chex.Array:               # [batch, aug_output_dim]
    """Transform observation using FRP word selected by env_index.

    Args:
        obs: Raw observations [batch, input_dim]
        env_index: Index of FRP word to use (scalar)
        frp_words: FRP transformation matrices

    Returns:
        Transformed observations [batch, aug_output_dim]
    """
    weight = get_weight_matrix(frp_words.words, env_index, input_dim, output_dim)
    return obs @ weight

# ❌ BAD: No types, no docs, unclear shapes
def transform_obs_bad(obs, idx, words):
    weight = get_weight_matrix(words.words, idx, 4, 64)
    return obs @ weight
```

---

### 4. Dimension Comments (CRITICAL)

**Always annotate array shapes in comments:**

```python
# ✅ GOOD: Explicit shape comments
def apply_frp_transform(
    obs: chex.Array,        # [num_envs, total_obs_dim]
    env_indices: chex.Array # [num_envs]
) -> chex.Array:            # [num_envs, frp_transformed_obs_size]
    # Split observation: [raw_obs (input_dim)] + [metadata (3)] + [wrapper (wrapper_dim)]
    raw_obs = obs[:, :input_dim]                    # [num_envs, input_dim]
    metadata = obs[:, input_dim:input_dim+3]        # [num_envs, 3]
    wrapper = obs[:, input_dim+3:]                  # [num_envs, wrapper_dim]

    # Build FRP input based on configuration
    frp_input_parts = [raw_obs]  # Start with raw obs
    if include_metadata:
        frp_input_parts.append(metadata)
    if include_wrapper:
        frp_input_parts.append(wrapper)

    frp_input = jnp.concatenate(frp_input_parts, axis=1)  # [num_envs, frp_input_dim]

    # Apply FRP transformation
    transformed = vmap_transform(frp_input, env_indices)   # [num_envs, aug_output_dim]

    # Build remaining parts (not transformed)
    remaining_parts = []
    if not include_metadata:
        remaining_parts.append(metadata)
    if not include_wrapper:
        remaining_parts.append(wrapper)

    # Reconstruct full observation
    if remaining_parts:
        new_obs = jnp.concatenate([transformed] + remaining_parts, axis=1)
    else:
        new_obs = transformed

    return new_obs  # [num_envs, aug_output_dim + non_transformed_size]

# ❌ BAD: No shape comments
def apply_frp_transform_bad(obs, env_indices):
    raw_obs = obs[:, :input_dim]
    metadata = obs[:, input_dim:input_dim+3]
    # ... what are the shapes? unclear!
    return transformed
```

---

### 5. JAX-Specific Control Flow

**Use `jax.lax` for control flow inside JIT:**

```python
# ✅ GOOD: JAX control flow
def conditional_transform(obs: chex.Array, use_frp: bool) -> chex.Array:
    """Conditionally apply FRP transformation."""
    return jax.lax.cond(
        use_frp,
        lambda x: transform_with_frp(x),   # True branch
        lambda x: x,                        # False branch
        obs
    )

# ✅ GOOD: JAX select for element-wise conditionals
def select_reset_or_step(done: bool, reset_obs: chex.Array, step_obs: chex.Array) -> chex.Array:
    """Select observation based on done flag."""
    return jax.lax.select(done, reset_obs, step_obs)

# ❌ BAD: Python if inside JIT (causes ConcretizationError)
@jax.jit
def conditional_transform_bad(obs: chex.Array, use_frp: bool) -> chex.Array:
    if use_frp:  # ❌ ConcretizationError!
        return transform_with_frp(obs)
    else:
        return obs
```

---

### 6. Variable Naming Conventions

**Consistent naming for common variables:**

```python
# Array dimensions
input_dim           # Raw environment observation dimension (before metadata/wrapper)
metadata_dim        # Metadata dimension (always 3: action_value, env_done, reset_flag)
wrapper_dim         # Wrapper dimension (n_actions+1 for discrete, 2 for continuous)
frp_input_dim       # Total dimension of FRP input (input_dim + optional metadata + optional wrapper)
aug_output_dim      # FRP augmented output dimension (meta_dim or truncated)

# Observation components
raw_obs             # Raw environment observation [batch, input_dim]
metadata            # Metadata [batch, 3]
wrapper             # Wrapper data [batch, wrapper_dim]
obs                 # Full observation (context-dependent)

# FRP-related
frp_words           # FRPWords dataclass (transformation matrices)
frp_manager         # FRPManager instance
env_index           # Index of FRP word to use (scalar or array)
env_indices         # Array of env_index values [num_envs]

# Meta-parameters
meta_depth          # Depth of word tree
meta_dim            # Output dimension of FRP
meta_max_depth      # Maximum depth (controls # of words: 2^max_depth)
meta_with_adjoint   # Include transpose matrices (bool)
```

---

### 7. Error Messages

**Provide actionable error messages:**

```python
# ✅ GOOD: Clear error with context
if obs.shape[1] != expected_obs_dim:
    raise ValueError(
        f"Observation dimension mismatch. "
        f"Expected {expected_obs_dim} (input_dim={input_dim} + metadata=3 + wrapper={wrapper_dim}), "
        f"but got {obs.shape[1]}. "
        f"Check FRP configuration: include_metadata={include_metadata}, include_wrapper={include_wrapper}"
    )

# ❌ BAD: Vague error
if obs.shape[1] != expected_obs_dim:
    raise ValueError("Wrong dimension")
```

---

### 8. File Organization

**Module structure:**

```python
"""Module docstring describing purpose and key concepts.

Example:
    >>> manager = FRPManager(...)
    >>> frp_words = manager.initialize_words(rng)
"""

# Standard library imports
from typing import Tuple, Optional, Dict

# Third-party imports (alphabetical)
import chex
from flax import struct
import jax
import jax.numpy as jnp

# Local imports (relative)
from .orthogonal import create_orthogonal_matrices

# Constants
DEFAULT_METADATA_DIM = 3

# Dataclasses
@struct.dataclass
class FRPState:
    """..."""
    pass

# Classes
class FRPManager:
    """..."""
    pass

# Functions
def create_frp_manager(...):
    """..."""
    pass
```

---

### 9. Comments and Documentation

**When to comment:**

```python
# ✅ GOOD: Comment for non-obvious logic
# Gymnax auto-reset mechanism: reset_env() is called every step,
# but the result is only used when done=True via jax.lax.select
obs_re, state_re = self.reset_env(key_reset, params)
state = jax.tree_map(
    lambda x, y: jax.lax.select(done, x, y),
    state_re,   # x: selected when done=True
    state_st    # y: selected when done=False
)

# ✅ GOOD: Comment for dimension calculations
# FRP input dimension (what gets transformed)
frp_input_dim = frp_manager.frp_input_dim
# Parts that are NOT transformed
non_transformed_size = base_env_obs_size - frp_input_dim

# ❌ BAD: Obvious comment
x = x + 1  # Increment x by 1
```

**Docstring format:**

```python
def function_name(arg1: Type1, arg2: Type2) -> ReturnType:
    """One-line summary.

    Longer description if needed. Explain the purpose, not the implementation.
    Mention important caveats or non-obvious behavior.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When and why this is raised

    Example:
        >>> result = function_name(val1, val2)
        >>> print(result)
        expected_output
    """
```

---

### 10. Performance Considerations

**JIT compilation:**

```python
# ✅ GOOD: Static arguments explicitly marked
@partial(jax.jit, static_argnums=(0,))
def step(self, key, state, action, params=None):
    """Step function with self as static argument."""
    pass

# ✅ GOOD: Avoid recompilation with consistent shapes
def process_batch(obs: chex.Array):  # [batch, dim]
    # Shapes are static, JIT will compile once
    return jnp.mean(obs, axis=0)

# ❌ BAD: Dynamic shapes cause recompilation
def process_variable_batch(obs: chex.Array):  # [variable_batch, dim]
    # Every different batch size triggers recompilation!
    return jnp.mean(obs, axis=0)
```

**Vectorization with vmap:**

```python
# ✅ GOOD: Use vmap for batch operations
transform_single = lambda obs, idx: transform_obs(obs, idx, frp_words)
vmap_transform = jax.vmap(transform_single, in_axes=(0, 0))
transformed_batch = vmap_transform(obs_batch, env_indices)  # Parallelized

# ❌ BAD: Manual loop (slow, not parallelized)
transformed_batch = []
for i in range(len(obs_batch)):
    transformed_batch.append(transform_obs(obs_batch[i], env_indices[i], frp_words))
transformed_batch = jnp.stack(transformed_batch)
```

---

## Python General Conventions

### Code Formatting

- **Line length:** 100 characters (flexible to 120 for readability)
- **Indentation:** 4 spaces
- **Imports:** Group and alphabetize as shown in File Organization
- **Strings:** Double quotes `"` for strings, single `'` for characters

### Naming

- **Classes:** `PascalCase` (e.g., `FRPManager`)
- **Functions/variables:** `snake_case` (e.g., `transform_obs`, `frp_input_dim`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `DEFAULT_METADATA_DIM`)
- **Private:** Prefix with `_` (e.g., `_internal_helper`)

---

## Project-Specific Patterns

### Dimension Calculation Pattern

**Always follow this pattern when calculating dimensions:**

```python
# 1. Get base observation size from environment
base_obs_size = env.observation_space(env_params).shape[0]
# base_obs_size = input_dim + metadata_dim + wrapper_dim

# 2. Calculate FRP input dimension based on configuration
frp_input_dim = frp_manager.frp_input_dim
# frp_input_dim = input_dim [+ metadata_dim] [+ wrapper_dim]

# 3. Calculate non-transformed parts
non_transformed_size = base_obs_size - frp_input_dim

# 4. Calculate final observation size after FRP
frp_transformed_obs_size = frp_manager.aug_output_dim + non_transformed_size
```

### FRP Transformation Pattern

**Standard pattern for applying FRP transformations:**

```python
# 1. Split observation into components
raw_obs = obs[:, :input_dim]
metadata = obs[:, input_dim:input_dim+3]
wrapper = obs[:, input_dim+3:]

# 2. Build FRP input based on configuration
frp_input_parts = [raw_obs]  # Always include raw obs
if include_metadata:
    frp_input_parts.append(metadata)
if include_wrapper:
    frp_input_parts.append(wrapper)
frp_input = jnp.concatenate(frp_input_parts, axis=1)

# 3. Apply FRP transformation
transformed = apply_frp(frp_input, env_indices, frp_words)

# 4. Build remaining parts
remaining_parts = []
if not include_metadata:
    remaining_parts.append(metadata)
if not include_wrapper:
    remaining_parts.append(wrapper)

# 5. Reconstruct observation
if remaining_parts:
    new_obs = jnp.concatenate([transformed] + remaining_parts, axis=1)
else:
    new_obs = transformed
```

---

## Common Anti-Patterns to Avoid

### ❌ Modifying arrays in-place

```python
# BAD
obs[0] = new_value  # JAX arrays are immutable!

# GOOD
obs = obs.at[0].set(new_value)
```

### ❌ Using Python lists for JAX operations

```python
# BAD
results = []
for x in data:
    results.append(process(x))
output = jnp.array(results)

# GOOD
output = jax.vmap(process)(data)
```

### ❌ Hardcoded dimensions

```python
# BAD
metadata = obs[:, 4:7]  # What is 4? What is 7?

# GOOD
metadata = obs[:, input_dim:input_dim+metadata_dim]  # Clear!
```

### ❌ Missing shape documentation

```python
# BAD
def transform(x, y):
    return jnp.dot(x, y)

# GOOD
def transform(
    x: chex.Array,  # [batch, input_dim]
    y: chex.Array   # [input_dim, output_dim]
) -> chex.Array:    # [batch, output_dim]
    return jnp.dot(x, y)
```

---

## Checklist for Code Review

Before submitting code, verify:

- [ ] All functions have type hints and docstrings
- [ ] All array shapes are documented in comments
- [ ] No Python control flow (if/for) on JAX arrays inside JIT
- [ ] All data structures use `flax.struct.dataclass`
- [ ] Dimension calculations are explicit and commented
- [ ] Error messages are actionable
- [ ] Code follows the transformation/dimension patterns
- [ ] No hardcoded magic numbers (use named constants)
- [ ] Imports are organized correctly
- [ ] JAX transformations (vmap/jit) are used appropriately
