# Testing Guidelines for frp_popjaxrl

**Last Updated:** 2026-01-05

---

## Testing Philosophy

In JAX-based RL systems, testing must verify:
1. **Dimension correctness:** Shape mismatches cause runtime errors
2. **Numerical correctness:** Transformations preserve expected properties
3. **Integration:** Components work together correctly
4. **Compilation:** JIT compilation succeeds without errors

---

## Mandatory Testing Requirements

### When Making Changes to FRP Logic

**MUST run ALL of the following tests:**

#### 1. Dimension Verification Test (Analytical)

```bash
# Run dimension verification script
python test_dimensions.py
```

**Purpose:** Verifies dimension calculations without requiring JAX/GPU

**What it tests:**
- Observation splitting logic
- FRP input dimension calculation
- Reconstruction dimension correctness
- All FRP configuration combinations

**Expected output:**
```
✓ All dimension tests PASSED

The implementation correctly:
  1. Splits observations into [raw_obs, metadata, wrapper]
  2. Selectively builds FRP input based on configuration
  3. Reconstructs the full observation with correct dimensions
```

#### 2. Execution Test (Compilation + Training)

```bash
# Test default configuration
WANDB_MODE=disabled python run_meta_popgym_separated.py \
    --env cartpole --arch gru --debug 1 --num_runs 1 --num_trials 2

# Expected: Successful compilation and training completion
# Look for: "GRU training completed in X.XXs"
```

**Purpose:** Verifies code actually runs and compiles

**What it tests:**
- JAX JIT compilation succeeds
- Training loop executes without errors
- Dimensions match at runtime
- Basic training functionality

#### 3. Configuration Tests (All FRP Scopes)

```bash
# Test 1: Default (raw obs only)
WANDB_MODE=disabled python run_meta_popgym_separated.py \
    --env cartpole --arch gru --num_runs 1 --num_trials 2

# Test 2: Raw obs + metadata
WANDB_MODE=disabled python run_meta_popgym_separated.py \
    --env cartpole --arch gru --frp_include_metadata 1 \
    --num_runs 1 --num_trials 2

# Test 3: Raw obs + wrapper
WANDB_MODE=disabled python run_meta_popgym_separated.py \
    --env cartpole --arch gru --frp_include_wrapper 1 \
    --num_runs 1 --num_trials 2

# Test 4: All components
WANDB_MODE=disabled python run_meta_popgym_separated.py \
    --env cartpole --arch gru \
    --frp_include_metadata 1 --frp_include_wrapper 1 \
    --num_runs 1 --num_trials 2
```

**Purpose:** Ensure all FRP configurations work

**What it tests:**
- Different FRP scope selections
- Dimension calculations for each configuration
- No hardcoded assumptions

#### 4. Architecture Tests (Both S5 and GRU)

```bash
# Test GRU
WANDB_MODE=disabled python run_meta_popgym_separated.py \
    --env cartpole --arch gru --num_runs 1 --num_trials 2

# Test S5
WANDB_MODE=disabled python run_meta_popgym_separated.py \
    --env cartpole --arch s5 --num_runs 1 --num_trials 2
```

**Purpose:** Verify both architectures work with changes

**What it tests:**
- GRU model compatibility
- S5 model compatibility
- Model-agnostic FRP implementation

---

## Test Script Structure

### Creating Dimension Verification Tests

**Template for `test_dimensions.py`:**

```python
#!/usr/bin/env python3
"""
Dimension verification test for FRP input configuration.
Tests the dimension calculations without requiring JAX.
"""

def calculate_frp_dimensions(input_dim, include_metadata, include_wrapper,
                            metadata_dim=3, wrapper_dim=0):
    """Calculate FRP input dimension based on configuration."""
    frp_input_dim = input_dim
    if include_metadata:
        frp_input_dim += metadata_dim
    if include_wrapper:
        frp_input_dim += wrapper_dim
    return frp_input_dim

def simulate_obs_transformation(raw_obs_dim, metadata_dim, wrapper_dim,
                                include_metadata, include_wrapper):
    """Simulate the observation transformation logic."""
    # Total observation dimension
    total_obs_dim = raw_obs_dim + metadata_dim + wrapper_dim

    # Build FRP input parts
    frp_input_parts = [raw_obs_dim]  # Always include raw obs
    if include_metadata:
        frp_input_parts.append(metadata_dim)
    if include_wrapper:
        frp_input_parts.append(wrapper_dim)

    frp_input_dim = sum(frp_input_parts)

    # Build remaining parts (not transformed)
    remaining_parts = []
    if not include_metadata:
        remaining_parts.append(metadata_dim)
    if not include_wrapper:
        remaining_parts.append(wrapper_dim)

    remaining_dim = sum(remaining_parts)

    # Reconstructed observation dimension
    # (In actual code, frp_input would be transformed to aug_output_dim,
    #  but for dimension checking we just verify total = frp_input + remaining)
    reconstructed_dim = frp_input_dim + remaining_dim

    return {
        'total_obs_dim': total_obs_dim,
        'frp_input_dim': frp_input_dim,
        'remaining_dim': remaining_dim,
        'reconstructed_dim': reconstructed_dim,
        'dimensions_match': total_obs_dim == reconstructed_dim
    }

def test_configuration(name, raw_obs_dim, metadata_dim, wrapper_dim,
                       include_metadata, include_wrapper):
    """Test a specific configuration."""
    print(f"\nTest: {name}")
    print(f"{'='*60}")

    result = simulate_obs_transformation(
        raw_obs_dim, metadata_dim, wrapper_dim,
        include_metadata, include_wrapper
    )

    print(f"Total obs: {result['total_obs_dim']}, "
          f"FRP input: {result['frp_input_dim']}, "
          f"Remaining: {result['remaining_dim']}, "
          f"Reconstructed: {result['reconstructed_dim']}")

    if result['dimensions_match']:
        print("✓ PASS")
    else:
        print("✗ FAIL")

    return result['dimensions_match']

def main():
    """Run all dimension tests."""
    all_passed = True

    # Test all configurations
    all_passed &= test_configuration(
        "Default (raw obs only)",
        raw_obs_dim=4, metadata_dim=3, wrapper_dim=3,
        include_metadata=False, include_wrapper=False
    )

    all_passed &= test_configuration(
        "Raw obs + metadata",
        raw_obs_dim=4, metadata_dim=3, wrapper_dim=3,
        include_metadata=True, include_wrapper=False
    )

    # Add more test cases...

    if all_passed:
        print("\n✓ All tests PASSED")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1

if __name__ == "__main__":
    exit(main())
```

---

## Integration Testing with pytest

### Test File Structure

```
frp_popjaxrl/
└── tests/
    ├── __init__.py
    ├── test_frp_manager.py           # FRP transformation tests
    ├── test_meta_environment.py      # Environment tests
    ├── test_ppo_frp_separated.py     # Algorithm tests
    └── test_dimensions.py            # Dimension verification
```

### Example Test Cases

#### Testing FRP Transformations

```python
import jax
import jax.numpy as jnp
import pytest
from frp_popjaxrl.frp.frp_manager import FRPManager

def test_frp_manager_initialization():
    """Test FRPManager initializes correctly."""
    manager = FRPManager(
        meta_depth=1,
        meta_dim=64,
        input_dim=4,
        meta_max_depth=8,
        meta_with_adjoint=False,
        meta_truncate_aug=0
    )

    assert manager.input_dim == 4
    assert manager.meta_dim == 64
    assert manager.frp_input_dim == 4  # Default: only raw obs

def test_frp_transformation_preserves_inner_products():
    """Test that FRP transformation preserves inner products."""
    manager = FRPManager(
        meta_depth=1, meta_dim=64, input_dim=4,
        meta_max_depth=8, meta_with_adjoint=False,
        meta_truncate_aug=0
    )

    rng = jax.random.PRNGKey(0)
    frp_words = manager.initialize_words(rng)

    # Create two observations
    obs1 = jnp.array([1.0, 2.0, 3.0, 4.0])
    obs2 = jnp.array([2.0, 3.0, 4.0, 5.0])

    # Calculate original inner product
    inner_product_original = jnp.dot(obs1, obs2)

    # Transform observations with same env_index
    env_index = 0
    transformed_obs1 = manager.transform_obs(obs1, env_index, frp_words)
    transformed_obs2 = manager.transform_obs(obs2, env_index, frp_words)

    # Calculate transformed inner product
    inner_product_transformed = jnp.dot(transformed_obs1, transformed_obs2)

    # Should be approximately equal (within numerical precision)
    assert jnp.allclose(inner_product_original, inner_product_transformed, rtol=1e-5)

def test_configurable_frp_scope():
    """Test FRP with different scope configurations."""
    configs = [
        {"include_metadata": False, "include_wrapper": False, "expected_dim": 4},
        {"include_metadata": True, "include_wrapper": False, "expected_dim": 7},
        {"include_metadata": False, "include_wrapper": True, "expected_dim": 7},
        {"include_metadata": True, "include_wrapper": True, "expected_dim": 10},
    ]

    for config in configs:
        manager = FRPManager(
            meta_depth=1, meta_dim=64, input_dim=4,
            meta_max_depth=8, meta_with_adjoint=False,
            meta_truncate_aug=0,
            include_metadata=config["include_metadata"],
            include_wrapper=config["include_wrapper"],
            metadata_dim=3,
            wrapper_dim=3
        )

        assert manager.frp_input_dim == config["expected_dim"]
```

#### Testing Environment Dimensions

```python
import jax
from frp_popjaxrl.envs.meta_environment_separated import create_meta_environment

def test_environment_observation_dimensions():
    """Test that environment produces correctly sized observations."""
    env_kwargs = {}
    meta_kwargs = {
        'meta_depth': 1,
        'meta_max_depth': 8,
        'meta_dim': 64,
        'meta_with_adjoint': False,
        'num_trials_per_episode': 2,
        'frp_include_metadata': False,
        'frp_include_wrapper': False,
    }
    norm_kwargs = {}

    env = create_meta_environment('cartpole', env_kwargs, meta_kwargs, norm_kwargs)

    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    # CartPole input_dim=4, metadata=3, so obs should be 7
    expected_obs_dim = env.input_dim + 3
    assert obs.shape[0] == expected_obs_dim
```

---

## Debugging Failed Tests

### Common Test Failures

#### 1. Shape Mismatch Errors

**Error:**
```
flax.errors.ScopeParamShapeError: Initializer expected to generate shape (134, 128)
but got shape (128, 128) instead
```

**Diagnosis:**
- Model expects different input dimension than provided
- Check `frp_transformed_obs_size` calculation in `ppo_frp_separated.py`

**Fix:**
```python
# Check if frp_input_dim is being used correctly
frp_input_dim = frp_manager.frp_input_dim  # NOT frp_manager.input_dim
non_transformed_size = base_env_obs_size - frp_input_dim
frp_transformed_obs_size = frp_manager.aug_output_dim + non_transformed_size
```

#### 2. ConcretizationError

**Error:**
```
jax._src.errors.ConcretizationTypeError: Abstract tracer value encountered where
concrete value is expected
```

**Diagnosis:**
- Using Python control flow on JAX arrays inside JIT
- Dynamic shapes or values

**Fix:**
```python
# Replace Python if with jax.lax.cond
# BAD
if some_jax_array > 0:
    result = do_something()

# GOOD
result = jax.lax.cond(
    some_jax_array > 0,
    lambda: do_something(),
    lambda: do_something_else()
)
```

#### 3. Dimension Reconstruction Mismatch

**Error:**
```
AssertionError: dimensions_match = False
Total: 10, Reconstructed: 9
```

**Diagnosis:**
- Missing component in reconstruction
- Incorrect splitting logic

**Fix:**
```python
# Verify all components are accounted for
# total_obs = raw_obs + metadata + wrapper
# frp_input = selected components
# remaining = non-selected components
# reconstructed = frp_output + remaining
#
# Must satisfy: total_obs = frp_input + remaining
```

---

## Continuous Integration Checks

### Pre-commit Checks

Before committing code, run:

```bash
# 1. Dimension verification
python test_dimensions.py

# 2. Quick execution test
WANDB_MODE=disabled python run_meta_popgym_separated.py \
    --env cartpole --arch gru --debug 1 --num_runs 1 --num_trials 2

# 3. Linting
flake8 frp_popjaxrl/ --max-line-length=120 --ignore=E203,W503

# 4. Type checking (if mypy is set up)
mypy frp_popjaxrl/ --ignore-missing-imports
```

### PR Checklist

- [ ] `test_dimensions.py` passes
- [ ] At least one execution test passes
- [ ] All FRP configurations tested (if FRP logic changed)
- [ ] Both S5 and GRU tested (if architecture-agnostic change)
- [ ] New tests added for new functionality
- [ ] Existing tests still pass

---

## Performance Testing

### Compilation Time

```bash
# Measure compilation time
time python run_meta_popgym_separated.py \
    --env cartpole --arch gru --num_runs 1 --num_trials 2 \
    2>&1 | grep "compilation completed"
```

**Expected:** Compilation should complete in < 60 seconds for simple environments

### Training Speed

```bash
# Measure training throughput
python run_meta_popgym_separated.py \
    --env cartpole --arch gru --num_runs 1 --num_trials 10 \
    2>&1 | grep "training completed"
```

**Monitor:** Training time should scale linearly with number of trials

---

## Test Data and Fixtures

### Reusable Test Fixtures

```python
import pytest
import jax

@pytest.fixture
def default_meta_kwargs():
    """Default meta-learning configuration."""
    return {
        'meta_depth': 1,
        'meta_max_depth': 8,
        'meta_dim': 64,
        'meta_with_adjoint': False,
        'num_trials_per_episode': 2,
        'frp_include_metadata': False,
        'frp_include_wrapper': False,
    }

@pytest.fixture
def rng():
    """JAX random key."""
    return jax.random.PRNGKey(42)

@pytest.fixture
def cartpole_env(default_meta_kwargs):
    """CartPole meta-environment."""
    from frp_popjaxrl.envs.meta_environment_separated import create_meta_environment
    return create_meta_environment('cartpole', {}, default_meta_kwargs, {})
```

---

## Regression Testing

### When to Add Regression Tests

Add a regression test when:
1. A bug is discovered and fixed
2. A dimension mismatch is resolved
3. An edge case is handled

### Regression Test Template

```python
def test_regression_frp_dimension_calculation_issue_2026_01():
    """
    Regression test for FRP dimension calculation bug discovered 2026-01-05.

    Bug: When include_metadata=True and include_wrapper=True, the dimension
    calculation used frp_manager.input_dim instead of frp_manager.frp_input_dim,
    causing shape mismatch errors.

    Fix: Use frp_manager.frp_input_dim which accounts for configurable scope.
    """
    manager = FRPManager(
        meta_depth=1, meta_dim=64, input_dim=4,
        meta_max_depth=8, meta_with_adjoint=False,
        meta_truncate_aug=0,
        include_metadata=True,
        include_wrapper=True,
        metadata_dim=3,
        wrapper_dim=3
    )

    # Should correctly calculate frp_input_dim = 4 + 3 + 3 = 10
    assert manager.frp_input_dim == 10

    # Verify dimension calculation for model input
    base_env_obs_size = 10  # input_dim + metadata + wrapper
    non_transformed_size = base_env_obs_size - manager.frp_input_dim
    frp_transformed_obs_size = manager.aug_output_dim + non_transformed_size

    # With full FRP scope, non_transformed_size should be 0
    assert non_transformed_size == 0
    assert frp_transformed_obs_size == manager.aug_output_dim
```

---

## Test Coverage Goals

**Minimum coverage for new features:**
- FRP transformations: Test all scope configurations
- Dimension calculations: Analytical + runtime verification
- Environment interactions: At least one full episode
- Model architectures: Both S5 and GRU

**Not required (but nice to have):**
- Full training convergence tests (too slow for CI)
- Extensive hyperparameter sweeps
- Multi-environment tests (can test on one environment)
