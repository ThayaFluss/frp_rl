# Gymnax Auto-Reset Mechanism

**Date:** 2026-01-01
**Purpose:** Reference documentation for coding AI and developers
**Location:** This document should be referenced by all MetaEnvironment implementations

---

## Overview

Gymnax provides an **automatic reset mechanism** that conditionally resets environment state when episodes complete (`done=True`). This mechanism is built into the base `gymnax.environments.environment.Environment` class.

**⚠️ CRITICAL FOR TESTING:** When testing environment behavior, always use `env.step()` (NOT `env.step_env()`) to match production training loop behavior.

---

## How Gymnax Auto-Reset Works

### Source Code

From `gymnax/environments/environment.py:26-47`:

```python
@partial(jax.jit, static_argnums=(0,))
def step(
    self,
    key: chex.PRNGKey,
    state: EnvState,
    action: Union[int, float],
    params: Optional[EnvParams] = None,
) -> Tuple[chex.Array, EnvState, float, bool, dict]:
    """Performs step transitions in the environment."""
    if params is None:
        params = self.default_params

    key, key_reset = jax.random.split(key)

    # 1. Execute environment step
    obs_st, state_st, reward, done, info = self.step_env(
        key, state, action, params
    )

    # 2. Generate reset state (CALLED EVERY STEP, regardless of done)
    obs_re, state_re = self.reset_env(key_reset, params)

    # 3. Conditionally select reset state when done=True
    state = jax.tree_map(
        lambda x, y: jax.lax.select(done, x, y),
        state_re,  # x: selected when done=True
        state_st   # y: selected when done=False
    )
    obs = jax.lax.select(done, obs_re, obs_st)

    return obs, state, reward, done, info
```

### Key Points

1. **`reset_env()` is called every step**
   - Due to JAX's functional programming and JIT compilation
   - The result is only **used** when `done=True`
   - This is **not** a performance issue (JIT optimizes it)

2. **Conditional state selection**
   - `jax.lax.select(done, reset_value, step_value)` chooses based on `done` flag
   - When `done=True`: state becomes `state_re` (from `reset_env()`)
   - When `done=False`: state becomes `state_st` (from `step_env()`)

3. **This mechanism is ALWAYS active**
   - Cannot be disabled without overriding `step()`
   - All environments inheriting from `gymnax.Environment` have this behavior

---

## Impact on MetaEnvironment Implementations

### What Gets Reset

**Whatever `reset_env()` returns is used when `done=True`.**

This means:
- ✅ Fields returned by `reset_env()` **WILL** be reset
- ❌ Fields **NOT** returned by `reset_env()` will be preserved from `step_env()`

### Legacy/Lazy Mode Behavior

**File:** `envs/meta_environment_legacy.py:159-194`, `envs/meta_environment_lazy.py`

```python
def reset_env(self, key: chex.PRNGKey, params: MetaEnvParams):
    env_key, obs_key, index_key = jax.random.split(key, 3)
    env_obs, env_state = self.env.reset_env(env_key, params.env_params)

    # Sample new env_index
    env_index = jax.random.randint(index_key, (), 0, self.total_words)

    state = MetaEnvState(
        env_index=env_index,  # ← NEW value sampled
        obs_words=self.words,
        trial_num=0,
        total_steps=0,
        env_state=env_state,
        init_state=env_state,
        init_obs=env_obs,
    )
    return obs, state
```

**Result:** When meta-episode completes (`done=True`):
- ✅ `env_index` gets a **new random value** (via gymnax auto-reset)
- ✅ `trial_num` resets to 0
- ✅ Base environment state resets
- ✅ `obs_words` stays the same (not resampled, uses `self.words`)

### Separated Mode Behavior

**File:** `envs/meta_environment.py:112-135`

```python
def reset_env(self, key: chex.PRNGKey, params: MetaEnvParams):
    env_key, _, _ = jax.random.split(key, 3)  # RNG compatibility
    env_obs, env_state = self.env.reset_env(env_key, params.env_params)

    state = MetaEnvState(
        trial_num=0,
        total_steps=0,
        env_state=env_state,
        init_state=env_state,
        init_obs=env_obs,
        # NOTE: No env_index field - managed externally
    )
    return obs, state
```

**Result:** When meta-episode completes (`done=True`):
- ❌ `env_index` is **NOT** reset here (doesn't exist in MetaEnvState)
- ✅ `trial_num` resets to 0
- ✅ Base environment state resets
- ❌ `env_index` reset must be handled **externally** (in algorithm code)

---

## Algorithm-Side env_index Management (Separated Mode Only)

### Original Separated Mode

**File:** `algorithms/ppo_in_context.py:201-206`

```python
def maybe_resample_env_index(env_idx, is_done, resample_rng):
    """Resample env_index when meta-episode completes"""
    new_idx = frp_manager.sample_env_index(frp_state.frp_words, resample_rng)
    return jax.lax.select(is_done, new_idx, env_idx)

# After env.step()
env_indices = jax.vmap(maybe_resample_env_index)(env_indices, done, resample_rngs)
```

**Behavior:** env_index resampled **1 time** when `done=True` (algorithm-side only)

---

## Comparison Table

| Mode | env_index in reset_env()? | Algorithm resampling? | Total resampling per meta-episode |
|------|---------------------------|----------------------|-----------------------------------|
| **Legacy** | ✅ Yes (via gymnax auto-reset) | ❌ No | **1 time** |
| **Lazy** | ✅ Yes (via gymnax auto-reset) | ❌ No | **1 time** |
| **Separated** | ❌ No (field doesn't exist) | ✅ Yes | **1 time** |

**Key Insight:** Legacy/Lazy and Separated (original) both resample **once**, but at **different locations**:
- Legacy/Lazy: Inside `reset_env()` (via gymnax auto-reset)
- Separated: Outside environment (in algorithm code)

---

## Testing Best Practices

### ⚠️ CRITICAL: Use `step()` NOT `step_env()`

**WRONG (bypasses gymnax auto-reset):**
```python
obs, state, reward, done, info = env.step_env(rng, state, action, params)
# ❌ This does NOT trigger gymnax auto-reset
# ❌ env_index will NOT be reset in Legacy/Lazy mode
```

**CORRECT (triggers gymnax auto-reset):**
```python
obs, state, reward, done, info = env.step(rng, state, action, params)
# ✅ This triggers gymnax auto-reset when done=True
# ✅ Matches production training loop behavior
```

### When to Use Each Method

**Use `env.step()`:**
- ✅ Integration tests (testing full training loop behavior)
- ✅ Behavioral tests (verifying reset behavior)
- ✅ End-to-end tests
- ✅ Any test that should match production training

**Use `env.step_env()` (rare):**
- Unit tests for `step_env()` method specifically
- Testing environment logic without auto-reset
- **Must document clearly:** "This test uses `step_env()` to bypass auto-reset"

### Test Docstring Template

```python
def test_something():
    """Test description here.

    IMPORTANT: This test uses env.step() to match production training loop behavior,
    which includes gymnax auto-reset on done=True.
    """
    # OR, if using step_env():
    """Test description here.

    NOTE: This test uses env.step_env() to bypass gymnax auto-reset and test
    the step_env() method in isolation. This does NOT match training loop behavior.
    """
```

---

## Common Misconceptions

### ❌ Misconception 1: "Legacy mode doesn't reset env_index"

**Reality:** Legacy mode **DOES** reset env_index via gymnax auto-reset

**Why confusion?** Tests using `step_env()` bypass auto-reset, making it appear fixed

### ❌ Misconception 2: "Separated mode resamples twice"

**Reality:** Separated mode resamples **once** (algorithm-side only)

**Why confusion?** Gymnax auto-reset still runs, but env_index isn't in MetaEnvState, so it has no effect on env_index

---

## RNG Consumption Patterns

### Legacy/Lazy Mode

On `done=True`:
1. `env.step()` splits RNG: `key, key_reset = jax.random.split(key)`
2. Calls `reset_env(key_reset, params)`
3. Inside `reset_env()`: `env_key, obs_key, index_key = jax.random.split(key_reset, 3)`
4. Uses `index_key` to sample new env_index

**RNG usage per meta-episode completion:** 2 splits (step + reset_env)

### Separated Mode (Original)

On `done=True`:
1. `env.step()` splits RNG: `key, key_reset = jax.random.split(key)`
2. Calls `reset_env(key_reset, params)` (but doesn't sample env_index)
3. Algorithm code: `rng, _rng = jax.random.split(rng)`
4. Uses `_rng` to sample new env_index

**RNG usage per meta-episode completion:** 2 splits (step + algorithm)

**Result:** Same RNG consumption, but at different code locations

---

## Debugging Tips

### How to verify gymnax auto-reset is working

```python
# Before done=True
initial_env_index = state.env_index  # (Legacy/Lazy only)

# Call env.step() when done should be True
obs, state, reward, done, info = env.step(rng, state, action, params)

assert done == True, "Meta-episode should be complete"

# After done=True
final_env_index = state.env_index  # (Legacy/Lazy only)

# Verify reset occurred
assert final_env_index != initial_env_index, "env_index should change"
# NOTE: May occasionally fail if same index randomly resampled
```

### How to trace reset calls

Add logging (temporary):
```python
def reset_env(self, key, params):
    print(f"reset_env called with key={key}")  # ← Add this
    # ... rest of implementation
```

---

## References

### Source Code

- `gymnax/environments/environment.py:26-47` - Gymnax auto-reset implementation
- `envs/meta_environment_legacy.py:159-194` - Legacy reset_env() with env_index
- `envs/meta_environment_lazy.py` - Lazy reset_env() with env_index
- `envs/meta_environment.py:112-135` - Separated reset_env() without env_index
- `algorithms/ppo_in_context.py:201-206` - Separated algorithm-side resampling

### Documentation

- `docs/dev/GYMNAX_AUTO_RESET_VERIFICATION_RESULTS.md` - Verification test results
- `docs/dev/NEXT_CONTEXT_ACTION_PLAN.md` - Current understanding and next steps

### Tests

- `tests/test_gymnax_auto_reset_behavior.py` - Verifies auto-reset works (uses `step()`)
- `tests/test_done_signal_distinction.py` - Trial vs meta-episode distinction
- `tests/test_legacy_lazy_meta_episode_reset.py` - Uses `step_env()` (unit tests)

---

**Author:** Gymnax Auto-Reset Documentation
**Date:** 2026-01-01
**Status:** ✅ Reference Documentation
**Intended Audience:** Coding AI, Developers, Future Maintainers
