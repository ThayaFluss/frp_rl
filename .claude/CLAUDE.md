# FRP-RL Project Guide for Claude

**Project:** Free Random Projection for In-Context Reinforcement Learning
**Focus Module:** `frp_popjaxrl/` - JAX-based meta-RL implementation
**Last Updated:** 2026-01-05

---

## Project Overview

This repository implements **Free Random Projection (FRP)**, a meta-learning technique for reinforcement learning that uses orthogonal matrix transformations to create diverse task variations while preserving inner products. This enables in-context adaptation without parameter updates.

### Core Concept: In-Context Learning via FRP

- **Training:** Agent sees observations transformed by randomly selected orthogonal matrices
- **Evaluation:** Agent adapts to new transformations using only hidden state updates (no parameter updates)
- **Key Innovation:** Orthogonal transformations preserve inner products while creating task diversity

---

## Architecture Overview

### Core Components

```
frp_popjaxrl/
├── frp/                          # FRP transformation core
│   ├── frp_manager.py           # FRPManager: centralized FRP logic (SEPARATED mode)
│   ├── orthogonal.py            # Core orthogonal matrix generation (SEPARATED mode)
│   ├── orthogonal_lazy.py       # Lazy mode implementation
│   ├── orthogonal_legacy.py     # Legacy mode implementation
│   └── orthogonal_v2.py         # Alternative FRP algorithm
├── envs/                         # Meta-environment framework
│   ├── meta_environment_separated.py  # SEPARATED mode (recommended)
│   ├── meta_environment_lazy.py       # LAZY mode
│   ├── meta_environment_legacy.py     # LEGACY mode (deprecated)
│   ├── wrappers.py              # Observation wrappers (AliasPrevActionV2)
│   └── environments/            # POPopGym environments
├── algorithms/                   # PPO implementations
│   ├── ppo_frp_separated.py     # SEPARATED mode (recommended)
│   ├── ppo_in_context_lazy.py   # LAZY mode
│   ├── ppo_in_context_legacy.py # LEGACY mode (deprecated)
│   ├── models.py                # S5/GRU network architectures
│   └── ppo_common.py            # Shared PPO utilities
├── run_meta_popgym_separated.py # Main entry point (SEPARATED mode)
└── docs/dev/                    # Development documentation
```

### Three Implementation Modes

1. **SEPARATED (Recommended):** FRP state managed externally from MetaEnvironment
   - Files: `*_separated.py`, `frp_manager.py`
   - Cleanest separation of concerns
   - Used in current development

2. **LAZY:** FRP state stored in MetaEnvState, regenerated on-demand
   - Files: `*_lazy.py`
   - Lazy word regeneration

3. **LEGACY (Deprecated):** Original implementation with FRP embedded in environment
   - Files: `*_legacy.py`
   - Do not use for new features

**⚠️ CRITICAL:** Always use SEPARATED mode files for new development unless explicitly instructed otherwise.

---

## Key Technical Concepts

### 1. FRP Transformation Pipeline

```
Raw Observation (input_dim)
    ↓
+ Metadata (3D: action_value, env_done, reset_flag)
    ↓
MetaEnvironment Output
    ↓
FRP Transformation (orthogonal matrix multiplication)
    ↓
+ AliasPrevActionV2 Wrapper (previous action + reset flag)
    ↓
Final Observation → S5/GRU Model
```

### 2. Observation Structure (SEPARATED mode)

**From MetaEnvironment:**
```
obs = [raw_obs (input_dim)] + [metadata (3)]
```

**After FRP Transformation:**
```
obs = [transformed_features] + [non_transformed_features]
```
- `transformed_features`: FRP applied (configurable: raw_obs, metadata, wrapper)
- `non_transformed_features`: Pass-through components

**After AliasPrevActionV2 Wrapper:**
```
obs = [frp_transformed] + [wrapper_data]
```
- Discrete actions: wrapper adds `(n_actions + 1)` dimensions (one-hot + reset)
- Continuous actions: wrapper adds `2` dimensions (action + reset)

### 3. FRP Configuration Parameters

**Core FRP Settings:**
- `meta_depth`: Depth of word tree (controls granularity) - typically 1-3
- `meta_dim`: Output dimension of FRP transformation - typically 64-128
- `meta_max_depth`: Maximum depth for parallel words (controls # of words: 2^max_depth) - typically 8-12
- `meta_with_adjoint`: Include transpose matrices (True/False)
- `meta_truncate_aug`: Truncate augmentation to input_dim (0/1)

**NEW: Configurable FRP Scope (v2026-01):**
- `frp_include_metadata`: Apply FRP to metadata (action, done, reset) - default: False
- `frp_include_wrapper`: Apply FRP to wrapper data (prev action) - default: False
- Raw observation is **always** included in FRP input

### 4. Dimension Calculations (CRITICAL)

**Understanding input_dim:**
- `input_dim` = raw environment observation dimension (e.g., 4 for CartPole)
- Does NOT include metadata or wrapper

**FRP input dimension calculation:**
```python
frp_input_dim = input_dim
if include_metadata:
    frp_input_dim += metadata_dim  # +3
if include_wrapper:
    frp_input_dim += wrapper_dim   # +n_actions+1 (discrete) or +2 (continuous)
```

**Observation size after FRP:**
```python
base_obs_size = input_dim + 3 + wrapper_dim  # Total before FRP
non_transformed_size = base_obs_size - frp_input_dim
frp_transformed_obs_size = aug_output_dim + non_transformed_size
```

**⚠️ When modifying FRP scope:** Always update both transformation logic AND dimension calculations in `make_train()`.

---

## Development Environment

### Python Environment Setup

**Package Management:**
- Use `uv` for dependency management with `pyproject.toml`
- Virtual environment: `.venv` (managed by `uv`)

**Setup commands:**
```bash
# Create and sync environment
uv sync

# Activate environment
source .venv/bin/activate

# Install/update dependencies
uv sync
```

---

## Development Guidelines

### Making Changes to frp_popjaxrl

#### 1. Before Writing Code

**Required Reading:**
- Review `GYMNAX_AUTO_RESET_MECHANISM.md` if touching environments

**Investigation First:**
```bash
# Find relevant files
rg "pattern" frp_popjaxrl/

# Check dimension usage
rg "input_dim|obs_size|frp_input_dim" frp_popjaxrl/

```

#### 2. Architecture Decisions

**When adding new features:**
1. Work in SEPARATED mode unless instructed otherwise
2. Keep FRP logic in `frp_manager.py`
3. Keep environment logic in `meta_environment_separated.py`
4. Keep algorithm logic in `ppo_frp_separated.py`

**Separation of Concerns:**
- **FRPManager:** Orthogonal matrix generation, transformation logic
- **MetaEnvironment:** Environment dynamics, metadata addition
- **PPO Algorithm:** Training loop, FRP state management
- **Models:** Neural network architectures (S5/GRU)

#### 3. Testing Requirements

**Dimension Verification (MANDATORY):**
```python
# Always verify dimensions after changes
python test_dimensions.py

# Run actual execution test
WANDB_MODE=disabled python run_meta_popgym_separated.py \
    --env cartpole --arch gru --debug 1 --num_runs 1 --num_trials 2
```

**Test all FRP configurations:**
```bash
# Default (raw obs only)
python run_meta_popgym_separated.py --env cartpole --arch gru

# With metadata
python run_meta_popgym_separated.py --env cartpole --arch gru --frp_include_metadata 1

# With wrapper
python run_meta_popgym_separated.py --env cartpole --arch gru --frp_include_wrapper 1

# All components
python run_meta_popgym_separated.py --env cartpole --arch gru \
    --frp_include_metadata 1 --frp_include_wrapper 1
```

#### 4. Documentation Requirements

**Always update documentation when:**
- Changing FRP behavior
- Adding new parameters
- Modifying dimension calculations
- Changing mode implementations

**Create/update files in:**
- `docs/dev/` - Technical documentation
- Inline comments for complex JAX operations
- Docstrings for all public functions

---

## Common Pitfalls

### 1. Dimension Mismatches
**Problem:** Shape errors like `(134, 128) but got (128, 128)`

**Cause:** `frp_transformed_obs_size` calculation doesn't match actual transformation

**Fix:**
```python
# WRONG (old logic)
metadata_and_wrapper_size = base_env_obs_size - frp_manager.input_dim
frp_transformed_obs_size = frp_manager.aug_output_dim + metadata_and_wrapper_size

# CORRECT (new logic)
frp_input_dim = frp_manager.frp_input_dim  # Accounts for include_metadata/wrapper
non_transformed_size = base_env_obs_size - frp_input_dim
frp_transformed_obs_size = frp_manager.aug_output_dim + non_transformed_size
```

### 2. Gymnax Auto-Reset Confusion
**Problem:** Environment resets unexpectedly in tests

**Cause:** `env.step()` automatically resets when `done=True` (Gymnax built-in)

**Fix:** This is expected behavior. Read `GYMNAX_AUTO_RESET_MECHANISM.md` for details.

### 3. Mode Confusion
**Problem:** Mixing SEPARATED/LAZY/LEGACY code

**Fix:** Always use SEPARATED mode files. Never import from other modes.

### 4. JAX JIT Violations
**Problem:** `ConcretizationTypeError` or slow compilation

**Cause:** Non-static values in JIT-compiled functions

**Fix:**
- Use `static_argnums` for non-array arguments
- Avoid Python control flow on JAX arrays
- Use `jax.lax.cond` instead of `if` statements on arrays

---

## Code Style Guidelines

See `.claude/rules/code-style.md` for detailed guidelines.

**Key Points:**
- Pure functional style (JAX requirement)
- Immutable data structures (use `flax.struct.dataclass`)
- Explicit dimension comments
- Type hints for all public functions

### Code Quality Checks

**Linting:**
- Run linters (pylint, flake8) when changes are complete
- Not required for every small change
- **Required before PR submission**

```bash
# Run when your changes are ready
flake8 frp_popjaxrl/ --max-line-length=120 --ignore=E203,W503
pylint frp_popjaxrl/
```

---

## Testing Strategy

See `.claude/rules/testing.md` for detailed guidelines.

**Mandatory Tests:**
1. Dimension verification (analytical)
2. Execution test (compilation + training)
3. All FRP configurations
4. Both architectures (S5 and GRU)

---

## File Organization Patterns

### When Creating New Features

**Pattern 1: New FRP variant**
```
frp/
├── orthogonal_<variant>.py          # New FRP algorithm
├── frp_manager.py                   # Update factory functions
└── tests/test_orthogonal_<variant>.py
```

**Pattern 2: New meta-environment feature**
```
envs/
├── meta_environment_separated.py    # Update main class
└── tests/test_meta_environment_separated.py
```

**Pattern 3: New algorithm feature**
```
algorithms/
├── ppo_frp_separated.py             # Update training loop
└── tests/test_ppo_frp_separated.py
```

---

## Useful Commands

```bash
# Run experiment
python run_meta_popgym_separated.py --env cartpole --arch gru --depth 2 --dim 64

# Test dimensions
python test_dimensions.py

# Find TODOs in code
rg "TODO|FIXME|HACK" frp_popjaxrl/

# Check for deprecated legacy code
rg "legacy|Legacy|LEGACY" frp_popjaxrl/

# List available environments
rg "elif env_name ==" frp_popjaxrl/envs/meta_environment_factory.py

# Profile compilation
JAX_TRACEBACK_FILTERING=off python run_meta_popgym_separated.py ...
```

---

## Git Workflow

**Branch Naming:**
- Feature: `feature/<description>`
- Fix: `fix/<description>`
- Claude branches: `claude/<auto-generated>`

**Commit Messages:**
```
<type>: <short summary>

<detailed description>

<optional: test results, breaking changes>
```

**PR to `dev` branch** (not `main`)

---

## Additional Resources

- **Paper:** https://arxiv.org/abs/2504.06983
- **Main README:** `/README.md`
- **Module README:** `frp_popjaxrl/README.md`
- **Dev Docs:** `frp_popjaxrl/docs/dev/`
- **Gymnax Docs:** https://github.com/RobertTLange/gymnax

---

## Getting Help

**If stuck:**
1. Check `frp_popjaxrl/docs/` for existing documentation
2. Search codebase: `rg "relevant_term" frp_popjaxrl/`
3. Read Gymnax auto-reset docs if environment-related
4. Check git history: `git log --oneline -- <file>`

**Common questions:**
- "Why is environment resetting?" → Read `GYMNAX_AUTO_RESET_MECHANISM.md`
- "Dimension mismatch error?" → Check FRP input dim calculation
- "Which mode to use?" → Always SEPARATED
- "Where to add FRP logic?" → `frp_manager.py`
