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

## Implementation Modes

### SEPARATED Mode (Recommended)

- FRP state managed externally from MetaEnvironment
- Cleanest separation of concerns
- Files: `*_separated.py`
- **Use this for all new development**

### LAZY Mode

- FRP state stored in environment, regenerated on-demand
- Files: `*_lazy.py`

### LEGACY Mode (Deprecated)

- Original implementation with embedded FRP
- Files: `*_legacy.py`
- **Do not use for new features**

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
```

---

## Development Workflow

### Before Writing Code

1. **Check documentation:**
   - Review `docs/dev/` for relevant information
   - ⚠️ Note: `docs/dev/` contains both current and historical documents
   - Always verify document date and relevance
   - Key docs: `GYMNAX_AUTO_RESET_MECHANISM.md`, recent `*_SUMMARY.md` files

2. **Search codebase:**

   ```bash
   # Find relevant files
   rg "pattern" frp_popjaxrl/

   # Check existing usage
   rg "function_name" frp_popjaxrl/
   ```

3. **Understand the architecture:**
   - See `.claude/rules/architecture.md` for component responsibilities
   - Identify which component your change belongs to

### Making Changes

**Architecture Decisions:**

- Work in SEPARATED mode unless instructed otherwise
- Keep FRP logic separate from environment logic
- Keep algorithm logic separate from model logic
- Maintain clear component boundaries

**Separation of Concerns:**

- **FRP Manager:** Transformation logic only
- **Meta Environment:** Environment dynamics only
- **PPO Algorithm:** Training orchestration only
- **Models:** Neural network architectures only

### Testing Requirements

**Mandatory tests when changing FRP or dimension logic:**

```bash
# 1. Dimension verification (analytical)
python test_dimensions.py

# 2. Execution test (compilation + training)
WANDB_MODE=disabled python run_meta_popgym_separated.py \
    --env cartpole --arch gru --debug 1 --num_runs 1 --num_trials 2

# 3. Test configurations (if FRP scope changed)
# - Default (raw obs only)
# - With metadata
# - With wrapper
# - All components

# 4. Test both architectures (if architecture-agnostic)
# - GRU
# - S5
```

### Documentation Requirements

**Update documentation when:**

- Changing component responsibilities
- Adding new parameters or configurations
- Modifying mode implementations
- Making architectural changes

**Where to document:**

- Inline comments for complex logic
- Docstrings for all public functions
- `docs/dev/` for significant changes

---

## Key Principles

### Dimension Management

**Critical concept:** Total observation size must equal transformed size plus non-transformed size.

**Best practices:**

- Always document array shapes in comments
- Use named variables for dimensions
- Validate dimensions at component boundaries
- Provide clear error messages with context

### JAX-Specific Requirements

- Pure functions only (no side effects)
- Immutable data structures (`flax.struct.dataclass`)
- Use `jax.lax.cond` instead of Python `if` inside JIT
- Use `jax.vmap` instead of Python loops
- Mark non-array arguments as static in JIT

### Code Quality

**Linting:**

- Run linters when changes are complete
- Not required for every small change
- **Required before PR submission**

```bash
# Run when your changes are ready
flake8 frp_popjaxrl/ --max-line-length=120 --ignore=E203,W503
pylint frp_popjaxrl/
```

---

## Common Issues

### Dimension Mismatches

**Symptom:** Shape errors during compilation or training

**Diagnosis:**

- Check dimension calculation in training setup
- Verify FRP input dimension usage
- Review observation reconstruction logic

### Gymnax Auto-Reset Behavior

**Symptom:** Environment resets unexpectedly in tests

**Explanation:** This is expected Gymnax behavior. `env.step()` automatically resets when `done=True`.

**Reference:** See `docs/dev/GYMNAX_AUTO_RESET_MECHANISM.md`

### Mode Confusion

**Symptom:** Mixing code from different modes

**Solution:** Always use SEPARATED mode files. Never import from LAZY or LEGACY modes.

### JAX JIT Errors (ConcretizationError)

**Symptom:** Abstract tracer value where concrete value expected

**Causes:**

- Python control flow on JAX arrays inside JIT
- Dynamic shapes or values
- Missing `static_argnums`

**Solution:** Use `jax.lax.cond` and mark static arguments appropriately.

---

## Evaluating Trained Models

### Overview

Two evaluation scripts are available depending on the training mode:

- **run_eval.py** - For legacy/lazy mode checkpoints
- **run_eval_separated.py** - For separated mode checkpoints

Both scripts:

- Load saved model checkpoints
- Evaluate over multiple episodes
- Track per-trial statistics (returns, steps, success rates)
- Generate plots (PNG) and data (CSV)
- Support WandB logging

### Legacy/Lazy Mode Evaluation

Use `run_eval.py` with explicit `--mode` flag:

```bash
# Legacy mode
python run_eval.py --checkpoint exp/20260106_190721/model_31_iter.pkl \
    --mode legacy \
    --eval_num_trials 32 \
    --num_episodes 10 \
    --eval_method tiling

# Lazy mode
python run_eval.py --checkpoint exp/20260106_190803/model_31_iter.pkl \
    --mode lazy \
    --eval_num_trials 32 \
    --num_episodes 10 \
    --eval_method padding
```

**Required arguments:**

- `--checkpoint`: Path to checkpoint file (.pkl)
- `--mode`: Implementation mode (`legacy` or `lazy`)

**Optional arguments:**

- `--eval_num_trials`: Number of trials per episode (default: 16)
- `--num_episodes`: Number of episodes to evaluate (default: 10)
- `--eval_method`: Evaluation method - `tiling`, `padding`, or `identity` (default: `tiling`)
- `--seed`: Random seed for evaluation (default: 0)
- `--log_wandb`: WandB project name (default: "popgym_eval", empty string disables)

### Separated Mode Evaluation

Use `run_eval_separated.py`:

```bash
python run_eval_separated.py --checkpoint exp/20260106_122337/model_31_iter.pkl \
    --eval_num_trials 32 \
    --num_episodes 10 \
    --eval_method tiling
```

**Required arguments:**

- `--checkpoint`: Path to checkpoint file (.pkl)

**Optional arguments:**

- `--eval_num_trials`: Number of trials per episode (default: 16)
- `--num_episodes`: Number of episodes to evaluate (default: 10)
- `--eval_method`: Evaluation method - `tiling`, `padding`, or `identity` (default: `tiling`)
- `--seed`: Random seed (default: None, uses checkpoint's eval_seed)
- `--log_wandb`: WandB project name (default: "popgym_eval_separated", empty string disables)

### Evaluation Methods

Three evaluation methods control how observations are transformed:

- **`tiling`** (recommended): Tiles small observations to fill meta_dim
  - Best for generalizing to different observation sizes
  - Repeats observation pattern to match training dimensions

- **`padding`**: Pads observations with zeros
  - Simpler approach, may affect performance
  - Adds zeros to match training dimensions

- **`identity`**: Uses identity transformation (no augmentation)
  - Evaluates without FRP transformation
  - Useful for ablation studies

### Evaluation Outputs

All evaluation scripts produce:

1. **Console output**: Per-episode and per-trial statistics

   ```text
   Trial | Mean Return | Std Return | Mean Steps | Std Steps | Success Rate
   ----------------------------------------------------------------------
       0 |        0.18 |       0.04 |       35.5 |       7.5 |        0.00%
       1 |        0.08 |       0.00 |       15.5 |       0.5 |        0.00%
   ```

2. **PNG plots**: Three subplots showing returns, steps, and success rates across trials with ±1 std bands
   - Saved as `<checkpoint>_trial_stats_n<trials>.png` (legacy/lazy)
   - Saved as `<checkpoint>_trial_stats_n<trials>_separated.png` (separated)

3. **CSV data**: Trial-wise statistics for further analysis
   - Saved as `<checkpoint>_trial_stats_n<trials>.csv` (legacy/lazy)
   - Saved as `<checkpoint>_trial_stats_n<trials>_separated.csv` (separated)

4. **WandB logs** (optional): Per-trial metrics logged with trial number as x-axis

### Error Handling

**Wrong script for checkpoint mode:**

```bash
# This will fail with a helpful error message
python run_eval.py --checkpoint <separated_checkpoint> --mode legacy
# Error: Checkpoint appears to be from SEPARATED mode (contains EVAL_SEED).
# Use run_eval_separated.py instead
```

### Examples

```bash
# Quick evaluation (2 trials, 2 episodes) for testing
python run_eval_separated.py --checkpoint exp/20260106_122337/model_31_iter.pkl \
    --eval_num_trials 2 --num_episodes 2 --log_wandb ""

# Full evaluation with WandB logging
python run_eval_separated.py --checkpoint exp/20260106_122337/model_31_iter.pkl \
    --eval_num_trials 32 --num_episodes 100 --eval_method tiling

# Evaluation with different methods for comparison
for method in tiling padding identity; do
    python run_eval_separated.py --checkpoint exp/model.pkl \
        --eval_method $method --log_wandb ""
done
```

---

## Useful Commands

```bash
# Run experiment
python run_meta_popgym_separated.py --env cartpole --arch gru --depth 2 --dim 64

# Test dimensions
python test_dimensions.py

# Find TODOs
rg "TODO|FIXME|HACK" frp_popjaxrl/

# Check for deprecated legacy code
rg "legacy|Legacy|LEGACY" frp_popjaxrl/

# List available environments
rg "elif env_name ==" frp_popjaxrl/envs/

# Profile with full traceback
JAX_TRACEBACK_FILTERING=off python run_meta_popgym_separated.py ...
```

---

## Git Workflow

**Branch Naming:**

- Feature: `feature/<description>`
- Fix: `fix/<description>`
- Claude branches: `claude/<auto-generated>`

**Commit Messages:**

```text
<type>: <short summary>

<detailed description>

<optional: test results, breaking changes>
```

**Pull Requests:**

- Target `dev` branch (not `main`)
- Include test results in PR description
- Document any breaking changes

---

## Additional Resources

- **Paper:** <https://arxiv.org/abs/2504.06983>
- **Main README:** `/README.md`
- **Module README:** `frp_popjaxrl/README.md`
- **Dev Docs:** `frp_popjaxrl/docs/dev/`
- **Architecture Guidelines:** `.claude/rules/architecture.md`
- **Code Style:** `.claude/rules/code-style.md`
- **Testing Guidelines:** `.claude/rules/testing.md`

---

## Getting Help

**If stuck:**

1. Search codebase: `rg "relevant_term" frp_popjaxrl/`
2. Check `docs/dev/` (verify document is current)
3. Review relevant `.claude/rules/` file
4. Check git history: `git log --oneline -- <file>`

**Common questions:**

- "Why is environment resetting?" → Read `GYMNAX_AUTO_RESET_MECHANISM.md`
- "Dimension mismatch error?" → Check component dimension calculations
- "Which mode to use?" → Always SEPARATED
- "Which files to modify?" → See architecture.md for component responsibilities
