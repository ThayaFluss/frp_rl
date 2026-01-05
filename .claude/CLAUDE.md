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
```
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

- **Paper:** https://arxiv.org/abs/2504.06983
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
