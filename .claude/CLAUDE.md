# FRP-RL Project Guide

**Project:** Free Random Projection for In-Context Reinforcement Learning
**Focus Module:** `frp_popjaxrl/` - JAX-based meta-RL implementation

---

## Project Overview

This repository implements **Free Random Projection (FRP)**, a meta-learning technique for reinforcement learning that uses orthogonal matrix transformations to create diverse task variations while preserving inner products.

### Core Concept

- **Training:** Agent sees observations transformed by randomly selected orthogonal matrices
- **Evaluation:** Agent adapts to new transformations using only hidden state updates (no parameter updates)
- **Key Innovation:** Orthogonal transformations preserve inner products while creating task diversity

---

## Implementation Modes

### SEPARATED Mode (Recommended)

- FRP state managed externally from MetaEnvironment
- Files: `*_separated.py`
- **Use this for all new development**

### LAZY Mode

- FRP state stored in environment, regenerated on-demand
- Files: `*_lazy.py`

### LEGACY Mode (Deprecated)

- Files: `*_legacy.py`
- **Do not use for new features**

---

## Development Environment

- Use `uv run python` instead of bare `python`

---

## Development Workflow

### Before Writing Code

1. Review `docs/dev/` (verify document date and relevance)
2. See `.claude/rules/architecture.md` for component responsibilities

### Making Changes

- Work in SEPARATED mode unless instructed otherwise
- Maintain component boundaries:
  - **FRP Manager:** Transformation logic only
  - **Meta Environment:** Environment dynamics only
  - **PPO Algorithm:** Training orchestration only
  - **Models:** Neural network architectures only

---

## Key Principles

### Dimension Management

**Invariant:** Total observation size = transformed size + non-transformed size

- Always document array shapes in comments
- Validate dimensions at component boundaries

### Gymnax Auto-Reset Behavior

`env.step()` automatically resets when `done=True`. This is expected behavior.

Reference: `docs/dev/GYMNAX_AUTO_RESET_MECHANISM.md`

---

## Evaluating Trained Models

### Scripts

- **run_eval_separated.py** - For separated mode checkpoints
- **run_eval.py** - For legacy/lazy mode checkpoints (requires `--mode` flag)

### Evaluation Methods

- **`tiling`** (recommended): Tiles small observations to fill meta_dim
- **`padding`**: Pads observations with zeros
- **`identity`**: No FRP transformation (for ablation studies)

### Example

```bash
python run_eval_separated.py --checkpoint exp/model.pkl \
    --eval_num_trials 32 --num_episodes 10 --eval_method tiling
```

---

## Useful Commands

```bash
# Run experiment
python run_meta_popgym_separated.py --env cartpole --arch gru --depth 2 --dim 64

# Test dimensions
python test_dimensions.py

# Quick test run
WANDB_MODE=disabled python run_meta_popgym_separated.py \
    --env cartpole --arch gru --debug 1 --num_runs 1 --num_trials 2
```

---

## Additional Resources

- **Paper:** <https://arxiv.org/abs/2504.06983>
- **Dev Docs:** `docs/dev/`
- **Architecture:** `.claude/rules/architecture.md`
- **Code Style:** `.claude/rules/code-style.md`
- **Testing:** `.claude/rules/testing.md`
