# Architecture Guidelines

**Last Updated:** 2026-01-28

---

## Core Principle: Separation of Concerns

- **FRP Logic:** Orthogonal transformations and matrix operations
- **Environment Logic:** Environment dynamics and episode structure
- **Algorithm Logic:** Training loop and optimization
- **Model Logic:** Neural network architectures

Keep these concerns decoupled and testable independently.

---

## Component Boundaries

### FRP Manager

Centralized FRP transformation management.

**Does NOT:** Handle environment dynamics, manage episode resets, store environment state

### Meta Environment

Wrap base environments with meta-learning structure.

**Does NOT:** Apply FRP transformations, store FRP state

### PPO Algorithm

Orchestrate training with FRP.

**Does NOT:** Generate orthogonal matrices directly, embed FRP logic in environment

### Neural Networks

Process transformed observations.

**Does NOT:** Know about FRP transformations, handle environment dynamics

---

## Implementation Modes

### SEPARATED Mode (Recommended)

- FRP state managed externally from environment
- Clean component boundaries
- **Use this for all new development**

### LAZY Mode

- FRP state in environment, regenerated on-demand
- Tighter coupling

### LEGACY Mode (Deprecated)

- Do not use for new features

---

## Dimension Management

### Invariant

Total observation size = FRP input + non-transformed parts

### Key Concepts

- **Base observation:** Environment output before FRP
- **FRP input:** What gets transformed
- **FRP output:** Transformed features
- **Reconstructed observation:** Final model input

---

## Extension Points

When adding new features:

- Identify which component owns the feature
- Maintain separation of concerns
- Consider: new FRP algorithms, environments, model architectures, training algorithms
