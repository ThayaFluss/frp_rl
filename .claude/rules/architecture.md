# Architecture Guidelines

**Last Updated:** 2026-01-05

---

## Core Architectural Principles

### Separation of Concerns

- **FRP Logic:** Orthogonal transformations and matrix operations
- **Environment Logic:** Environment dynamics and episode structure
- **Algorithm Logic:** Training loop and optimization
- **Model Logic:** Neural network architectures

Keep these concerns decoupled and testable independently.

---

## Component Responsibilities

### FRP Manager

**Purpose:** Centralized FRP transformation management

**Responsibilities:**

- Generate orthogonal transformation matrices
- Sample transformation indices
- Transform observations
- Manage configurable scope

**Does NOT:**

- Handle environment dynamics
- Manage episode resets
- Store environment state

### Meta Environment

**Purpose:** Wrap base environments with meta-learning structure

**Responsibilities:**

- Wrap base environments
- Add metadata to observations
- Manage trial structure
- Implement environment interface

**Does NOT:**

- Apply FRP transformations
- Store FRP state
- Handle observation wrappers

### PPO Algorithm

**Purpose:** Orchestrate training with FRP

**Responsibilities:**

- Initialize managers and environments
- Manage FRP state externally
- Apply transformations at appropriate points
- Implement PPO update logic

**Does NOT:**

- Generate orthogonal matrices directly
- Embed FRP logic in environment

### Neural Networks

**Purpose:** Process transformed observations

**Responsibilities:**

- Encode observations
- Maintain recurrent state
- Output policy and value predictions

**Does NOT:**

- Know about FRP transformations
- Handle environment dynamics

---

## Implementation Modes

### SEPARATED Mode (Recommended)

**Philosophy:** Complete separation of FRP from environment

**Characteristics:**

- FRP state managed externally
- Clean component boundaries
- Easy to test independently

### LAZY Mode

**Philosophy:** FRP state in environment, regenerated on-demand

**Characteristics:**

- State co-located with environment
- Simpler algorithm code
- Tighter coupling

### LEGACY Mode (Deprecated)

**Philosophy:** Original implementation with embedded FRP

**Status:** Do not use for new features

---

## Design Patterns

### Immutable State

- Use dataclasses for all state
- Update via `.replace()` method
- No in-place mutations

### Factory Functions

- Use factories for configuration-driven construction
- Centralize initialization logic
- Make dependencies explicit

### Vectorization

- Use `jax.vmap` for batch operations
- Parallelize across environments
- Leverage hardware accelerators

### Static Arguments

- Mark non-array arguments as static in JIT
- Prevent unnecessary recompilation
- Improve performance

### Configuration-Driven Behavior

- Use flags instead of subclasses
- Reduce code duplication
- Test all configurations easily

---

## Dimension Management

### Key Concepts

- **Base observation:** Environment output before FRP
- **FRP input:** What gets transformed
- **FRP output:** Transformed features
- **Reconstructed observation:** Final model input

### Invariant

Total observation size must equal FRP input plus non-transformed parts.

### Best Practices

- Document all array shapes in comments
- Use named variables for dimensions
- Validate dimensions at boundaries
- Provide clear error messages

---

## Extension Points

### Adding New Features

- Identify which component owns the feature
- Maintain separation of concerns
- Add tests for new functionality
- Update relevant documentation

### Common Extensions

- New FRP algorithms
- New environments
- New model architectures
- New training algorithms

---

## Testing Architecture

### Component Testing

- Test each component independently
- Use mocks for dependencies
- Verify interfaces

### Integration Testing

- Test component interactions
- Verify end-to-end flow
- Check all configurations

### Regression Testing

- Document historical issues
- Prevent reoccurrence
- Clear test naming

---

## Performance Considerations

### JIT Compilation

- Compilation happens once per shape
- Avoid dynamic shapes
- Mark static arguments
- Minimize control flow variations

### Memory Management

- Arrays are immutable (copy-on-write)
- Avoid unnecessary operations in loops
- Use in-place updates with `.at[]`

### Vectorization Best Practices

- Always prefer `vmap` over loops
- Enables automatic parallelization
- Better hardware utilization

---

## Documentation Standards

### When to Document

- Non-obvious design decisions
- Component responsibilities
- Interface contracts
- Performance considerations

### Where to Document

- High-level: This file
- Implementation: Inline comments and docstrings
- Specific patterns: CLAUDE.md

### What NOT to Document

- Implementation details that change frequently
- Specific variable names or function signatures
- Code that is self-explanatory

---

## Architectural Evolution

### When Refactoring

- Maintain existing interfaces where possible
- Provide migration path for users
- Update tests before changing implementation
- Document breaking changes

### Adding New Modes

- Follow existing mode patterns
- Maintain consistency with other modes
- Provide clear migration guidance
- Test thoroughly before release
