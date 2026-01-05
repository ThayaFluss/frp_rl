# Code Style Guidelines

**Last Updated:** 2026-01-05

---

## General Principles

### Pure Functional Programming
- All JAX functions must be pure (no side effects)
- No global state modifications
- Predictable behavior for same inputs

### Immutable Data Structures
- Use `flax.struct.dataclass` for all stateful data
- Never modify arrays in-place
- Update state using `.replace()` method

### Type Hints Required
- All public functions must have type hints
- Parameters and return values must be annotated
- Use `chex.Array` for JAX arrays

---

## Documentation Standards

### Docstrings
- Required for all public functions and classes
- Include Args, Returns, and Raises sections
- Add Examples for non-trivial functions

### Inline Comments
- Annotate all array shapes in comments
- Explain non-obvious logic
- Document JAX-specific patterns
- Note performance considerations

---

## JAX-Specific Requirements

### Control Flow
- Use `jax.lax.cond` instead of Python `if` inside JIT
- Use `jax.lax.select` for element-wise conditionals
- Mark non-array arguments as static in JIT

### Vectorization
- Prefer `jax.vmap` over Python loops
- Avoid dynamic shapes that trigger recompilation

---

## Naming Conventions

### Style
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: prefix with `_`

### Clarity
- Use descriptive names
- Avoid abbreviations unless standard
- Be consistent within the codebase

---

## Code Organization

### File Structure
```python
# Standard library
# Third-party libraries (alphabetical)
# Local imports (relative)

# Constants
# Dataclasses
# Classes
# Functions
```

### Line Length
- Target: 100 characters
- Flexible to 120 for readability

---

## Error Handling

- Provide actionable error messages
- Include context and expected values
- Suggest fixes when possible

---

## Performance

- Avoid recompilation by keeping shapes static
- Use appropriate JAX transformations (jit, vmap)
- Minimize unnecessary array operations

---

## Code Review Checklist

- [ ] Type hints on all public functions
- [ ] Docstrings present and complete
- [ ] Array shapes documented in comments
- [ ] No Python control flow on JAX arrays inside JIT
- [ ] Immutable data structures used
- [ ] Error messages are clear and actionable
