# Code Style Guidelines

**Last Updated:** 2026-01-28

---

## Documentation Principles for `.claude/`

- Write abstract, change-resistant principles rather than implementation details
- Omit rules that are obvious to Claude Code (standard Python conventions, etc.)
- Include repository-specific notable points (JAX patterns, dimension handling, etc.)

---

## JAX/Flax Principles

### Pure Functional Programming

- All JAX functions must be pure (no side effects)
- Use `flax.struct.dataclass` for stateful data
- Update state via `.replace()` method, never modify in-place

### Control Flow in JIT

- Use `jax.lax.cond` instead of Python `if` on JAX arrays
- Use `jax.lax.select` for element-wise conditionals
- Mark non-array arguments as `static_argnums`

### Performance

- Avoid dynamic shapes that trigger recompilation
- Prefer `jax.vmap` over Python loops

---

## Documentation

### Array Shape Annotations

- Annotate all array shapes in comments (e.g., `# (batch, seq, dim)`)
- Document dimension transformations at function boundaries

---

## Markdown Formatting

- Add blank lines after headings and around lists/code blocks
- Use unique heading names within a document
