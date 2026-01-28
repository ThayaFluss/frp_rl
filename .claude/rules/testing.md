# Testing Guidelines

**Last Updated:** 2026-01-28

---

## Testing Philosophy

- Verify dimension correctness (shape mismatches cause runtime errors in JAX)
- Test numerical correctness of FRP transformations
- Verify JIT compilation succeeds

---

## Required Tests for FRP Changes

1. **Dimension Verification** - Analytical verification without requiring JAX
2. **Execution Test** - Compilation and training completion
3. **Configuration Tests** - All FRP scope combinations (if FRP scope changed)
4. **Architecture Tests** - Both S5 and GRU models (if architecture-agnostic)

---

## Common Test Failures

### Shape Mismatch Errors

**Symptom:** Model expects different input dimension than provided

**Diagnosis:**

- Check dimension calculation logic
- Verify FRP input dimension usage
- Review observation reconstruction

### ConcretizationError

**Symptom:** Abstract tracer value where concrete value expected

**Diagnosis:**

- Python control flow on JAX arrays inside JIT
- Dynamic shapes or values
- Missing `static_argnums`

### Dimension Reconstruction Mismatch

**Symptom:** Total observation size doesn't match reconstructed size

**Diagnosis:**

- Missing component in reconstruction
- Incorrect splitting logic
- Verify: total = frp_input + remaining
