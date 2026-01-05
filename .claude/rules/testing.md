# Testing Guidelines

**Last Updated:** 2026-01-05

---

## Testing Philosophy

### Core Principles
- Verify dimension correctness (shape mismatches cause runtime errors)
- Test numerical correctness of transformations
- Ensure components integrate properly
- Verify JIT compilation succeeds

---

## Mandatory Testing Requirements

### When Making Changes to FRP Logic

**Required tests:**
1. **Dimension Verification** - Analytical verification without requiring JAX
2. **Execution Test** - Compilation and training completion
3. **Configuration Tests** - All FRP scope combinations
4. **Architecture Tests** - Both S5 and GRU models

### Test Coverage
- All FRP configurations (if FRP logic changed)
- Both model architectures (if architecture-agnostic change)
- New functionality must have corresponding tests
- Existing tests must continue to pass

---

## Testing Strategy

### Unit Tests
- FRP transformations
- Environment dynamics
- Model forward passes
- Helper functions

### Integration Tests
- Full training loop
- End-to-end dimension flow
- Component interactions

### Regression Tests
- Document historical bugs
- Prevent reoccurrence
- Clear test names explaining the issue

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
- Missing static_argnums

### Dimension Reconstruction Mismatch
**Symptom:** Total observation size doesn't match reconstructed size

**Diagnosis:**
- Missing component in reconstruction
- Incorrect splitting logic
- Verify: total = frp_input + remaining

---

## Pre-commit Checks

Before committing code:
1. Run dimension verification
2. Execute at least one quick test
3. Check code formatting
4. Run type checking (if configured)

---

## Pull Request Checklist

- [ ] Dimension verification passes
- [ ] At least one execution test passes
- [ ] All FRP configurations tested (if relevant)
- [ ] Both architectures tested (if relevant)
- [ ] New tests added for new functionality
- [ ] Existing tests still pass

---

## Performance Testing

### Compilation Time
- Should complete in reasonable time
- Monitor for regression

### Training Speed
- Should scale linearly with problem size
- Check for unexpected slowdowns

---

## Test Organization

### Test File Structure
```
tests/
├── test_frp_manager.py
├── test_meta_environment.py
├── test_ppo_frp_separated.py
└── test_dimensions.py
```

### Test Naming
- Clear, descriptive names
- Indicate what is being tested
- Include issue number for regression tests

---

## Continuous Integration

### CI Pipeline
- Automated testing on push
- Verify all configurations
- Check code quality
- Report failures clearly

### When CI Fails
- Review error messages
- Check recent changes
- Run tests locally
- Fix and push again
