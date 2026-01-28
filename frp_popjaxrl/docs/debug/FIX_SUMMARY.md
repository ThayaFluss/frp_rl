# RNG Divergence Fix - Quick Summary

## Problem
Legacy and Separated modes produced different training metrics despite identical environment logic.

## Root Cause
**Missing RNG split during initialization** in Separated mode ([ppo_in_context.py:121](algorithms/ppo_in_context.py#L121))

Legacy mode:
```python
rng, _rng = jax.random.split(rng)  # Consumes RNG
train_words = _create_words(_rng)
```

Separated mode (BEFORE):
```python
frp_word_rng = config["META_KWARGS"]["meta_rng"]  # Separate RNG
frp_words = frp_manager.initialize_words(frp_word_rng)
# Main RNG NOT consumed → divergence!
```

## Solution
Added 1 line to match RNG consumption:

```python
rng, _rng = jax.random.split(rng)  # ← Added this line
frp_word_rng = config["META_KWARGS"]["meta_rng"]
frp_words = frp_manager.initialize_words(frp_word_rng)
```

## Verification
✅ All RNG states match perfectly:
- First step RNG: `[1297055832 3528801448]` - ✅ Match
- Eval start: `[118048102 2348238176]` - ✅ Match  
- Update end: `[1343288570 3289136665]` - ✅ Match
- Eval end: `[861276174 4088136895]` - ✅ Match

## Files Changed
- `frp_popjaxrl/algorithms/ppo_in_context.py` - Line 121 (1 line added + comments)

## Status
✅ **FIXED** - RNG determinism fully restored

## Documentation
See [RNG_DIVERGENCE_FIX_REPORT.md](RNG_DIVERGENCE_FIX_REPORT.md) for complete investigation details.
