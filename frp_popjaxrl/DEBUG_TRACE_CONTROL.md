# DEBUG_TRACE Control - Usage Guide

## Summary

DEBUG_TRACEフラグを`--debug`コマンドライン引数で制御できるようになりました。

## Usage

### デフォルト実行（トレースなし）

```bash
python run_meta_popgym.py --mode separated --env cartpole --seed 42
# または明示的に
python run_meta_popgym.py --mode separated --env cartpole --seed 42 --debug 0
```

**結果:** RNGトレース出力は表示されません（パフォーマンス向上）

### デバッグトレース有効化

```bash
python run_meta_popgym.py --mode separated --env cartpole --seed 42 --debug 2
```

**結果:** RNGトレース出力が表示されます
```
[SEPARATED] Step start RNG: [1297055832 3528801448], env_indices: [250 121  58 ...]
[SEPARATED] action=[1 1 0 0 1 1 1 1], last_done=[False False ...]
[SEPARATED] done=[False False ...], reward=[0.005 0.005 ...]
[SEPARATED] Eval start RNG: [118048102 2348238176]
[SEPARATED] Eval end RNG: [861276174 4088136895]
```

## Implementation Details

### Modified Files

**`run_meta_popgym.py`** (Lines 112, 146)

変更前:
```python
"DEBUG_TRACE": True,  # 常に有効
```

変更後:
```python
"DEBUG_TRACE": (args.debug >= 2),  # --debug 2以上で有効
```

### Debug Level Semantics

| `--debug` Level | DEBUG_TRACE | 用途 |
|-----------------|-------------|------|
| 0 (default) | False | 通常実行（トレースなし） |
| 1 | False | 一般的なデバッグ情報 |
| 2+ | True | RNGトレース有効（詳細デバッグ） |

## Benefits

1. **パフォーマンス向上**: デフォルトでトレースコールバックが無効化され、不要なオーバーヘッドを削減
2. **柔軟性**: コマンドライン引数で簡単に制御可能
3. **下位互換性**: 既存のコードは変更不要（デフォルトでトレース無効）

## Related Changes

- RNG divergence fix: [RNG_DIVERGENCE_FIX_REPORT.md](RNG_DIVERGENCE_FIX_REPORT.md)
- Debug callbacks location: 
  - `algorithms/ppo_in_context.py` (Lines 182-183, 195-197, 242-244, 378-379, 412-413, 454-456)
  - `algorithms/ppo_in_context_legacy.py` (Lines 144-145, 157-158, 168-169, 268-269, 283-284, 322-323)

## Testing

```bash
# テスト1: デフォルト（トレースなし）
python run_meta_popgym.py --mode separated --env cartpole --seed 42 --log_wandb disabled 2>&1 | grep "SEPARATED"
# 出力: なし（成功）

# テスト2: トレース有効
python run_meta_popgym.py --mode separated --env cartpole --seed 42 --debug 2 --log_wandb disabled 2>&1 | grep "Step start RNG"
# 出力: [SEPARATED] Step start RNG: [...] （成功）
```

## Date

2026-01-02
