# Lazy Evaluation Implementation for FRP

## 概要

Free Random Projection (FRP)の**lazy evaluation**実装により、**メモリ使用量を最大96.9%削減**しました。

## 主な成果

- ✅ **メモリ削減**: State sizeを67.8%〜96.9%削減（Nに依存）
- ✅ **コンパイル時間**: v1とほぼ同等（わずかに高速化）
- ✅ **正確性**: v1と同一の出力を保証（誤差 < 1e-8）

## 使い方

### 基本的な実行

```bash
# Lazy版を使用
python run_meta_popgym.py --env cartpole --arch gru --mode lazy

# V1版を使用（比較用）
python run_meta_popgym.py --env cartpole --arch gru --mode v1
```

### パラメータ

```bash
--depth 4          # FRP depth（デフォルト: 4）
--max_depth 8      # 最大depth、word数 = 2^max_depth（デフォルト: 8）
--dim 128          # 出力次元（デフォルト: 128）
--mode lazy        # lazy または v1（デフォルト: v1）
```

## テスト

```bash
# 単体テスト
python -m pytest tests/frp/test_orthogonal_lazy.py -v

# 統合テスト（v1との一致確認）
python -m pytest tests/integration/test_lazy_vs_v1.py -v

# ベンチマーク
python tests/benchmarks/compare_compile_time.py
```

## 実装の詳細

詳細は [lazy_implementation_summary.md](./lazy_implementation_summary.md) を参照してください。

## ファイル構成

```
frp/
  orthogonal_lazy.py              # Lazy評価のコア実装
envs/
  meta_environment_lazy.py        # Lazy評価版環境
algorithms/
  ppo_gru_in_context_lazy.py      # GRU版PPO
  ppo_s5_in_context_lazy.py       # S5版PPO
tests/
  frp/test_orthogonal_lazy.py     # 単体テスト
  integration/test_lazy_vs_v1.py  # 統合テスト
  benchmarks/compare_compile_time.py  # ベンチマーク
docs/
  lazy_implementation_summary.md  # 実装詳細
```
