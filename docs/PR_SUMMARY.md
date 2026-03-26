# PR Summary: Full Benchmark + Strict Evaluation Upgrade

## What Changed
- Added a benchmark runner stack under `src/outofstep_ml/benchmark/` with:
  - common model zoo,
  - strict train/validation/final-holdout protocol,
  - raw results export,
  - publication-table generation.
- Added benchmark scripts:
  - `scripts/train_model.py`
  - `scripts/run_full_benchmark.py`
  - `scripts/generate_tables.py`
- Added benchmark configs for model families and end-to-end execution:
  - `configs/logreg.yaml`, `configs/svm.yaml`, `configs/random_forest.yaml`, `configs/xgboost_or_lightgbm.yaml`, `configs/catboost_baseline.yaml`, `configs/fttransformer.yaml`, `configs/tabnet.yaml`, `configs/proposed_physics_model.yaml`, `configs/full_benchmark.yaml`, `configs/full_benchmark_smoke.yaml`.
- Added benchmark documentation:
  - `docs/REPO_AUDIT_BENCHMARK_UPGRADE.md`
  - `docs/TRAINING_BUDGET_POLICY.md`
  - `docs/MODEL_COMPARISON_REPORT.md`
- Updated `README.md` to document full benchmark commands and strict evaluation policy.

## Strict Evaluation Protocol (Implemented)
1. Train / Validation / Final Holdout Test split.
2. Model fitting uses Train only.
3. Hyperparameter tuning, calibration, threshold selection, and early-stopping decisions use Validation only.
4. Final holdout Test is used only for final unbiased comparison.
5. Robustness experiments use shifted test sets without any tuning on shifted data.
6. Split indices and seeds are saved for reproducibility.

Artifacts:
- `results/splits/strict_split_seed*_*.csv`
- `results/splits/strict_split_seed*_*.npz`
- `results/splits/split_manifest.csv`
- `results/splits/strict_evaluation_protocol.json`

## Why It Changed
- To move from a single-experiment pipeline to a benchmark-grade, publication-ready framework.
- To make model comparisons fair, reproducible, and audit-ready for Q1 journal expectations.

## How to Run
```bash
python scripts/run_full_benchmark.py --config configs/full_benchmark.yaml
python scripts/generate_figures.py --mode benchmark --config configs/full_benchmark.yaml
```

## Smoke Validation
- `python scripts/run_full_benchmark.py --config configs/full_benchmark_smoke.yaml`
- `pytest -q`

## Remaining Limitations
- FT-Transformer and TabNet are optional and dependency-gated.
- Temporal baselines remain scaffold-only unless real transient sequence data is available.
