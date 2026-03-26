# Phase 1 Audit: Static OOS Q1-Grade Validation Upgrade

## Scope Guard (Phase 1)
- This document is **audit + plan only**.
- No major refactors, no new training execution, no benchmark-wide rewrites in this phase.
- Static-risk only; no fabricated time-series data.

## 1) Concise Repository Audit Summary

### Repository structure (current)
- Legacy pipeline:
  - `main.py` (monolithic end-to-end run)
  - `src/data.py`, `src/features.py`, `src/models.py`, `src/eval.py`, `src/plots.py`, `src/report_problem.py`, `src/report_results.py`, `src/monitoring.py`, `src/retrain.py`, `src/api_app.py`
- Modular package layer:
  - `src/outofstep_ml/data/*`
  - `src/outofstep_ml/features/*`
  - `src/outofstep_ml/models/*`
  - `src/outofstep_ml/evaluation/*`
  - `src/outofstep_ml/explainability/*`
  - `src/outofstep_ml/deployment/*`
  - `src/outofstep_ml/benchmark/*`
- Script entry points:
  - `scripts/train_static.py`
  - `scripts/evaluate_static.py`
  - `scripts/run_ablation.py`
  - `scripts/run_robustness.py`
  - `scripts/run_full_benchmark.py`
  - `scripts/train_model.py`
  - `scripts/generate_tables.py`
  - `scripts/generate_figures.py`

### Current training / evaluation entry points
- `scripts/run_full_benchmark.py` is the strongest current comparison runner.
- `src/outofstep_ml/benchmark/runner.py` enforces strict train/val/test usage and writes split manifests.
- `main.py` still supports broad scenario orchestration and manifest generation, but mixes many responsibilities.
- `scripts/train_static.py` and `scripts/evaluate_static.py` remain useful for deployable single-model flows.

### Data loading / preprocessing flow
- Canonicalization and robust schema handling are in `src/data.py` (legacy).
- Modular loader (`src/outofstep_ml/data/loaders.py`) wraps legacy loader and feature construction.
- Engineered features currently used:
  - `invH`, `Sgn_over_H`, `Sgn_over_Ik`, `Ik_over_H` (+ optional logs)
- Missing handling and constant/duplicate checks are already implemented.

### Current model stack
- Baselines and physics-aware models already exist:
  - Tier A: logistic, shallow tree
  - Tier B: RF, GBM fallback stack
  - Tier C: monotonic HGB, physics-regularized logistic, two-stage hybrid
- Advanced benchmark extras:
  - SVM (RBF), optional CatBoost, optional TabNet, FT-Transformer scaffold.

### Current metrics and outputs
- Metrics include PR-AUC, ROC-AUC, Precision, Recall, F1, Specificity, Balanced Accuracy, FNR, Brier, ECE, MAE, RMSE.
- Threshold policy table includes `tau_default`, `tau_F1`, `tau_cost`, `tau_HR`.
- Calibration table and robustness table are already produced in benchmark mode.
- Strict protocol artifacts are saved (`strict_split_*.csv/.npz`, `split_manifest.csv`, `strict_evaluation_protocol.json`).

### Current saved artifacts
- Long-run benchmark outputs already available under:
  - `outputs/benchmark_long/tables/`
  - `outputs/benchmark_long/figures/`
  - `outputs/benchmark_long/splits/`

### Existing training evidence (already executed)
- From `outputs/benchmark_long/tables/best_method_summary.json`:
  - Best pure predictive: `SVM (RBF)`
  - Best calibrated: `SVM (RBF)`
  - Best deployable: `Proposed Physics-Aware Model`
  - Best overall recommended: `Proposed Physics-Aware Model`

## 2) Gaps vs requested Q1-grade static validation framework
- Requested 10 named validation scenarios are only partially formalized as dedicated scenario modules.
- Low-inertia, weak-grid, and stress-loading migrations are not yet first-class scenario runners with controlled levels and dedicated outputs.
- Cross-interaction heatmaps over `(H, I, S)` are available via plotting primitives but not standardized in benchmark scenario export.
- Explicit monotonic consistency checks (quantitative violation rates) are not yet exported as a formal table.
- Imbalance-aware ablation exists partially, but should be promoted to explicit no-weight vs class-weight vs current-approach comparison protocol.
- Counterfactual evaluation exists, but needs scenario-level structured reporting and tie-in to strict holdout policy.

## 3) Key ambiguities / risks affecting implementation
- Dual architecture (legacy `src/*.py` + modular `src/outofstep_ml/*`) creates risk of duplicated logic and drift.
- Some scripts use strict holdout protocol, while some legacy CV paths do not enforce single final holdout in the same way.
- Missing-data stress currently masks one feature to NaN, but explicit imputation policy comparison is not yet scenarioized.
- Dependency variability (CatBoost/TabNet/FT-Transformer optional) can create non-uniform benchmark availability across environments.

## 4) Suggested implementation branch name (Phase 2)
- `codex/phase2-static-oos-q1-validation`

