# Repository Audit for Full Benchmark Upgrade

## 1) Current folder structure (pre-upgrade snapshot)
- Top-level executable legacy pipeline: `main.py`
- Legacy modules: `src/data.py`, `src/features.py`, `src/models.py`, `src/eval.py`, `src/plots.py`, `src/monitoring.py`, `src/retrain.py`, reporting modules
- Existing outputs and artifacts under `outputs/`
- Tests under `tests/`
- Newly added modular layer from previous phase under `src/outofstep_ml/`, `scripts/`, `configs/`, `docs/`

## 2) Current model code
- Legacy ladder includes:
  - Logistic baseline
  - Decision-tree baseline
  - Random forest
  - GBM backend (XGBoost/LightGBM/HistGB fallback)
  - Monotonic constrained GBM
  - Physics-regularized logistic model
  - Two-stage hybrid model
- Existing API and monitoring paths already available in legacy modules.

## 3) Current data format
- Static/tabular operating-point format.
- Core expected fields: `Tag_rate`, `Ikssmin_kA`, `Sgn_eff_MVA`, `H_s`, `Out_of_step` (`GenName` optional).
- Extra columns are tolerated and standardized in legacy loader.

## 4) Current train/eval scripts
- Legacy monolithic run: `main.py`
- Modular scripts: `scripts/train_static.py`, `scripts/evaluate_static.py`, `scripts/run_ablation.py`, `scripts/run_robustness.py`, `scripts/generate_figures.py`, `scripts/export_model.py`

## 5) Current outputs and saved artifacts
- Legacy: `outputs/` (model artifacts, figures, tables, docs, monitoring)
- New modular path: `results/` with `tables/`, `figures/`, `model/`, `splits/`

## 6) Time-series/PMU availability
- No real transient sequence/PMU-window dataset is part of the current repository workflow.
- Temporal baseline should remain scaffold-only unless true sequential data is added.

## 7) README quality
- README already significantly improved in prior phase.
- Additional benchmark-specific sections were still required for full model-ladder comparison and strict evaluation protocol.

## 8) Reusable components
- Robust legacy data standardization and cleaning
- Feature engineering primitives
- CV evaluation and thresholding primitives
- Monitoring primitives (PSI/KS)
- Existing physics-aware and hybrid models

## 9) Gaps addressed in this benchmark upgrade
- Full benchmark ladder with strict holdout protocol
- Model availability tracking and optional advanced baselines
- Train/val/test strict usage enforcement
- Split manifest and seed persistence
- Publication-table generation (Tables 1-8 + mathematical table)
- Benchmark report docs and budget policy docs
