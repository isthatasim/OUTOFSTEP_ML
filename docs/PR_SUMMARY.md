# PR Summary: Q1 Physics-Aware OOS Upgrade

## What Changed
- Added a new modular package under `src/outofstep_ml/` for data, features, models, evaluation, explainability, deployment, and utilities.
- Added config-driven scripts for training/evaluation/ablation/robustness/figure generation/export.
- Preserved legacy compatibility: existing `main.py` pipeline remains intact.
- Added split-manifest support, calibration comparison helpers, threshold optimization, and dynamic-refinement scaffold.
- Added benchmark docs and updated README for reproducible research workflow.

## Why It Changed
- Move from single-run experiment code to benchmark-grade, paper-ready research infrastructure.
- Improve reproducibility, robustness testing, and deployment readiness expected for Q1 submissions.

## Main Files Added/Modified
- Added: `configs/*`, `scripts/*`, `src/outofstep_ml/*`, `docs/CASE_STUDY_PLAN.md`
- Updated: `README.md`, tests

## How to Run
```bash
python scripts/train_static.py --config configs/model_static_physics.yaml
python scripts/evaluate_static.py --config configs/model_static_physics.yaml
python scripts/run_ablation.py --config configs/base.yaml
python scripts/run_robustness.py --config configs/base.yaml
python scripts/generate_figures.py --config configs/base.yaml
```

## Remaining Limitations
- Dynamic refinement is scaffold-only until transient time-series windows are available.
- Advanced uncertainty quantification can be expanded (e.g., conformal wrappers per regime).
- Additional large-system datasets should be integrated for external validation.
