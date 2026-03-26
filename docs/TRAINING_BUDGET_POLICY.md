# Training Budget Policy

This benchmark uses **strict and fair** training budgets across model families.

## Core strict protocol
1. Split into Train / Validation / Final Holdout Test.
2. Fit model parameters on Train only.
3. Use Validation only for hyperparameter tuning, calibration fitting, threshold selection, and early-stopping decisions.
4. Use Final Test once for unbiased final comparison.
5. Robustness is measured on shifted test sets only (no training/tuning on shifted sets).
6. Save split indices and seeds for reproducibility.

## Configurable budget knobs
- `training_budget.n_trials`
- `training_budget.max_epochs`
- `training_budget.patience`
- `training_budget.early_stopping`
- `training_budget.batch_size`
- `training_budget.learning_rate`
- `benchmark.n_seeds`
- `strict_eval.test_size`
- `strict_eval.val_size`

## Family-specific policy
- Classical baselines (LR/SVM/RF): randomized search over compact but competitive spaces.
- Boosting baselines (XGBoost/LightGBM/CatBoost): stronger randomized search where supported.
- Deep tabular baselines (FT-Transformer/TabNet): optional; run only when dependencies are available, with early stopping budget.
- Proposed physics-aware model: equal-or-stronger tuning budget while maintaining fairness.

## Fairness safeguards
- Same split strategy and seed schedule for all available models.
- Same target metric for tuning (`PR-AUC` on validation).
- Same threshold policy definitions (`tau_default`, `tau_F1`, `tau_cost`, `tau_HR`).
- No hyperparameter/calibrator/threshold tuning on final test.
