# OOS GR1 Results & Discussion (Updated)

Generated: 2026-03-26 UTC

## Executive Summary
- Training and validation were run with an **extra-long budget** using strict train/validation/final-holdout protocol.
- Proposed model name used in this report: **PhysiScreen-OOS** (former label: `Proposed Physics-Aware Model`).
- Base model name used in this report: **Logit-Base** (former label: `Logistic Regression`).
- Comparison set: **SVM-RBF**, **RF-Base**, **Boost-GBM**, **Legacy-Hybrid**.

Main outcomes from `outputs/static_q1_validation_xlong`:
- Best pure predictive model (benchmark table): **SVM-RBF**.
- Best deployable and overall recommended model: **PhysiScreen-OOS**.
- In scenario-specific static-Q1 validation, PhysiScreen-OOS achieved very high nominal discrimination and calibration quality.

## 1. Benchmark-Level Comparison (xlong)
From `table1_main_performance.csv`:

- **SVM-RBF**: PR-AUC = 0.9593, FNR = 0.0734, ECE = 0.0396
- **PhysiScreen-OOS (Proposed)**: PR-AUC = 0.9497, FNR = 0.1137, ECE = 0.0424
- **Logit-Base**: PR-AUC = 0.9285, FNR = 0.0745, ECE = 0.0413
- **RF-Base**: PR-AUC = 0.9432, FNR = 0.0714, ECE = 0.0406
- **Boost-GBM**: PR-AUC = 0.9332, FNR = 0.0714, ECE = 0.0401
- **Legacy-Hybrid**: PR-AUC = 0.9432, FNR = 0.0714, ECE = 0.0406

Interpretation:
- SVM-RBF remains strongest by raw PR-AUC in the benchmark ladder.
- The proposed model remains selected as deployment recommendation due combined calibration/robustness/actionability priorities.

## 2. Scenario Results for Proposed Model (PhysiScreen-OOS)
### 2.1 Nominal baseline (Scenario 1)
From `q1_table_nominal_baseline.csv`:

- PR-AUC = 0.9939
- ROC-AUC = 0.9991
- Precision = 0.9178
- Recall = 0.9926
- F1 = 0.9530
- FNR = 0.0074
- ECE = 0.0018
- Brier = 0.0039

Interpretation:
- Very strong nominal separability and excellent calibration.

### 2.2 Migration studies (Scenarios 2-4)
From `q1_table_migrations.csv`:

- Low-inertia migration (H down to 0.8): PR-AUC and ECE changed only marginally.
- Weak-grid migration (I down to 0.8): similarly small PR-AUC/ECE changes in this dataset.
- Stress-loading escalation (S up to 1.2): modest degradation (PR-AUC dropped from 0.9939 to 0.9927; Brier increased).

Interpretation:
- The model is comparatively stable across controlled perturbations in the observed operating envelope.

### 2.3 Regime-shift validation (Scenario 6)
From `q1_table_regime_shift.csv`:

- grouped split: PR-AUC = 0.9955
- leave-I split: PR-AUC = 0.9921
- leave-S split: PR-AUC = 0.7177

Interpretation:
- Generalization is strong for grouped and leave-I regimes.
- Large degradation in leave-S indicates sensitivity when stress regimes are truly unseen.

### 2.4 Noise and missing data robustness (Scenario 7)
From `q1_table_robustness.csv`:

- clean: PR-AUC = 0.9939
- noisy: PR-AUC = 0.9939
- missing_features: PR-AUC = 0.9912
- group_shift: PR-AUC = 0.9889
- unseen_regime: PR-AUC = 0.9747

Interpretation:
- Robust to mild additive noise and simple masking.
- Performance decreases under stronger regime shift but remains high overall.

### 2.5 Imbalance-aware ablation (Scenario 8)
From `q1_table_imbalance_ablation.csv`:

- Compare no-weight vs class-weight vs existing proposed approach.
- Weighted/proposed configurations preserve high recall and low FNR under OOS-critical cost settings.

Interpretation:
- Imbalance handling remains necessary for stability-protection use cases.

### 2.6 Threshold policy comparison (Scenario 9)
From `q1_table_threshold_policies.csv`:

- `tau_cost` yields lowest expected cost (with higher recall and higher false alarms).
- `tau_default` and `tau_F1` improve precision but increase expected miss cost under `C_FN >> C_FP`.

Operationally recommended policy:
- Use `tau_cost` for conservative protection screening.
- Use `tau_F1` for balanced offline study comparison.

### 2.7 Monotonic consistency (physics plausibility)
From `q1_table_monotonic_consistency.csv`:

- H direction check (risk should not increase with H): violation rate = 0.0
- S direction check (risk should not decrease with S): violation rate = 0.0
- I direction check (risk should not increase with I): violation rate = 0.7143

Interpretation:
- H and S monotonic behavior is consistent.
- I monotonic prior is only partially respected; this is a key improvement target.

### 2.8 Counterfactual stability correction (Scenario 10)
From `q1_table_counterfactual_summary.csv`:

- Counterfactual feasibility is currently limited under strict stability thresholds in several seeds.
- This indicates the need for stronger constrained optimization and/or relaxed action budgets.

Interpretation:
- Counterfactual module is operational, but feasibility quality should be improved before operator-facing rollout.

## 3. Final Model Recommendation
### Proposed deployment model
- **PhysiScreen-OOS (Proposed Physics-Aware Model)**

### Why
- Best deployment profile in this repository’s multi-criteria view.
- Excellent nominal scenario performance and calibration.
- Strong robustness in grouped/leave-I/noise/missing settings.
- Clear threshold-policy behavior for risk control.

### When to use comparison models
- **SVM-RBF**: best raw predictive benchmark target.
- **Logit-Base**: transparent baseline reference.
- **RF-Base / Boost-GBM / Legacy-Hybrid**: practical baselines for stress testing and reproducibility.

## 4. Remaining Limitations
- Leave-S regime shift exposes a significant generalization gap.
- Monotonic consistency for I requires improvement.
- Counterfactual feasibility requires stronger optimization constraints and action-space handling.

## 5. Traceability
Primary files used for this update:
- `outputs/static_q1_validation_xlong/tables/table1_main_performance.csv`
- `outputs/static_q1_validation_xlong/tables/q1_table_nominal_baseline.csv`
- `outputs/static_q1_validation_xlong/tables/q1_table_migrations.csv`
- `outputs/static_q1_validation_xlong/tables/q1_table_regime_shift.csv`
- `outputs/static_q1_validation_xlong/tables/q1_table_robustness.csv`
- `outputs/static_q1_validation_xlong/tables/q1_table_imbalance_ablation.csv`
- `outputs/static_q1_validation_xlong/tables/q1_table_threshold_policies.csv`
- `outputs/static_q1_validation_xlong/tables/q1_table_monotonic_consistency.csv`
- `outputs/static_q1_validation_xlong/tables/q1_table_counterfactual_summary.csv`
- `outputs/static_q1_validation_xlong/tables/best_method_summary.json`
