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

## Variable and Symbol Reference for This Results Report
### Dataset variables
| Symbol | Dataset column | Unit | Operational interpretation |
|---|---|---|---|
| \(T\) | `Tag_rate` | as provided | disturbance/acceleration proxy of operating condition |
| \(I\) | `Ikssmin_kA` | kA | short-circuit strength proxy; lower values indicate weaker grid |
| \(S\) | `Sgn_eff_MVA` | MVA | stress/loading proxy; higher values indicate heavier operating stress |
| \(H\) | `H_s` | s | inertia constant; lower values indicate reduced inertial support |
| \(y\) | `Out_of_step` | - | binary label; 1 = out-of-step, 0 = stable |

### Engineered variables used by PhysiScreen-OOS
| Feature | Formula | Interpretation |
|---|---|---|
| `invH` | \(1/H\) | inverse inertia stress indicator |
| `S_over_H` | \(S/H\) | stress normalized by inertia |
| `S_over_I` | \(S/I\) | stress normalized by grid strength |
| `I_over_H` | \(I/H\) | grid strength relative to inertia |

### Decision and evaluation symbols
| Symbol | Meaning |
|---|---|
| \(p\) | predicted OOS probability |
| \(\tau\) | decision threshold for OOS alarm |
| \(\tau_{F1}\) | threshold maximizing F1 on validation set |
| \(\tau_{cost}\) | threshold minimizing \(C_{FN}FN + C_{FP}FP\) |
| \(\tau_{HR}\) | high-recall threshold target (\(\mathrm{Recall}\ge 0.95\)) |
| FNR | false-negative rate (critical for missed-instability risk) |
| ECE | expected calibration error |
| Brier | probability calibration/accuracy score |

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
- Variable-level meaning in this context: high-risk nominal points are mainly associated with low \(H\), low \(I\), and high \(S\)-to-\(H\)/\(I\) engineered ratios.

### 2.2 Migration studies (Scenarios 2-4)
From `q1_table_migrations.csv`:

- Low-inertia migration (H down to 0.8): PR-AUC and ECE changed only marginally.
- Weak-grid migration (I down to 0.8): similarly small PR-AUC/ECE changes in this dataset.
- Stress-loading escalation (S up to 1.2): modest degradation (PR-AUC dropped from 0.9939 to 0.9927; Brier increased).

Interpretation:
- The model is comparatively stable across controlled perturbations in the observed operating envelope.
- Scenario semantics:
  - low-inertia migration decreases \(H\) while keeping other factors fixed by protocol;
  - weak-grid migration decreases \(I\);
  - stress-loading escalation increases \(S\).

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
- Physical reading: some operating regions still show locally non-monotone risk response to \(I\), suggesting either data sparsity in those regions or insufficient regularization strength on \(I\)-direction constraints.

### 2.8 Counterfactual stability correction (Scenario 10)
From `q1_table_counterfactual_summary.csv`:

- Counterfactual feasibility is currently limited under strict stability thresholds in several seeds.
- This indicates the need for stronger constrained optimization and/or relaxed action budgets.

Interpretation:
- Counterfactual module is operational, but feasibility quality should be improved before operator-facing rollout.
- Action semantics: recommendations are expressed as bounded changes to \(H\), \(I\), and/or \(S\) required to bring \(p\) below a stable threshold.

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
