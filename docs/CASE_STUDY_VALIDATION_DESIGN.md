# Static OOS Case-Study Validation Design (Phase 1)

## Scope
- Static operating-point risk screening for OOS (no dynamic/PMU fabrication).
- Feature set:
  - Raw: `T=Tag_rate`, `I=Ikssmin_kA`, `S=Sgn_eff_MVA`, `H=H_s`
  - Engineered: `1/H`, `S/H`, `S/I`, `I/H`

## Strict evaluation protocol (must remain invariant)
1. Train / Validation / Final holdout Test split.
2. Fit model parameters on Train only.
3. Use Validation only for:
   - hyperparameter tuning
   - calibration fitting
   - threshold selection
   - early stopping decisions
4. Use final holdout Test only once for final unbiased reporting.
5. No tuning on final holdout test.
6. Save split indices and seeds.

Note: this is already enforced in `src/outofstep_ml/benchmark/runner.py` and evidenced by:
- `outputs/benchmark_long/splits/strict_evaluation_protocol.json`
- `outputs/benchmark_long/splits/split_manifest.csv`

---

## Scenario design for the 10 requested validations

## 1) Nominal baseline
- **Purpose:** establish clean-condition reference.
- **Implementation:** evaluate all supported models on strict holdout under nominal inputs.
- **Current status:** already available via `run_full_benchmark` clean shift and table outputs.

## 2) Low-inertia migration
- **Purpose:** simulate inertia reduction trend.
- **Implementation plan:** apply controlled multiplicative levels to `H` (e.g., 100%, 90%, 80%, 70% or quantile regime shift) on test-only copies; evaluate risk shift, calibration drift, threshold stability.
- **Outputs:** migration curve table + figure (`risk vs H_level`) per model.

## 3) Weak-grid migration
- **Purpose:** simulate weaker short-circuit strength.
- **Implementation plan:** controlled reductions on `I` with fixed protocol and test-only perturbation.
- **Outputs:** `risk vs I_level` curves and delta-to-nominal metrics.

## 4) Stress-loading escalation
- **Purpose:** simulate increased loading/stress.
- **Implementation plan:** controlled increases on `S` with test-only perturbation levels.
- **Outputs:** `risk vs S_level` curves and threshold sensitivity.

## 5) Cross-interaction heatmaps over H, I, S
- **Purpose:** visualize nonlinear boundaries and interactions.
- **Implementation plan:** generate 2D heatmaps for `(H,I)`, `(H,S)`, `(I,S)` with remaining variables fixed at median or stratified by regime.
- **Outputs:** publication heatmaps + boundary contours at selected `tau`.

## 6) Regime-shift validation (grouped + leave-one-level-out)
- **Purpose:** evaluate out-of-regime generalization.
- **Implementation plan:** grouped splits on rounded operating bins; leave-one-level-out over `S` bins and `I` bins.
- **Current status:** already implemented in strict runner (`grouped`, `leave_S`, `leave_I`).

## 7) Noise and missing-data robustness
- **Purpose:** assess measurement uncertainty and data quality resilience.
- **Implementation plan:** additive feature noise at configured levels; masked feature tests with fixed imputation policy.
- **Current status:** implemented as shifted tests (`noisy`, `missing_features`) in strict benchmark.

## 8) Imbalance-aware ablation
- **Purpose:** quantify effect of imbalance handling strategy.
- **Implementation plan:** compare at least:
  - no weighting
  - class weighting
  - current imbalance-aware method (existing weighted/physics models)
- **Outputs:** ablation table with PR-AUC/FNR/ECE deltas.

## 9) Cost-sensitive threshold policies
- **Purpose:** operational trade-off between misses and nuisance trips.
- **Implementation plan:** compare `tau_default`, `tau_F1`, `tau_cost`, `tau_HR` with confusion/cost metrics.
- **Current status:** threshold comparison table exists (`table2_threshold_comparison.csv`); extend to scenario-level reporting.

## 10) Counterfactual stability correction
- **Purpose:** actionable adjustments to reduce OOS risk.
- **Implementation plan:** for high-risk points, solve minimal-change adjustments in `H`, `I`, `S` to move below stable threshold.
- **Current status:** counterfactual utility exists; needs scenario-integrated summary KPIs and protocol-aware exports.

---

## How this leverages completed training so far
- Existing long-run benchmark (`outputs/benchmark_long`) provides a strong baseline for:
  - nominal comparison,
  - strict protocol evidence,
  - current robustness and threshold tables.
- These outputs are reused as baseline references for scenario deltas rather than retraining from scratch in Phase 1.

---

## Assumptions that must remain explicit
1. Dataset is static operating-point data; no sequential dynamics are inferred.
2. Disturbance context is implicit in labels, not explicit in features.
3. Calibration and thresholds are validation-fitted only.
4. Regime-shift tests are proxy shifts unless external datasets are added.
5. Optional models (CatBoost/TabNet/FT-Transformer) are dependency-gated.

---

## Risks to existing functionality
1. Introducing scenario modules may duplicate existing logic if not carefully wrapped.
2. Changing output schemas can break existing report scripts.
3. Mixing legacy and modular pipelines can create inconsistent defaults.
4. Overwriting existing table names can break downstream comparisons.

Mitigation:
- Add new scenario-specific filenames.
- Keep backward-compatible CLI and current outputs intact.
- Add smoke tests for new scripts and schema tests for new tables.

