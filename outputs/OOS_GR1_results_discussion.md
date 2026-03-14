# OOS GR1 Results & Discussion

Generated: 2026-03-14 15:34:40 UTC

## Executive Summary
- Best composite performer across evaluated Step x Validation runs: **B2 (Gradient Boosting)**.
- Recommended deployment model: **C3 Two-stage hybrid** with calibration **none**.
- Recommended thresholds: tau_F1=0.7596153846153846, tau_HR=0.999, tau_cost=0.147.
- Operational emphasis: minimize missed instability (FNR) while preserving calibration quality (ECE/Brier).

## Tier-by-Tier Comparison
### Tier A (interpretability-first)
- Mean PR-AUC: 0.7920
- Mean FNR: 0.0623
- Mean ECE: 0.0388
- Mean CompositeScore: 0.8753

### Tier B (accuracy-first)
- Mean PR-AUC: 0.7762
- Mean FNR: 0.0001
- Mean ECE: 0.0282
- Mean CompositeScore: 0.8954

### Tier C (physics-aware/hybrid)
- Mean PR-AUC: 0.5641
- Mean FNR: 0.0740
- Mean ECE: 0.0645
- Mean CompositeScore: 0.7657

## Scenario Discussion (Step 1 -> Step 5)
### Step 1: Static OOS prediction
Key leaderboard and calibration artifacts:
- [ART-0006] leaderboard_step1_static_V1_stratified  (code: `main.py::_evaluate_step_protocol`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\leaderboard_step1_static_V1_stratified.csv`)
- [ART-0008] leaderboard_step1_static_V2_grouped  (code: `main.py::_evaluate_step_protocol`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\leaderboard_step1_static_V2_grouped.csv`)
- [ART-0010] leaderboard_step1_static_V3_leave_Sgn  (code: `main.py::_evaluate_step_protocol`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\leaderboard_step1_static_V3_leave_Sgn.csv`)
- [ART-0012] leaderboard_step1_static_V3_leave_Ik  (code: `main.py::_evaluate_step_protocol`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\leaderboard_step1_static_V3_leave_Ik.csv`)
- [ART-0005] calibration_comparison_B2  (code: `main.py::_build_tier_models`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\calibration_comparison_B2.csv`)

Interpretation:
- Step 1 establishes baseline separability and threshold policy under offline conditions.

### Step 2: Robustness on operating maps and unseen levels
Key robustness artifacts:
- [ART-0022] leaderboard_step2_robustness_V1_stratified  (code: `main.py::run_pipeline`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\leaderboard_step2_robustness_V1_stratified.csv`)
- [ART-0024] leaderboard_step2_robustness_V2_grouped  (code: `main.py::run_pipeline`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\leaderboard_step2_robustness_V2_grouped.csv`)
- [ART-0026] leaderboard_step2_robustness_V3_leave_Sgn  (code: `main.py::run_pipeline`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\leaderboard_step2_robustness_V3_leave_Sgn.csv`)
- [ART-0028] leaderboard_step2_robustness_V3_leave_Ik  (code: `main.py::run_pipeline`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\leaderboard_step2_robustness_V3_leave_Ik.csv`)
- [ART-0023] delta_step_step2_robustness_V1_stratified  (code: `main.py::_delta_vs_baseline`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\delta_step_step2_robustness_V1_stratified.csv`)
- [ART-0025] delta_step_step2_robustness_V2_grouped  (code: `main.py::_delta_vs_baseline`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\delta_step_step2_robustness_V2_grouped.csv`)
- [ART-0027] delta_step_step2_robustness_V3_leave_Sgn  (code: `main.py::_delta_vs_baseline`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\delta_step_step2_robustness_V3_leave_Sgn.csv`)
- [ART-0029] delta_step_step2_robustness_V3_leave_Ik  (code: `main.py::_delta_vs_baseline`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\delta_step_step2_robustness_V3_leave_Ik.csv`)

Interpretation:
- Grouped and leave-level protocols test generalization under unseen operating regimes.

### Step 3: Measurement noise and uncertainty realism
Key noise artifacts:
- [ART-0014] leaderboard_step3_noise_V1_stratified  (code: `main.py::_evaluate_step_protocol`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\leaderboard_step3_noise_V1_stratified.csv`)
- [ART-0016] leaderboard_step3_noise_V2_grouped  (code: `main.py::_evaluate_step_protocol`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\leaderboard_step3_noise_V2_grouped.csv`)
- [ART-0018] leaderboard_step3_noise_V3_leave_Sgn  (code: `main.py::_evaluate_step_protocol`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\leaderboard_step3_noise_V3_leave_Sgn.csv`)
- [ART-0020] leaderboard_step3_noise_V3_leave_Ik  (code: `main.py::_evaluate_step_protocol`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\leaderboard_step3_noise_V3_leave_Ik.csv`)
- [ART-0038] noise_robustness_best_ABC  (code: `main.py::run_pipeline`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\noise_robustness_best_ABC.csv`)
- [ART-0039] noise_robustness_prauc  (code: `src/plots.py::plot_noise_robustness`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\figures\noise_robustness_prauc.png`)
- [ART-0040] noise_robustness_prauc  (code: `src/plots.py::plot_noise_robustness`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\figures\noise_robustness_prauc.pdf`)
- [ART-0041] noise_robustness_fnr  (code: `src/plots.py::plot_noise_robustness`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\figures\noise_robustness_fnr.png`)
- [ART-0042] noise_robustness_fnr  (code: `src/plots.py::plot_noise_robustness`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\figures\noise_robustness_fnr.pdf`)

Interpretation:
- Robustness curves quantify degradation with increased input uncertainty.

### Step 4: Deployment prototype
Key deployment artifacts:
- [ART-0061] model_card  (code: `main.py::_model_card`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\model\model_card.md`)
- [ART-0062] deployment_metrics  (code: `main.py::run_pipeline`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\tables\deployment_metrics.csv`)
- [ART-0063] api_api_app  (code: `main.py::_copy_api_files`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\api\api_app.py`)
- [ART-0064] api_requirements  (code: `main.py::_copy_api_files`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\api\requirements.txt`)
- [ART-0065] api_curl_example  (code: `main.py::_copy_api_files`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\api\curl_example.txt`)

### Step 5: Monitoring, drift, and retraining policy
Key monitoring artifacts:
- [ART-0070] psi  (code: `src/monitoring.py`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\monitoring\psi.csv`)
- [ART-0074] retrain_policy  (code: `src/monitoring.py`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\monitoring\retrain_policy.json`)

Interpretation:
- Retraining is triggered only when drift alarms coincide with sufficient new data and challenger superiority criteria.

## Physics Plausibility Evidence
- Partial dependence and boundary comparison artifacts:
- [ART-0050] pdp_main_features  (code: `src/plots.py::plot_pdp`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\figures\pdp_main_features.png`)
- [ART-0051] pdp_main_features  (code: `src/plots.py::plot_pdp`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\figures\pdp_main_features.pdf`)
- [ART-0052] boundary_comparison  (code: `src/plots.py::plot_boundary_comparison`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\figures\boundary_comparison.png`)
- [ART-0053] boundary_comparison  (code: `src/plots.py::plot_boundary_comparison`; file: `C:\Users\masim\OneDrive\Desktop\Out of step ML\outputs\figures\boundary_comparison.pdf`)

## Deployment Recommendation
- Model: **C3 Two-stage hybrid**
- Calibration: **none**
- Threshold policy: tau_cost=0.147, tau_F1=0.7596153846153846, tau_HR=0.999
- Rationale: composite score ranking with explicit penalty on FNR and poor calibration.

## Failure Modes and Limitations
- Static descriptors approximate dynamic transient behavior; extrapolation outside modeled operating envelope requires caution.
- Leave-level-out gaps indicate potential boundary shift risk under unseen stress/strength combinations.
- Periodic retraining and threshold re-validation are required after drift alerts.

## Traceability to Code and Artifacts
The complete registry is stored in `outputs/results_manifest.json`. Every table/figure above is linked via artifact IDs and `code_path` references.
