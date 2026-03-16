# OOS Prediction for GR1: Physics-aware, ML-based, Deployable Framework

Generated: 2026-03-16 15:01:48 UTC

## 1. Problem Statement (Applied Energy framing)
Develop a practical and deployable out-of-step (OOS) predictor for **GR1** that supports screening, calibrated risk, decision-support counterfactuals, and production monitoring.

## 2. Data Definition
For sample $i=1,\dots,N$:

$$
T^{(i)}:=\text{Tag rate at sample }i,\quad
I^{(i)}:=\text{Ikssmin (kA) at sample }i,\quad
S^{(i)}:=\text{Sgn eff (MVA) at sample }i,\quad
H^{(i)}:=\text{inertia value at sample }i
$$

$$
x^{(i)}=[T^{(i)},\ I^{(i)},\ S^{(i)},\ H^{(i)}]
$$

$$
y^{(i)}\in\{0,1\},\quad y^{(i)}=1\ \text{means out-of-step}
$$

Physics-motivated engineered features:

$$
z^{(i,1)}=\frac{1}{H^{(i)}}\ (\text{invH}),\quad
z^{(i,2)}=\frac{S^{(i)}}{H^{(i)}}\ (\text{Sgn over H}),\quad
z^{(i,3)}=\frac{S^{(i)}}{I^{(i)}}\ (\text{Sgn over Ik}),\quad
z^{(i,4)}=\frac{I^{(i)}}{H^{(i)}}\ (\text{Ik over H})
$$

## 3. Physics Background
Swing equation perspective:

$$
\dot{\delta}=\omega
$$

$$
M\dot{\omega}=P_m-P_e(\delta,V,\text{network})-D\omega
$$

$$
M=\frac{2H}{\omega_{\text{sync}}}
$$

OOS corresponds to loss of synchronism after disturbance. This static dataset is used to learn a surrogate stability boundary.

## 4. Mathematical Formulation
### 4.1 Probabilistic model
$$
p^{(i)}=f_\theta(x^{(i)})\approx P\!\left(y^{(i)}=1\mid x^{(i)}\right)
$$

$$
\hat{y}^{(i)}(\tau)=\mathbb{1}\!\left[p^{(i)}\ge\tau\right]
$$

### 4.2 Imbalance-aware loss
$$
\mathcal{L}^{\mathrm{CE}}(\theta)=
-\sum_{i=1}^{N}\left[w^{+}y^{(i)}\log p^{(i)}+w^{-}(1-y^{(i)})\log(1-p^{(i)})\right]
$$

Optional focal alternative:

$$
\mathcal{L}_{Focal}(\theta)=
-\sum_{i=1}^{N}\left[\alpha y^{(i)}(1-p^{(i)})^\gamma\log p^{(i)}+(1-\alpha)(1-y^{(i)})(p^{(i)})^\gamma\log(1-p^{(i)})\right]
$$

### 4.3 Physics-informed soft constraints
Monotonic priors:
$$
\frac{\partial f}{\partial H}\le 0,\qquad
\frac{\partial f}{\partial I}\le 0,\qquad
\frac{\partial f}{\partial S}\ge 0
$$

Finite-difference penalty:
$$
\mathcal{R}^{\mathrm{phys}}(\theta)=
\lambda^{H}\mathbb{E}\left[\max\left(0,\frac{\Delta f}{\Delta H}\right)\right]
+\lambda^{I}\mathbb{E}\left[\max\left(0,\frac{\Delta f}{\Delta I}\right)\right]
+\lambda^{S}\mathbb{E}\left[\max\left(0,-\frac{\Delta f}{\Delta S}\right)\right]
$$

Total objective:
$$
\min_{\theta}\ \mathcal{L}(\theta)=\mathcal{L}^{\mathrm{CE}}(\theta)+\mathcal{R}^{\mathrm{phys}}(\theta)
$$

### 4.4 Cost-sensitive thresholding
$$
\tau^*=\arg\min_{\tau}\left(C^{FN}\,FN(\tau)+C^{FP}\,FP(\tau)\right),
\qquad C^{FN}\gg C^{FP}
$$

Also report:

$$
\tau^{F1}=\arg\max_{\tau}F_1(\tau),\qquad
\tau^{HR}:\ \mathrm{Recall}(\tau)\ge 0.95
$$

### 4.5 Counterfactual support
$$
\min_{\Delta x}\|W\Delta x\|_1
\quad\text{s.t.}\quad
f_\theta(x+\Delta x)\le\tau^{\text{stable}},
\quad
x^{\text{min}}\le x+\Delta x\le x^{\text{max}}
$$

## 5. Workflow
Data ingestion -> audit/cleaning -> feature engineering -> split protocols -> Tier A/B/C models -> calibration + thresholds -> scenario Steps 1..5 -> explainability/maps -> counterfactual support -> API/tests -> monitoring/drift/retraining -> exports + manifest.

## 6. Validation Protocols
- V1 StratifiedKFold
- V2 GroupKFold on rounded operating grid
- V3 Leave-one-level-out for Sgn_eff and Ikssmin bins

## 7. Current Execution Summary
- Best model: pending
- Calibration: pending
- Thresholds: tau_F1=0.7596153846153846, tau_HR=0.999, tau_cost=0.147
- Notes: Equation-compatibility rewrite for strict renderers.

## 8. Artifact Index
### Tables
- `outputs/tables/ablation_study.csv`
- `outputs/tables/all_delta_vs_A1.csv`
- `outputs/tables/all_tier_results.csv`
- `outputs/tables/calibration_comparison_B2.csv`
- `outputs/tables/counterfactual_examples.csv`
- `outputs/tables/dataset_summary.csv`
- `outputs/tables/delta_step_step1_static_V1_stratified.csv`
- `outputs/tables/delta_step_step1_static_V2_grouped.csv`
- `outputs/tables/delta_step_step1_static_V3_leave_Ik.csv`
- `outputs/tables/delta_step_step1_static_V3_leave_Sgn.csv`
- `outputs/tables/delta_step_step2_robustness_V1_stratified.csv`
- `outputs/tables/delta_step_step2_robustness_V2_grouped.csv`
- `outputs/tables/delta_step_step2_robustness_V3_leave_Ik.csv`
- `outputs/tables/delta_step_step2_robustness_V3_leave_Sgn.csv`
- `outputs/tables/delta_step_step3_noise_V1_stratified.csv`
- `outputs/tables/delta_step_step3_noise_V2_grouped.csv`
- `outputs/tables/delta_step_step3_noise_V3_leave_Ik.csv`
- `outputs/tables/delta_step_step3_noise_V3_leave_Sgn.csv`
- `outputs/tables/deployment_metrics.csv`
- `outputs/tables/leaderboard_step1_static_V1_stratified.csv`
- `outputs/tables/leaderboard_step1_static_V2_grouped.csv`
- `outputs/tables/leaderboard_step1_static_V3_leave_Ik.csv`
- `outputs/tables/leaderboard_step1_static_V3_leave_Sgn.csv`
- `outputs/tables/leaderboard_step2_robustness_V1_stratified.csv`
- `outputs/tables/leaderboard_step2_robustness_V2_grouped.csv`
- `outputs/tables/leaderboard_step2_robustness_V3_leave_Ik.csv`
- `outputs/tables/leaderboard_step2_robustness_V3_leave_Sgn.csv`
- `outputs/tables/leaderboard_step3_noise_V1_stratified.csv`
- `outputs/tables/leaderboard_step3_noise_V2_grouped.csv`
- `outputs/tables/leaderboard_step3_noise_V3_leave_Ik.csv`
- `outputs/tables/leaderboard_step3_noise_V3_leave_Sgn.csv`
- `outputs/tables/model_comparison_step1.csv`
- `outputs/tables/model_comparison_step2_robustness.csv`
- `outputs/tables/noise_robustness_best_ABC.csv`
- `outputs/tables/noise_robustness_step3.csv`

### Figures
- `outputs/figures/boundary_comparison.png`
- `outputs/figures/calibration_reliability.png`
- `outputs/figures/drift_monitoring.png`
- `outputs/figures/feature_distributions.png`
- `outputs/figures/feature_importance_permutation.png`
- `outputs/figures/flowchart_oos_pipeline.png`
- `outputs/figures/noise_robustness_fnr.png`
- `outputs/figures/noise_robustness_prauc.png`
- `outputs/figures/pdp_main_features.png`
- `outputs/figures/stability_map_Ik_vs_Sgn.png`
- `outputs/figures/tradeoff_prauc_vs_complexity.png`
- `outputs/figures/tradeoff_prauc_vs_ece.png`
- `outputs/figures/tradeoff_recall_vs_fpr_hr.png`
