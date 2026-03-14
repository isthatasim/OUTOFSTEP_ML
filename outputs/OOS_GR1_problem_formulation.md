# OOS Prediction for GR1: Physics-aware, ML-based, Deployable Framework

Generated: 2026-03-14 16:28:35 UTC

## 1. Problem Statement (Applied Energy framing)
Develop a practical and deployable out-of-step (OOS) predictor for **GR1** that supports screening, calibrated risk, decision-support counterfactuals, and production monitoring.

## 2. Data Definition
For sample $i=1,\dots,N$:

$$
T_i:=\text{Tag rate at sample }i,\quad
I_i:=\text{Ikssmin (kA) at sample }i,\quad
S_i:=\text{Sgn eff (MVA) at sample }i,\quad
H_i:=\text{inertia }H_s\text{ at sample }i
$$

$$
x_i=[T_i,\ I_i,\ S_i,\ H_i]
$$

$$
y_i\in\{0,1\},\quad y_i=1\ \text{means out-of-step}
$$

Physics-motivated engineered features:

$$
z_i^{(1)}=\frac{1}{H_i}\ (\text{invH}),\quad
z_i^{(2)}=\frac{S_i}{H_i}\ (\text{Sgn over H}),\quad
z_i^{(3)}=\frac{S_i}{I_i}\ (\text{Sgn over Ik}),\quad
z_i^{(4)}=\frac{I_i}{H_i}\ (\text{Ik over H})
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
M=\frac{2H}{\omega_s}
$$

OOS corresponds to loss of synchronism after disturbance. This static dataset is used to learn a surrogate stability boundary.

## 4. Mathematical Formulation
### 4.1 Probabilistic model
$$
p_i=f_\theta(x_i)\approx P(y_i=1\mid x_i)
$$

$$
\hat{y}_i(\tau)=\mathbb{1}[p_i\ge\tau]
$$

### 4.2 Imbalance-aware loss
$$
\mathcal{L}_{CE}(\theta)=
-\sum_i\left[w_1y_i\log p_i+w_0(1-y_i)\log(1-p_i)\right]
$$

Optional focal alternative:

$$
\mathcal{L}_{Focal}(\theta)=
-\sum_i\left[\alpha y_i(1-p_i)^\gamma\log p_i+(1-\alpha)(1-y_i)p_i^\gamma\log(1-p_i)\right]
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
\mathcal{R}_{phys}(\theta)=
\lambda_H\mathbb{E}\left[\max\left(0,\frac{\Delta f}{\Delta H}\right)\right]
+\lambda_I\mathbb{E}\left[\max\left(0,\frac{\Delta f}{\Delta I}\right)\right]
+\lambda_S\mathbb{E}\left[\max\left(0,-\frac{\Delta f}{\Delta S}\right)\right]
$$

Total objective:
$$
\min_\theta\ \mathcal{L}(\theta)=\mathcal{L}_{CE}(\theta)+\mathcal{R}_{phys}(\theta)
$$

### 4.4 Cost-sensitive thresholding
$$
\tau^*=\arg\min_\tau\left(C_{FN}FN(\tau)+C_{FP}FP(\tau)\right),
\qquad C_{FN}\gg C_{FP}
$$

Also report:

$$
\tau_{F1}=\arg\max_\tau F_1(\tau),\qquad
\tau_{HR}:\ \mathrm{Recall}(\tau)\ge 0.95
$$

### 4.5 Counterfactual support
$$
\min_{\Delta x}\|W\Delta x\|_1
\quad\text{s.t.}\quad
f_\theta(x+\Delta x)\le\tau_{stable},
\quad
x_{min}\le x+\Delta x\le x_{max}
$$

## 5. Workflow
Data ingestion -> audit/cleaning -> feature engineering -> split protocols -> Tier A/B/C models -> calibration + thresholds -> scenario Steps 1..5 -> explainability/maps -> counterfactual support -> API/tests -> monitoring/drift/retraining -> exports + manifest.

## 6. Validation Protocols
- V1 StratifiedKFold
- V2 GroupKFold on rounded operating grid
- V3 Leave-one-level-out for Sgn_eff and Ikssmin bins

## 7. Current Execution Summary
- Best model: Two-stage hybrid
- Calibration: none
- Thresholds: tau_F1=0.7596153846153846, tau_HR=0.999, tau_cost=0.147
- Notes: Equation notation normalized across all report generators.

## 8. Artifact Index
### Tables
- `outputs\tables\ablation_study.csv`
- `outputs\tables\all_delta_vs_A1.csv`
- `outputs\tables\all_tier_results.csv`
- `outputs\tables\calibration_comparison_B2.csv`
- `outputs\tables\counterfactual_examples.csv`
- `outputs\tables\dataset_summary.csv`
- `outputs\tables\delta_step_step1_static_V1_stratified.csv`
- `outputs\tables\delta_step_step1_static_V2_grouped.csv`
- `outputs\tables\delta_step_step1_static_V3_leave_Ik.csv`
- `outputs\tables\delta_step_step1_static_V3_leave_Sgn.csv`
- `outputs\tables\delta_step_step2_robustness_V1_stratified.csv`
- `outputs\tables\delta_step_step2_robustness_V2_grouped.csv`
- `outputs\tables\delta_step_step2_robustness_V3_leave_Ik.csv`
- `outputs\tables\delta_step_step2_robustness_V3_leave_Sgn.csv`
- `outputs\tables\delta_step_step3_noise_V1_stratified.csv`
- `outputs\tables\delta_step_step3_noise_V2_grouped.csv`
- `outputs\tables\delta_step_step3_noise_V3_leave_Ik.csv`
- `outputs\tables\delta_step_step3_noise_V3_leave_Sgn.csv`
- `outputs\tables\deployment_metrics.csv`
- `outputs\tables\leaderboard_step1_static_V1_stratified.csv`
- `outputs\tables\leaderboard_step1_static_V2_grouped.csv`
- `outputs\tables\leaderboard_step1_static_V3_leave_Ik.csv`
- `outputs\tables\leaderboard_step1_static_V3_leave_Sgn.csv`
- `outputs\tables\leaderboard_step2_robustness_V1_stratified.csv`
- `outputs\tables\leaderboard_step2_robustness_V2_grouped.csv`
- `outputs\tables\leaderboard_step2_robustness_V3_leave_Ik.csv`
- `outputs\tables\leaderboard_step2_robustness_V3_leave_Sgn.csv`
- `outputs\tables\leaderboard_step3_noise_V1_stratified.csv`
- `outputs\tables\leaderboard_step3_noise_V2_grouped.csv`
- `outputs\tables\leaderboard_step3_noise_V3_leave_Ik.csv`
- `outputs\tables\leaderboard_step3_noise_V3_leave_Sgn.csv`
- `outputs\tables\model_comparison_step1.csv`
- `outputs\tables\model_comparison_step2_robustness.csv`
- `outputs\tables\noise_robustness_best_ABC.csv`
- `outputs\tables\noise_robustness_step3.csv`

### Figures
- `outputs\figures\boundary_comparison.png`
- `outputs\figures\calibration_reliability.png`
- `outputs\figures\drift_monitoring.png`
- `outputs\figures\feature_distributions.png`
- `outputs\figures\feature_importance_permutation.png`
- `outputs\figures\flowchart_oos_pipeline.png`
- `outputs\figures\noise_robustness_fnr.png`
- `outputs\figures\noise_robustness_prauc.png`
- `outputs\figures\pdp_main_features.png`
- `outputs\figures\stability_map_Ik_vs_Sgn.png`
- `outputs\figures\tradeoff_prauc_vs_complexity.png`
- `outputs\figures\tradeoff_prauc_vs_ece.png`
- `outputs\figures\tradeoff_recall_vs_fpr_hr.png`
