# OOS Prediction for GR1: Physics-Aware Static Risk Screening

Generated: 2026-03-26 UTC

## 1. Problem Statement
Develop a deployable out-of-step (OOS) risk screener for GR1 using static operating-point descriptors.

Model taxonomy used in this repository:
- Proposed model: **PhysiScreen-OOS** (physics-aware regularized static classifier)
- Base model: **Logit-Base** (balanced logistic baseline)
- Comparison models: **SVM-RBF**, **RF-Base**, **Boost-GBM**, **Legacy-Hybrid**

## 2. Data Definition
For sample \(i=1,\ldots,N\):

$$
\mathbf{x}^{(i)}=\left[T^{(i)},\ I^{(i)},\ S^{(i)},\ H^{(i)}\right], \qquad y^{(i)}\in\{0,1\}.
$$

where:
- \(T\): Tag_rate
- \(I\): Ikssmin_kA
- \(S\): Sgn_eff_MVA
- \(H\): H_s

### 2.1 Variable Dictionary (Raw Inputs)
| Symbol | Dataset column | Type | Unit | Physical meaning | Expected effect on OOS risk |
|---|---|---|---|---|---|
| \(T\) | `Tag_rate` | continuous | as provided in dataset | grid acceleration/disturbance-severity proxy | data-driven; estimated from data |
| \(I\) | `Ikssmin_kA` | continuous | kA | minimum short-circuit current (grid strength proxy) | higher \(I\) should reduce risk |
| \(S\) | `Sgn_eff_MVA` | continuous | MVA | effective generator loading/stress proxy | higher \(S\) may increase risk |
| \(H\) | `H_s` | continuous | s | inertia constant | higher \(H\) should reduce risk |
| \(y\) | `Out_of_step` | binary | - | stability label: 1 = out-of-step, 0 = stable | target variable |
| GenName | `GenName` | categorical | - | generator identifier (GR1 in this study) | metadata/filter field |

Engineered physics-aware features:

$$
\mathbf{z}^{(i)}=\left[
\frac{1}{H^{(i)}},\
\frac{S^{(i)}}{H^{(i)}},\
\frac{S^{(i)}}{I^{(i)}},\
\frac{I^{(i)}}{H^{(i)}}
\right].
$$

### 2.2 Variable Dictionary (Engineered Features)
| Symbol | Engineered feature | Unit | Interpretation |
|---|---|---|---|
| \(\mathrm{invH}\) | \(1/H\) | s\(^{-1}\) | inverse inertia; larger values mean lower inertial support |
| \(\mathrm{S\_over\_H}\) | \(S/H\) | MVA/s | stress normalized by inertia |
| \(\mathrm{S\_over\_I}\) | \(S/I\) | MVA/kA | stress normalized by grid strength |
| \(\mathrm{I\_over\_H}\) | \(I/H\) | kA/s | grid strength per inertia |

## 3. Physics Background
Swing-equation viewpoint:

$$
\dot{\delta}=\omega,
$$

$$
M\dot{\omega}=P_m-P_e(\delta,V,\text{network})-D\omega,
$$

$$
M=\frac{2H}{\omega_s}.
$$

OOS is treated as a static surrogate label from simulation/protection outcomes.

## 4. Mathematical Formulation
### 4.1 Probabilistic Model

$$
p^{(i)}=f_{\theta}\!\left(\mathbf{x}^{(i)}\right)\approx \Pr\!\left(y^{(i)}=1\mid \mathbf{x}^{(i)}\right),
$$

$$
\hat{y}^{(i)}(\tau)=\mathbb{1}\!\left[p^{(i)}\ge\tau\right].
$$

### 4.1.1 Symbol Dictionary (Model and Optimization)
| Symbol | Meaning |
|---|---|
| \(N\) | number of samples |
| \(i\) | sample index |
| \(\mathbf{x}^{(i)}\) | raw feature vector of sample \(i\) |
| \(\mathbf{z}^{(i)}\) | engineered feature vector of sample \(i\) |
| \(f_{\theta}(\cdot)\) | model mapping features to OOS probability |
| \(\theta\) | trainable model parameters |
| \(p^{(i)}\) | predicted OOS probability for sample \(i\) |
| \(\hat{y}^{(i)}(\tau)\) | hard class prediction at threshold \(\tau\) |
| \(\tau\) | decision threshold for converting probability to class |
| \(\mathbb{1}[\cdot]\) | indicator function |

### 4.2 Imbalance-Aware Objective

$$
\mathcal{L}_{\mathrm{CE}}(\theta)=
-\sum_{i=1}^{N}
\left[
 w_{1}y^{(i)}\log p^{(i)}
 +w_{0}\left(1-y^{(i)}\right)\log\left(1-p^{(i)}\right)
\right].
$$

where:
- \(w_1\): positive-class weight (OOS class weight)
- \(w_0\): negative-class weight (stable class weight)
- larger \(w_1\) prioritizes missed-instability reduction (lower FNR)

Optional focal form:

$$
\mathcal{L}_{\mathrm{Focal}}(\theta)=
-\sum_{i=1}^{N}
\left[
\alpha y^{(i)}\left(1-p^{(i)}\right)^{\gamma}\log p^{(i)}
+\left(1-\alpha\right)\left(1-y^{(i)}\right)\left(p^{(i)}\right)^{\gamma}\log\left(1-p^{(i)}\right)
\right].
$$

where:
- \(\alpha\): class-balancing factor in focal loss
- \(\gamma\): focusing parameter (down-weights easy samples)

### 4.3 Physics-Informed Soft Constraints
Monotonic priors:

$$
\frac{\partial f}{\partial H}\le 0,\qquad
\frac{\partial f}{\partial I}\le 0,\qquad
\frac{\partial f}{\partial S}\ge 0.
$$

Finite-difference penalty:

$$
\mathcal{R}_{\mathrm{phys}}(\theta)=
\lambda_H\,\mathbb{E}\!\left[\max\!\left(0,\frac{\Delta f}{\Delta H}\right)\right]
+\lambda_I\,\mathbb{E}\!\left[\max\!\left(0,\frac{\Delta f}{\Delta I}\right)\right]
+\lambda_S\,\mathbb{E}\!\left[\max\!\left(0,-\frac{\Delta f}{\Delta S}\right)\right].
$$

where:
- \(\lambda_H,\lambda_I,\lambda_S \ge 0\): regularization strengths for monotonic priors
- \(\Delta f/\Delta H\): finite-difference slope approximation with respect to \(H\)
- \(\Delta f/\Delta I\): finite-difference slope approximation with respect to \(I\)
- \(\Delta f/\Delta S\): finite-difference slope approximation with respect to \(S\)
- \(\max(0,\cdot)\): hinge penalty enforcing one-sided monotonic trend

Total objective:

$$
\min_{\theta}\ \mathcal{L}(\theta)=\mathcal{L}_{\mathrm{CE}}(\theta)+\mathcal{R}_{\mathrm{phys}}(\theta).
$$

### 4.4 Cost-Sensitive Thresholding

$$
\tau^{\star}=\arg\min_{\tau}
\left[
C_{\mathrm{FN}}\,\mathrm{FN}(\tau)+C_{\mathrm{FP}}\,\mathrm{FP}(\tau)
\right],
\qquad C_{\mathrm{FN}}\gg C_{\mathrm{FP}}.
$$

where:
- \(C_{\mathrm{FN}}\): cost of false negative (missed unstable case)
- \(C_{\mathrm{FP}}\): cost of false positive (nuisance alarm)
- \(\mathrm{FN}(\tau)\), \(\mathrm{FP}(\tau)\): false-negative and false-positive counts at threshold \(\tau\)

Also reported:

$$
\tau_{F1}=\arg\max_{\tau}F_1(\tau),
\qquad
\tau_{HR}:\ \mathrm{Recall}(\tau)\ge 0.95.
$$

### 4.5 Counterfactual Stability Correction

$$
\min_{\Delta\mathbf{x}}\ \lVert W\Delta\mathbf{x}\rVert_1
\quad
\text{s.t.}
\quad
f_{\theta}(\mathbf{x}+\Delta\mathbf{x})\le \tau_{\mathrm{stable}},
\quad
\mathbf{x}_{\min}\le \mathbf{x}+\Delta\mathbf{x}\le \mathbf{x}_{\max}.
$$

## 5. Strict Validation Protocol
1. Train/validation/final holdout split.
2. Fit only on training data.
3. Tune hyperparameters, calibrators, thresholds, and early-stopping decisions only on validation data.
4. Use final holdout once for final reporting.
5. Evaluate robustness on shifted test sets only.
6. Save split indices and seeds.

## 6. Scenario Program (Integrated with Training)
1. Nominal baseline
2. Low-inertia migration
3. Weak-grid migration
4. Stress-loading escalation
5. Cross-interaction heatmaps (H, I, S)
6. Regime-shift validation (grouped + leave-level-out)
7. Noise/missing-data robustness
8. Imbalance-aware ablation
9. Threshold policy comparison
10. Counterfactual stability correction

## 7. Artifact Paths (Current)
- Main benchmark tables: `outputs/static_q1_validation_xlong/tables/`
- Scenario tables: `outputs/static_q1_validation_xlong/tables/`
- Scenario figures: `outputs/static_q1_validation_xlong/figures/`
- Split protocol manifests: `outputs/static_q1_validation_xlong/splits/`
