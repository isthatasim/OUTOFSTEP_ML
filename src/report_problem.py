from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def _fmt_table_paths(paths: List[str]) -> str:
    if not paths:
        return "- (to be filled after execution)"
    return "\n".join([f"- `{p}`" for p in paths])


def build_problem_formulation_markdown(
    output_path: str | Path,
    summary: Optional[Dict[str, str]] = None,
    table_paths: Optional[List[str]] = None,
    figure_paths: Optional[List[str]] = None,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = summary or {}
    table_paths = table_paths or []
    figure_paths = figure_paths or []

    template = r"""# OOS Prediction for GR1: Physics-aware, ML-based, Deployable Framework

Generated: __DATE__ UTC

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
\mathcal{L}^{\mathrm{CE}}(\theta)=
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
\mathcal{R}^{\mathrm{phys}}(\theta)=
\lambda_H\mathbb{E}\left[\max\left(0,\frac{\Delta f}{\Delta H}\right)\right]
+\lambda_I\mathbb{E}\left[\max\left(0,\frac{\Delta f}{\Delta I}\right)\right]
+\lambda_S\mathbb{E}\left[\max\left(0,-\frac{\Delta f}{\Delta S}\right)\right]
$$

Total objective:
$$
\min_{\theta}\ \mathcal{L}(\theta)=\mathcal{L}^{\mathrm{CE}}(\theta)+\mathcal{R}^{\mathrm{phys}}(\theta)
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
- Best model: __BEST_MODEL__
- Calibration: __BEST_CALIBRATION__
- Thresholds: tau_F1=__TAU_F1__, tau_HR=__TAU_HR__, tau_cost=__TAU_COST__
- Notes: __NOTES__

## 8. Artifact Index
### Tables
__TABLES__

### Figures
__FIGURES__
"""

    md = template
    md = md.replace("__DATE__", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    md = md.replace("__BEST_MODEL__", str(summary.get("best_model", "pending")))
    md = md.replace("__BEST_CALIBRATION__", str(summary.get("best_calibration", "pending")))
    md = md.replace("__TAU_F1__", str(summary.get("tau_f1", "pending")))
    md = md.replace("__TAU_HR__", str(summary.get("tau_hr", "pending")))
    md = md.replace("__TAU_COST__", str(summary.get("tau_cost", "pending")))
    md = md.replace("__NOTES__", str(summary.get("notes", "Awaiting dataset execution.")))
    md = md.replace("__TABLES__", _fmt_table_paths(table_paths))
    md = md.replace("__FIGURES__", _fmt_table_paths(figure_paths))

    out.write_text(md, encoding="utf-8")
    return out
