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
