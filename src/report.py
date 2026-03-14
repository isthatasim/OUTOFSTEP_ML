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

    md = fr"""# OOS Prediction for GR1: Physics-aware, ML-based, Deployable Framework

Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

## 1. Problem Statement (Applied Energy framing)
This work develops an operationally useful out-of-step (OOS) risk predictor for generator **GR1**. The objective is to estimate a calibrated probability of instability and support system operation via: (i) risk screening, (ii) calibrated probability outputs, (iii) what-if counterfactual planning, and (iv) deployable software with drift monitoring and retraining policy.

## 2. Data and Variables
For sample index \(i = 1, \dots, N\):
- Input vector \(x_i = [\text{{Tag\_rate}}_i, \text{{Ikssmin\_kA}}_i, \text{{Sgn\_eff\_MVA}}_i, H_{{s,i}}]\)
- Label \(y_i \in \{{0,1\}}\), where 1 denotes out-of-step (unstable).

Physics-motivated engineered features include:
- \(\text{{invH}} = 1/H_s\)
- \(\text{{Sgn\_over\_H}} = \text{{Sgn\_eff\_MVA}}/H_s\)
- \(\text{{Sgn\_over\_Ik}} = \text{{Sgn\_eff\_MVA}}/\text{{Ikssmin\_kA}}\)
- \(\text{{Ik\_over\_H}} = \text{{Ikssmin\_kA}}/H_s\)

## 3. Physics Background
We use the swing-equation viewpoint for transient stability interpretation:
- \(\dot{{\delta}} = \omega\)
- \(M\dot{{\omega}} = P_m - P_e(\delta, V, \text{{network}}) - D\omega\)
- \(M = 2H/\omega_s\)

Out-of-step corresponds to loss of synchronism after disturbance. Because this dataset is static/parametric, we learn a surrogate stability boundary from descriptors linked to inertia, short-circuit strength, and operating stress.

## 4. Mathematical Formulation
### 4.1 Probabilistic classifier
Learn \(f_\theta: \mathbb{{R}}^d \rightarrow [0,1]\):
\[
p_i = f_\theta(x_i) \approx P(y_i=1|x_i)
\]
Decision with threshold \(\tau\):
\[
\hat{{y}}_i(\tau)=\mathbb{{1}}[p_i\ge\tau]
\]

### 4.2 Imbalance-aware objective
Weighted BCE:
\[
\mathcal{{L}}_{{CE}}(\theta) = -\sum_i\left[w_1 y_i\log p_i + w_0 (1-y_i)\log(1-p_i)\right]
\]
(Optionally focal loss when needed.)

### 4.3 Physics-informed regularization
Soft monotonic priors:
- \(\partial f/\partial H_s \le 0\)
- \(\partial f/\partial \text{{Ikssmin\_kA}} \le 0\)
- \(\partial f/\partial \text{{Sgn\_eff\_MVA}} \ge 0\) when supported by data

Finite-difference penalty form:
\[
\mathcal{{R}}_{{phys}}(\theta)=\lambda_H\mathbb{{E}}[\max(0,\Delta f/\Delta H)]
+\lambda_I\mathbb{{E}}[\max(0,\Delta f/\Delta I)]
+\lambda_S\mathbb{{E}}[\max(0,-\Delta f/\Delta S)]
\]
Total objective:
\[
\min_\theta \; \mathcal{{L}}(\theta)=\mathcal{{L}}_{{CE}}(\theta)+\mathcal{{R}}_{{phys}}(\theta)
\]

### 4.4 Cost-sensitive thresholding
For \(C_{{FN}} \gg C_{{FP}}\):
\[
\tau^* = \arg\min_\tau \left(C_{{FN}}FN(\tau)+C_{{FP}}FP(\tau)\right)
\]
Also report:
- \(\tau_{{F1}}\): maximizes F1
- \(\tau_{{HR}}\): enforces high recall target (e.g., Recall \(\ge 0.95\))

### 4.5 Counterfactual planning
Given unstable point \(x\), solve:
\[
\min_{{\Delta x}} \|W\Delta x\|_1
\quad\text{{s.t.}}\quad f_\theta(x+\Delta x)\le\tau_{{stable}},\;
x_{{min}}\le x+\Delta x\le x_{{max}}
\]
Used to derive operator/planner guidance (e.g., required inertia or short-circuit margin).

## 5. Workflow (Step 1..Step 5)
1. Static OOS prediction with baseline and strong models.
2. Robustness via grouped CV and leave-level-out conditions.
3. Noise injection and uncertainty/stability checks.
4. Production API with validation, OOD checks, tests, artifacts.
5. Monitoring + drift detection + retraining trigger and champion/challenger policy.

## 6. Validation Protocol
Splits:
- StratifiedKFold
- GroupKFold over rounded operating descriptors
- Leave-one-feature-level-out (Sgn_eff or Ikssmin bins)

Primary metric: PR-AUC. Secondary metrics: ROC-AUC, Precision, Recall, F1, Specificity, Balanced Accuracy, Brier, ECE, FNR, cost-weighted risk.

## 7. Deployment and MLOps
Pipeline: train -> validate -> calibrate -> deploy -> monitor drift/performance -> trigger retraining -> redeploy with champion/challenger gate.

Champion/challenger deploy condition: deploy challenger only if PR-AUC improves **and** FNR decreases under leave-level-out evaluation.

## 8. Artifact Index
### Tables
{_fmt_table_paths(table_paths)}

### Figures
{_fmt_table_paths(figure_paths)}

## 9. Current Execution Summary
- Best model: {summary.get('best_model', 'pending')}
- Calibration: {summary.get('best_calibration', 'pending')}
- Thresholds: tau_F1={summary.get('tau_f1', 'pending')}, tau_HR={summary.get('tau_hr', 'pending')}, tau_cost={summary.get('tau_cost', 'pending')}
- Notes: {summary.get('notes', 'Awaiting dataset execution.')}

## 10. Limitations and Future Work
- Static descriptors approximate dynamic transient behavior; wider disturbance coverage may improve transferability.
- Label quality and scenario representativeness dominate field performance.
- Future work: richer dynamic features, domain adaptation across grids, and protection logic co-design.
"""
    out.write_text(md, encoding="utf-8")
    return out
