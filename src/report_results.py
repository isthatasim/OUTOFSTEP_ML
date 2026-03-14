from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _artifact_line(rec: Dict) -> str:
    return (
        f"- [{rec.get('artifact_id')}] {rec.get('artifact_name')}  "
        f"(code: `{rec.get('code_path')}`; file: `{rec.get('file_path')}`)"
    )


def _section_artifacts(records: List[Dict], keyword: str, limit: int = 8) -> str:
    hits = [r for r in records if keyword.lower() in str(r.get("artifact_name", "")).lower()]
    if not hits:
        return "- (no artifacts found)"
    return "\n".join(_artifact_line(r) for r in hits[:limit])


def build_results_discussion_markdown(
    output_path: str | Path,
    results_df: pd.DataFrame,
    tier_summary_df: pd.DataFrame,
    recommendation: Dict[str, str],
    manifest_records: List[Dict],
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if len(results_df) > 0:
        top = results_df.sort_values("CompositeScore", ascending=False).iloc[0]
        exec_best = f"{top['model_code']} ({top['model_name']})"
    else:
        exec_best = "pending"

    def tier_block(tier: str) -> str:
        d = tier_summary_df[tier_summary_df["tier"] == tier]
        if len(d) == 0:
            return "- no results"
        row = d.iloc[0]
        return (
            f"- Mean PR-AUC: {row['PR_AUC']:.4f}\n"
            f"- Mean FNR: {row['FNR']:.4f}\n"
            f"- Mean ECE: {row['ECE']:.4f}\n"
            f"- Mean CompositeScore: {row['CompositeScore']:.4f}"
        )

    md = f"""# OOS GR1 Results & Discussion

Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

## Executive Summary
- Best composite performer across evaluated Step x Validation runs: **{exec_best}**.
- Recommended deployment model: **{recommendation.get('model_name', 'pending')}** with calibration **{recommendation.get('calibration', 'pending')}**.
- Recommended thresholds: tau_F1={recommendation.get('tau_f1','pending')}, tau_HR={recommendation.get('tau_hr','pending')}, tau_cost={recommendation.get('tau_cost','pending')}.
- Operational emphasis: minimize missed instability (FNR) while preserving calibration quality (ECE/Brier).

## Tier-by-Tier Comparison
### Tier A (interpretability-first)
{tier_block('Tier A')}

### Tier B (accuracy-first)
{tier_block('Tier B')}

### Tier C (physics-aware/hybrid)
{tier_block('Tier C')}

## Scenario Discussion (Step 1 -> Step 5)
### Step 1: Static OOS prediction
Key leaderboard and calibration artifacts:
{_section_artifacts(manifest_records, 'leaderboard_step1')}
{_section_artifacts(manifest_records, 'calibration_comparison')}

Interpretation:
- Step 1 establishes baseline separability and threshold policy under offline conditions.

### Step 2: Robustness on operating maps and unseen levels
Key robustness artifacts:
{_section_artifacts(manifest_records, 'leaderboard_step2')}
{_section_artifacts(manifest_records, 'delta_step_step2_robustness')}

Interpretation:
- Grouped and leave-level protocols test generalization under unseen operating regimes.

### Step 3: Measurement noise and uncertainty realism
Key noise artifacts:
{_section_artifacts(manifest_records, 'leaderboard_step3')}
{_section_artifacts(manifest_records, 'noise_robustness')}

Interpretation:
- Robustness curves quantify degradation with increased input uncertainty.

### Step 4: Deployment prototype
Key deployment artifacts:
{_section_artifacts(manifest_records, 'model_card')}
{_section_artifacts(manifest_records, 'deployment_metrics')}
{_section_artifacts(manifest_records, 'api_')}

### Step 5: Monitoring, drift, and retraining policy
Key monitoring artifacts:
{_section_artifacts(manifest_records, 'psi')}
{_section_artifacts(manifest_records, 'retrain_policy')}

Interpretation:
- Retraining is triggered only when drift alarms coincide with sufficient new data and challenger superiority criteria.

## Physics Plausibility Evidence
- Partial dependence and boundary comparison artifacts:
{_section_artifacts(manifest_records, 'pdp')}
{_section_artifacts(manifest_records, 'boundary_comparison')}

## Deployment Recommendation
- Model: **{recommendation.get('model_name', 'pending')}**
- Calibration: **{recommendation.get('calibration', 'pending')}**
- Threshold policy: tau_cost={recommendation.get('tau_cost','pending')}, tau_F1={recommendation.get('tau_f1','pending')}, tau_HR={recommendation.get('tau_hr','pending')}
- Rationale: composite score ranking with explicit penalty on FNR and poor calibration.

## Failure Modes and Limitations
- Static descriptors approximate dynamic transient behavior; extrapolation outside modeled operating envelope requires caution.
- Leave-level-out gaps indicate potential boundary shift risk under unseen stress/strength combinations.
- Periodic retraining and threshold re-validation are required after drift alerts.

## Traceability to Code and Artifacts
The complete registry is stored in `outputs/results_manifest.json`. Every table/figure above is linked via artifact IDs and `code_path` references.
"""
    out.write_text(md, encoding="utf-8")
    return out
