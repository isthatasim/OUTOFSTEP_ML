from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.base import clone

from src.eval import evaluate_model_cv


def run_validation(
    models: Dict[str, object],
    X: pd.DataFrame,
    y,
    protocols: List[dict],
    groups,
    leaveout_frame: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    for model_code, est in models.items():
        for p in protocols:
            summary, _, _ = evaluate_model_cv(
                model_name=model_code,
                estimator=clone(est),
                X=X,
                y=y,
                split_mode=p["split_mode"],
                scenario=p.get("scenario", "validation"),
                groups=groups,
                n_splits=int(p.get("n_splits", 3)),
                random_state=random_state,
                leaveout_feature=p.get("leave_feature", "Sgn_eff_MVA"),
                leaveout_frame=leaveout_frame,
            )
            summary.update({"model_code": model_code, "validation_protocol": p["name"]})
            rows.append(summary)
    return pd.DataFrame(rows)
