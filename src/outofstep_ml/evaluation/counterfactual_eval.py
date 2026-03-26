from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.outofstep_ml.evaluation.counterfactuals import generate_counterfactual_recommendations


def evaluate_counterfactual_stability_correction(
    model,
    X: pd.DataFrame,
    feature_bounds: Dict[str, tuple[float, float]],
    stable_threshold: float,
    max_examples: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cf = generate_counterfactual_recommendations(
        model=model,
        X=X,
        feature_bounds=feature_bounds,
        threshold=stable_threshold,
        max_examples=max_examples,
    )
    if len(cf) == 0:
        summary = pd.DataFrame(
            [
                {
                    "n_counterfactuals": 0,
                    "mean_start_risk": np.nan,
                    "mean_achieved_risk": np.nan,
                    "mean_delta_H_s_pct": np.nan,
                    "mean_delta_Ikssmin_kA_pct": np.nan,
                    "mean_delta_Sgn_eff_MVA_pct": np.nan,
                    "success_rate": np.nan,
                }
            ]
        )
        return summary, cf

    success = (cf["achieved_risk"] <= float(stable_threshold)).astype(int)
    summary = pd.DataFrame(
        [
            {
                "n_counterfactuals": int(len(cf)),
                "mean_start_risk": float(cf["start_risk"].mean()),
                "mean_achieved_risk": float(cf["achieved_risk"].mean()),
                "mean_delta_H_s_pct": float(cf["delta_H_s_pct"].mean()),
                "mean_delta_Ikssmin_kA_pct": float(cf["delta_Ikssmin_kA_pct"].mean()),
                "mean_delta_Sgn_eff_MVA_pct": float(cf["delta_Sgn_eff_MVA_pct"].mean()),
                "success_rate": float(success.mean()),
            }
        ]
    )
    return summary, cf

