from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.eval import minimal_counterfactual


def _to_percent(delta: float, base: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return float(100.0 * delta / base)


def generate_counterfactual_recommendations(
    model,
    X: pd.DataFrame,
    feature_bounds: Dict[str, tuple[float, float]],
    threshold: float,
    max_examples: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    p = model.predict_proba(X)[:, 1]
    risky_idx = np.where(p >= threshold)[0][:max_examples]
    rows: List[Dict] = []
    for idx in risky_idx:
        cf = minimal_counterfactual(model, X.iloc[idx], feature_bounds, threshold=threshold, random_state=random_state + int(idx))
        h0 = float(X.iloc[idx].get("H_s", 0.0))
        i0 = float(X.iloc[idx].get("Ikssmin_kA", 0.0))
        s0 = float(X.iloc[idx].get("Sgn_eff_MVA", 0.0))
        dH = float(cf.get("delta_H_s", 0.0))
        dI = float(cf.get("delta_Ikssmin_kA", 0.0))
        dS = float(cf.get("delta_Sgn_eff_MVA", 0.0))
        rows.append(
            {
                "sample_idx": int(idx),
                "start_risk": float(p[idx]),
                "achieved_risk": float(cf.get("new_risk", np.nan)),
                "delta_H_s_pct": _to_percent(dH, h0),
                "delta_Ikssmin_kA_pct": _to_percent(dI, i0),
                "delta_Sgn_eff_MVA_pct": _to_percent(dS, s0),
                "recommendation": f"Increase H by {abs(_to_percent(dH, h0)):.2f}% | Increase I by {abs(_to_percent(dI, i0)):.2f}% | Decrease S by {abs(_to_percent(dS, s0)):.2f}%",
            }
        )
    return pd.DataFrame(rows)
