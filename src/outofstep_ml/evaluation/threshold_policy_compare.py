from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def compare_threshold_policies(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Dict[str, float],
    c_fn: float,
    c_fp: float,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    policies = {
        "tau_default": 0.5,
        "tau_F1": float(thresholds["tau_f1"]),
        "tau_cost": float(thresholds["tau_cost"]),
        "tau_HR": float(thresholds["tau_hr"]),
    }
    rows: List[Dict] = []
    for name, tau in policies.items():
        y_hat = (y_prob >= tau).astype(int)
        fn = int(((y_true == 1) & (y_hat == 0)).sum())
        fp = int(((y_true == 0) & (y_hat == 1)).sum())
        rows.append(
            {
                "policy": name,
                "tau": float(tau),
                "Precision": float(precision_score(y_true, y_hat, zero_division=0)),
                "Recall": float(recall_score(y_true, y_hat, zero_division=0)),
                "F1": float(f1_score(y_true, y_hat, zero_division=0)),
                "FN": fn,
                "FP": fp,
                "Cost": float(c_fn * fn + c_fp * fp),
            }
        )
    return pd.DataFrame(rows)


def aggregate_threshold_policy_tables(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    valid = [f for f in frames if f is not None and len(f) > 0]
    if not valid:
        return pd.DataFrame(columns=["policy", "tau", "Precision", "Recall", "F1", "FN", "FP", "Cost"])
    df = pd.concat(valid, ignore_index=True)
    out = (
        df.groupby("policy", as_index=False)
        .agg(
            tau=("tau", "mean"),
            Precision=("Precision", "mean"),
            Recall=("Recall", "mean"),
            F1=("F1", "mean"),
            FN=("FN", "mean"),
            FP=("FP", "mean"),
            Cost=("Cost", "mean"),
        )
    )
    return out

