from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval import select_thresholds


def optimize_thresholds(y_true, y_prob, c_fn: float, c_fp: float, min_recall: float) -> dict:
    th = select_thresholds(y_true, y_prob, c_fn=c_fn, c_fp=c_fp, min_recall=min_recall)
    return {"tau_cost": th.tau_cost, "tau_f1": th.tau_f1, "tau_hr": th.tau_hr}


def threshold_curve(y_true, y_prob, c_fn: float = 10.0, c_fp: float = 1.0) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    rows = []
    for tau in np.linspace(0, 1, 201):
        y_hat = (y_prob >= tau).astype(int)
        fn = int(((y_true == 1) & (y_hat == 0)).sum())
        fp = int(((y_true == 0) & (y_hat == 1)).sum())
        rows.append({"tau": float(tau), "FN": fn, "FP": fp, "Cost": c_fn * fn + c_fp * fp})
    return pd.DataFrame(rows)
