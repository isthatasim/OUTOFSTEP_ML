from __future__ import annotations

import numpy as np

from src.outofstep_ml.evaluation.metrics import compute_all_metrics


def test_metrics_include_forecast_terms() -> None:
    y = np.array([0, 0, 1, 1], dtype=int)
    p = np.array([0.1, 0.3, 0.7, 0.9], dtype=float)
    m = compute_all_metrics(y, p)

    for key in ["MSE", "RMSE", "MAE", "R2", "PR_AUC", "FNR", "ECE"]:
        assert key in m
    assert m["MSE"] >= 0.0
    assert m["RMSE"] >= 0.0
