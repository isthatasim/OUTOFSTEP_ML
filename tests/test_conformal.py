from __future__ import annotations

import numpy as np

from src.outofstep_ml.evaluation.conformal import binary_conformal_prediction_sets, summarize_conformal_sets


def test_binary_conformal_summary_shape():
    y_calib = np.array([0, 0, 1, 1])
    p_calib = np.array([0.05, 0.20, 0.80, 0.95])
    y_test = np.array([0, 1, 1])
    p_test = np.array([0.10, 0.70, 0.90])

    conformal = binary_conformal_prediction_sets(y_calib, p_calib, p_test, alpha=0.10)
    summary = summarize_conformal_sets(y_test, conformal)

    assert len(conformal["sets"]) == len(y_test)
    assert {"coverage", "oos_coverage", "average_set_size", "ambiguous_rate"}.issubset(summary.columns)
    assert 0.0 <= float(summary["coverage"].iloc[0]) <= 1.0

