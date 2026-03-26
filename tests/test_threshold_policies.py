from __future__ import annotations

import numpy as np

from src.outofstep_ml.evaluation.threshold_policy_compare import compare_threshold_policies


def test_threshold_policy_outputs():
    y = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    p = np.array([0.05, 0.20, 0.70, 0.90, 0.35, 0.80], dtype=float)
    thresholds = {"tau_f1": 0.65, "tau_cost": 0.40, "tau_hr": 0.25}
    out = compare_threshold_policies(y, p, thresholds=thresholds, c_fn=10.0, c_fp=1.0)
    assert set(out["policy"].tolist()) == {"tau_default", "tau_F1", "tau_cost", "tau_HR"}
    assert np.all(out["tau"] >= 0.0)
    assert np.all(out["tau"] <= 1.0)
    assert "Cost" in out.columns

