from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features import build_feature_frame, resolve_engineered_feature_columns
from src.outofstep_ml.product.grid_sync import GridSyncCompatibilityService


def test_grid_sync_product_verdicts() -> None:
    raw = [
        {"Tag_rate": 100.0, "Ikssmin_kA": 3.0, "Sgn_eff_MVA": 10.0, "H_s": 0.2, "GenName": "GR1"},
        {"Tag_rate": 1000.0, "Ikssmin_kA": 15.0, "Sgn_eff_MVA": 0.2, "H_s": 5.0, "GenName": "GR1"},
        {"Tag_rate": 100.0, "Ikssmin_kA": 5.0, "Sgn_eff_MVA": 8.0, "H_s": 0.5, "GenName": "GR1"},
        {"Tag_rate": 1000.0, "Ikssmin_kA": 12.0, "Sgn_eff_MVA": 1.0, "H_s": 4.0, "GenName": "GR1"},
    ]
    frame = build_feature_frame(pd.DataFrame(raw))
    features = ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"] + resolve_engineered_feature_columns(frame)
    y = [1, 0, 1, 0]
    model = LogisticRegression(max_iter=500).fit(frame[features], y)
    bundle = {
        "model": model,
        "calibrator": None,
        "calibration_method": "none",
        "feature_columns": features,
        "threshold_used": 0.5,
        "include_logs": True,
        "feature_bounds": {c: [float(frame[c].min()), float(frame[c].max())] for c in features},
        "reference_stats": {c: {"q25": float(frame[c].quantile(0.25)), "q75": float(frame[c].quantile(0.75))} for c in ["H_s", "Ikssmin_kA", "Sgn_eff_MVA"]},
    }
    service = GridSyncCompatibilityService(bundle)
    out = service.predict_one({"DeviceId": "D1", "Tag_rate": 1000.0, "Ikssmin_kA": 12.0, "Sgn_eff_MVA": 1.0, "H_s": 4.0})

    assert 0.0 <= out.p_oos <= 1.0
    assert out.verdict in {
        "COMPATIBLE_FOR_GRID_SYNC",
        "NOT_COMPATIBLE_HIGH_OOS_RISK",
        "ENGINEERING_REVIEW_REQUIRED_OUT_OF_DOMAIN",
    }
    assert out.device_id == "D1"
    assert out.recommendations
