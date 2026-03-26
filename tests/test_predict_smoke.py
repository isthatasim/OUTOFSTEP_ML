from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.outofstep_ml.deployment.predict import InferenceBundle, predict_one
from src.outofstep_ml.models.static_physics_model import StaticModelConfig, StaticPhysicsRiskModel


def test_predict_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "Tag_rate": rng.uniform(0.8, 1.8, 120),
            "Ikssmin_kA": rng.uniform(5, 20, 120),
            "Sgn_eff_MVA": rng.uniform(50, 180, 120),
            "H_s": rng.uniform(1.0, 5.0, 120),
            "GenName": ["GR1"] * 120,
        }
    )
    y = ((df["Sgn_eff_MVA"] / df["H_s"] - df["Ikssmin_kA"]) > 20).astype(int).values

    numeric = ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"]
    model = StaticPhysicsRiskModel(
        config=StaticModelConfig(model_name="tierA_logistic", calibrate=False, random_state=7),
        numeric_features=numeric,
        categorical_features=["GenName"],
    ).fit(df[numeric + ["GenName"]], y)

    bundle_path = tmp_path / "bundle.joblib"
    model.save(bundle_path)

    cfg = {
        "features": numeric + ["GenName"],
        "thresholds": {"tau_cost": 0.5},
        "feature_bounds": {k: [float(df[k].min()), float(df[k].max())] for k in numeric},
    }
    cfg_path = tmp_path / "inference.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    bundle = InferenceBundle.load(bundle_path, cfg_path)
    out = predict_one(
        bundle,
        {
            "Tag_rate": 1.1,
            "Ikssmin_kA": 12.0,
            "Sgn_eff_MVA": 120.0,
            "H_s": 2.5,
            "GenName": "GR1",
        },
    )

    assert 0.0 <= out["p_oos"] <= 1.0
    assert out["decision"] in [0, 1]
    assert "explanation" in out
