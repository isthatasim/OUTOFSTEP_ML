from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None

from src.features import build_preprocessor


@pytest.mark.skipif(TestClient is None, reason="fastapi test client unavailable")
def test_api_predict_roundtrip(tmp_path):
    from src.api_app import create_app

    X = pd.DataFrame(
        {
            "Tag_rate": np.random.uniform(0.5, 2.0, 80),
            "Ikssmin_kA": np.random.uniform(5, 35, 80),
            "Sgn_eff_MVA": np.random.uniform(50, 250, 80),
            "H_s": np.random.uniform(1.5, 8.0, 80),
            "GenName": ["GR1"] * 80,
        }
    )
    y = ((X["Sgn_eff_MVA"] / X["H_s"] - X["Ikssmin_kA"]) > 10).astype(int).values

    prep = build_preprocessor(["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"], ["GenName"], include_interactions=False)
    model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X, y)

    import joblib

    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "config.yaml").write_text(
        "model_version: \"vtest\"\ncalibration_version: \"none\"\nthresholds:\n  tau_cost: 0.5\nfeature_bounds:\n  Tag_rate: [0.5, 2.0]\n  Ikssmin_kA: [5, 35]\n  Sgn_eff_MVA: [50, 250]\n  H_s: [1.5, 8.0]\n",
        encoding="utf-8",
    )

    app = create_app(model_dir)
    client = TestClient(app)
    resp = client.post(
        "/predict",
        json={"Tag_rate": 1.0, "Ikssmin_kA": 15.0, "Sgn_eff_MVA": 150.0, "H_s": 3.0, "GenName": "GR1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "p_oos" in body
    assert "ood_flag" in body
