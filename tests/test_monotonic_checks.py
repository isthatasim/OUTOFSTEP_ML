from __future__ import annotations

import numpy as np
import pandas as pd

from src.outofstep_ml.evaluation.monotonic_checks import monotonic_consistency_check


class _DummyLinearRisk:
    def predict_proba(self, X: pd.DataFrame):
        z = (
            -0.8 * pd.to_numeric(X["H_s"], errors="coerce").astype(float).values
            - 0.6 * pd.to_numeric(X["Ikssmin_kA"], errors="coerce").astype(float).values
            + 0.7 * pd.to_numeric(X["Sgn_eff_MVA"], errors="coerce").astype(float).values / 100.0
        )
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


def test_monotonic_consistency_schema():
    X = pd.DataFrame(
        {
            "H_s": np.linspace(1.0, 5.0, 50),
            "Ikssmin_kA": np.linspace(5.0, 20.0, 50),
            "Sgn_eff_MVA": np.linspace(60.0, 200.0, 50),
        }
    )
    model = _DummyLinearRisk()
    out = monotonic_consistency_check(
        model=model,
        X=X,
        monotonic_dirs={"H_s": -1, "Ikssmin_kA": -1, "Sgn_eff_MVA": 1},
        step_frac=0.02,
        n_samples=30,
        random_state=7,
    )
    assert len(out) == 3
    assert set(["feature", "direction", "violation_rate"]).issubset(out.columns)
    assert np.all((out["violation_rate"] >= 0.0) & (out["violation_rate"] <= 1.0))

