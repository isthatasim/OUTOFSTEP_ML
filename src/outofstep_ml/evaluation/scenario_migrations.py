from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MigrationScenario:
    name: str
    feature: str
    levels: List[float]
    mode: str  # "multiply" or "quantile_level"


def default_migration_scenarios() -> List[MigrationScenario]:
    return [
        MigrationScenario(
            name="low_inertia_migration",
            feature="H_s",
            levels=[1.00, 0.95, 0.90, 0.85, 0.80],
            mode="multiply",
        ),
        MigrationScenario(
            name="weak_grid_migration",
            feature="Ikssmin_kA",
            levels=[1.00, 0.95, 0.90, 0.85, 0.80],
            mode="multiply",
        ),
        MigrationScenario(
            name="stress_loading_escalation",
            feature="Sgn_eff_MVA",
            levels=[1.00, 1.05, 1.10, 1.15, 1.20],
            mode="multiply",
        ),
    ]


def apply_migration_level(
    X: pd.DataFrame,
    feature: str,
    level: float,
    bounds: Dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    out = X.copy()
    if feature not in out.columns:
        return out
    out[feature] = pd.to_numeric(out[feature], errors="coerce") * float(level)
    if bounds and feature in bounds:
        lo, hi = bounds[feature]
        out[feature] = out[feature].clip(lower=float(lo), upper=float(hi))
    return out


def evaluate_migration_curves(
    model,
    calibrator_method: str,
    calibrator_obj,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    scenarios: Iterable[MigrationScenario],
    apply_calibrator_fn,
    prob_metrics_fn,
    class_metrics_fn,
    thresholds: Dict[str, float],
    feature_bounds: Dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for sc in scenarios:
        for level in sc.levels:
            X_shift = apply_migration_level(X_test, sc.feature, level=level, bounds=feature_bounds)
            p_raw = np.clip(model.predict_proba(X_shift)[:, 1], 1e-6, 1 - 1e-6)
            p = np.clip(apply_calibrator_fn(calibrator_obj, calibrator_method, p_raw), 1e-6, 1 - 1e-6)
            pm = prob_metrics_fn(y_test, p)
            y_hat = (p >= float(thresholds["tau_cost"])).astype(int)
            cm = class_metrics_fn(y_test, y_hat)
            rows.append(
                {
                    "scenario": sc.name,
                    "feature": sc.feature,
                    "level": float(level),
                    **pm,
                    **cm,
                }
            )
    return pd.DataFrame(rows)

