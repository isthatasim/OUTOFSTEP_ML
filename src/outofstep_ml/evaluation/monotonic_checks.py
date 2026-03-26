from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def monotonic_consistency_check(
    model,
    X: pd.DataFrame,
    monotonic_dirs: Dict[str, int],
    step_frac: float = 0.02,
    n_samples: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Local finite-difference consistency check.
    direction = -1 means risk should not increase when feature increases.
    direction = +1 means risk should not decrease when feature increases.
    """
    rng = np.random.default_rng(random_state)
    if len(X) == 0:
        return pd.DataFrame(columns=["feature", "direction", "n_checked", "violation_rate", "mean_delta_p", "mean_abs_delta_p"])

    idx = rng.choice(np.arange(len(X)), size=min(n_samples, len(X)), replace=False)
    Xs = X.iloc[idx].copy()
    p0 = np.clip(model.predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)

    rows: List[Dict] = []
    for feat, direction in monotonic_dirs.items():
        if feat not in Xs.columns or direction == 0:
            continue
        x = pd.to_numeric(Xs[feat], errors="coerce").astype(float).values
        scale = np.maximum(np.abs(x), 1e-6)
        delta = step_frac * scale
        Xp = Xs.copy()
        Xp[feat] = x + delta
        p1 = np.clip(model.predict_proba(Xp)[:, 1], 1e-6, 1 - 1e-6)
        dp = p1 - p0

        if int(direction) < 0:
            viol = dp > 0.0
        else:
            viol = dp < 0.0

        rows.append(
            {
                "feature": feat,
                "direction": int(direction),
                "n_checked": int(len(dp)),
                "violation_rate": float(np.mean(viol)),
                "mean_delta_p": float(np.mean(dp)),
                "mean_abs_delta_p": float(np.mean(np.abs(dp))),
            }
        )

    return pd.DataFrame(rows)


def aggregate_monotonic_checks(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    valid = [f for f in frames if f is not None and len(f) > 0]
    if not valid:
        return pd.DataFrame(columns=["feature", "direction", "n_checked", "violation_rate", "mean_delta_p", "mean_abs_delta_p"])
    df = pd.concat(valid, ignore_index=True)
    agg = (
        df.groupby(["feature", "direction"], as_index=False)
        .agg(
            n_checked=("n_checked", "sum"),
            violation_rate=("violation_rate", "mean"),
            mean_delta_p=("mean_delta_p", "mean"),
            mean_abs_delta_p=("mean_abs_delta_p", "mean"),
        )
    )
    return agg

