from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.inspection import partial_dependence


def compute_pdp_table(model, X: pd.DataFrame, features: Iterable[str], grid_resolution: int = 50) -> pd.DataFrame:
    rows = []
    for feat in features:
        if feat not in X.columns:
            continue
        pdp = partial_dependence(model, X, [feat], grid_resolution=grid_resolution)
        xs = pdp.get("grid_values", pdp.get("values"))[0]
        ys = pdp["average"][0]
        for x_val, y_val in zip(xs, ys):
            rows.append({"feature": feat, "x": float(x_val), "pdp": float(y_val)})
    return pd.DataFrame(rows)
