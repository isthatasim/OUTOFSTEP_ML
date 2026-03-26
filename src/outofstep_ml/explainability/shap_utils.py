from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def feature_importance_table(model, X: pd.DataFrame, y, random_state: int = 42) -> pd.DataFrame:
    try:
        import shap  # type: ignore

        explainer = shap.Explainer(model.predict_proba, X)
        shap_values = explainer(X)
        vals = np.abs(shap_values.values[..., 1]).mean(axis=0)
        return pd.DataFrame({"feature": X.columns, "importance": vals, "method": "shap"}).sort_values("importance", ascending=False)
    except Exception:
        perm = permutation_importance(model, X, y, n_repeats=8, random_state=random_state, scoring="average_precision")
        return pd.DataFrame({"feature": X.columns, "importance": perm.importances_mean, "method": "permutation"}).sort_values("importance", ascending=False)


def save_importance(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p
