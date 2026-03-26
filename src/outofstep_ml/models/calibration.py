from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.eval import evaluate_probabilities
from src.models import make_calibrated_model


def compare_calibration_methods(estimator, X, y, random_state: int = 42) -> pd.DataFrame:
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_state)
    rows = []

    base = clone(estimator)
    base.fit(Xtr, ytr)
    p0 = base.predict_proba(Xva)[:, 1]
    rows.append({"method": "none", **evaluate_probabilities(yva, p0)})

    for method in ["sigmoid", "isotonic"]:
        cal = make_calibrated_model(clone(estimator), method=method, cv=3)
        cal.fit(Xtr, ytr)
        p = cal.predict_proba(Xva)[:, 1]
        rows.append({"method": method, **evaluate_probabilities(yva, p)})

    return pd.DataFrame(rows)
