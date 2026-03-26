from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor


def build_imbalance_ablation_models(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
) -> Dict[str, Any]:
    prep = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        include_interactions=True,
    )
    return {
        "no_weight_logistic": Pipeline(
            [
                ("prep", prep),
                ("clf", LogisticRegression(max_iter=3000, class_weight=None, solver="liblinear", random_state=random_state)),
            ]
        ),
        "class_weight_logistic": Pipeline(
            [
                ("prep", prep),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=random_state)),
            ]
        ),
    }


def evaluate_imbalance_ablation(
    model_variants: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    fit_budget_fn,
    select_calibration_fn,
    apply_calibration_fn,
    optimize_thresholds_fn,
    prob_metrics_fn,
    class_metrics_fn,
    c_fn: float,
    c_fp: float,
    min_recall: float,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for model_name, model in model_variants.items():
        # No hyperparameter sweep for these controlled ablations.
        model.fit(X_train, y_train)
        p_val_raw = np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
        cal_name, cal_obj, _ = select_calibration_fn(y_val, p_val_raw)
        p_val = np.clip(apply_calibration_fn(cal_obj, cal_name, p_val_raw), 1e-6, 1 - 1e-6)
        th = optimize_thresholds_fn(y_val, p_val, c_fn=c_fn, c_fp=c_fp, min_recall=min_recall)

        p_test = np.clip(apply_calibration_fn(cal_obj, cal_name, model.predict_proba(X_test)[:, 1]), 1e-6, 1 - 1e-6)
        pm = prob_metrics_fn(y_test, p_test)
        y_hat = (p_test >= float(th["tau_cost"])).astype(int)
        cm = class_metrics_fn(y_test, y_hat)
        rows.append(
            {
                "imbalance_mode": model_name,
                "calibration": cal_name,
                "tau_cost": float(th["tau_cost"]),
                "tau_f1": float(th["tau_f1"]),
                "tau_hr": float(th["tau_hr"]),
                **pm,
                **cm,
            }
        )
    return pd.DataFrame(rows)

