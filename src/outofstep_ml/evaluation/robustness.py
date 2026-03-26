from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.base import clone

from src.eval import add_noise, evaluate_model_cv_noisy_test


def run_noise_robustness(estimator, X: pd.DataFrame, y, noise_cfg: Dict[str, float], random_state: int = 42) -> dict:
    X_noisy = add_noise(X, noise_cfg, random_state=random_state)
    summary, _, _ = evaluate_model_cv_noisy_test(
        model_name="robustness",
        estimator=clone(estimator),
        X_clean=X,
        X_noisy=X_noisy,
        y=y,
        split_mode="stratified",
        scenario="noise_robustness",
        n_splits=3,
        random_state=random_state,
    )
    return summary


def run_missing_feature_stress(estimator, X: pd.DataFrame, y, drop_column: str, random_state: int = 42) -> dict:
    X_stress = X.copy()
    if drop_column in X_stress.columns:
        X_stress[drop_column] = float("nan")
    summary, _, _ = evaluate_model_cv_noisy_test(
        model_name="missing_stress",
        estimator=clone(estimator),
        X_clean=X,
        X_noisy=X_stress,
        y=y,
        split_mode="stratified",
        scenario="missing_feature_stress",
        n_splits=3,
        random_state=random_state,
    )
    return summary
