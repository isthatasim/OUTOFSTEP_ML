from __future__ import annotations

from typing import Dict

import pandas as pd

from src.monitoring import ks_table, psi_table


def feature_drift_report(train_df: pd.DataFrame, current_df: pd.DataFrame, features: list[str]) -> Dict[str, pd.DataFrame]:
    return {
        "psi": psi_table(train_df[features], current_df[features], features),
        "ks": ks_table(train_df[features], current_df[features], features),
    }


def score_drift_report(reference_scores: pd.Series, current_scores: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reference_mean": [float(reference_scores.mean())],
            "current_mean": [float(current_scores.mean())],
            "reference_std": [float(reference_scores.std())],
            "current_std": [float(current_scores.std())],
        }
    )
