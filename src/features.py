from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

BASE_NUMERIC_FEATURES = ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"]
ENGINEERED_FEATURES = [
    "invH",
    "S_over_H",
    "S_over_I",
    "I_over_H",
    "log_Sgn_eff_MVA",
    "log_Ikssmin_kA",
]
ENGINEERED_LEGACY_ALIASES = {
    "S_over_H": "Sgn_over_H",
    "S_over_I": "Sgn_over_Ik",
}
ALL_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_FEATURES
OPTIONAL_CATEGORICAL = ["GenName"]


@dataclass
class FeatureConfig:
    include_logs: bool = True
    include_interactions: bool = False
    add_categorical: bool = True


def build_feature_frame(df: pd.DataFrame, include_logs: bool = True) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-8
    out["invH"] = 1.0 / np.clip(out["H_s"].astype(float), eps, None)
    out["S_over_H"] = out["Sgn_eff_MVA"].astype(float) / np.clip(out["H_s"].astype(float), eps, None)
    out["S_over_I"] = out["Sgn_eff_MVA"].astype(float) / np.clip(out["Ikssmin_kA"].astype(float), eps, None)
    out["I_over_H"] = out["Ikssmin_kA"].astype(float) / np.clip(out["H_s"].astype(float), eps, None)

    # Backward-compatible aliases for legacy scripts/configs.
    out["Sgn_over_H"] = out["S_over_H"]
    out["Sgn_over_Ik"] = out["S_over_I"]
    out["Ik_over_H"] = out["I_over_H"]

    if include_logs:
        out["log_Sgn_eff_MVA"] = np.log(np.clip(out["Sgn_eff_MVA"].astype(float), eps, None))
        out["log_Ikssmin_kA"] = np.log(np.clip(out["Ikssmin_kA"].astype(float), eps, None))
    else:
        out["log_Sgn_eff_MVA"] = out["Sgn_eff_MVA"].astype(float)
        out["log_Ikssmin_kA"] = out["Ikssmin_kA"].astype(float)

    if "GenName" not in out.columns:
        out["GenName"] = "GR1"
    out["GenName"] = out["GenName"].astype(str)
    return out


def resolve_engineered_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Resolve engineered columns with canonical names first, then legacy aliases.
    Returns a de-duplicated list in deterministic order.
    """
    ordered_candidates = [
        ["invH"],
        ["S_over_H", "Sgn_over_H"],
        ["S_over_I", "Sgn_over_Ik"],
        ["I_over_H", "Ik_over_H"],
        ["log_Sgn_eff_MVA"],
        ["log_Ikssmin_kA"],
    ]
    resolved: List[str] = []
    seen = set()
    for candidates in ordered_candidates:
        chosen = next((c for c in candidates if c in df.columns), None)
        if chosen and chosen not in seen:
            resolved.append(chosen)
            seen.add(chosen)
    return resolved


def get_feature_columns(df: pd.DataFrame, add_categorical: bool = True) -> List[str]:
    cols = [c for c in ALL_NUMERIC_FEATURES if c in df.columns]
    if add_categorical and "GenName" in df.columns:
        cols.append("GenName")
    return cols


def build_preprocessor(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str] | None = None,
    include_interactions: bool = False,
) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if include_interactions:
        num_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)))
    num_steps.append(("scaler", StandardScaler()))

    transformers = [("num", Pipeline(num_steps), list(numeric_features))]
    cat = list(categorical_features) if categorical_features else []
    if len(cat) > 0:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_monotonic_constraints(feature_names: List[str], stress_monotonic_positive: bool = True) -> List[int]:
    signs: Dict[str, int] = {
        "Tag_rate": 0,
        "Ikssmin_kA": -1,
        "Sgn_eff_MVA": 1 if stress_monotonic_positive else 0,
        "H_s": -1,
        "invH": 1,
        "S_over_H": 1,
        "S_over_I": 1,
        "Sgn_over_H": 1,
        "Sgn_over_Ik": 1,
        "I_over_H": 0,
        "Ik_over_H": 0,
        "log_Sgn_eff_MVA": 1 if stress_monotonic_positive else 0,
        "log_Ikssmin_kA": -1,
    }
    return [int(signs.get(col, 0)) for col in feature_names]


def derive_feature_bounds(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, tuple[float, float]]:
    bounds: Dict[str, tuple[float, float]] = {}
    for c in feature_names:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            mn, mx = float(df[c].min()), float(df[c].max())
            bounds[c] = (mn, mx)
    return bounds
