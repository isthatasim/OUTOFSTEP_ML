from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PreprocessConfig:
    clip_iqr_multiplier: float = 3.0
    fill_numeric: str = "median"


def clip_outliers_iqr(df: pd.DataFrame, numeric_columns: Iterable[str], k: float = 3.0) -> pd.DataFrame:
    out = df.copy()
    for c in numeric_columns:
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce")
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = float(q3 - q1)
        if iqr <= 0 or np.isnan(iqr):
            continue
        lo, hi = float(q1 - k * iqr), float(q3 + k * iqr)
        out[c] = x.clip(lower=lo, upper=hi)
    return out


def fill_missing_numeric(df: pd.DataFrame, numeric_columns: Iterable[str], strategy: str = "median") -> pd.DataFrame:
    out = df.copy()
    for c in numeric_columns:
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce")
        val = float(x.mean()) if strategy == "mean" else float(x.median())
        if np.isnan(val):
            val = 0.0
        out[c] = x.fillna(val)
    return out
