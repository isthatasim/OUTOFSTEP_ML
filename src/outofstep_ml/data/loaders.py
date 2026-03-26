from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.data import DataAudit, load_dataset
from src.features import build_feature_frame


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_validated_dataset(path: str | Path, include_logs: bool = True) -> Tuple[pd.DataFrame, DataAudit]:
    """Load dataset with legacy robust standardization and add engineered physics features."""
    df, audit = load_dataset(path)
    df = build_feature_frame(df, include_logs=include_logs)
    return df, audit


def expected_dataset_description() -> str:
    return (
        "Expected columns include Tag_rate, Ikssmin_kA, Sgn_eff_MVA, H_s, GenName (optional), "
        "Out_of_step (target). Extra columns are allowed."
    )
