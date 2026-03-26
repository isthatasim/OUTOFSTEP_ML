from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.eval import generate_splits, make_group_labels


def build_groups(df: pd.DataFrame, cols: List[str], round_decimals: int = 2) -> np.ndarray:
    return make_group_labels(df, cols=cols, round_decimals=round_decimals)


def create_split_manifest(
    y: np.ndarray,
    split_mode: str,
    n_splits: int,
    random_state: int,
    groups: np.ndarray | None = None,
    leaveout_feature: str = "Sgn_eff_MVA",
    leaveout_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    splits = generate_splits(
        y=y,
        split_mode=split_mode,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
        leaveout_feature=leaveout_feature,
        leaveout_frame=leaveout_frame,
    )
    rows: List[Dict] = []
    for fold, (tr, te) in enumerate(splits, start=1):
        for idx in tr:
            rows.append({"fold": fold, "index": int(idx), "subset": "train"})
        for idx in te:
            rows.append({"fold": fold, "index": int(idx), "subset": "test"})
    return pd.DataFrame(rows)


def save_split_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p
