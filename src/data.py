from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


CANONICAL_COLUMNS = {
    "tag_rate": "Tag_rate",
    "tagrate": "Tag_rate",
    "tag_rete": "Tag_rate",
    "tagrete": "Tag_rate",
    "tag": "Tag_rate",
    "ikssmin_ka": "Ikssmin_kA",
    "ikssmin": "Ikssmin_kA",
    "ikssminka": "Ikssmin_kA",
    "ikss_min_ka": "Ikssmin_kA",
    "sgn_eff_mva": "Sgn_eff_MVA",
    "sgneffmva": "Sgn_eff_MVA",
    "sgn_eff": "Sgn_eff_MVA",
    "sgn": "Sgn_eff_MVA",
    "h_s": "H_s",
    "hs": "H_s",
    "h": "H_s",
    "genname": "GenName",
    "generator": "GenName",
    "generatore": "GenName",
    "out_of_step": "Out_of_step",
    "outofstep": "Out_of_step",
    "oos": "Out_of_step",
    "label": "Out_of_step",
    "target": "Out_of_step",
}

CORE_FEATURES = ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s", "GenName"]
TARGET_COLUMN = "Out_of_step"


@dataclass
class DataAudit:
    n_rows_raw: int
    n_rows_clean: int
    n_columns_raw: int
    n_columns_clean: int
    missing_per_column: Dict[str, int]
    duplicate_rows_removed: int
    constant_columns: List[str]
    class_counts: Dict[int, int]
    class_ratio_positive: float
    leakage_hint_tag_rate_auc: float | None
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_column_name(col: str) -> str:
    lowered = col.strip().lower()
    simplified = (
        lowered.replace("(", "_")
        .replace(")", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
        .strip("_")
    )
    return CANONICAL_COLUMNS.get(simplified, col.strip())


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: _normalize_column_name(col) for col in df.columns}
    out = df.rename(columns=rename_map).copy()
    return out


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _coerce_target(df: pd.DataFrame, col: str = TARGET_COLUMN) -> pd.DataFrame:
    if col not in df.columns:
        return df
    mapping = {
        "0": 0,
        "1": 1,
        "stable": 0,
        "unstable": 1,
        "out_of_step": 1,
        "out-of-step": 1,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
    }
    if df[col].dtype == object:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
            .where(lambda s: s.notna(), pd.to_numeric(df[col], errors="coerce"))
        )
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            med = df[c].median()
            if pd.isna(med):
                med = 0.0
            df[c] = df[c].fillna(med)
        else:
            df[c] = df[c].fillna("UNKNOWN")
    return df


def _drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    constants: List[str] = []
    for c in df.columns:
        nun = df[c].nunique(dropna=False)
        if nun <= 1 and c != TARGET_COLUMN:
            constants.append(c)
    if constants:
        df = df.drop(columns=constants)
    return df, constants


def _leakage_tag_rate_auc(df: pd.DataFrame) -> float | None:
    if "Tag_rate" not in df.columns or TARGET_COLUMN not in df.columns:
        return None
    y = df[TARGET_COLUMN].values
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, df["Tag_rate"].values))
    except Exception:
        return None


def validate_required_columns(df: pd.DataFrame) -> List[str]:
    missing = [c for c in ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s", TARGET_COLUMN] if c not in df.columns]
    return missing


def load_dataset(csv_path: str | Path) -> Tuple[pd.DataFrame, DataAudit]:
    path = Path(csv_path)
    raw = pd.read_csv(path)
    n_rows_raw, n_cols_raw = raw.shape

    df = standardize_columns(raw)
    duplicate_rows_removed = int(df.duplicated().sum())
    if duplicate_rows_removed:
        df = df.drop_duplicates().reset_index(drop=True)

    df = _coerce_numeric(df, ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"])
    df = _coerce_target(df, TARGET_COLUMN)

    if "GenName" in df.columns:
        df["GenName"] = df["GenName"].astype(str).str.strip().replace("", "UNKNOWN")
    else:
        df["GenName"] = "GR1"

    df = _fill_missing(df)
    df, constants = _drop_constant_columns(df)

    missing_required = validate_required_columns(df)
    notes: List[str] = []
    if missing_required:
        notes.append(f"Missing expected columns after standardization: {missing_required}")

    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else pd.Series(dtype=float)
    if len(y) > 0:
        y = y.astype(int)
        class_counts = y.value_counts().sort_index().to_dict()
        pos_ratio = float(y.mean())
    else:
        class_counts = {}
        pos_ratio = float("nan")

    leakage_auc = _leakage_tag_rate_auc(df)
    if leakage_auc is not None and leakage_auc > 0.95:
        notes.append(
            "Tag_rate alone has very high ROC-AUC; verify this is not a hidden leakage pathway."
        )

    audit = DataAudit(
        n_rows_raw=n_rows_raw,
        n_rows_clean=int(df.shape[0]),
        n_columns_raw=n_cols_raw,
        n_columns_clean=int(df.shape[1]),
        missing_per_column={c: int(v) for c, v in df.isna().sum().to_dict().items()},
        duplicate_rows_removed=duplicate_rows_removed,
        constant_columns=constants,
        class_counts={int(k): int(v) for k, v in class_counts.items()},
        class_ratio_positive=pos_ratio,
        leakage_hint_tag_rate_auc=leakage_auc,
        notes=notes,
    )
    return df, audit


def print_audit_report(audit: DataAudit) -> None:
    print("\n=== DATA AUDIT REPORT ===")
    for k, v in audit.to_dict().items():
        print(f"- {k}: {v}")
    print("=========================\n")


def dataset_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.describe(include="all").transpose().reset_index().rename(columns={"index": "feature"})
    if TARGET_COLUMN in df.columns:
        summary["target_positive_rate"] = float(df[TARGET_COLUMN].mean())
    return summary
