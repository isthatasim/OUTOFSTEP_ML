from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

REQUIRED_COLUMNS = ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s", "Out_of_step"]
OPTIONAL_COLUMNS = ["GenName"]


class DataSchemaError(ValueError):
    """Raised when dataset schema is incompatible with expected columns."""


@dataclass
class SchemaReport:
    missing_required: List[str]
    present_optional: List[str]
    extra_columns: List[str]


def validate_schema(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> SchemaReport:
    required = list(required)
    missing = [c for c in required if c not in df.columns]
    present_optional = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in required and c not in OPTIONAL_COLUMNS]
    return SchemaReport(missing_required=missing, present_optional=present_optional, extra_columns=extra)


def assert_schema(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    report = validate_schema(df, required=required)
    if report.missing_required:
        raise DataSchemaError(f"Missing required columns: {report.missing_required}")


def coerce_numeric_columns(df: pd.DataFrame, numeric_columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
