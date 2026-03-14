from __future__ import annotations

import pandas as pd

from src.data import TARGET_COLUMN, load_dataset, standardize_columns, validate_required_columns


def test_standardize_and_required_columns(tmp_path):
    df = pd.DataFrame(
        {
            "Tag rate": [1.0, 1.1],
            "Ikssmin (kA)": [10, 11],
            "Sgn_eff_MVA": [100, 120],
            "H_s": [3.2, 3.5],
            "Out of step": [0, 1],
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)

    loaded, audit = load_dataset(path)
    missing = validate_required_columns(loaded)
    assert missing == []
    assert TARGET_COLUMN in loaded.columns
    assert audit.n_rows_clean == 2


def test_standardize_columns_aliases():
    df = pd.DataFrame({"tagrate": [1], "ikssmin": [2], "h": [3], "sgn": [4], "oos": [0]})
    out = standardize_columns(df)
    assert "Tag_rate" in out.columns
    assert "Ikssmin_kA" in out.columns
    assert "H_s" in out.columns
    assert "Sgn_eff_MVA" in out.columns
    assert "Out_of_step" in out.columns
