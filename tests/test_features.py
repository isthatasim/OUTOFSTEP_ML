from __future__ import annotations

import pandas as pd

from src.features import build_feature_frame


def test_feature_engineering_columns():
    df = pd.DataFrame(
        {
            "Tag_rate": [1.0, 2.0],
            "Ikssmin_kA": [10.0, 20.0],
            "Sgn_eff_MVA": [100.0, 200.0],
            "H_s": [2.0, 4.0],
            "Out_of_step": [0, 1],
        }
    )
    out = build_feature_frame(df)
    for c in ["invH", "S_over_H", "S_over_I", "Ik_over_H", "log_Sgn_eff_MVA", "log_Ikssmin_kA"]:
        assert c in out.columns
    for c in ["Sgn_over_H", "Sgn_over_Ik"]:
        assert c in out.columns
    assert "GenName" in out.columns
