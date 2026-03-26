from __future__ import annotations

import numpy as np
import pandas as pd

from src.outofstep_ml.data.splitters import create_split_manifest


def test_split_manifest_reproducibility() -> None:
    n = 40
    y = np.array([0] * 30 + [1] * 10)
    df = pd.DataFrame({"Sgn_eff_MVA": np.linspace(1, 10, n)})

    m1 = create_split_manifest(
        y=y,
        split_mode="leave-level-out",
        n_splits=4,
        random_state=123,
        leaveout_feature="Sgn_eff_MVA",
        leaveout_frame=df,
    )
    m2 = create_split_manifest(
        y=y,
        split_mode="leave-level-out",
        n_splits=4,
        random_state=123,
        leaveout_feature="Sgn_eff_MVA",
        leaveout_frame=df,
    )

    assert len(m1) > 0
    assert m1.equals(m2)
