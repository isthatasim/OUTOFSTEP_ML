from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class PhysicsFeatureConfig:
    tag_col: str = "Tag_rate"
    i_col: str = "Ikssmin_kA"
    s_col: str = "Sgn_eff_MVA"
    h_col: str = "H_s"
    eps: float = 1e-8


def add_physics_features(df: pd.DataFrame, cfg: PhysicsFeatureConfig = PhysicsFeatureConfig()) -> pd.DataFrame:
    out = df.copy()
    H = np.clip(pd.to_numeric(out[cfg.h_col], errors="coerce").astype(float), cfg.eps, None)
    I = np.clip(pd.to_numeric(out[cfg.i_col], errors="coerce").astype(float), cfg.eps, None)
    S = pd.to_numeric(out[cfg.s_col], errors="coerce").astype(float)

    out["invH"] = 1.0 / H
    out["S_over_H"] = S / H
    out["S_over_I"] = S / I
    out["I_over_H"] = I / H
    return out


def monotonic_prior_directions() -> Dict[str, int]:
    """Return monotonic prior signs: +1 risk increases with feature, -1 decreases."""
    return {
        "H_s": -1,
        "Ikssmin_kA": -1,
        "Sgn_eff_MVA": 1,
    }
