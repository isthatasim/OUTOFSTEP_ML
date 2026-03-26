from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
import yaml

from src.features import build_feature_frame


class InferenceBundle:
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config

    @staticmethod
    def load(model_path: str | Path, config_path: str | Path) -> "InferenceBundle":
        loaded = joblib.load(model_path)
        # Support both direct estimator dumps and wrapped payloads from StaticPhysicsRiskModel.save.
        if isinstance(loaded, dict) and "estimator" in loaded:
            model = loaded["estimator"]
        else:
            model = loaded
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        return InferenceBundle(model=model, config=cfg)


def prepare_input_frame(payload: Dict) -> pd.DataFrame:
    return build_feature_frame(pd.DataFrame([payload]))


def _ood_flag(x: pd.Series, bounds: Dict[str, list]) -> bool:
    for k, lim in bounds.items():
        if k not in x.index:
            continue
        lo, hi = float(lim[0]), float(lim[1])
        v = float(x[k])
        if v < lo or v > hi:
            return True
    return False


def predict_one(bundle: InferenceBundle, payload: Dict) -> Dict:
    X = prepare_input_frame(payload)
    features = bundle.config.get("features", [c for c in X.columns if c != "Out_of_step"])
    X = X[[c for c in features if c in X.columns]]

    p = float(bundle.model.predict_proba(X)[0, 1])
    thr = float(bundle.config.get("thresholds", {}).get("tau_cost", 0.5))
    ood = _ood_flag(X.iloc[0], bundle.config.get("feature_bounds", {}))

    return {
        "p_oos": p,
        "p_oos_calibrated": p,
        "decision": int(p >= thr),
        "threshold_used": thr,
        "explanation": "Risk driven by operating-point pattern on H, I, S, T and engineered ratios.",
        "ood_flag": bool(ood),
    }
