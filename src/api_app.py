from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class PredictRequest(BaseModel):
    Tag_rate: float = Field(..., description="Grid acceleration proxy")
    Ikssmin_kA: float = Field(..., description="Short-circuit current in kA")
    Sgn_eff_MVA: float = Field(..., description="Nominal machine power in MVA")
    H_s: float = Field(..., description="Inertia constant in seconds")
    GenName: str = Field("GR1", description="Generator name")


class PredictResponse(BaseModel):
    p_oos: float
    class_label: int
    threshold_used: float
    model_version: str
    calibration_version: str
    ood_flag: int


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(txt) or {}
    # Very small fallback parser for key: value pairs
    out: Dict[str, Any] = {}
    for line in txt.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip().strip("\"")
    return out


def _is_ood(row: pd.Series, bounds: Dict[str, Any], margin_frac: float = 0.05) -> int:
    for c in ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"]:
        if c not in bounds:
            continue
        mn, mx = bounds[c]
        span = max(mx - mn, 1e-9)
        lo = mn - margin_frac * span
        hi = mx + margin_frac * span
        if float(row[c]) < lo or float(row[c]) > hi:
            return 1
    return 0


def create_app(model_dir: str | Path = "outputs/model") -> FastAPI:
    app = FastAPI(title="OOS Risk Service", version="1.0.0")

    model_dir = Path(model_dir)
    model_path = model_dir / "model.joblib"
    config_path = model_dir / "config.yaml"

    if not model_path.exists():
        app.state.model = None
        app.state.config = {}
    else:
        app.state.model = joblib.load(model_path)
        app.state.config = _load_yaml(config_path)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {
            "status": "ok",
            "model_loaded": str(app.state.model is not None).lower(),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest):
        model = app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model artifact not loaded. Train/export first.")

        row = pd.DataFrame([payload.model_dump()])
        for c in ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"]:
            if not np.isfinite(float(row[c].iloc[0])):
                raise HTTPException(status_code=422, detail=f"Invalid numeric value for {c}")

        p = float(model.predict_proba(row)[:, 1][0])
        tau = float(app.state.config.get("thresholds", {}).get("tau_cost", 0.5))
        cls = int(p >= tau)
        bounds = app.state.config.get("feature_bounds", {})
        ood_flag = _is_ood(row.iloc[0], bounds=bounds) if isinstance(bounds, dict) else 0

        return PredictResponse(
            p_oos=p,
            class_label=cls,
            threshold_used=tau,
            model_version=str(app.state.config.get("model_version", "v0")),
            calibration_version=str(app.state.config.get("calibration_version", "none")),
            ood_flag=ood_flag,
        )

    return app


app = create_app()
