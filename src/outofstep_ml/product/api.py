from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.outofstep_ml.product import GridSyncCompatibilityService


class DeviceSyncRequest(BaseModel):
    Tag_rate: float = Field(..., description="Disturbance / acceleration proxy T.")
    Ikssmin_kA: float = Field(..., description="Short-circuit grid strength proxy I in kA.")
    Sgn_eff_MVA: float = Field(..., description="Stress/loading proxy S in MVA.")
    H_s: float = Field(..., description="Inertia constant H in seconds.")
    GenName: str = Field("GR1", description="Generator/grid study name.")
    DeviceId: str = Field("unknown_device", description="Device identifier for reporting.")


class DeviceSyncResponse(BaseModel):
    device_id: str
    p_oos: float
    p_grid_sync_compatible: float
    threshold_used: float
    ood_flag: bool
    risk_band: str
    compatible: bool
    verdict: str
    explanation: str
    recommendations: List[str]


def create_app(model_dir: str | Path = "outputs/product") -> FastAPI:
    app = FastAPI(
        title="Grid Sync Compatibility Service",
        version="1.0.0",
        description="Product API for static OOS risk screening and grid synchronization compatibility.",
    )
    try:
        app.state.service = GridSyncCompatibilityService.load(model_dir)
        app.state.load_error = ""
    except Exception as exc:  # pragma: no cover
        app.state.service = None
        app.state.load_error = str(exc)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok" if app.state.service is not None else "model_unavailable",
            "model_loaded": app.state.service is not None,
            "load_error": app.state.load_error,
        }

    @app.post("/compatibility", response_model=DeviceSyncResponse)
    def compatibility(payload: DeviceSyncRequest) -> Dict[str, Any]:
        if app.state.service is None:
            raise HTTPException(status_code=503, detail=f"Product model unavailable: {app.state.load_error}")
        try:
            return app.state.service.predict_one(payload.model_dump()).to_dict()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/predict", response_model=DeviceSyncResponse)
    def predict(payload: DeviceSyncRequest) -> Dict[str, Any]:
        return compatibility(payload)

    return app


app = create_app()
