from __future__ import annotations

from pydantic import BaseModel, Field


class OOSRequest(BaseModel):
    Tag_rate: float = Field(...)
    Ikssmin_kA: float = Field(...)
    Sgn_eff_MVA: float = Field(...)
    H_s: float = Field(...)
    GenName: str = Field("GR1")


class OOSResponse(BaseModel):
    p_oos: float
    p_oos_calibrated: float
    decision: int
    threshold_used: float
    explanation: str
    ood_flag: bool
