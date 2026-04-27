from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd

from src.features import build_feature_frame


RAW_FEATURES = ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"]


@dataclass
class GridSyncDecision:
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "p_oos": self.p_oos,
            "p_grid_sync_compatible": self.p_grid_sync_compatible,
            "threshold_used": self.threshold_used,
            "ood_flag": self.ood_flag,
            "risk_band": self.risk_band,
            "compatible": self.compatible,
            "verdict": self.verdict,
            "explanation": self.explanation,
            "recommendations": self.recommendations,
        }


def _apply_calibrator(calibrator: Any, method: str, p: np.ndarray) -> np.ndarray:
    if calibrator is None or method == "none":
        return p
    if method in {"platt", "sigmoid"}:
        return calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return calibrator.transform(p)
    return p


def _risk_band(p_oos: float, threshold: float) -> str:
    if p_oos >= threshold:
        return "high"
    if p_oos >= 0.75 * threshold:
        return "borderline"
    return "low"


def _coerce_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    aliases = {
        "T": "Tag_rate",
        "I": "Ikssmin_kA",
        "S": "Sgn_eff_MVA",
        "H": "H_s",
        "device_name": "DeviceId",
        "device_id": "DeviceId",
    }
    for old, new in aliases.items():
        if old in out and new not in out:
            out[new] = out[old]
    if "GenName" not in out:
        out["GenName"] = "GR1"
    if "DeviceId" not in out:
        out["DeviceId"] = str(out.get("GenName", "unknown_device"))
    missing = [c for c in RAW_FEATURES if c not in out]
    if missing:
        raise ValueError(f"Missing required device/grid input(s): {missing}")
    for c in RAW_FEATURES:
        value = float(out[c])
        if not np.isfinite(value):
            raise ValueError(f"Invalid non-finite value for {c}: {out[c]}")
        out[c] = value
    return out


class GridSyncCompatibilityService:
    """Product wrapper for static GR1 grid-synchronization compatibility screening."""

    def __init__(self, bundle: Dict[str, Any]):
        self.bundle = bundle
        self.model = bundle["model"]
        self.calibrator = bundle.get("calibrator")
        self.calibration_method = str(bundle.get("calibration_method", "none"))
        self.feature_columns = list(bundle["feature_columns"])
        self.threshold = float(bundle.get("threshold_used", 0.5))
        self.feature_bounds = bundle.get("feature_bounds", {})
        self.reference_stats = bundle.get("reference_stats", {})
        self.model_version = str(bundle.get("model_version", "grid-sync-oos-v1"))

    @classmethod
    def load(cls, model_dir: str | Path = "outputs/product") -> "GridSyncCompatibilityService":
        path = Path(model_dir) / "grid_sync_bundle.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Product bundle not found: {path}. Run scripts/build_grid_sync_product.py first.")
        return cls(joblib.load(path))

    def _ood_flag(self, row: pd.Series, margin_frac: float = 0.05) -> bool:
        for col, lim in self.feature_bounds.items():
            if col not in row.index or lim is None:
                continue
            lo, hi = float(lim[0]), float(lim[1])
            span = max(hi - lo, 1e-9)
            if float(row[col]) < lo - margin_frac * span or float(row[col]) > hi + margin_frac * span:
                return True
        return False

    def _recommendations(self, row: pd.Series, p_oos: float, ood: bool) -> List[str]:
        recs: List[str] = []
        stats = self.reference_stats
        needs_mitigation = ood or p_oos >= 0.75 * self.threshold
        if ood:
            recs.append("Input is outside the training operating envelope; require engineering review before grid synchronization.")
        if p_oos >= self.threshold:
            recs.append("Do not synchronize automatically; predicted OOS risk is above the safety threshold.")
        if needs_mitigation and "H_s" in stats and float(row["H_s"]) < float(stats["H_s"].get("q25", row["H_s"])):
            recs.append("Consider increasing effective inertia H or enabling synthetic/grid-forming inertia support.")
        if needs_mitigation and "Ikssmin_kA" in stats and float(row["Ikssmin_kA"]) < float(stats["Ikssmin_kA"].get("q25", row["Ikssmin_kA"])):
            recs.append("Consider strengthening grid short-circuit support I before synchronization.")
        if needs_mitigation and "Sgn_eff_MVA" in stats and float(row["Sgn_eff_MVA"]) > float(stats["Sgn_eff_MVA"].get("q75", row["Sgn_eff_MVA"])):
            recs.append("Consider derating/reducing loading S or redispatching before synchronization.")
        if not recs:
            recs.append("Operating point is inside the learned domain and below the OOS risk threshold.")
        return recs

    def predict_one(self, payload: Dict[str, Any]) -> GridSyncDecision:
        clean = _coerce_payload(payload)
        frame = build_feature_frame(pd.DataFrame([clean]), include_logs=bool(self.bundle.get("include_logs", True)))
        X = frame[self.feature_columns].copy()
        p_raw = np.clip(self.model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        p_oos = float(np.clip(_apply_calibrator(self.calibrator, self.calibration_method, p_raw)[0], 1e-6, 1 - 1e-6))
        ood = self._ood_flag(frame.iloc[0])
        band = _risk_band(p_oos, self.threshold)
        compatible = bool((p_oos < self.threshold) and not ood)
        if ood:
            verdict = "ENGINEERING_REVIEW_REQUIRED_OUT_OF_DOMAIN"
        elif compatible:
            verdict = "COMPATIBLE_FOR_GRID_SYNC"
        else:
            verdict = "NOT_COMPATIBLE_HIGH_OOS_RISK"
        explanation = (
            f"Predicted calibrated OOS risk is {p_oos:.4f}; threshold is {self.threshold:.4f}. "
            f"The device is classified as {verdict} using raw features T/I/S/H and physics ratios "
            "1/H, S/H, S/I, and I/H."
        )
        return GridSyncDecision(
            device_id=str(clean.get("DeviceId", clean.get("GenName", "unknown_device"))),
            p_oos=p_oos,
            p_grid_sync_compatible=1.0 - p_oos,
            threshold_used=self.threshold,
            ood_flag=ood,
            risk_band=band,
            compatible=compatible,
            verdict=verdict,
            explanation=explanation,
            recommendations=self._recommendations(frame.iloc[0], p_oos, ood),
        )

    def predict_many(self, records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame([self.predict_one(r).to_dict() for r in records])
