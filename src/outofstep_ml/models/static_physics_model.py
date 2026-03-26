from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.base import clone

from src.features import get_monotonic_constraints
from src.models import make_model_registry, select_best_calibration


@dataclass
class StaticModelConfig:
    model_name: str = "tierC_two_stage_hybrid"
    calibrate: bool = True
    random_state: int = 42


class StaticPhysicsRiskModel:
    """Deployable static screening model with optional post-hoc calibration."""

    def __init__(self, config: StaticModelConfig, numeric_features: List[str], categorical_features: List[str]):
        self.config = config
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.estimator_ = None
        self.calibration_name_ = "none"

    def _build_base(self):
        mono = get_monotonic_constraints(self.numeric_features, stress_monotonic_positive=True)
        specs = make_model_registry(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            monotonic_cst=mono,
            include_physics_nn=False,
            random_state=self.config.random_state,
        )
        spec_map = {s.name: s.estimator for s in specs}
        if self.config.model_name not in spec_map:
            raise ValueError(f"Unknown model_name: {self.config.model_name}")
        return clone(spec_map[self.config.model_name])

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        base = self._build_base()
        base.fit(X, y)

        if self.config.calibrate and len(np.unique(y)) == 2 and len(y) >= 50:
            n = len(y)
            cut = max(int(n * 0.2), 1)
            X_tr, X_va = X.iloc[:-cut], X.iloc[-cut:]
            y_tr, y_va = y[:-cut], y[-cut:]
            if len(np.unique(y_tr)) == 2 and len(np.unique(y_va)) == 2:
                calibrated, method, _ = select_best_calibration(base, X_tr, y_tr, X_va, y_va)
                self.estimator_ = calibrated
                self.calibration_name_ = method
            else:
                self.estimator_ = base
        else:
            self.estimator_ = base
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "calibration_name": self.calibration_name_,
            "estimator": self.estimator_,
        }
        joblib.dump(payload, p)
        return p

    @staticmethod
    def load(path: str | Path) -> "StaticPhysicsRiskModel":
        payload = joblib.load(path)
        model = StaticPhysicsRiskModel(
            config=StaticModelConfig(**payload["config"]),
            numeric_features=payload["numeric_features"],
            categorical_features=payload["categorical_features"],
        )
        model.calibration_name_ = payload.get("calibration_name", "none")
        model.estimator_ = payload["estimator"]
        return model
