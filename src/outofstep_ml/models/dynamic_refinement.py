from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DynamicRefinementConfig:
    enabled: bool = False
    sequence_length: int = 64
    feature_names: tuple[str, ...] = ("delta", "omega", "pe", "pm")


class DynamicRefinementScaffold:
    """Optional scaffold for future transient-window refinement."""

    def __init__(self, config: DynamicRefinementConfig | None = None):
        self.config = config or DynamicRefinementConfig()

    def expected_input_format(self) -> dict:
        return {
            "shape": "[n_samples, sequence_length, n_features]",
            "required_features": list(self.config.feature_names),
            "notes": "Provide aligned transient windows per contingency event.",
        }

    def fit(self, X, y):
        raise NotImplementedError("Scaffold-only: add temporal training once transient-window data is available.")

    def predict_proba(self, X):
        raise NotImplementedError("Scaffold-only: add temporal inference once transient-window data is available.")
