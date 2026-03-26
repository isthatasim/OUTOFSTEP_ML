from __future__ import annotations

from typing import Dict, List

from sklearn.base import clone

from src.features import get_monotonic_constraints
from src.models import make_model_registry


def build_baseline_ladder(numeric_features: List[str], categorical_features: List[str], random_state: int) -> Dict[str, object]:
    mono = get_monotonic_constraints(numeric_features, stress_monotonic_positive=True)
    specs = make_model_registry(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        monotonic_cst=mono,
        include_physics_nn=False,
        random_state=random_state,
    )
    spec_map = {s.name: s.estimator for s in specs}
    return {
        "A1_logistic": clone(spec_map["tierA_logistic"]),
        "A2_tree": clone(spec_map["tierA_tree"]),
        "B1_rf": clone(spec_map["tierB_random_forest"]),
        "B2_gbm": clone(spec_map["tierB_gradient_boosting"]),
        "C1_monotonic": clone(spec_map["tierC_monotonic_hgb"]),
        "C2_physics_logit": clone(spec_map["tierC_physics_logit"]),
        "C3_hybrid": clone(spec_map["tierC_two_stage_hybrid"]),
    }
