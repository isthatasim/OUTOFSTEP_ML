from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from .eval import evaluate_model_cv
from .features import get_monotonic_constraints
from .models import make_model_registry


@dataclass
class ChallengerDecision:
    deploy: int
    champion_pr_auc: float
    champion_fnr: float
    challenger_pr_auc: float
    challenger_fnr: float
    reason: str


def train_candidate_models(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int = 42,
) -> List[Tuple[str, BaseEstimator]]:
    monotonic = get_monotonic_constraints(numeric_features, stress_monotonic_positive=True)
    specs = make_model_registry(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        monotonic_cst=monotonic,
        random_state=random_state,
    )
    return [(s.name, s.estimator) for s in specs]


def champion_challenger_gate(
    champion_model: BaseEstimator,
    challenger_model: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    leaveout_frame: pd.DataFrame,
    leaveout_feature: str = "Sgn_eff_MVA",
) -> ChallengerDecision:
    champ_summary, _, _ = evaluate_model_cv(
        model_name="champion",
        estimator=clone(champion_model),
        X=X,
        y=y,
        split_mode="leave-level-out",
        scenario="retrain_gate",
        leaveout_frame=leaveout_frame,
        leaveout_feature=leaveout_feature,
        n_splits=5,
    )
    chall_summary, _, _ = evaluate_model_cv(
        model_name="challenger",
        estimator=clone(challenger_model),
        X=X,
        y=y,
        split_mode="leave-level-out",
        scenario="retrain_gate",
        leaveout_frame=leaveout_frame,
        leaveout_feature=leaveout_feature,
        n_splits=5,
    )

    champ_pr, champ_fnr = champ_summary["PR_AUC"], champ_summary["FNR"]
    chall_pr, chall_fnr = chall_summary["PR_AUC"], chall_summary["FNR"]

    deploy = int((chall_pr > champ_pr) and (chall_fnr < champ_fnr))
    reason = "deploy challenger" if deploy else "keep champion"
    return ChallengerDecision(
        deploy=deploy,
        champion_pr_auc=float(champ_pr),
        champion_fnr=float(champ_fnr),
        challenger_pr_auc=float(chall_pr),
        challenger_fnr=float(chall_fnr),
        reason=reason,
    )


def retrain_challenger(
    champion_model: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    leaveout_frame: pd.DataFrame,
    random_state: int = 42,
) -> Dict[str, object]:
    candidates = train_candidate_models(
        X,
        y,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
    )

    best = None
    best_summary = None
    for name, est in candidates:
        s, _, _ = evaluate_model_cv(
            model_name=name,
            estimator=clone(est),
            X=X,
            y=y,
            split_mode="stratified",
            scenario="retrain_select",
            n_splits=5,
            random_state=random_state,
        )
        if best is None or s["PR_AUC"] > best_summary["PR_AUC"]:
            best = (name, clone(est))
            best_summary = s

    challenger_name, challenger = best
    decision = champion_challenger_gate(
        champion_model=champion_model,
        challenger_model=challenger,
        X=X,
        y=y,
        leaveout_frame=leaveout_frame,
        leaveout_feature="Sgn_eff_MVA",
    )

    return {
        "challenger_name": challenger_name,
        "challenger_summary": best_summary,
        "decision": decision,
        "challenger_model": challenger,
    }
