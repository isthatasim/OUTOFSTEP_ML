from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.features import build_preprocessor
from src.outofstep_ml.models.baselines import build_baseline_ladder


@dataclass
class BenchmarkModelSpec:
    model_id: str
    model_name: str
    family: str
    estimator: Optional[object]
    param_distributions: Dict[str, list]
    probabilistic: bool = True
    available: bool = True
    skip_reason: str = ""
    is_existing_repo_model: bool = False
    is_proposed_model: bool = False


def _optional_catboost(preprocessor, random_state: int) -> BenchmarkModelSpec:
    try:
        from catboost import CatBoostClassifier

        est = Pipeline(
            [
                ("prep", preprocessor),
                (
                    "clf",
                    CatBoostClassifier(
                        verbose=False,
                        random_seed=random_state,
                        loss_function="Logloss",
                        auto_class_weights="Balanced",
                    ),
                ),
            ]
        )
        params = {
            "clf__depth": [4, 6, 8],
            "clf__learning_rate": [0.02, 0.05, 0.1],
            "clf__iterations": [200, 400, 700],
            "clf__l2_leaf_reg": [1, 3, 5, 7],
        }
        return BenchmarkModelSpec(
            model_id="catboost_tuned",
            model_name="CatBoost (tuned)",
            family="strong_tabular",
            estimator=est,
            param_distributions=params,
            probabilistic=True,
            available=True,
        )
    except Exception as exc:
        return BenchmarkModelSpec(
            model_id="catboost_tuned",
            model_name="CatBoost (tuned)",
            family="strong_tabular",
            estimator=None,
            param_distributions={},
            probabilistic=True,
            available=False,
            skip_reason=f"catboost unavailable: {exc}",
        )


def _optional_ft_transformer() -> BenchmarkModelSpec:
    # Scaffold-only unless compatible implementation dependencies are available.
    return BenchmarkModelSpec(
        model_id="ft_transformer",
        model_name="FT-Transformer",
        family="deep_tabular",
        estimator=None,
        param_distributions={},
        available=False,
        skip_reason="Optional baseline scaffold: implement with dedicated FT-Transformer package/runtime.",
    )


def _optional_tabnet() -> BenchmarkModelSpec:
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore

        class _TabNetWrapper:
            def __init__(self, seed: int = 42):
                self.seed = seed
                self.model = TabNetClassifier(seed=seed, verbose=0)

            def fit(self, X, y):
                self.model.fit(X.values, y, max_epochs=120, patience=15, batch_size=512, virtual_batch_size=128)
                return self

            def predict_proba(self, X):
                return self.model.predict_proba(X.values)

        return BenchmarkModelSpec(
            model_id="tabnet",
            model_name="TabNet",
            family="deep_tabular",
            estimator=_TabNetWrapper(),
            param_distributions={},
            available=True,
        )
    except Exception as exc:
        return BenchmarkModelSpec(
            model_id="tabnet",
            model_name="TabNet",
            family="deep_tabular",
            estimator=None,
            param_distributions={},
            available=False,
            skip_reason=f"pytorch_tabnet unavailable: {exc}",
        )


def build_benchmark_model_specs(numeric_features: List[str], categorical_features: List[str], random_state: int = 42) -> List[BenchmarkModelSpec]:
    ladder = build_baseline_ladder(numeric_features, categorical_features, random_state=random_state)
    prep = build_preprocessor(numeric_features=numeric_features, categorical_features=categorical_features, include_interactions=False)

    specs: List[BenchmarkModelSpec] = [
        BenchmarkModelSpec(
            model_id="logreg",
            model_name="Logistic Regression",
            family="classical",
            estimator=ladder["A1_logistic"],
            param_distributions={"clf__C": [0.1, 0.5, 1.0, 3.0, 10.0]},
        ),
        BenchmarkModelSpec(
            model_id="svm_rbf",
            model_name="SVM (RBF)",
            family="classical",
            estimator=Pipeline(
                [
                    ("prep", prep),
                    (
                        "clf",
                        SVC(
                            kernel="rbf",
                            probability=True,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            param_distributions={
                "clf__C": [0.5, 1.0, 2.0, 5.0],
                "clf__gamma": ["scale", 0.01, 0.05, 0.1],
            },
        ),
        BenchmarkModelSpec(
            model_id="random_forest",
            model_name="Random Forest",
            family="classical",
            estimator=ladder["B1_rf"],
            param_distributions={
                "clf__n_estimators": [120, 200, 350],
                "clf__max_depth": [None, 8, 14],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        ),
        BenchmarkModelSpec(
            model_id="boosting_strong",
            model_name="XGBoost/LightGBM/HistGB",
            family="strong_tabular",
            estimator=ladder["B2_gbm"],
            param_distributions={},
        ),
        BenchmarkModelSpec(
            model_id="existing_repo_model",
            model_name="Existing Repo Model (Hybrid)",
            family="repo_specific",
            estimator=ladder["C3_hybrid"],
            param_distributions={},
            is_existing_repo_model=True,
        ),
        BenchmarkModelSpec(
            model_id="proposed_physics_model",
            model_name="Proposed Physics-Aware Model",
            family="proposed",
            estimator=ladder["C2_physics_logit"],
            param_distributions={
                "clf__learning_rate": [0.01, 0.03, 0.05],
                "clf__epochs": [800, 1200, 1600],
                "clf__lambda_phys": [0.1, 0.2, 0.4],
            },
            is_proposed_model=True,
        ),
    ]

    specs.append(_optional_catboost(prep, random_state=random_state))
    specs.append(_optional_ft_transformer())
    specs.append(_optional_tabnet())

    return specs
