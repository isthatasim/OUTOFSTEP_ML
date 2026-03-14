from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .features import build_preprocessor


@dataclass
class ModelSpec:
    name: str
    estimator: BaseEstimator
    tier: str
    calibrated: bool = False


class TwoStageHybridClassifier(BaseEstimator, ClassifierMixin):
    """Stage-1 classifier followed by Stage-2 isotonic smoothing on stage-1 risk."""

    def __init__(self, base_estimator: Optional[BaseEstimator] = None):
        self.base_estimator = base_estimator
        self._base = None
        self._iso = None

    def fit(self, X, y):
        base = clone(self.base_estimator) if self.base_estimator is not None else HistGradientBoostingClassifier(random_state=42)
        base.fit(X, y)
        p = base.predict_proba(X)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        self._base = base
        self._iso = iso
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self._base.predict_proba(X)[:, 1]
        p2 = self._iso.transform(p)
        p2 = np.clip(p2, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p2, p2])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PhysicsRegularizedLogistic(BaseEstimator, ClassifierMixin):
    """Logistic regression trained with sign-based physics priors."""

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        monotonic_signs: Optional[Dict[str, int]] = None,
        learning_rate: float = 0.03,
        epochs: int = 1200,
        l2: float = 1e-3,
        lambda_phys: float = 0.2,
        random_state: int = 42,
    ):
        self.feature_names = feature_names
        self.monotonic_signs = monotonic_signs or {"H_s": -1, "Ikssmin_kA": -1, "Sgn_eff_MVA": 1}
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.lambda_phys = lambda_phys
        self.random_state = random_state

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        Xs = (X - self.mean_) / self.std_

        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(0.0, 0.01, size=d)
        self.b_ = 0.0

        feat_names = self.feature_names if self.feature_names is not None else [f"x{i}" for i in range(d)]
        idx_sign = {i: self.monotonic_signs[name] for i, name in enumerate(feat_names) if name in self.monotonic_signs}

        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1.0)

        for _ in range(self.epochs):
            z = Xs @ self.w_ + self.b_
            p = self._sigmoid(z)

            err = (p - y) * np.where(y > 0.5, pos_weight, 1.0)
            grad_w = (Xs.T @ err) / n + self.l2 * self.w_
            grad_b = np.mean(err)

            for idx, sign in idx_sign.items():
                if sign < 0:
                    viol = max(0.0, self.w_[idx])
                    grad_w[idx] += self.lambda_phys * 2.0 * viol
                elif sign > 0:
                    viol = max(0.0, -self.w_[idx])
                    grad_w[idx] += self.lambda_phys * (-2.0 * viol)

            self.w_ -= self.learning_rate * grad_w
            self.b_ -= self.learning_rate * grad_b

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        p = self._sigmoid(Xs @ self.w_ + self.b_)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PhysicsAwareNN(BaseEstimator, ClassifierMixin):
    """Tiny physics-aware NN with finite-difference monotonic penalties (torch, optional)."""

    def __init__(
        self,
        feature_names: List[str],
        monotonic_signs: Optional[Dict[str, int]] = None,
        hidden_dim: int = 16,
        lr: float = 5e-3,
        epochs: int = 300,
        lambda_phys: float = 0.3,
        fd_eps: float = 1e-2,
        random_state: int = 42,
    ):
        self.feature_names = feature_names
        self.monotonic_signs = monotonic_signs or {"H_s": -1, "Ikssmin_kA": -1, "Sgn_eff_MVA": 1}
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.lambda_phys = lambda_phys
        self.fd_eps = fd_eps
        self.random_state = random_state

    def fit(self, X, y):
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PhysicsAwareNN requires torch. Use PhysicsRegularizedLogistic if torch is unavailable.") from exc

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n, d = X.shape

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        Xs = (X - self.mean_) / self.std_

        torch.manual_seed(self.random_state)
        x_t = torch.tensor(Xs, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        model = nn.Sequential(
            nn.Linear(d, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        pos = y_t.sum().item()
        neg = float(len(y_t) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        idx_sign = {
            i: self.monotonic_signs[name]
            for i, name in enumerate(self.feature_names)
            if name in self.monotonic_signs
        }

        for _ in range(self.epochs):
            logits = model(x_t)
            loss = bce(logits, y_t)
            p = torch.sigmoid(logits)

            phys_pen = torch.tensor(0.0)
            for idx, sign in idx_sign.items():
                x_eps = x_t.clone()
                x_eps[:, idx] += self.fd_eps
                p_eps = torch.sigmoid(model(x_eps))
                delta = (p_eps - p) / self.fd_eps
                if sign < 0:
                    phys_pen = phys_pen + torch.relu(delta).mean()
                elif sign > 0:
                    phys_pen = phys_pen + torch.relu(-delta).mean()

            obj = loss + self.lambda_phys * phys_pen
            opt.zero_grad()
            obj.backward()
            opt.step()

        self.model_ = model
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        import torch

        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        with torch.no_grad():
            logits = self.model_(torch.tensor(Xs, dtype=torch.float32)).numpy().reshape(-1)
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _try_xgboost(random_state: int):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )
    except Exception:
        return None


def _try_lightgbm(random_state: int):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=random_state,
        )
    except Exception:
        return None


def _build_histgb(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=200,
        l2_regularization=1e-3,
        random_state=random_state,
    )


def _build_monotonic_histgb(random_state: int, monotonic_cst: List[int]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=180,
        l2_regularization=1e-3,
        monotonic_cst=monotonic_cst,
        random_state=random_state,
    )


def make_model_registry(
    numeric_features: List[str],
    categorical_features: List[str],
    monotonic_cst: List[int],
    include_physics_nn: bool = False,
    random_state: int = 42,
) -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    prep_linear: ColumnTransformer = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        include_interactions=True,
    )
    prep_tree: ColumnTransformer = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        include_interactions=False,
    )

    lr = Pipeline(
        [
            ("prep", prep_linear),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    specs.append(ModelSpec(name="tierA_logistic", estimator=lr, tier="A"))

    dt = Pipeline(
        [
            ("prep", prep_tree),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    specs.append(ModelSpec(name="tierA_tree", estimator=dt, tier="A"))

    rf = Pipeline(
        [
            ("prep", prep_tree),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=220,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    specs.append(ModelSpec(name="tierB_random_forest", estimator=rf, tier="B"))

    gb_base = _try_xgboost(random_state) or _try_lightgbm(random_state) or _build_histgb(random_state)
    gb = Pipeline([("prep", prep_tree), ("clf", gb_base)])
    specs.append(ModelSpec(name="tierB_gradient_boosting", estimator=gb, tier="B"))

    mono_prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            )
        ],
        remainder="drop",
    )
    mono = Pipeline(
        [
            ("prep", mono_prep),
            ("clf", _build_monotonic_histgb(random_state, monotonic_cst=monotonic_cst)),
        ]
    )
    specs.append(ModelSpec(name="tierC_monotonic_hgb", estimator=mono, tier="C"))

    phys_logit = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "num",
                            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                            numeric_features,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            (
                "clf",
                PhysicsRegularizedLogistic(
                    feature_names=numeric_features,
                    monotonic_signs={"H_s": -1, "Ikssmin_kA": -1, "Sgn_eff_MVA": 1},
                    random_state=random_state,
                ),
            ),
        ]
    )
    specs.append(ModelSpec(name="tierC_physics_logit", estimator=phys_logit, tier="C"))

    # Optional finite-difference neural model (requires torch, can be slow)
    if include_physics_nn:
        try:
            _ = __import__("torch")
            phys_nn = Pipeline(
                [
                    (
                        "prep",
                        ColumnTransformer(
                            [
                                (
                                    "num",
                                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                                    numeric_features,
                                )
                            ],
                            remainder="drop",
                        ),
                    ),
                    (
                        "clf",
                        PhysicsAwareNN(
                            feature_names=numeric_features,
                            monotonic_signs={"H_s": -1, "Ikssmin_kA": -1, "Sgn_eff_MVA": 1},
                            epochs=80,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            specs.append(ModelSpec(name="tierC_physics_nn", estimator=phys_nn, tier="C"))
        except Exception:
            pass

    hybrid = Pipeline(
        [
            ("prep", prep_tree),
            (
                "clf",
                TwoStageHybridClassifier(
                    base_estimator=RandomForestClassifier(
                        n_estimators=150,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=1,
                    )
                ),
            ),
        ]
    )
    specs.append(ModelSpec(name="tierC_two_stage_hybrid", estimator=hybrid, tier="C"))

    return specs


def make_calibrated_model(model: BaseEstimator, method: str = "isotonic", cv: int = 3) -> BaseEstimator:
    try:
        return CalibratedClassifierCV(estimator=clone(model), method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=clone(model), method=method, cv=cv)


def calibration_candidates(model: BaseEstimator, cv: int = 3) -> Dict[str, BaseEstimator]:
    return {
        "sigmoid": make_calibrated_model(model, method="sigmoid", cv=cv),
        "isotonic": make_calibrated_model(model, method="isotonic", cv=cv),
    }

from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RandomizedSearchCV


def controlled_random_search(
    estimator: BaseEstimator,
    param_distributions: Dict[str, list],
    X,
    y,
    cv,
    n_iter: int = 12,
    random_state: int = 42,
    scoring: str = "average_precision",
):
    search = RandomizedSearchCV(
        estimator=clone(estimator),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


def select_best_calibration(
    fitted_model: BaseEstimator,
    X_train,
    y_train,
    X_valid,
    y_valid,
) -> tuple[BaseEstimator, str, float]:
    best_model = fitted_model
    best_name = "none"
    p0 = fitted_model.predict_proba(X_valid)[:, 1]
    best_brier = brier_score_loss(y_valid, p0)

    for method in ["sigmoid", "isotonic"]:
        cal = make_calibrated_model(fitted_model, method=method, cv=3)
        cal.fit(X_train, y_train)
        p = cal.predict_proba(X_valid)[:, 1]
        brier = brier_score_loss(y_valid, p)
        if brier < best_brier:
            best_brier = brier
            best_model = cal
            best_name = method

    return best_model, best_name, float(best_brier)
