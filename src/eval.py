from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold


@dataclass
class ThresholdPack:
    tau_cost: float
    tau_f1: float
    tau_hr: float


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / len(y_true)) * abs(acc - conf)
    return float(ece)


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / max(tn + fp, 1))


def _fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn / max(fn + tp, 1))


def _cost_score(y_true: np.ndarray, y_pred: np.ndarray, c_fn: float = 10.0, c_fp: float = 1.0) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(c_fn * fn + c_fp * fp)


def _threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.sum(y_true) == 0:
        return 1.0
    if np.sum(y_true) == len(y_true):
        return 0.0
    p, r, t = precision_recall_curve(y_true, y_prob)
    if len(t) == 0:
        return 0.5
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    idx = int(np.nanargmax(f1))
    return float(t[idx])


def _threshold_for_high_recall(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.95) -> float:
    if np.sum(y_true) == 0:
        return 1.0
    thresholds = np.linspace(0.0, 1.0, 1001)
    eligible = []
    for tau in thresholds:
        yp = (y_prob >= tau).astype(int)
        rec = recall_score(y_true, yp, zero_division=0)
        if rec >= min_recall:
            eligible.append(tau)
    if not eligible:
        return 0.5
    return float(max(eligible))


def _threshold_for_cost(y_true: np.ndarray, y_prob: np.ndarray, c_fn: float = 10.0, c_fp: float = 1.0) -> float:
    thresholds = np.linspace(0.0, 1.0, 1001)
    costs = []
    for tau in thresholds:
        yp = (y_prob >= tau).astype(int)
        costs.append(_cost_score(y_true, yp, c_fn=c_fn, c_fp=c_fp))
    return float(thresholds[int(np.argmin(costs))])


def select_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    c_fn: float = 10.0,
    c_fp: float = 1.0,
    min_recall: float = 0.95,
) -> ThresholdPack:
    return ThresholdPack(
        tau_cost=_threshold_for_cost(y_true, y_prob, c_fn=c_fn, c_fp=c_fp),
        tau_f1=_threshold_for_f1(y_true, y_prob),
        tau_hr=_threshold_for_high_recall(y_true, y_prob, min_recall=min_recall),
    )


def evaluate_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    c_fn: float = 10.0,
    c_fp: float = 1.0,
    min_recall: float = 0.95,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    uniq = np.unique(y_true)
    single_class = len(uniq) < 2

    if single_class:
        if int(np.sum(y_true)) == 0:
            pr_auc = 0.0
        else:
            pr_auc = 1.0
        roc_auc = float("nan")
        r2_val = float("nan")
    else:
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        r2_val = float(r2_score(y_true, y_prob))

    th = select_thresholds(y_true, y_prob, c_fn=c_fn, c_fp=c_fp, min_recall=min_recall)
    y_f1 = (y_prob >= th.tau_f1).astype(int)
    y_hr = (y_prob >= th.tau_hr).astype(int)
    y_cost = (y_prob >= th.tau_cost).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_f1, labels=[0, 1]).ravel()
    out = {
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "Precision": float(precision_score(y_true, y_f1, zero_division=0)),
        "Recall": float(recall_score(y_true, y_f1, zero_division=0)),
        "F1": float(f1_score(y_true, y_f1, zero_division=0)),
        "Specificity": _specificity(y_true, y_f1),
        "Balanced_Acc": float(balanced_accuracy_score(y_true, y_f1)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        # Forecast-style continuous metrics on probability forecast p(y=1|x)
        "MSE": float(mean_squared_error(y_true, y_prob)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_prob))),
        "MAE": float(mean_absolute_error(y_true, y_prob)),
        "R2": r2_val,
        "ECE": expected_calibration_error(y_true, y_prob, n_bins=10),
        "FNR": _fnr(y_true, y_f1),
        "CostRisk": _cost_score(y_true, y_cost, c_fn=c_fn, c_fp=c_fp),
        "tau_F1": float(th.tau_f1),
        "tau_HR": float(th.tau_hr),
        "tau_cost": float(th.tau_cost),
        "cm_F1_tn": int(tn),
        "cm_F1_fp": int(fp),
        "cm_F1_fn": int(fn),
        "cm_F1_tp": int(tp),
    }

    tn2, fp2, fn2, tp2 = confusion_matrix(y_true, y_hr, labels=[0, 1]).ravel()
    out.update(
        {
            "cm_HR_tn": int(tn2),
            "cm_HR_fp": int(fp2),
            "cm_HR_fn": int(fn2),
            "cm_HR_tp": int(tp2),
            "Recall_HR": float(recall_score(y_true, y_hr, zero_division=0)),
            "Precision_HR": float(precision_score(y_true, y_hr, zero_division=0)),
            "FPR_HR": float(fp2 / max(fp2 + tn2, 1)),
        }
    )
    return out


def generate_splits(
    y: np.ndarray,
    split_mode: str,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    leaveout_feature: str = "Sgn_eff_MVA",
    leaveout_frame: pd.DataFrame | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(y)
    if split_mode == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return [(tr, te) for tr, te in splitter.split(np.zeros(n), y)]
    if split_mode == "grouped":
        if groups is None:
            raise ValueError("groups must be provided for grouped CV")
        splitter = GroupKFold(n_splits=n_splits)
        return [(tr, te) for tr, te in splitter.split(np.zeros(n), y, groups=groups)]
    if split_mode == "leave-level-out":
        if leaveout_frame is None:
            raise ValueError("leaveout_frame required for leave-level-out")
        return [
            (tr, te)
            for tr, te in leave_one_feature_level_out(leaveout_frame, y, feature=leaveout_feature, n_bins=n_splits)
        ]
    raise ValueError(f"Unknown split_mode: {split_mode}")


def make_group_labels(df: pd.DataFrame, cols: List[str], round_decimals: int = 2) -> np.ndarray:
    frame = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            continue
        v = df[c]
        if pd.api.types.is_numeric_dtype(v):
            frame[c] = v.round(round_decimals).astype(str)
        else:
            frame[c] = v.astype(str)
    labels = frame.apply(lambda r: "|".join(r.values), axis=1)
    return labels.values


def leave_one_feature_level_out(
    df: pd.DataFrame,
    y: np.ndarray,
    feature: str,
    n_bins: int = 5,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    vals = pd.qcut(df[feature], q=n_bins, duplicates="drop")
    levels = vals.astype(str).unique()
    idx = np.arange(len(df))
    for lev in levels:
        test_mask = vals.astype(str) == lev
        train_idx = idx[~test_mask]
        test_idx = idx[test_mask]
        if len(np.unique(y[train_idx])) < 2 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def cross_validated_oof(
    estimator: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    split_mode: str = "stratified",
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    leaveout_feature: str = "Sgn_eff_MVA",
    leaveout_frame: pd.DataFrame | None = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    n = len(y)
    oof = np.zeros(n, dtype=float)
    fold_rows: List[Dict[str, float]] = []

    splits = generate_splits(
        y=y,
        split_mode=split_mode,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
        leaveout_feature=leaveout_feature,
        leaveout_frame=leaveout_frame,
    )

    for fold, (tr, te) in enumerate(splits, start=1):
        model = clone(estimator)
        if isinstance(X, pd.DataFrame):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
        else:
            X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        model.fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        oof[te] = p_te

        fold_metrics = evaluate_probabilities(y_te, p_te)
        fold_metrics["fold"] = fold
        fold_rows.append(fold_metrics)

    return oof, fold_rows


def evaluate_model_cv_noisy_test(
    model_name: str,
    estimator: BaseEstimator,
    X_clean: pd.DataFrame | np.ndarray,
    X_noisy: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    split_mode: str,
    scenario: str,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    leaveout_feature: str = "Sgn_eff_MVA",
    leaveout_frame: pd.DataFrame | None = None,
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    n = len(y)
    oof = np.zeros(n, dtype=float)
    fold_rows: List[Dict[str, float]] = []
    splits = generate_splits(
        y=y,
        split_mode=split_mode,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
        leaveout_feature=leaveout_feature,
        leaveout_frame=leaveout_frame,
    )

    for fold, (tr, te) in enumerate(splits, start=1):
        model = clone(estimator)
        if isinstance(X_clean, pd.DataFrame):
            X_tr = X_clean.iloc[tr]
            X_te = X_noisy.iloc[te] if isinstance(X_noisy, pd.DataFrame) else X_noisy[te]
        else:
            X_tr = X_clean[tr]
            X_te = X_noisy[te]
        y_tr, y_te = y[tr], y[te]
        model.fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        oof[te] = p_te
        fold_metrics = evaluate_probabilities(y_te, p_te)
        fold_metrics["fold"] = fold
        fold_rows.append(fold_metrics)

    summary = evaluate_probabilities(y, oof)
    summary["model"] = model_name
    summary["scenario"] = scenario
    summary["split"] = split_mode
    fold_df = pd.DataFrame(fold_rows)
    if len(fold_df) > 0:
        summary["PR_AUC_std"] = float(fold_df["PR_AUC"].std(ddof=0))
        summary["FNR_std"] = float(fold_df["FNR"].std(ddof=0))
    else:
        summary["PR_AUC_std"] = np.nan
        summary["FNR_std"] = np.nan
    return summary, fold_df, oof


def evaluate_model_cv(
    model_name: str,
    estimator: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    split_mode: str,
    scenario: str,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    leaveout_feature: str = "Sgn_eff_MVA",
    leaveout_frame: pd.DataFrame | None = None,
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    oof, fold_rows = cross_validated_oof(
        estimator=estimator,
        X=X,
        y=y,
        split_mode=split_mode,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
        leaveout_feature=leaveout_feature,
        leaveout_frame=leaveout_frame,
    )
    summary = evaluate_probabilities(y, oof)
    summary["model"] = model_name
    summary["scenario"] = scenario
    summary["split"] = split_mode

    fold_df = pd.DataFrame(fold_rows)
    if len(fold_df) > 0:
        summary["PR_AUC_std"] = float(fold_df["PR_AUC"].std(ddof=0))
        summary["FNR_std"] = float(fold_df["FNR"].std(ddof=0))
    else:
        summary["PR_AUC_std"] = np.nan
        summary["FNR_std"] = np.nan

    return summary, fold_df, oof


def add_noise(df: pd.DataFrame, noise_cfg: Dict[str, float], random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = df.copy()
    for col, frac in noise_cfg.items():
        if col not in out.columns:
            continue
        base = out[col].astype(float).values
        sigma = np.abs(base) * frac
        noise = rng.normal(0.0, sigma)
        out[col] = base + noise
    return out


def boundary_consistency_index(fold_prob_surfaces: Iterable[np.ndarray]) -> float:
    mats = [np.asarray(m) for m in fold_prob_surfaces if m is not None]
    if len(mats) < 2:
        return float("nan")
    stacked = np.stack(mats, axis=0)
    return float(np.mean(np.std(stacked, axis=0)))


def minimal_counterfactual(
    model: BaseEstimator,
    x: pd.Series,
    feature_bounds: Dict[str, Tuple[float, float]],
    threshold: float,
    max_iter: int = 2000,
    random_state: int = 42,
) -> Dict[str, float]:
    """Randomized local search for minimal weighted L1 change to achieve stability."""
    rng = np.random.default_rng(random_state)
    base = x.copy()
    current_p = float(model.predict_proba(base.to_frame().T)[:, 1][0])
    if current_p <= threshold:
        return {
            "found": 1,
            "base_p": current_p,
            "counterfactual_p": current_p,
            "l1_change": 0.0,
        }

    direction = {
        "H_s": 1.0,
        "Ikssmin_kA": 1.0,
        "Sgn_eff_MVA": -1.0,
        "Tag_rate": -1.0,
    }
    best = None
    best_l1 = float("inf")

    for _ in range(max_iter):
        cand = base.copy()
        for f, d in direction.items():
            if f not in cand.index or f not in feature_bounds:
                continue
            mn, mx = feature_bounds[f]
            span = mx - mn
            step = rng.uniform(0.0, 0.12) * span * d
            cand[f] = np.clip(cand[f] + step, mn, mx)
        p = float(model.predict_proba(cand.to_frame().T)[:, 1][0])
        if p <= threshold:
            l1 = float(np.sum(np.abs((cand - base).values.astype(float))))
            if l1 < best_l1:
                best_l1 = l1
                best = cand

    if best is None:
        return {"found": 0, "base_p": current_p, "counterfactual_p": current_p, "l1_change": float("nan")}

    res = {"found": 1, "base_p": current_p, "counterfactual_p": float(model.predict_proba(best.to_frame().T)[:, 1][0]), "l1_change": best_l1}
    for k in ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"]:
        if k in base.index and k in best.index:
            res[f"delta_{k}"] = float(best[k] - base[k])
    return res
