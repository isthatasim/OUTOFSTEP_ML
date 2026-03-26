from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.eval import add_noise, expected_calibration_error
from src.features import resolve_engineered_feature_columns
from src.outofstep_ml.benchmark.model_zoo import BenchmarkModelSpec, build_benchmark_model_specs
from src.outofstep_ml.data.loaders import load_validated_dataset
from src.outofstep_ml.data.splitters import build_groups
from src.outofstep_ml.models.thresholds import optimize_thresholds
from src.outofstep_ml.utils.io import ensure_dir, save_json
from src.outofstep_ml.utils.seed import set_global_seed


@dataclass
class StrictSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    strategy: str
    seed: int


def _assert_disjoint_split(split: StrictSplit, n_rows: int) -> None:
    tr = set(split.train_idx.tolist())
    va = set(split.val_idx.tolist())
    te = set(split.test_idx.tolist())
    if tr & va or tr & te or va & te:
        raise ValueError(
            f"Split leakage detected for strategy={split.strategy}, seed={split.seed}: "
            "train/val/test indices overlap."
        )
    union = tr | va | te
    if len(union) != n_rows:
        raise ValueError(
            f"Incomplete split coverage for strategy={split.strategy}, seed={split.seed}: "
            f"covered={len(union)} expected={n_rows}."
        )


def _prob_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    p = np.asarray(p).astype(float)
    out: Dict[str, float] = {}
    if len(np.unique(y_true)) >= 2:
        out["ROC_AUC"] = float(roc_auc_score(y_true, p))
        out["PR_AUC"] = float(average_precision_score(y_true, p))
    else:
        out["ROC_AUC"] = float("nan")
        out["PR_AUC"] = float("nan")
    out["Brier"] = float(brier_score_loss(y_true, p))
    out["ECE"] = float(expected_calibration_error(y_true, p, n_bins=10))
    out["MAE"] = float(np.mean(np.abs(y_true - p)))
    out["RMSE"] = float(np.sqrt(np.mean((y_true - p) ** 2)))
    return out


def _class_metrics(y_true: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_hat = y_hat.astype(int)
    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    tn = int(((y_true == 0) & (y_hat == 0)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())
    spec = float(tn / max(tn + fp, 1))
    fnr = float(fn / max(fn + tp, 1))
    return {
        "Precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "Recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "F1": float(f1_score(y_true, y_hat, zero_division=0)),
        "Specificity": spec,
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_hat)),
        "FNR": fnr,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def _sample_params(param_distributions: Dict[str, list], rng: np.random.Generator) -> Dict[str, Any]:
    out = {}
    for k, vals in param_distributions.items():
        if not vals:
            continue
        out[k] = vals[int(rng.integers(0, len(vals)))]
    return out


def _fit_with_budget(
    spec: BenchmarkModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int,
    seed: int,
) -> Tuple[Any, Dict[str, Any], float]:
    rng = np.random.default_rng(seed)
    best_score = -np.inf
    best_model = None
    best_params: Dict[str, Any] = {}

    trials = max(1, n_trials if spec.param_distributions else 1)
    for _ in range(trials):
        params = _sample_params(spec.param_distributions, rng)
        model = clone(spec.estimator)
        if params:
            model.set_params(**params)

        model.fit(X_train, y_train)
        p_val = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, p_val) if len(np.unique(y_val)) >= 2 else float(np.mean(p_val))
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    return best_model, best_params, float(best_score)


def _fit_calibrator(y_val: np.ndarray, p_val: np.ndarray, method: str):
    if len(np.unique(y_val)) < 2:
        return None
    if method == "platt":
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(p_val.reshape(-1, 1), y_val)
        return lr
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_val, y_val)
        return iso
    return None


def _apply_calibrator(calibrator, method: str, p: np.ndarray) -> np.ndarray:
    if calibrator is None or method == "none":
        return p
    if method == "platt":
        return calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return calibrator.transform(p)
    return p


def _select_best_calibration(y_val: np.ndarray, p_val: np.ndarray) -> Tuple[str, Any, Dict[str, float]]:
    if len(np.unique(y_val)) < 2:
        brier = float(brier_score_loss(y_val, np.clip(p_val, 1e-6, 1 - 1e-6)))
        ece = float(expected_calibration_error(y_val, np.clip(p_val, 1e-6, 1 - 1e-6), n_bins=10))
        return "none", None, {"none_brier": brier, "none_ece": ece}

    methods = ["none", "platt", "isotonic"]
    best_method = "none"
    best_calibrator = None
    best_brier = np.inf
    stats: Dict[str, float] = {}

    for m in methods:
        cal = _fit_calibrator(y_val, p_val, m)
        pc = np.clip(_apply_calibrator(cal, m, p_val), 1e-6, 1 - 1e-6)
        brier = float(brier_score_loss(y_val, pc))
        ece = float(expected_calibration_error(y_val, pc, n_bins=10))
        stats[f"{m}_brier"] = brier
        stats[f"{m}_ece"] = ece
        if brier < best_brier:
            best_brier = brier
            best_method = m
            best_calibrator = cal

    return best_method, best_calibrator, stats


def make_strict_split(
    df: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    strategy: str,
    test_size: float,
    val_size: float,
    group_cols: List[str],
    leave_feature: str,
) -> StrictSplit:
    idx = np.arange(len(df))

    if strategy == "grouped":
        groups = build_groups(df, group_cols, round_decimals=2)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trv_i, te_i = next(gss.split(idx, y, groups=groups))
        train_val_idx = idx[trv_i]
        test_idx = idx[te_i]
    elif strategy.startswith("leave_"):
        feat = leave_feature
        bins = pd.qcut(df[feat], q=6, duplicates="drop")
        levels = list(pd.Series(bins.astype(str)).unique())
        chosen = levels[seed % len(levels)]
        test_mask = bins.astype(str) == chosen
        test_idx = idx[test_mask]
        train_val_idx = idx[~test_mask]
        if len(test_idx) == 0 or len(np.unique(y[test_idx])) < 1:
            train_val_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    else:
        train_val_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)

    y_trv = y[train_val_idx]
    val_frac_within_trv = val_size / max(1.0 - test_size, 1e-9)
    if len(np.unique(y_trv)) >= 2:
        tr_i, va_i = train_test_split(train_val_idx, test_size=val_frac_within_trv, stratify=y_trv, random_state=seed)
    else:
        tr_i, va_i = train_test_split(train_val_idx, test_size=val_frac_within_trv, random_state=seed)

    return StrictSplit(train_idx=np.asarray(tr_i), val_idx=np.asarray(va_i), test_idx=np.asarray(test_idx), strategy=strategy, seed=seed)


def _shifted_tests(X_test: pd.DataFrame, y_test: np.ndarray, noise_cfg: Dict[str, float], missing_feature: str) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    out = {"clean": (X_test.copy(), y_test.copy())}
    out["noisy"] = (add_noise(X_test, noise_cfg, random_state=123), y_test.copy())

    X_miss = X_test.copy()
    if missing_feature in X_miss.columns:
        X_miss[missing_feature] = np.nan
    out["missing_features"] = (X_miss, y_test.copy())

    # Unseen-regime proxy: high-stress tail on test
    if "Sgn_eff_MVA" in X_test.columns:
        thr = float(np.quantile(X_test["Sgn_eff_MVA"], 0.8))
        mask = X_test["Sgn_eff_MVA"] >= thr
        if mask.sum() > 10:
            out["unseen_regime"] = (X_test.loc[mask].copy(), y_test[mask.values])

    # Group shift proxy: low-I tail on test
    if "Ikssmin_kA" in X_test.columns:
        thr_i = float(np.quantile(X_test["Ikssmin_kA"], 0.2))
        mask = X_test["Ikssmin_kA"] <= thr_i
        if mask.sum() > 10:
            out["group_shift"] = (X_test.loc[mask].copy(), y_test[mask.values])

    return out


def run_full_benchmark(cfg: Dict[str, Any]) -> Dict[str, Path]:
    seed0 = int(cfg.get("seed", 42))
    set_global_seed(seed0)

    out_root = ensure_dir(cfg.get("outputs", {}).get("root", "results"))
    out_tables = ensure_dir(cfg.get("outputs", {}).get("table_dir", Path(out_root) / "tables"))
    out_splits = ensure_dir(cfg.get("outputs", {}).get("split_dir", Path(out_root) / "splits"))
    out_models = ensure_dir(cfg.get("outputs", {}).get("model_dir", Path(out_root) / "model"))

    df, audit = load_validated_dataset(cfg.get("data", {}).get("path"), include_logs=bool(cfg.get("features", {}).get("include_logs", True)))
    save_json(Path(out_tables) / "benchmark_data_audit.json", audit.to_dict())

    y = df["Out_of_step"].astype(int).values
    base_numeric = ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"]
    engineered = resolve_engineered_feature_columns(df)
    use_engineered = bool(cfg.get("features", {}).get("use_engineered", True))
    numeric = base_numeric + engineered if use_engineered else base_numeric
    numeric = [c for c in numeric if c in df.columns]
    categorical = ["GenName"] if "GenName" in df.columns else []
    X = df[numeric + categorical].copy()

    specs = build_benchmark_model_specs(numeric_features=numeric, categorical_features=categorical, random_state=seed0)
    pd.DataFrame(
        [
            {
                "model_id": s.model_id,
                "model_name": s.model_name,
                "family": s.family,
                "available": s.available,
                "skip_reason": s.skip_reason,
            }
            for s in specs
        ]
    ).to_csv(Path(out_tables) / "model_availability.csv", index=False)

    n_seeds = int(cfg.get("benchmark", {}).get("n_seeds", 3))
    seeds = [seed0 + i for i in range(n_seeds)]
    split_strategies = cfg.get("benchmark", {}).get("split_strategies", ["stratified", "grouped", "leave_S", "leave_I"])

    strict_cfg = cfg.get("strict_eval", {})
    test_size = float(strict_cfg.get("test_size", 0.2))
    val_size = float(strict_cfg.get("val_size", 0.2))
    if not bool(strict_cfg.get("enforce_no_test_tuning", True)):
        raise ValueError("strict_eval.enforce_no_test_tuning must be true for this benchmark runner.")
    n_trials = int(cfg.get("training_budget", {}).get("n_trials", 15))

    c_fn = float(cfg.get("thresholds", {}).get("c_fn", 10.0))
    c_fp = float(cfg.get("thresholds", {}).get("c_fp", 1.0))
    min_recall = float(cfg.get("thresholds", {}).get("min_recall", 0.95))

    noise_cfg = cfg.get("robustness", {}).get("noise", {"Tag_rate": 0.01, "Ikssmin_kA": 0.02, "Sgn_eff_MVA": 0.02, "H_s": 0.01})
    missing_feature = cfg.get("robustness", {}).get("missing_feature", "Sgn_eff_MVA")

    rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    robustness_rows: List[Dict[str, Any]] = []
    efficiency_rows: List[Dict[str, Any]] = []
    split_manifest_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        for strategy in split_strategies:
            leave_feature = "Sgn_eff_MVA" if strategy == "leave_S" else "Ikssmin_kA"
            split = make_strict_split(
                df=df,
                y=y,
                seed=seed,
                strategy=strategy,
                test_size=test_size,
                val_size=val_size,
                group_cols=["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"],
                leave_feature=leave_feature,
            )
            _assert_disjoint_split(split, n_rows=len(df))

            split_frame = pd.DataFrame(
                {
                    "index": np.concatenate([split.train_idx, split.val_idx, split.test_idx]),
                    "subset": ["train"] * len(split.train_idx) + ["val"] * len(split.val_idx) + ["test"] * len(split.test_idx),
                    "seed": seed,
                    "strategy": strategy,
                }
            )
            split_frame.to_csv(Path(out_splits) / f"strict_split_seed{seed}_{strategy}.csv", index=False)
            np.savez_compressed(
                Path(out_splits) / f"strict_split_seed{seed}_{strategy}.npz",
                train_idx=split.train_idx,
                val_idx=split.val_idx,
                test_idx=split.test_idx,
            )
            split_manifest_rows.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "n_train": int(len(split.train_idx)),
                    "n_val": int(len(split.val_idx)),
                    "n_test": int(len(split.test_idx)),
                    "split_csv": str(Path(out_splits) / f"strict_split_seed{seed}_{strategy}.csv"),
                    "split_npz": str(Path(out_splits) / f"strict_split_seed{seed}_{strategy}.npz"),
                }
            )

            X_train, y_train = X.iloc[split.train_idx], y[split.train_idx]
            X_val, y_val = X.iloc[split.val_idx], y[split.val_idx]
            X_test, y_test = X.iloc[split.test_idx], y[split.test_idx]

            for spec in specs:
                base_row = {
                    "seed": seed,
                    "strategy": strategy,
                    "model_id": spec.model_id,
                    "model_name": spec.model_name,
                    "family": spec.family,
                    "available": spec.available,
                }

                if not spec.available or spec.estimator is None:
                    rows.append({**base_row, "status": "skipped", "skip_reason": spec.skip_reason})
                    continue

                train_t0 = time.perf_counter()
                model, best_params, val_score = _fit_with_budget(
                    spec,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    n_trials=n_trials,
                    seed=seed,
                )
                train_time = float(time.perf_counter() - train_t0)

                # Calibration fit and threshold selection strictly on validation only
                p_val_raw = np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
                best_calib_name, best_calib, calib_stats = _select_best_calibration(y_val, p_val_raw)
                p_val_cal = np.clip(_apply_calibrator(best_calib, best_calib_name, p_val_raw), 1e-6, 1 - 1e-6)

                th = optimize_thresholds(y_val, p_val_cal, c_fn=c_fn, c_fp=c_fp, min_recall=min_recall)

                # Final holdout evaluation (single-pass; no tuning on test)
                infer_t0 = time.perf_counter()
                p_test_raw = np.clip(model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)
                p_test = np.clip(_apply_calibrator(best_calib, best_calib_name, p_test_raw), 1e-6, 1 - 1e-6)
                infer_ms = float((time.perf_counter() - infer_t0) * 1000.0 / max(len(X_test), 1))

                prob_m = _prob_metrics(y_test, p_test)
                y_hat_cost = (p_test >= th["tau_cost"]).astype(int)
                cls_m = _class_metrics(y_test, y_hat_cost)

                row = {
                    **base_row,
                    "status": "ok",
                    **prob_m,
                    **cls_m,
                    "Training_Time_s": train_time,
                    "Inference_ms_per_sample": infer_ms,
                    "Validation_PR_AUC": val_score,
                    "Calibration": best_calib_name,
                    "best_params": best_params,
                    "is_existing_repo_model": spec.is_existing_repo_model,
                    "is_proposed_model": spec.is_proposed_model,
                }
                rows.append(row)

                threshold_rows.append(
                    {
                        **base_row,
                        "tau_default": 0.5,
                        "tau_F1": th["tau_f1"],
                        "tau_cost": th["tau_cost"],
                        "tau_HR": th["tau_hr"],
                        "Precision@tau_cost": float(precision_score(y_test, y_hat_cost, zero_division=0)),
                        "Recall@tau_cost": float(recall_score(y_test, y_hat_cost, zero_division=0)),
                        "F1@tau_cost": float(f1_score(y_test, y_hat_cost, zero_division=0)),
                        "Cost@tau_cost": float(c_fn * ((y_test == 1) & (y_hat_cost == 0)).sum() + c_fp * ((y_test == 0) & (y_hat_cost == 1)).sum()),
                    }
                )

                calibration_rows.append(
                    {
                        **base_row,
                        "Uncalibrated_Brier": float(brier_score_loss(y_test, p_test_raw)),
                        "Calibrated_Brier": float(brier_score_loss(y_test, p_test)),
                        "Uncalibrated_ECE": float(expected_calibration_error(y_test, p_test_raw, n_bins=10)),
                        "Calibrated_ECE": float(expected_calibration_error(y_test, p_test, n_bins=10)),
                        "Best_Calibration": best_calib_name,
                        **calib_stats,
                    }
                )

                shifts = _shifted_tests(X_test, y_test, noise_cfg=noise_cfg, missing_feature=missing_feature)
                clean_ref = prob_m["PR_AUC"]
                for shift_name, (Xs, ys) in shifts.items():
                    p_shift = np.clip(_apply_calibrator(best_calib, best_calib_name, model.predict_proba(Xs)[:, 1]), 1e-6, 1 - 1e-6)
                    m_shift = _prob_metrics(ys, p_shift)
                    robustness_rows.append(
                        {
                            **base_row,
                            "shift": shift_name,
                            "PR_AUC": m_shift["PR_AUC"],
                            "Performance_Drop": float(clean_ref - m_shift["PR_AUC"]) if not np.isnan(clean_ref) and not np.isnan(m_shift["PR_AUC"]) else float("nan"),
                        }
                    )

                temp_model_path = Path(out_models) / f"tmp_{spec.model_id}_{seed}_{strategy}.joblib"
                joblib.dump(model, temp_model_path)
                model_size_mb = float(temp_model_path.stat().st_size / (1024.0 * 1024.0))
                temp_model_path.unlink(missing_ok=True)

                efficiency_rows.append(
                    {
                        **base_row,
                        "Training_Time_s": train_time,
                        "Inference_ms_per_sample": infer_ms,
                        "Model_Size_MB": model_size_mb,
                        "Suitable_for_Online_Screening": bool(infer_ms <= 10.0),
                    }
                )

    raw_df = pd.DataFrame(rows)
    th_df = pd.DataFrame(threshold_rows)
    cal_df = pd.DataFrame(calibration_rows)
    rob_df = pd.DataFrame(robustness_rows)
    eff_df = pd.DataFrame(efficiency_rows)

    raw_df.to_csv(Path(out_tables) / "benchmark_raw_results.csv", index=False)
    th_df.to_csv(Path(out_tables) / "benchmark_thresholds_raw.csv", index=False)
    cal_df.to_csv(Path(out_tables) / "benchmark_calibration_raw.csv", index=False)
    rob_df.to_csv(Path(out_tables) / "benchmark_robustness_raw.csv", index=False)
    eff_df.to_csv(Path(out_tables) / "benchmark_efficiency_raw.csv", index=False)
    pd.DataFrame(split_manifest_rows).to_csv(Path(out_splits) / "split_manifest.csv", index=False)
    strict_protocol = {
        "strict_evaluation_protocol": {
            "1_train_val_test_split": True,
            "2_fit_on_train_only": True,
            "3_tune_calibrate_threshold_on_validation_only": True,
            "4_final_test_used_once_for_unbiased_comparison": True,
            "5_shifted_tests_not_used_for_tuning": True,
            "6_split_indices_and_seeds_saved": True,
            "notes": "No hyperparameter, calibrator, or threshold tuning is performed on final holdout test.",
        },
        "seed0": seed0,
        "seeds": seeds,
        "split_strategies": split_strategies,
        "test_size": test_size,
        "val_size": val_size,
        "train_size": float(1.0 - test_size - val_size),
    }
    save_json(Path(out_splits) / "strict_evaluation_protocol.json", strict_protocol)

    return {
        "raw": Path(out_tables) / "benchmark_raw_results.csv",
        "thresholds": Path(out_tables) / "benchmark_thresholds_raw.csv",
        "calibration": Path(out_tables) / "benchmark_calibration_raw.csv",
        "robustness": Path(out_tables) / "benchmark_robustness_raw.csv",
        "efficiency": Path(out_tables) / "benchmark_efficiency_raw.csv",
        "split_manifest": Path(out_splits) / "split_manifest.csv",
        "strict_protocol": Path(out_splits) / "strict_evaluation_protocol.json",
    }
