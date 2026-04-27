from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import derive_feature_bounds, resolve_engineered_feature_columns
from src.outofstep_ml.benchmark.runner import (
    _apply_calibrator,
    _class_metrics,
    _prob_metrics,
    _select_best_calibration,
    make_strict_split,
)
from src.outofstep_ml.data.loaders import load_validated_dataset
from src.outofstep_ml.models.baselines import build_baseline_ladder
from src.outofstep_ml.models.thresholds import optimize_thresholds
from src.outofstep_ml.utils.io import ensure_dir, load_yaml, save_json
from src.outofstep_ml.utils.seed import set_global_seed


def _reference_stats(df: pd.DataFrame, columns: list[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in columns:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            stats[col] = {
                "min": float(s.min()),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build product artifact for grid synchronization compatibility screening.")
    ap.add_argument("--config", default="configs/logic_ladder.yaml")
    ap.add_argument("--model-policy", choices=["best_predictive", "deployment_safety"], default="deployment_safety")
    ap.add_argument("--output-dir", default="outputs/product")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    df, audit = load_validated_dataset(
        cfg.get("data", {}).get("path"),
        include_logs=bool(cfg.get("features", {}).get("include_logs", True)),
    )
    y = df["Out_of_step"].astype(int).values
    base = ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"]
    engineered = resolve_engineered_feature_columns(df)
    numeric = [c for c in base + engineered if c in df.columns]
    categorical = ["GenName"] if "GenName" in df.columns else []
    feature_columns = numeric + categorical
    X = df[feature_columns].copy()

    split = make_strict_split(
        df=df,
        y=y,
        seed=seed,
        strategy=str(cfg.get("strict_eval", {}).get("strategy", "stratified")),
        test_size=float(cfg.get("strict_eval", {}).get("test_size", 0.2)),
        val_size=float(cfg.get("strict_eval", {}).get("val_size", 0.2)),
        group_cols=base,
        leave_feature="Sgn_eff_MVA",
    )
    X_train, y_train = X.iloc[split.train_idx], y[split.train_idx]
    X_val, y_val = X.iloc[split.val_idx], y[split.val_idx]
    X_test, y_test = X.iloc[split.test_idx], y[split.test_idx]

    ladder = build_baseline_ladder(numeric_features=numeric, categorical_features=categorical, random_state=seed)
    if args.model_policy == "best_predictive":
        model_name = "PhysiScreen-OOS product predictor (1+2 raw + engineered)"
        model = ladder["A1_logistic"]
        use_calibration = False
        use_cost_threshold = False
    else:
        model_name = "PhysiScreen-OOS product safety stack (1+2+4+5 calibrated cost-sensitive)"
        model = ladder["C2_physics_logit"]
        use_calibration = True
        use_cost_threshold = True

    model.fit(X_train, y_train)
    p_val_raw = np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
    calib_name, calib_obj = "none", None
    if use_calibration:
        calib_name, calib_obj, _ = _select_best_calibration(y_val, p_val_raw)
    p_val = np.clip(_apply_calibrator(calib_obj, calib_name, p_val_raw), 1e-6, 1 - 1e-6)
    thresholds = optimize_thresholds(
        y_val,
        p_val,
        c_fn=float(cfg.get("thresholds", {}).get("c_fn", 10.0)),
        c_fp=float(cfg.get("thresholds", {}).get("c_fp", 1.0)),
        min_recall=float(cfg.get("thresholds", {}).get("min_recall", 0.95)),
    )
    threshold_used = float(thresholds["tau_cost"] if use_cost_threshold else 0.5)

    p_test_raw = np.clip(model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)
    p_test = np.clip(_apply_calibrator(calib_obj, calib_name, p_test_raw), 1e-6, 1 - 1e-6)
    y_hat = (p_test >= threshold_used).astype(int)

    out_dir = ensure_dir(args.output_dir)
    bundle: Dict[str, Any] = {
        "model": model,
        "calibrator": calib_obj,
        "calibration_method": calib_name,
        "feature_columns": feature_columns,
        "raw_features": base,
        "engineered_features": engineered,
        "include_logs": bool(cfg.get("features", {}).get("include_logs", True)),
        "threshold_used": threshold_used,
        "threshold_policy": "tau_cost" if use_cost_threshold else "tau_default",
        "thresholds": thresholds,
        "model_name": model_name,
        "model_policy": args.model_policy,
        "model_version": "grid-sync-oos-v1",
        "feature_bounds": derive_feature_bounds(df, numeric),
        "reference_stats": _reference_stats(df, base + engineered),
        "training_data_audit": audit.to_dict(),
        "strict_split": {
            "strategy": split.strategy,
            "seed": split.seed,
            "n_train": int(len(split.train_idx)),
            "n_val": int(len(split.val_idx)),
            "n_test": int(len(split.test_idx)),
        },
        "holdout_metrics": {
            **_prob_metrics(y_test, p_test),
            **_class_metrics(y_test, y_hat),
        },
    }
    joblib.dump(bundle, Path(out_dir) / "grid_sync_bundle.joblib")
    save_json(Path(out_dir) / "grid_sync_model_card.json", {k: v for k, v in bundle.items() if k not in {"model", "calibrator"}})

    pd.DataFrame(
        [
            {
                "model_name": model_name,
                "model_policy": args.model_policy,
                "threshold_used": threshold_used,
                "calibration": calib_name,
                **bundle["holdout_metrics"],
            }
        ]
    ).to_csv(Path(out_dir) / "grid_sync_holdout_metrics.csv", index=False)
    print(f"Saved product bundle to {Path(out_dir) / 'grid_sync_bundle.joblib'}")
    print(f"Verdict threshold: p_oos >= {threshold_used:.4f} => NOT_COMPATIBLE_HIGH_OOS_RISK")


if __name__ == "__main__":
    main()
