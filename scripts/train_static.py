from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import derive_feature_bounds
from src.outofstep_ml.data.loaders import load_validated_dataset
from src.outofstep_ml.data.splitters import build_groups, create_split_manifest, save_split_manifest
from src.outofstep_ml.evaluation.metrics import compute_all_metrics
from src.outofstep_ml.evaluation.validation import run_validation
from src.outofstep_ml.models.baselines import build_baseline_ladder
from src.outofstep_ml.models.static_physics_model import StaticModelConfig, StaticPhysicsRiskModel
from src.outofstep_ml.models.thresholds import optimize_thresholds
from src.outofstep_ml.utils.io import ensure_dir, load_yaml, save_json, save_yaml
from src.outofstep_ml.utils.logging_utils import get_logger
from src.outofstep_ml.utils.seed import set_global_seed


def _merge_cfg(path: Path) -> Dict[str, Any]:
    cfg = load_yaml(path)
    ext = cfg.get("extends")
    if not ext:
        return cfg
    base = load_yaml(path.parent / "base.yaml") if ext == "base" else load_yaml(path.parent / ext)

    def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out

    return merge(base, cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = _merge_cfg(cfg_path)

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    out_root = ensure_dir(cfg.get("outputs", {}).get("root", "results"))
    out_logs = ensure_dir(Path(out_root) / "logs")
    out_tables = ensure_dir(cfg.get("outputs", {}).get("table_dir", out_root / "tables"))
    out_model = ensure_dir(cfg.get("outputs", {}).get("model_dir", out_root / "model"))
    out_splits = ensure_dir(cfg.get("outputs", {}).get("split_dir", out_root / "splits"))
    logger = get_logger("train_static", Path(out_logs) / "train_static.log")

    data_path = cfg.get("data", {}).get("path")
    if not data_path:
        raise ValueError("config.data.path is required")

    logger.info("Loading dataset from %s", data_path)
    df, audit = load_validated_dataset(data_path, include_logs=bool(cfg.get("features", {}).get("include_logs", True)))
    save_json(Path(out_tables) / "data_audit_train_static.json", audit.to_dict())

    y = df["Out_of_step"].astype(int).values
    base_numeric = ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"]
    engineered = [c for c in ["invH", "Sgn_over_H", "Sgn_over_Ik", "Ik_over_H", "log_Sgn_eff_MVA", "log_Ikssmin_kA"] if c in df.columns]
    use_engineered = bool(cfg.get("features", {}).get("use_engineered", True))
    numeric_features = base_numeric + engineered if use_engineered else base_numeric
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical = ["GenName"] if "GenName" in df.columns else []

    X = df[numeric_features + categorical].copy()
    groups = build_groups(df, ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"], round_decimals=2)

    protocols = cfg.get("validation", {}).get("protocols", [])
    for p in protocols:
        manifest = create_split_manifest(
            y=y,
            split_mode=p["split_mode"],
            n_splits=int(p.get("n_splits", 5)),
            random_state=seed,
            groups=groups,
            leaveout_feature=p.get("leave_feature", "Sgn_eff_MVA"),
            leaveout_frame=df,
        )
        save_split_manifest(manifest, Path(out_splits) / f"split_manifest_{p['name']}.csv")

    ladder = build_baseline_ladder(numeric_features=numeric_features, categorical_features=categorical, random_state=seed)
    validation_df = run_validation(
        models=ladder,
        X=X,
        y=y,
        protocols=protocols,
        groups=groups,
        leaveout_frame=df,
        random_state=seed,
    )
    validation_path = Path(out_tables) / "train_static_validation.csv"
    validation_df.to_csv(validation_path, index=False)

    if len(validation_df) == 0:
        raise RuntimeError("No validation results produced.")

    score = validation_df.copy()
    score["_rank"] = score["PR_AUC"] - 0.2 * score["FNR"] - 0.05 * score["ECE"]
    best_model_code = str(score.sort_values("_rank", ascending=False).iloc[0]["model_code"])

    model_key_map = {
        "A1_logistic": "tierA_logistic",
        "A2_tree": "tierA_tree",
        "B1_rf": "tierB_random_forest",
        "B2_gbm": "tierB_gradient_boosting",
        "C1_monotonic": "tierC_monotonic_hgb",
        "C2_physics_logit": "tierC_physics_logit",
        "C3_hybrid": "tierC_two_stage_hybrid",
    }
    requested_key = cfg.get("model", {}).get("key")
    chosen_key = requested_key if requested_key else model_key_map.get(best_model_code, "tierC_two_stage_hybrid")

    model = StaticPhysicsRiskModel(
        config=StaticModelConfig(
            model_name=chosen_key,
            calibrate=bool(cfg.get("model", {}).get("calibrate", True)),
            random_state=seed,
        ),
        numeric_features=numeric_features,
        categorical_features=categorical,
    ).fit(X, y)

    p = model.predict_proba(X)[:, 1]
    th = optimize_thresholds(
        y_true=y,
        y_prob=p,
        c_fn=float(cfg.get("thresholds", {}).get("c_fn", 10.0)),
        c_fp=float(cfg.get("thresholds", {}).get("c_fp", 1.0)),
        min_recall=float(cfg.get("thresholds", {}).get("min_recall", 0.95)),
    )
    metrics = compute_all_metrics(
        y_true=y,
        y_prob=p,
        c_fn=float(cfg.get("thresholds", {}).get("c_fn", 10.0)),
        c_fp=float(cfg.get("thresholds", {}).get("c_fp", 1.0)),
    )

    bundle_path = model.save(Path(out_model) / "static_model_bundle.joblib")
    bounds = derive_feature_bounds(df, numeric_features)
    inference_cfg = {
        "model_code": chosen_key,
        "features": numeric_features + categorical,
        "thresholds": th,
        "feature_bounds": {k: [float(v[0]), float(v[1])] for k, v in bounds.items()},
        "calibration": model.calibration_name_,
    }
    save_yaml(Path(out_model) / "inference_config.yaml", inference_cfg)

    summary = {
        "selected_validation_model": best_model_code,
        "trained_model_key": chosen_key,
        "bundle_path": str(bundle_path),
        "thresholds": th,
        "metrics": metrics,
    }
    save_json(Path(out_tables) / "train_static_summary.json", summary)
    logger.info("Training complete. Best validation model=%s | trained key=%s", best_model_code, chosen_key)


if __name__ == "__main__":
    main()
