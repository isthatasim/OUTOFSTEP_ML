from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.outofstep_ml.data.loaders import load_validated_dataset
from src.outofstep_ml.evaluation.metrics import compute_all_metrics
from src.outofstep_ml.models.thresholds import optimize_thresholds, threshold_curve
from src.outofstep_ml.models.calibration import compare_calibration_methods
from src.outofstep_ml.models.baselines import build_baseline_ladder
from src.outofstep_ml.models.static_physics_model import StaticPhysicsRiskModel
from src.outofstep_ml.utils.io import ensure_dir, load_yaml, save_json


def _merge_cfg(path: Path) -> dict:
    cfg = load_yaml(path)
    ext = cfg.get("extends")
    if not ext:
        return cfg
    base = load_yaml(path.parent / "base.yaml") if ext == "base" else load_yaml(path.parent / ext)
    merged = dict(base)
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-bundle", default="results/model/static_model_bundle.joblib")
    ap.add_argument("--inference-config", default="results/model/inference_config.yaml")
    args = ap.parse_args()

    cfg = _merge_cfg(Path(args.config))
    out_root = ensure_dir(cfg.get("outputs", {}).get("root", "results"))
    out_tables = ensure_dir(cfg.get("outputs", {}).get("table_dir", out_root / "tables"))

    df, _ = load_validated_dataset(cfg.get("data", {}).get("path"), include_logs=bool(cfg.get("features", {}).get("include_logs", True)))
    y = df["Out_of_step"].astype(int).values

    bundle = StaticPhysicsRiskModel.load(args.model_bundle)
    feature_cols = bundle.numeric_features + bundle.categorical_features
    X = df[feature_cols].copy()
    p = bundle.predict_proba(X)[:, 1]

    metrics = compute_all_metrics(
        y_true=y,
        y_prob=p,
        c_fn=float(cfg.get("thresholds", {}).get("c_fn", 10.0)),
        c_fp=float(cfg.get("thresholds", {}).get("c_fp", 1.0)),
    )
    thresholds = optimize_thresholds(
        y_true=y,
        y_prob=p,
        c_fn=float(cfg.get("thresholds", {}).get("c_fn", 10.0)),
        c_fp=float(cfg.get("thresholds", {}).get("c_fp", 1.0)),
        min_recall=float(cfg.get("thresholds", {}).get("min_recall", 0.95)),
    )

    pd.DataFrame([metrics]).to_csv(Path(out_tables) / "evaluate_static_metrics.csv", index=False)
    pd.DataFrame([thresholds]).to_csv(Path(out_tables) / "evaluate_static_thresholds.csv", index=False)
    threshold_curve(y, p, c_fn=float(cfg.get("thresholds", {}).get("c_fn", 10.0)), c_fp=float(cfg.get("thresholds", {}).get("c_fp", 1.0))).to_csv(
        Path(out_tables) / "threshold_curve.csv", index=False
    )

    ladder = build_baseline_ladder(bundle.numeric_features, bundle.categorical_features, random_state=int(cfg.get("seed", 42)))
    model_key = cfg.get("model", {}).get("key", "tierC_two_stage_hybrid")
    key_map = {
        "tierA_logistic": "A1_logistic",
        "tierA_tree": "A2_tree",
        "tierB_random_forest": "B1_rf",
        "tierB_gradient_boosting": "B2_gbm",
        "tierC_monotonic_hgb": "C1_monotonic",
        "tierC_physics_logit": "C2_physics_logit",
        "tierC_two_stage_hybrid": "C3_hybrid",
    }
    calib_base = ladder[key_map.get(model_key, "C3_hybrid")]
    compare_calibration_methods(calib_base, X, y, random_state=int(cfg.get("seed", 42))).to_csv(
        Path(out_tables) / "evaluate_static_calibration.csv", index=False
    )

    save_json(Path(out_tables) / "evaluate_static_summary.json", {"metrics": metrics, "thresholds": thresholds})


if __name__ == "__main__":
    main()
