from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.outofstep_ml.benchmark.model_zoo import build_benchmark_model_specs
from src.outofstep_ml.benchmark.runner import _fit_with_budget, _select_best_calibration, _apply_calibrator, make_strict_split, _prob_metrics, _class_metrics
from src.outofstep_ml.data.loaders import load_validated_dataset
from src.outofstep_ml.models.thresholds import optimize_thresholds
from src.outofstep_ml.utils.io import ensure_dir, load_yaml, save_json
from src.outofstep_ml.utils.seed import set_global_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-id", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_global_seed(int(cfg.get("seed", 42)))

    out_root = ensure_dir(cfg.get("outputs", {}).get("root", "results"))
    out_model = ensure_dir(cfg.get("outputs", {}).get("model_dir", Path(out_root) / "model"))
    out_tables = ensure_dir(cfg.get("outputs", {}).get("table_dir", Path(out_root) / "tables"))

    df, _ = load_validated_dataset(cfg.get("data", {}).get("path"), include_logs=bool(cfg.get("features", {}).get("include_logs", True)))
    y = df["Out_of_step"].astype(int).values

    base = ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"]
    eng = [c for c in ["invH", "Sgn_over_H", "Sgn_over_Ik", "Ik_over_H", "log_Sgn_eff_MVA", "log_Ikssmin_kA"] if c in df.columns]
    use_engineered = bool(cfg.get("features", {}).get("use_engineered", True))
    numeric = [c for c in (base + eng if use_engineered else base) if c in df.columns]
    categorical = ["GenName"] if "GenName" in df.columns else []
    X = df[numeric + categorical].copy()

    specs = {s.model_id: s for s in build_benchmark_model_specs(numeric, categorical, random_state=int(cfg.get("seed", 42)))}
    if args.model_id not in specs:
        raise ValueError(f"Unknown model-id: {args.model_id}. Available: {list(specs)}")
    spec = specs[args.model_id]
    if not spec.available or spec.estimator is None:
        raise RuntimeError(f"Model unavailable: {spec.skip_reason}")

    split = make_strict_split(
        df=df,
        y=y,
        seed=int(cfg.get("seed", 42)),
        strategy="stratified",
        test_size=float(cfg.get("strict_eval", {}).get("test_size", 0.2)),
        val_size=float(cfg.get("strict_eval", {}).get("val_size", 0.2)),
        group_cols=["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"],
        leave_feature="Sgn_eff_MVA",
    )

    X_train, y_train = X.iloc[split.train_idx], y[split.train_idx]
    X_val, y_val = X.iloc[split.val_idx], y[split.val_idx]
    X_test, y_test = X.iloc[split.test_idx], y[split.test_idx]

    model, best_params, val_score = _fit_with_budget(
        spec,
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials=int(cfg.get("training_budget", {}).get("n_trials", 10)),
        seed=int(cfg.get("seed", 42)),
    )
    p_val_raw = model.predict_proba(X_val)[:, 1]
    calib_name, calib_obj, _ = _select_best_calibration(y_val, p_val_raw)
    p_test = _apply_calibrator(calib_obj, calib_name, model.predict_proba(X_test)[:, 1])

    th = optimize_thresholds(
        y_true=y_val,
        y_prob=p_val_raw,
        c_fn=float(cfg.get("thresholds", {}).get("c_fn", 10.0)),
        c_fp=float(cfg.get("thresholds", {}).get("c_fp", 1.0)),
        min_recall=float(cfg.get("thresholds", {}).get("min_recall", 0.95)),
    )
    y_hat = (p_test >= th["tau_cost"]).astype(int)

    summary = {
        "model_id": args.model_id,
        "model_name": spec.model_name,
        "best_params": best_params,
        "validation_pr_auc": val_score,
        "calibration": calib_name,
        "thresholds": th,
        "test_prob_metrics": _prob_metrics(y_test, p_test),
        "test_class_metrics_at_tau_cost": _class_metrics(y_test, y_hat),
    }

    import joblib

    model_path = Path(out_model) / f"{args.model_id}_strict_model.joblib"
    joblib.dump(model, model_path)
    save_json(Path(out_tables) / f"{args.model_id}_strict_summary.json", summary)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
