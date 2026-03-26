from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.outofstep_ml.data.loaders import load_validated_dataset
from src.outofstep_ml.deployment.predict import InferenceBundle
from src.outofstep_ml.evaluation.counterfactuals import generate_counterfactual_recommendations
from src.outofstep_ml.explainability.pdp_utils import compute_pdp_table
from src.outofstep_ml.explainability.shap_utils import feature_importance_table
from src.outofstep_ml.models.thresholds import threshold_curve
from src.outofstep_ml.utils.io import ensure_dir, load_yaml


def _static_mode(cfg: dict, model_path: str, inference_config: str) -> None:
    out_fig = ensure_dir(cfg.get("outputs", {}).get("figure_dir", "results/figures"))
    out_tab = ensure_dir(cfg.get("outputs", {}).get("table_dir", "results/tables"))

    df, _ = load_validated_dataset(cfg.get("data", {}).get("path"), include_logs=bool(cfg.get("features", {}).get("include_logs", True)))
    y = df["Out_of_step"].astype(int).values

    bundle = InferenceBundle.load(model_path, inference_config)
    feature_cols = bundle.config.get("features", [c for c in df.columns if c != "Out_of_step"])
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    p = bundle.model.predict_proba(X)[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y, p, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(Path(out_fig) / "roc_curve.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y, p, ax=ax)
    ax.set_title("PR Curve")
    fig.tight_layout()
    fig.savefig(Path(out_fig) / "pr_curve.png", dpi=220)
    plt.close(fig)

    bins = pd.cut(pd.Series(p), bins=10, include_lowest=True)
    rel = pd.DataFrame({"y": y, "p": p, "bin": bins}).groupby("bin", observed=False).agg(obs=("y", "mean"), conf=("p", "mean")).dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(rel["conf"], rel["obs"], marker="o")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagram")
    fig.tight_layout()
    fig.savefig(Path(out_fig) / "reliability_diagram_static.png", dpi=220)
    plt.close(fig)

    tc = threshold_curve(
        y_true=y,
        y_prob=p,
        c_fn=float(cfg.get("thresholds", {}).get("c_fn", 10.0)),
        c_fp=float(cfg.get("thresholds", {}).get("c_fp", 1.0)),
    )
    tc.to_csv(Path(out_tab) / "cost_threshold_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(tc["tau"], tc["Cost"])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Cost")
    ax.set_title("Cost vs Threshold")
    fig.tight_layout()
    fig.savefig(Path(out_fig) / "cost_threshold_curve.png", dpi=220)
    plt.close(fig)

    fi = feature_importance_table(bundle.model, X, y)
    fi.to_csv(Path(out_tab) / "feature_importance_research.csv", index=False)

    pdp = compute_pdp_table(bundle.model, X, [f for f in ["H_s", "Ikssmin_kA", "Sgn_eff_MVA", "Tag_rate"] if f in X.columns])
    pdp.to_csv(Path(out_tab) / "pdp_research.csv", index=False)

    bounds = {k: tuple(v) for k, v in bundle.config.get("feature_bounds", {}).items()}
    thr = float(bundle.config.get("thresholds", {}).get("tau_cost", 0.5))
    cf = generate_counterfactual_recommendations(bundle.model, X, bounds, threshold=thr, max_examples=20)
    cf.to_csv(Path(out_tab) / "counterfactual_research.csv", index=False)


def _benchmark_mode(cfg: dict) -> None:
    out_root = Path(cfg.get("outputs", {}).get("root", "results"))
    tdir = out_root / "tables"
    fdir = ensure_dir(out_root / "figures")

    t1 = pd.read_csv(tdir / "table1_main_performance.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(t1))
    ax.bar(x - 0.15, t1["ROC-AUC"], width=0.3, label="ROC-AUC")
    ax.bar(x + 0.15, t1["PR-AUC"], width=0.3, label="PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(t1["Model"], rotation=45, ha="right")
    ax.set_title("ROC/PR Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fdir / "benchmark_roc_pr_comparison.png", dpi=220)
    plt.close(fig)

    t3 = pd.read_csv(tdir / "table3_calibration_comparison.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(t3))
    ax.bar(x - 0.15, t3["Uncalibrated ECE"], width=0.3, label="Uncalibrated ECE")
    ax.bar(x + 0.15, t3["Calibrated ECE"], width=0.3, label="Calibrated ECE")
    ax.set_xticks(x)
    ax.set_xticklabels(t3["Model"], rotation=45, ha="right")
    ax.set_title("Calibration Improvement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fdir / "benchmark_calibration_comparison.png", dpi=220)
    plt.close(fig)

    t4 = pd.read_csv(tdir / "table4_robustness.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(t4))
    ax.bar(x, t4["Performance Drop"].fillna(0))
    ax.set_xticks(x)
    ax.set_xticklabels(t4["Model"], rotation=45, ha="right")
    ax.set_title("Robustness Degradation (PR-AUC drop)")
    fig.tight_layout()
    fig.savefig(fdir / "benchmark_robustness_drop.png", dpi=220)
    plt.close(fig)

    t5 = pd.read_csv(tdir / "table5_efficiency.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(t5))
    ax.bar(x, t5["Inference Time"])
    ax.set_xticks(x)
    ax.set_xticklabels(t5["Model"], rotation=45, ha="right")
    ax.set_title("Inference Time Comparison")
    fig.tight_layout()
    fig.savefig(fdir / "benchmark_inference_time.png", dpi=220)
    plt.close(fig)

    t8 = pd.read_csv(tdir / "table8_best_method_summary.csv")
    cols = ["Predictive Score", "Calibration Score", "Robustness Score", "Efficiency Score", "Physics Consistency", "Explainability", "Deployment Readiness"]
    if len(t8):
        top = t8.iloc[0]
        vals = top[cols].values.astype(float)
        angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False)
        vals = np.concatenate([vals, [vals[0]]])
        angles = np.concatenate([angles, [angles[0]]])
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, vals)
        ax.fill(angles, vals, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cols, fontsize=8)
        ax.set_title(f"Radar: {top['Model']}")
        fig.tight_layout()
        fig.savefig(fdir / "benchmark_best_model_radar.png", dpi=220)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["static", "benchmark"], default="static")
    ap.add_argument("--model", default="results/model/static_model_bundle.joblib")
    ap.add_argument("--inference-config", default="results/model/inference_config.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.mode == "benchmark":
        _benchmark_mode(cfg)
    else:
        _static_mode(cfg, model_path=args.model, inference_config=args.inference_config)


if __name__ == "__main__":
    main()
