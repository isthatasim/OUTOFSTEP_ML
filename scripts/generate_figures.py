from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="results/model/static_model_bundle.joblib")
    ap.add_argument("--inference-config", default="results/model/inference_config.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_fig = ensure_dir(cfg.get("outputs", {}).get("figure_dir", "results/figures"))
    out_tab = ensure_dir(cfg.get("outputs", {}).get("table_dir", "results/tables"))

    df, _ = load_validated_dataset(cfg.get("data", {}).get("path"), include_logs=bool(cfg.get("features", {}).get("include_logs", True)))
    y = df["Out_of_step"].astype(int).values

    bundle = InferenceBundle.load(args.model, args.inference_config)
    feature_cols = bundle.config.get("features", [c for c in df.columns if c != "Out_of_step"])
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    p = bundle.model.predict_proba(X)[:, 1]

    # ROC + PR
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

    # Reliability
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

    # Cost-threshold curve
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

    # Explainability summaries
    fi = feature_importance_table(bundle.model, X, y)
    fi.to_csv(Path(out_tab) / "feature_importance_research.csv", index=False)

    pdp = compute_pdp_table(bundle.model, X, [f for f in ["H_s", "Ikssmin_kA", "Sgn_eff_MVA", "Tag_rate"] if f in X.columns])
    pdp.to_csv(Path(out_tab) / "pdp_research.csv", index=False)

    # Counterfactual table + simple illustration
    bounds = {k: tuple(v) for k, v in bundle.config.get("feature_bounds", {}).items()}
    thr = float(bundle.config.get("thresholds", {}).get("tau_cost", 0.5))
    cf = generate_counterfactual_recommendations(bundle.model, X, bounds, threshold=thr, max_examples=20)
    cf.to_csv(Path(out_tab) / "counterfactual_research.csv", index=False)
    if len(cf):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(cf)), cf["start_risk"], alpha=0.7, label="start")
        ax.bar(range(len(cf)), cf["achieved_risk"], alpha=0.7, label="after CF")
        ax.axhline(thr, color="red", linestyle="--", label="threshold")
        ax.set_title("Counterfactual Risk Reduction")
        ax.set_xlabel("Example")
        ax.set_ylabel("Risk")
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(out_fig) / "counterfactual_illustration.png", dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    main()
