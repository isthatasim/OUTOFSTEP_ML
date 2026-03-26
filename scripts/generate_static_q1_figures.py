from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.outofstep_ml.utils.io import load_yaml


def _save(fig, path_stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=260)
    fig.savefig(path_stem.with_suffix(".pdf"), dpi=260)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    out_root = Path(cfg.get("outputs", {}).get("root", "outputs/static_q1_validation"))
    tdir = out_root / "tables"
    fdir = out_root / "figures"
    fdir.mkdir(parents=True, exist_ok=True)

    # 1) Migrations
    mig = pd.read_csv(tdir / "scenario2_3_4_migrations.csv")
    if len(mig):
        for metric in ["PR_AUC", "FNR", "ECE"]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for sc in sorted(mig["scenario"].unique()):
                d = mig[mig["scenario"] == sc].groupby("level", as_index=False).mean(numeric_only=True).sort_values("level")
                ax.plot(d["level"], d[metric], marker="o", label=sc)
            ax.set_xlabel("Migration level multiplier")
            ax.set_ylabel(metric)
            ax.set_title(f"Migration Scenario Curves: {metric}")
            ax.grid(alpha=0.25)
            ax.legend()
            _save(fig, fdir / f"q1_migrations_{metric.lower()}")

    # 2) Regime shift
    reg = pd.read_csv(tdir / "scenario6_regime_shift.csv")
    if len(reg):
        d = reg.groupby("strategy", as_index=False).mean(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(d["strategy"], d["PR_AUC"], color="#1f77b4")
        ax.set_title("Regime Shift: PR-AUC by split strategy")
        ax.set_ylabel("PR-AUC")
        ax.grid(axis="y", alpha=0.25)
        _save(fig, fdir / "q1_regime_shift_pr_auc")

    # 3) Robustness
    rob = pd.read_csv(tdir / "scenario7_noise_missing.csv")
    if len(rob):
        d = rob.groupby("shift", as_index=False).mean(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(d["shift"], d["PR_AUC"], color="#2ca02c")
        ax.set_title("Noise/Missing Robustness (PR-AUC)")
        ax.set_ylabel("PR-AUC")
        ax.grid(axis="y", alpha=0.25)
        _save(fig, fdir / "q1_robustness_pr_auc")

    # 4) Imbalance ablation
    imb = pd.read_csv(tdir / "scenario8_imbalance_ablation.csv")
    if len(imb):
        d = imb.groupby("imbalance_mode", as_index=False).mean(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(d))
        ax.bar(x - 0.15, d["PR_AUC"], width=0.3, label="PR-AUC")
        ax.bar(x + 0.15, 1.0 - d["FNR"], width=0.3, label="1-FNR")
        ax.set_xticks(x)
        ax.set_xticklabels(d["imbalance_mode"], rotation=30, ha="right")
        ax.set_title("Imbalance-aware Ablation")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        _save(fig, fdir / "q1_imbalance_ablation")

    # 5) Threshold policies
    tp = pd.read_csv(tdir / "scenario9_threshold_policies_aggregated.csv")
    if len(tp):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(tp))
        ax.bar(x - 0.15, tp["Recall"], width=0.3, label="Recall")
        ax.bar(x + 0.15, tp["Precision"], width=0.3, label="Precision")
        ax.set_xticks(x)
        ax.set_xticklabels(tp["policy"], rotation=20, ha="right")
        ax.set_title("Threshold Policy Comparison")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        _save(fig, fdir / "q1_threshold_policy_precision_recall")

    # 6) Monotonic consistency
    mono = pd.read_csv(tdir / "scenario_monotonic_consistency.csv")
    if len(mono):
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = [f"{r.feature} (dir={int(r.direction)})" for _, r in mono.iterrows()]
        ax.bar(labels, mono["violation_rate"], color="#d62728")
        ax.set_ylabel("Violation rate")
        ax.set_title("Monotonic Consistency Check")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
        _save(fig, fdir / "q1_monotonic_violation_rates")

    # 7) Counterfactual summary
    cf = pd.read_csv(tdir / "scenario10_counterfactual_summary.csv")
    if len(cf):
        d = cf.mean(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(["Start risk", "Achieved risk"], [d["mean_start_risk"], d["mean_achieved_risk"]], color=["#ff7f0e", "#1f77b4"])
        ax.set_ylim(0, 1)
        ax.set_title("Counterfactual Stability Correction")
        ax.grid(axis="y", alpha=0.25)
        _save(fig, fdir / "q1_counterfactual_risk_reduction")

    print(f"Static Q1 figures generated in: {fdir}")


if __name__ == "__main__":
    main()

