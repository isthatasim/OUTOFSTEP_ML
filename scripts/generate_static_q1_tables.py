from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.outofstep_ml.utils.io import load_yaml


def _write_md(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_root = Path(cfg.get("outputs", {}).get("root", "outputs/static_q1_validation"))
    tdir = out_root / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    # Scenario leaderboard from nominal + regime shift
    nominal = pd.read_csv(tdir / "scenario1_nominal_baseline.csv")
    regime = pd.read_csv(tdir / "scenario6_regime_shift.csv")
    overview = pd.read_csv(tdir / "scenario_overview_summary.csv")

    table_nominal = nominal.groupby(["model_id", "model_name"], as_index=False).mean(numeric_only=True)
    table_nominal = table_nominal[
        [
            "model_name",
            "PR_AUC",
            "ROC_AUC",
            "Precision",
            "Recall",
            "F1",
            "Specificity",
            "Balanced_Accuracy",
            "FNR",
            "Brier",
            "ECE",
            "MAE",
            "RMSE",
            "tau_cost",
            "tau_f1",
            "tau_hr",
        ]
    ].rename(columns={"model_name": "Model"})
    table_nominal.to_csv(tdir / "q1_table_nominal_baseline.csv", index=False)
    _write_md(table_nominal, tdir / "q1_table_nominal_baseline.md")

    table_regime = regime.groupby("strategy", as_index=False).mean(numeric_only=True)[
        ["strategy", "PR_AUC", "ROC_AUC", "FNR", "ECE", "Brier", "Recall", "Precision"]
    ]
    table_regime.to_csv(tdir / "q1_table_regime_shift.csv", index=False)
    _write_md(table_regime, tdir / "q1_table_regime_shift.md")

    # Migration summary
    migrations = pd.read_csv(tdir / "scenario2_3_4_migrations.csv")
    table_mig = migrations.groupby(["scenario", "feature", "level"], as_index=False).mean(numeric_only=True)[
        ["scenario", "feature", "level", "PR_AUC", "FNR", "ECE", "Brier", "Recall"]
    ]
    table_mig.to_csv(tdir / "q1_table_migrations.csv", index=False)
    _write_md(table_mig, tdir / "q1_table_migrations.md")

    # Robustness summary
    robustness = pd.read_csv(tdir / "scenario7_noise_missing.csv")
    table_rob = robustness.groupby("shift", as_index=False).mean(numeric_only=True)[
        ["shift", "PR_AUC", "ROC_AUC", "FNR", "ECE", "Brier", "Recall", "Precision", "F1"]
    ]
    table_rob.to_csv(tdir / "q1_table_robustness.csv", index=False)
    _write_md(table_rob, tdir / "q1_table_robustness.md")

    # Imbalance ablation
    imbalance = pd.read_csv(tdir / "scenario8_imbalance_ablation.csv")
    table_imb = imbalance.groupby("imbalance_mode", as_index=False).mean(numeric_only=True)[
        ["imbalance_mode", "PR_AUC", "ROC_AUC", "FNR", "ECE", "Brier", "Recall", "Precision", "F1"]
    ]
    table_imb.to_csv(tdir / "q1_table_imbalance_ablation.csv", index=False)
    _write_md(table_imb, tdir / "q1_table_imbalance_ablation.md")

    # Threshold and monotonic/counterfactual
    threshold = pd.read_csv(tdir / "scenario9_threshold_policies_aggregated.csv")
    threshold.to_csv(tdir / "q1_table_threshold_policies.csv", index=False)
    _write_md(threshold, tdir / "q1_table_threshold_policies.md")

    monotonic = pd.read_csv(tdir / "scenario_monotonic_consistency.csv")
    monotonic.to_csv(tdir / "q1_table_monotonic_consistency.csv", index=False)
    _write_md(monotonic, tdir / "q1_table_monotonic_consistency.md")

    cf_summary = pd.read_csv(tdir / "scenario10_counterfactual_summary.csv")
    cf_summary.to_csv(tdir / "q1_table_counterfactual_summary.csv", index=False)
    _write_md(cf_summary, tdir / "q1_table_counterfactual_summary.md")

    overview.to_csv(tdir / "q1_table_scenario_overview.csv", index=False)
    _write_md(overview, tdir / "q1_table_scenario_overview.md")

    print(f"Static Q1 tables generated in: {tdir}")


if __name__ == "__main__":
    main()

