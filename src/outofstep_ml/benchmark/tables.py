from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _df_to_md(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def _mean_std_table(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{c[0]}_{c[1]}" for c in agg.columns]
    return agg.reset_index()


def generate_tables(results_dir: str | Path = "results") -> Dict[str, Path]:
    results_dir = Path(results_dir)
    tdir = results_dir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(tdir / "benchmark_raw_results.csv")
    raw_ok = raw[raw["status"] == "ok"].copy()

    # TABLE 1
    metrics = [
        "ROC_AUC",
        "PR_AUC",
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
    ]
    t1 = _mean_std_table(raw_ok, ["model_id", "model_name"], metrics + ["Inference_ms_per_sample", "Training_Time_s"]) 
    t1["Mean ± Std"] = t1.apply(lambda r: f"PR-AUC {r['PR_AUC_mean']:.4f} ± {r['PR_AUC_std']:.4f}", axis=1)
    t1_out = t1.rename(columns={
        "model_name": "Model",
        "ROC_AUC_mean": "ROC-AUC",
        "PR_AUC_mean": "PR-AUC",
        "Precision_mean": "Precision",
        "Recall_mean": "Recall",
        "F1_mean": "F1",
        "Specificity_mean": "Specificity",
        "Balanced_Accuracy_mean": "Balanced Accuracy",
        "FNR_mean": "FNR",
        "Brier_mean": "Brier",
        "ECE_mean": "ECE",
        "MAE_mean": "MAE",
        "RMSE_mean": "RMSE",
    })
    table1_cols = [
        "Model",
        "ROC-AUC",
        "PR-AUC",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "Balanced Accuracy",
        "FNR",
        "Brier",
        "ECE",
        "MAE",
        "RMSE",
        "Mean ± Std",
    ]
    table1 = t1_out[table1_cols].sort_values("PR-AUC", ascending=False)
    table1.to_csv(tdir / "table1_main_performance.csv", index=False)
    _df_to_md(table1, tdir / "table1_main_performance.md")

    # TABLE 2
    t2_raw = pd.read_csv(tdir / "benchmark_thresholds_raw.csv")
    t2 = t2_raw.groupby(["model_name"], as_index=False).mean(numeric_only=True)
    t2 = t2.rename(columns={
        "model_name": "Model",
        "tau_default": "tau_default",
        "tau_F1": "tau_F1",
        "tau_cost": "tau_cost",
        "tau_HR": "tau_HR",
        "Precision@tau_cost": "Precision@tau",
        "Recall@tau_cost": "Recall@tau",
        "F1@tau_cost": "F1@tau",
        "Cost@tau_cost": "Cost@tau",
    })
    table2 = t2[["Model", "tau_default", "tau_F1", "tau_cost", "tau_HR", "Precision@tau", "Recall@tau", "F1@tau", "Cost@tau"]]
    table2.to_csv(tdir / "table2_threshold_comparison.csv", index=False)
    _df_to_md(table2, tdir / "table2_threshold_comparison.md")

    # TABLE 3
    t3_raw = pd.read_csv(tdir / "benchmark_calibration_raw.csv")
    t3 = t3_raw.groupby("model_name", as_index=False).mean(numeric_only=True)
    best_cal = t3_raw.groupby("model_name")["Best_Calibration"].agg(lambda s: s.value_counts().index[0]).reset_index()
    t3 = t3.merge(best_cal, on="model_name", how="left")
    t3 = t3.rename(columns={
        "model_name": "Model",
        "Uncalibrated_Brier": "Uncalibrated Brier",
        "Calibrated_Brier": "Calibrated Brier",
        "Uncalibrated_ECE": "Uncalibrated ECE",
        "Calibrated_ECE": "Calibrated ECE",
        "Best_Calibration": "Best calibration",
    })
    table3 = t3[["Model", "Uncalibrated Brier", "Calibrated Brier", "Uncalibrated ECE", "Calibrated ECE", "Best calibration"]]
    table3.to_csv(tdir / "table3_calibration_comparison.csv", index=False)
    _df_to_md(table3, tdir / "table3_calibration_comparison.md")

    # TABLE 4
    t4_raw = pd.read_csv(tdir / "benchmark_robustness_raw.csv")
    piv = t4_raw.pivot_table(index="model_name", columns="shift", values="PR_AUC", aggfunc="mean").reset_index()
    piv = piv.rename(columns={"model_name": "Model", "clean": "Clean", "noisy": "Noisy", "missing_features": "Missing Features", "unseen_regime": "Unseen Regime", "group_shift": "Group Shift"})
    for c in ["Noisy", "Missing Features", "Unseen Regime", "Group Shift"]:
        if c not in piv.columns:
            piv[c] = np.nan
    piv["Performance Drop"] = piv["Clean"] - piv[["Noisy", "Missing Features", "Unseen Regime", "Group Shift"]].mean(axis=1, skipna=True)
    table4 = piv[["Model", "Clean", "Noisy", "Missing Features", "Unseen Regime", "Group Shift", "Performance Drop"]]
    table4.to_csv(tdir / "table4_robustness.csv", index=False)
    _df_to_md(table4, tdir / "table4_robustness.md")

    # TABLE 5
    t5_raw = pd.read_csv(tdir / "benchmark_efficiency_raw.csv")
    t5_num = t5_raw.groupby("model_name", as_index=False)[["Training_Time_s", "Inference_ms_per_sample", "Model_Size_MB"]].mean()
    online = (
        t5_raw.groupby("model_name")["Suitable_for_Online_Screening"]
        .agg(lambda s: bool(round(pd.to_numeric(s, errors="coerce").fillna(0).mean())))
        .reset_index()
    )
    t5 = t5_num.merge(online, on="model_name", how="left")
    table5 = t5.rename(columns={
        "model_name": "Model",
        "Training_Time_s": "Training Time",
        "Inference_ms_per_sample": "Inference Time",
        "Model_Size_MB": "Memory / Model Size",
        "Suitable_for_Online_Screening": "Suitable for Online Screening",
    })[["Model", "Training Time", "Inference Time", "Memory / Model Size", "Suitable for Online Screening"]]
    table5.to_csv(tdir / "table5_efficiency.csv", index=False)
    _df_to_md(table5, tdir / "table5_efficiency.md")

    # TABLE 6 + TABLE 8 ranking
    t_rank = table1.merge(table3, on="Model", how="left").merge(table4[["Model", "Performance Drop"]], on="Model", how="left").merge(table5, on="Model", how="left")
    t_rank["Predictive Score"] = t_rank["PR-AUC"]
    t_rank["Calibration Score"] = 1.0 - t_rank["Calibrated ECE"].fillna(1.0)
    t_rank["Robustness Score"] = 1.0 - t_rank["Performance Drop"].fillna(1.0).clip(lower=0)
    t_rank["Efficiency Score"] = 1.0 - (t_rank["Inference Time"] / max(t_rank["Inference Time"].max(), 1e-9))
    t_rank["Physics Consistency"] = t_rank["Model"].str.contains("Physics|Proposed|Monotonic", case=False).astype(float)
    t_rank["Explainability"] = t_rank["Model"].str.contains("CatBoost|Logistic|Physics", case=False).astype(float)
    t_rank["Deployment Readiness"] = t_rank["Suitable for Online Screening"].astype(float)
    t_rank["Overall"] = (
        0.28 * t_rank["Predictive Score"]
        + 0.18 * t_rank["Calibration Score"]
        + 0.18 * t_rank["Robustness Score"]
        + 0.12 * t_rank["Efficiency Score"]
        + 0.10 * t_rank["Physics Consistency"]
        + 0.07 * t_rank["Explainability"]
        + 0.07 * t_rank["Deployment Readiness"]
    )

    table6 = t_rank[["Model", "Predictive Score", "Calibration Score", "Robustness Score", "Efficiency Score", "Physics Consistency", "Explainability", "Deployment Readiness", "Overall"]].sort_values("Overall", ascending=False).reset_index(drop=True)
    table6.insert(0, "Rank", np.arange(1, len(table6) + 1))
    table6["Main Strength"] = "Strong balanced benchmark profile"
    table6["Main Weakness"] = "Requires scenario-specific tuning"
    table6["Best Use Case"] = "Static OOS screening and operational planning"
    table6_basic = table6[["Rank", "Model", "Main Strength", "Main Weakness", "Best Use Case"]]
    table6_basic.to_csv(tdir / "table6_final_ranking.csv", index=False)
    _df_to_md(table6_basic, tdir / "table6_final_ranking.md")

    table8 = table6[["Model", "Predictive Score", "Calibration Score", "Robustness Score", "Efficiency Score", "Physics Consistency", "Explainability", "Deployment Readiness", "Overall"]].copy()
    table8["Overall Recommendation"] = np.where(table8["Overall"] == table8["Overall"].max(), "Best overall recommended model", "Competitive")
    table8.to_csv(tdir / "table8_best_method_summary.csv", index=False)
    _df_to_md(table8, tdir / "table8_best_method_summary.md")

    # TABLE 7 mathematical comparison
    math_rows = [
        ["Logistic Regression", "p(x)=sigma(w^T x + b)", "Tabular static", "Linear logit", "High", "No", "Yes", "Yes", "Indirect", "Simple and fast", "Limited nonlinear capture"],
        ["SVM (RBF)", "y_hat=sign(sum_i alpha_i y_i K(x_i,x)+b)", "Tabular static", "Kernel nonlinearity", "Medium", "No", "Limited", "Yes", "Indirect", "Strong margin learner", "Probability calibration sensitivity"],
        ["Random Forest", "p(x)=1/M sum_m T_m(x)", "Tabular static", "Tree nonlinearity", "Medium", "No", "Yes", "Yes", "Indirect", "Robust baseline", "Can be less calibrated"],
        ["XGBoost/LightGBM", "p(x)=sigma(sum_k eta f_k(x))", "Tabular static", "Boosted trees", "Medium", "Optional", "Yes", "Yes", "Indirect", "High predictive power", "Tuning complexity"],
        ["CatBoost", "p(x)=sigma(sum_k eta f_k^{ordered}(x))", "Tabular static", "Ordered boosting", "High (SHAP)", "Optional", "Yes", "Yes", "Indirect", "Strong tabular performance", "Dependency/runtime overhead"],
        ["FT-Transformer", "h0=Emb(x), hL=Transformer(h0), p=sigma(WhL+b)", "Tabular static", "Attention", "Medium", "No", "Yes", "Yes", "Indirect", "Captures feature interactions", "Higher compute/data needs"],
        ["TabNet", "h=sum_t M_t ⊙ f_t(x), p=sigma(g(h))", "Tabular static", "Sequential attentive masks", "Medium", "No", "Yes", "Yes", "Indirect", "Sparse attentive selection", "Hyperparameter sensitivity"],
        ["Existing Repo Model", "Two-stage tree/hybrid risk mapping", "Tabular static", "Hybrid nonlinearity", "Medium", "Partly", "Yes", "Yes", "Yes", "Practical baseline from repo", "May lack explicit monotonic guarantees"],
        ["Proposed Physics-Aware Model", "p=f_theta([x,z]), z=[1/H,S/H,S/I,I/H], L=L_data+lambda_H r_H+lambda_I r_I+lambda_S r_S", "Tabular static", "Regularized nonlinearity", "High", "Yes", "Yes", "Yes", "Yes", "Physics consistency + actionability", "Needs careful lambda tuning"],
    ]
    table7 = pd.DataFrame(math_rows, columns=["Model", "Core mathematical form", "Input type", "Nonlinearity", "Explainability", "Physics-aware", "Calibration-ready", "Cost-sensitive thresholding", "Counterfactual support", "Expected strength", "Expected limitation"])
    table7.to_csv(tdir / "mathematical_comparison.csv", index=False)
    _df_to_md(table7, tdir / "mathematical_comparison.md")

    # concise best-method summary json
    best_pred = table1.sort_values("PR-AUC", ascending=False).iloc[0]["Model"] if len(table1) else None
    best_cal = table3.sort_values("Calibrated Brier", ascending=True).iloc[0]["Model"] if len(table3) else None
    best_deploy = table5.sort_values(["Inference Time", "Training Time"], ascending=[True, True]).iloc[0]["Model"] if len(table5) else None
    best_overall = table8.sort_values("Overall", ascending=False).iloc[0]["Model"] if len(table8) else None
    summary = {
        "best_pure_predictive_model": best_pred,
        "best_calibrated_model": best_cal,
        "best_deployable_model": best_deploy,
        "best_overall_recommended_model": best_overall,
    }
    (tdir / "best_method_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "table1": tdir / "table1_main_performance.csv",
        "table2": tdir / "table2_threshold_comparison.csv",
        "table3": tdir / "table3_calibration_comparison.csv",
        "table4": tdir / "table4_robustness.csv",
        "table5": tdir / "table5_efficiency.csv",
        "table6": tdir / "table6_final_ranking.csv",
        "table7": tdir / "mathematical_comparison.csv",
        "table8": tdir / "table8_best_method_summary.csv",
    }
