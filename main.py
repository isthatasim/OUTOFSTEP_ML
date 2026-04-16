from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import export_text

from src.data import TARGET_COLUMN, dataset_summary_table, load_dataset, print_audit_report
from src.eval import (
    add_noise,
    evaluate_model_cv,
    evaluate_model_cv_noisy_test,
    evaluate_probabilities,
    make_group_labels,
    minimal_counterfactual,
    select_thresholds,
)
from src.features import ALL_NUMERIC_FEATURES, build_feature_frame, derive_feature_bounds, get_monotonic_constraints
from src.models import (
    controlled_random_search,
    make_calibrated_model,
    make_model_registry,
    select_best_calibration,
)
from src.monitoring import (
    concept_drift_scan,
    ks_table,
    psi_table,
    retrain_trigger_policy,
    rolling_performance,
    simulate_stream,
)
from src.plots import (
    plot_boundary_comparison,
    plot_drift_monitoring,
    plot_feature_distributions,
    plot_feature_importance,
    plot_flowchart_figure,
    plot_noise_robustness,
    plot_pdp,
    plot_reliability_diagram,
    plot_stability_map,
    plot_tradeoff_scatter,
    print_ascii_flowchart,
)
from src.report_problem import build_problem_formulation_markdown
from src.report_results import build_results_discussion_markdown


@dataclass
class TierModel:
    model_code: str
    model_name: str
    tier: str
    estimator: object
    complexity_score: float
    config: Dict


class ManifestRegistry:
    def __init__(self) -> None:
        self.records: List[Dict] = []

    @staticmethod
    def _config_hash(config: Dict) -> str:
        blob = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]

    def add(
        self,
        artifact_name: str,
        artifact_type: str,
        step: str,
        validation_protocol: str,
        model_name: str,
        config: Dict,
        code_path: str,
        file_path: Path,
    ) -> str:
        artifact_id = f"ART-{len(self.records) + 1:04d}"
        rec = {
            "artifact_id": artifact_id,
            "artifact_name": artifact_name,
            "artifact_type": artifact_type,
            "step": step,
            "validation_protocol": validation_protocol,
            "model_name": model_name,
            "hyperparameter_hash": self._config_hash(config),
            "code_path": code_path,
            "file_path": str(file_path.resolve()),
        }
        self.records.append(rec)
        return artifact_id

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")


def _version_report() -> Dict[str, str]:
    import sklearn

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _write_yaml(path: Path, obj: Dict) -> None:
    lines = []
    for k, v in obj.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {json.dumps(vv)}")
        else:
            lines.append(f"{k}: {json.dumps(v)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_output_dirs(output_root: Path) -> Dict[str, Path]:
    dirs = {
        "root": output_root,
        "tables": output_root / "tables",
        "figures": output_root / "figures",
        "model": output_root / "model",
        "api": output_root / "api",
        "tests": output_root / "tests",
        "monitoring": output_root / "monitoring",
        "logs": output_root / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _copy_api_files(project_root: Path, out_api: Path) -> List[Path]:
    out = []
    dst = out_api / "api_app.py"
    shutil.copy2(project_root / "src" / "api_app.py", dst)
    out.append(dst)
    req = out_api / "requirements.txt"
    req.write_text(
        "\n".join(
            [
                "fastapi>=0.115",
                "uvicorn>=0.30",
                "pydantic>=2.8",
                "numpy>=1.26",
                "pandas>=2.2",
                "scikit-learn>=1.5",
                "joblib>=1.4",
                "PyYAML>=6.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out.append(req)
    curl = out_api / "curl_example.txt"
    curl.write_text(
        "curl -X POST http://127.0.0.1:8000/predict "
        "-H \"Content-Type: application/json\" "
        "-d '{\"Tag_rate\": 1.1, \"Ikssmin_kA\": 20.0, \"Sgn_eff_MVA\": 100.0, \"H_s\": 3.5, \"GenName\": \"GR1\"}'\n",
        encoding="utf-8",
    )
    out.append(curl)
    return out


def _copy_tests(project_root: Path, out_tests: Path) -> List[Path]:
    paths: List[Path] = []
    for f in (project_root / "tests").glob("*.py"):
        dst = out_tests / f.name
        shutil.copy2(f, dst)
        paths.append(dst)
    return paths


def _profile_latency_ms(model, X: pd.DataFrame, y: np.ndarray, n_runs: int = 120) -> float:
    m = clone(model)
    fit_n = min(len(X), 6000)
    m.fit(X.iloc[:fit_n], y[:fit_n])
    sample = X.iloc[[0]]
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = m.predict_proba(sample)
    return float((time.perf_counter() - t0) * 1000.0 / n_runs)


def _composite_score(df: pd.DataFrame) -> pd.Series:
    brier_term = 1.0 - np.clip(df["Brier"].astype(float) / 0.25, 0, 1)
    ece_term = 1.0 - np.clip(df["ECE"].astype(float), 0, 1)
    fnr_term = 1.0 - np.clip(df["FNR"].astype(float), 0, 1)
    pr_term = np.clip(df["PR_AUC"].astype(float), 0, 1)
    rec_term = np.clip(df["Recall"].astype(float), 0, 1)
    return 0.40 * pr_term + 0.25 * fnr_term + 0.15 * ece_term + 0.10 * rec_term + 0.10 * brier_term


def _build_tier_models(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
) -> tuple[List[TierModel], pd.DataFrame]:
    monotonic = get_monotonic_constraints(numeric_features, stress_monotonic_positive=True)
    specs = make_model_registry(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        monotonic_cst=monotonic,
        include_physics_nn=False,
        random_state=random_state,
    )
    spec_map = {s.name: s for s in specs}

    # Controlled tuning on B1 Random Forest
    b1_base = clone(spec_map["tierB_random_forest"].estimator)
    cv_small = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
    sample_n = min(len(X), 8000)
    Xs, ys = X.iloc[:sample_n], y[:sample_n]
    b1_best, b1_params, b1_score = controlled_random_search(
        estimator=b1_base,
        param_distributions={
            "clf__n_estimators": [80, 100, 140],
            "clf__max_depth": [None, 8, 12],
            "clf__min_samples_leaf": [1, 2, 4],
        },
        X=Xs,
        y=ys,
        cv=cv_small,
        n_iter=6,
        random_state=random_state,
        scoring="average_precision",
    )

    # Calibration comparison for B2
    b2 = clone(spec_map["tierB_gradient_boosting"].estimator)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_state)
    b2.fit(Xtr, ytr)
    cal_rows = []
    p_none = b2.predict_proba(Xva)[:, 1]
    m_none = evaluate_probabilities(yva, p_none)
    cal_rows.append({"method": "none", **m_none})
    best_method = "sigmoid"
    best_brier = float("inf")
    for method in ["sigmoid", "isotonic"]:
        cal = make_calibrated_model(clone(spec_map["tierB_gradient_boosting"].estimator), method=method, cv=2)
        cal.fit(Xtr, ytr)
        p = cal.predict_proba(Xva)[:, 1]
        m = evaluate_probabilities(yva, p)
        cal_rows.append({"method": method, **m})
        if m["Brier"] < best_brier:
            best_brier = m["Brier"]
            best_method = method
    calibration_df = pd.DataFrame(cal_rows)
    b3_est = make_calibrated_model(clone(spec_map["tierB_gradient_boosting"].estimator), method=best_method, cv=2)

    models = [
        TierModel("A1", "Logistic + interactions", "Tier A", clone(spec_map["tierA_logistic"].estimator), 1.0, {}),
        TierModel("A2", "Shallow Decision Tree", "Tier A", clone(spec_map["tierA_tree"].estimator), 1.6, {}),
        TierModel("B1", "Random Forest (tuned)", "Tier B", b1_best, 3.0, {"tuned_params": b1_params, "cv_score": b1_score}),
        TierModel("B2", "Gradient Boosting", "Tier B", clone(spec_map["tierB_gradient_boosting"].estimator), 3.4, {}),
        TierModel("B3", f"Calibrated B2 ({best_method})", "Tier B", b3_est, 3.8, {"calibration_method": best_method}),
        TierModel("C1", "Monotonic-constrained GBM", "Tier C", clone(spec_map["tierC_monotonic_hgb"].estimator), 4.4, {}),
        TierModel("C2", "Physics-regularized logistic", "Tier C", clone(spec_map["tierC_physics_logit"].estimator), 2.6, {}),
        TierModel("C3", "Two-stage hybrid", "Tier C", clone(spec_map["tierC_two_stage_hybrid"].estimator), 5.0, {}),
    ]
    return models, calibration_df

def _evaluate_step_protocol(
    step_name: str,
    protocol: Dict,
    models: List[TierModel],
    X_clean: pd.DataFrame,
    X_noisy: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    leaveout_frame: pd.DataFrame,
    random_state: int,
    latency_map: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    for tm in models:
        if step_name == "step3_noise":
            summary, _, _ = evaluate_model_cv_noisy_test(
                model_name=tm.model_code,
                estimator=clone(tm.estimator),
                X_clean=X_clean,
                X_noisy=X_noisy,
                y=y,
                split_mode=protocol["split_mode"],
                scenario=step_name,
                groups=groups,
                n_splits=protocol["n_splits"],
                random_state=random_state,
                leaveout_feature=protocol.get("leave_feature", "Sgn_eff_MVA"),
                leaveout_frame=leaveout_frame,
            )
        else:
            summary, _, _ = evaluate_model_cv(
                model_name=tm.model_code,
                estimator=clone(tm.estimator),
                X=X_clean,
                y=y,
                split_mode=protocol["split_mode"],
                scenario=step_name,
                groups=groups,
                n_splits=protocol["n_splits"],
                random_state=random_state,
                leaveout_feature=protocol.get("leave_feature", "Sgn_eff_MVA"),
                leaveout_frame=leaveout_frame,
            )
        summary["model_code"] = tm.model_code
        summary["model_name"] = tm.model_name
        summary["tier"] = tm.tier
        summary["complexity_score"] = tm.complexity_score
        summary["validation_protocol"] = protocol["name"]
        summary["step"] = step_name
        summary["Latency_ms"] = latency_map.get(tm.model_code, np.nan)
        rows.append(summary)

    df = pd.DataFrame(rows)
    df["CompositeScore"] = _composite_score(df)

    for metric in [
        "PR_AUC",
        "ROC_AUC",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "Balanced_Acc",
        "CompositeScore",
        "R2",
    ]:
        best = df[metric].astype(float).max()
        df[f"best_{metric}"] = (df[metric].astype(float) == best).astype(int)
    for metric in ["FNR", "Brier", "MSE", "RMSE", "MAE", "ECE", "Latency_ms"]:
        best = df[metric].astype(float).min()
        df[f"best_{metric}"] = (df[metric].astype(float) == best).astype(int)

    return df


def _delta_vs_baseline(df: pd.DataFrame, baseline_code: str = "A1") -> pd.DataFrame:
    base = df[df["model_code"] == baseline_code].iloc[0]
    out = df[["model_code", "model_name", "tier", "validation_protocol", "step"]].copy()
    out["Delta_PR_AUC"] = df["PR_AUC"] - float(base["PR_AUC"])
    out["Delta_FNR"] = df["FNR"] - float(base["FNR"])
    out["Delta_ECE"] = df["ECE"] - float(base["ECE"])
    out["Delta_Brier"] = df["Brier"] - float(base["Brier"])
    return out


def _extract_a2_rules(model, output_path: Path) -> Path:
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    txt = export_text(clf, feature_names=names, max_depth=4)
    output_path.write_text(txt, encoding="utf-8")
    return output_path


def _operator_rules_from_a1(model, output_path: Path) -> Path:
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    coefs = clf.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]
    lines = ["# Operator Rules of Thumb (from A1 Logistic)", "", "Largest positive risk contributors:"]
    for i in order[:5]:
        if coefs[i] > 0:
            lines.append(f"- {names[i]}: coefficient={coefs[i]:.4f}")
    lines.append("")
    lines.append("Largest negative risk contributors:")
    for i in order[:5]:
        if coefs[i] < 0:
            lines.append(f"- {names[i]}: coefficient={coefs[i]:.4f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _model_card(path: Path, model_name: str, calibration: str, thresholds: Dict[str, float], metrics: Dict[str, float]) -> None:
    txt = f"""# OOS GR1 Model Card

## Model Identity
- Model: {model_name}
- Calibration: {calibration}
- Version: v1.1.0
- Task framing: pattern-based binary classification for operating-point risk forecasting (not time-series forecasting).

## Inputs
Tag_rate, Ikssmin_kA, Sgn_eff_MVA, H_s, GenName

## Outputs
p_oos and class label under threshold policy.

## Threshold Policy
- tau_cost: {thresholds['tau_cost']:.4f}
- tau_F1: {thresholds['tau_f1']:.4f}
- tau_HR: {thresholds['tau_hr']:.4f}

## Validation Snapshot
- PR_AUC: {metrics.get('PR_AUC', float('nan')):.4f}
- ROC_AUC: {metrics.get('ROC_AUC', float('nan')):.4f}
- FNR: {metrics.get('FNR', float('nan')):.4f}
- ECE: {metrics.get('ECE', float('nan')):.4f}
"""
    path.write_text(txt, encoding="utf-8")


def run_pipeline(data_path: Path, output_root: Path, random_state: int = 42) -> List[Path]:
    manifest = ManifestRegistry()
    dirs = _ensure_output_dirs(output_root)
    exported: List[Path] = []

    print_ascii_flowchart()
    f_png, f_pdf = plot_flowchart_figure(dirs["figures"])
    exported.extend([f_png, f_pdf])
    manifest.add("flowchart", "figure", "all", "all", "ALL", {}, "src/plots.py::plot_flowchart_figure", f_png)
    manifest.add("flowchart", "figure", "all", "all", "ALL", {}, "src/plots.py::plot_flowchart_figure", f_pdf)

    version = _version_report()
    print("Version report:", version)

    df_raw, audit = load_dataset(data_path)
    print_audit_report(audit)
    df = build_feature_frame(df_raw)

    summary_path = dirs["tables"] / "dataset_summary.csv"
    dataset_summary_table(df).to_csv(summary_path, index=False)
    exported.append(summary_path)
    manifest.add("dataset_summary", "table", "all", "all", "ALL", {}, "src/data.py::dataset_summary_table", summary_path)

    audit_path = dirs["tables"] / "data_audit.json"
    audit_path.write_text(json.dumps(audit.to_dict(), indent=2), encoding="utf-8")
    exported.append(audit_path)
    manifest.add("data_audit", "table", "all", "all", "ALL", {}, "src/data.py::load_dataset", audit_path)

    y = df[TARGET_COLUMN].astype(int).values
    numeric_features = [c for c in ALL_NUMERIC_FEATURES if c in df.columns]
    categorical_features = ["GenName"] if "GenName" in df.columns else []
    model_cols = numeric_features + categorical_features
    X = df[model_cols].copy()

    groups = make_group_labels(df, ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"], round_decimals=2)

    # Runtime-aware evaluation subset for Step 1-3 comparisons.
    if len(X) > 12000:
        rng = np.random.default_rng(random_state)
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        n_eval = 12000
        n_pos = max(1, int(n_eval * float(np.mean(y))))
        n_neg = n_eval - n_pos
        pick_pos = rng.choice(idx_pos, size=min(len(idx_pos), n_pos), replace=False)
        pick_neg = rng.choice(idx_neg, size=min(len(idx_neg), n_neg), replace=False)
        eval_idx = np.sort(np.concatenate([pick_pos, pick_neg]))
    else:
        eval_idx = np.arange(len(X))

    X_eval = X.iloc[eval_idx].reset_index(drop=True)
    y_eval = y[eval_idx]
    df_eval = df.iloc[eval_idx].reset_index(drop=True)
    groups_eval = make_group_labels(df_eval, ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"], round_decimals=2)

    models, calibration_df = _build_tier_models(
        X=X_eval,
        y=y_eval,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
    )
    cal_path = dirs["tables"] / "calibration_comparison_B2.csv"
    calibration_df.to_csv(cal_path, index=False)
    exported.append(cal_path)
    manifest.add("calibration_comparison_B2", "table", "step1_static", "V1_stratified", "B3", {}, "main.py::_build_tier_models", cal_path)

    # latency profile for leaderboard columns
    latency_map = {}
    for tm in models:
        latency_map[tm.model_code] = _profile_latency_ms(tm.estimator, X_eval, y_eval, n_runs=80)

    protocols = [
        {"name": "V1_stratified", "split_mode": "stratified", "n_splits": 2},
        {"name": "V2_grouped", "split_mode": "grouped", "n_splits": 2},
        {"name": "V3_leave_Sgn", "split_mode": "leave-level-out", "n_splits": 3, "leave_feature": "Sgn_eff_MVA"},
        {"name": "V3_leave_Ik", "split_mode": "leave-level-out", "n_splits": 3, "leave_feature": "Ikssmin_kA"},
    ]
    protocol_map = {p["name"]: p for p in protocols}
    scenario_protocols = {
        # Step 1: baseline static screening on standard stratified CV.
        "step1_static": [protocol_map["V1_stratified"]],
        # Step 2: explicit generalization stress test on grouped and leave-level-out splits.
        "step2_robustness": [
            protocol_map["V2_grouped"],
            protocol_map["V3_leave_Sgn"],
            protocol_map["V3_leave_Ik"],
        ],
        # Step 3: noisy-measurement realism evaluated on all protocols.
        "step3_noise": protocols,
    }

    noise_cfg = {"Ikssmin_kA": 0.02, "Sgn_eff_MVA": 0.02, "H_s": 0.01, "Tag_rate": 0.01}
    X_noisy = add_noise(X_eval, noise_cfg, random_state=random_state)

    all_results = []
    all_deltas = []
    for step_name, protocol_list in scenario_protocols.items():
        for protocol in protocol_list:
            lb = _evaluate_step_protocol(
                step_name=step_name,
                protocol=protocol,
                models=models,
                X_clean=X_eval,
                X_noisy=X_noisy,
                y=y_eval,
                groups=groups_eval,
                leaveout_frame=df_eval,
                random_state=random_state,
                latency_map=latency_map,
            )
            lb_path = dirs["tables"] / f"leaderboard_{step_name}_{protocol['name']}.csv"
            lb.to_csv(lb_path, index=False)
            exported.append(lb_path)
            manifest.add(
                artifact_name=f"leaderboard_{step_name}_{protocol['name']}",
                artifact_type="table",
                step=step_name,
                validation_protocol=protocol["name"],
                model_name="ALL",
                config={"n_splits": protocol["n_splits"]},
                code_path="main.py::_evaluate_step_protocol",
                file_path=lb_path,
            )
            all_results.append(lb)

            dd = _delta_vs_baseline(lb, baseline_code="A1")
            dd_path = dirs["tables"] / f"delta_step_{step_name}_{protocol['name']}.csv"
            dd.to_csv(dd_path, index=False)
            exported.append(dd_path)
            manifest.add(
                artifact_name=f"delta_step_{step_name}_{protocol['name']}",
                artifact_type="table",
                step=step_name,
                validation_protocol=protocol["name"],
                model_name="ALL",
                config={},
                code_path="main.py::_delta_vs_baseline",
                file_path=dd_path,
            )
            all_deltas.append(dd)

    results_df = pd.concat(all_results, ignore_index=True)
    results_all_path = dirs["tables"] / "all_tier_results.csv"
    results_df.to_csv(results_all_path, index=False)
    exported.append(results_all_path)
    manifest.add("all_tier_results", "table", "all", "all", "ALL", {}, "main.py::run_pipeline", results_all_path)

    deltas_df = pd.concat(all_deltas, ignore_index=True)
    deltas_all_path = dirs["tables"] / "all_delta_vs_A1.csv"
    deltas_df.to_csv(deltas_all_path, index=False)
    exported.append(deltas_all_path)
    manifest.add("all_delta_vs_A1", "table", "all", "all", "ALL", {}, "main.py::run_pipeline", deltas_all_path)

    # Step-2 robustness consistency relative to Step-1 V1 baseline per model.
    s1 = results_df[
        (results_df["step"] == "step1_static") & (results_df["validation_protocol"] == "V1_stratified")
    ][["model_code", "PR_AUC", "FNR", "ECE"]].rename(
        columns={"PR_AUC": "PR_AUC_step1_v1", "FNR": "FNR_step1_v1", "ECE": "ECE_step1_v1"}
    )
    s2 = results_df[results_df["step"] == "step2_robustness"][
        ["model_code", "model_name", "tier", "validation_protocol", "PR_AUC", "FNR", "ECE"]
    ].copy()
    bc = s2.merge(s1, on="model_code", how="left")
    bc["Delta_PR_AUC_vs_step1"] = bc["PR_AUC"] - bc["PR_AUC_step1_v1"]
    bc["Delta_FNR_vs_step1"] = bc["FNR"] - bc["FNR_step1_v1"]
    bc["Delta_ECE_vs_step1"] = bc["ECE"] - bc["ECE_step1_v1"]
    bc_path = dirs["tables"] / "boundary_consistency_step2.csv"
    bc.to_csv(bc_path, index=False)
    exported.append(bc_path)
    manifest.add(
        "boundary_consistency_step2",
        "table",
        "step2_robustness",
        "V2_grouped+V3_leave_level",
        "ALL",
        {},
        "main.py::run_pipeline",
        bc_path,
    )

    # Trade-off plots from Step1 + V1
    ref = results_df[(results_df["step"] == "step1_static") & (results_df["validation_protocol"] == "V1_stratified")].copy()
    t1_png, t1_pdf = plot_tradeoff_scatter(
        ref,
        x_col="complexity_score",
        y_col="PR_AUC",
        label_col="model_code",
        color_col="tier",
        title="PR-AUC vs Model Complexity",
        output_stem=dirs["figures"] / "tradeoff_prauc_vs_complexity",
    )
    t2_png, t2_pdf = plot_tradeoff_scatter(
        ref,
        x_col="ECE",
        y_col="PR_AUC",
        label_col="model_code",
        color_col="tier",
        title="PR-AUC vs ECE",
        output_stem=dirs["figures"] / "tradeoff_prauc_vs_ece",
    )
    t3_png, t3_pdf = plot_tradeoff_scatter(
        ref,
        x_col="FPR_HR",
        y_col="Recall_HR",
        label_col="model_code",
        color_col="tier",
        title="Recall vs FPR at tau_HR",
        output_stem=dirs["figures"] / "tradeoff_recall_vs_fpr_hr",
    )
    for p in [t1_png, t1_pdf, t2_png, t2_pdf, t3_png, t3_pdf]:
        exported.append(p)
        manifest.add(p.stem, "figure", "step1_static", "V1_stratified", "ALL", {}, "src/plots.py::plot_tradeoff_scatter", p)

    # Noise robustness for best A/B/C
    best_ref = ref.sort_values("CompositeScore", ascending=False)
    best_a = best_ref[best_ref["tier"] == "Tier A"].iloc[0]["model_code"]
    best_b = best_ref[best_ref["tier"] == "Tier B"].iloc[0]["model_code"]
    best_c = best_ref[best_ref["tier"] == "Tier C"].iloc[0]["model_code"]
    best_set = {best_a, best_b, best_c}
    model_lookup = {m.model_code: m for m in models}

    noise_rows = []
    for lvl in [0.0, 0.01, 0.02, 0.03]:
        cfg = {"Ikssmin_kA": lvl, "Sgn_eff_MVA": lvl, "H_s": lvl * 0.5, "Tag_rate": lvl * 0.4}
        xn = add_noise(X_eval, cfg, random_state=random_state + int(1000 * lvl))
        for code in sorted(best_set):
            tm = model_lookup[code]
            s, _, _ = evaluate_model_cv_noisy_test(
                model_name=code,
                estimator=clone(tm.estimator),
                X_clean=X_eval,
                X_noisy=xn,
                y=y_eval,
                split_mode="stratified",
                scenario="noise_curve",
                n_splits=2,
                random_state=random_state,
            )
            s.update({"model_code": code, "tier": tm.tier, "noise_level": lvl})
            noise_rows.append(s)
    noise_curve = pd.DataFrame(noise_rows)
    noise_curve_path = dirs["tables"] / "noise_robustness_best_ABC.csv"
    noise_curve.to_csv(noise_curve_path, index=False)
    exported.append(noise_curve_path)
    manifest.add("noise_robustness_best_ABC", "table", "step3_noise", "V1_stratified", "ALL", {}, "main.py::run_pipeline", noise_curve_path)

    nr1_png, nr1_pdf = plot_noise_robustness(
        noise_curve,
        x_col="noise_level",
        y_col="PR_AUC",
        group_col="model_code",
        title="Noise Robustness: PR-AUC vs Noise",
        output_stem=dirs["figures"] / "noise_robustness_prauc",
    )
    nr2_png, nr2_pdf = plot_noise_robustness(
        noise_curve,
        x_col="noise_level",
        y_col="FNR",
        group_col="model_code",
        title="Noise Robustness: FNR vs Noise",
        output_stem=dirs["figures"] / "noise_robustness_fnr",
    )
    for p in [nr1_png, nr1_pdf, nr2_png, nr2_pdf]:
        exported.append(p)
        manifest.add(p.stem, "figure", "step3_noise", "V1_stratified", "ALL", {}, "src/plots.py::plot_noise_robustness", p)

    # Select deployment model by robust average score
    robust_view = results_df[results_df["step"].isin(["step2_robustness", "step3_noise"])].copy()
    model_rank = (
        robust_view.groupby(["model_code", "model_name", "tier"], as_index=False)[["CompositeScore", "FNR", "ECE"]]
        .mean()
        .sort_values(["CompositeScore", "FNR", "ECE"], ascending=[False, True, True])
    )
    chosen = model_rank.iloc[0]
    chosen_code = str(chosen["model_code"])
    chosen_model = model_lookup[chosen_code]

    # thresholds from Step1 V1 row of chosen model
    chosen_row = ref[ref["model_code"] == chosen_code].iloc[0]
    thresholds = {
        "tau_cost": float(chosen_row["tau_cost"]),
        "tau_f1": float(chosen_row["tau_F1"]),
        "tau_hr": float(chosen_row["tau_HR"]),
    }

    final_model = clone(chosen_model.estimator).fit(X, y)
    final_metrics = evaluate_probabilities(y, final_model.predict_proba(X)[:, 1])

    # Figures for explainability and boundary
    fd_png, fd_pdf = plot_feature_distributions(df, TARGET_COLUMN, dirs["figures"])
    for p in [fd_png, fd_pdf]:
        exported.append(p)
        manifest.add("feature_distributions", "figure", "step1_static", "V1_stratified", "ALL", {}, "src/plots.py::plot_feature_distributions", p)

    smap = plot_stability_map(
        final_model,
        df,
        target_col=TARGET_COLUMN,
        x_feature="Ikssmin_kA",
        y_feature="Sgn_eff_MVA",
        output_stem=dirs["figures"] / "stability_map_Ik_vs_Sgn",
        fixed_values={"H_s": float(df["H_s"].median()), "Tag_rate": float(df["Tag_rate"].median())},
        threshold=thresholds["tau_cost"],
    )
    np.save(dirs["figures"] / "stability_surface.npy", smap)
    exported.append(dirs["figures"] / "stability_surface.npy")
    manifest.add("stability_surface", "figure-data", "step1_static", "V1_stratified", chosen_code, {}, "src/plots.py::plot_stability_map", dirs["figures"] / "stability_surface.npy")

    p_train = final_model.predict_proba(X)[:, 1]
    cr_png, cr_pdf = plot_reliability_diagram(y, p_train, dirs["figures"] / "calibration_reliability")
    for p in [cr_png, cr_pdf]:
        exported.append(p)
        manifest.add("calibration_reliability", "figure", "step1_static", "V1_stratified", chosen_code, {}, "src/plots.py::plot_reliability_diagram", p)

    fi_png, fi_pdf = plot_feature_importance(final_model, X, y, dirs["figures"] / "feature_importance_permutation")
    for p in [fi_png, fi_pdf]:
        exported.append(p)
        manifest.add("feature_importance_permutation", "figure", "step1_static", "V1_stratified", chosen_code, {}, "src/plots.py::plot_feature_importance", p)

    pdp_png, pdp_pdf = plot_pdp(final_model, X, [c for c in ["H_s", "Ikssmin_kA", "Sgn_eff_MVA", "Tag_rate"] if c in X.columns], dirs["figures"] / "pdp_main_features")
    for p in [pdp_png, pdp_pdf]:
        exported.append(p)
        manifest.add("pdp_main_features", "figure", "step1_static", "V1_stratified", chosen_code, {}, "src/plots.py::plot_pdp", p)

    c1_model = clone(model_lookup["C1"].estimator).fit(X, y)
    bc_png, bc_pdf = plot_boundary_comparison(final_model, c1_model, df, TARGET_COLUMN, "Ikssmin_kA", "Sgn_eff_MVA", dirs["figures"] / "boundary_comparison")
    for p in [bc_png, bc_pdf]:
        exported.append(p)
        manifest.add("boundary_comparison", "figure", "step2_robustness", "V2_grouped", "ALL", {}, "src/plots.py::plot_boundary_comparison", p)

    # Interpretable rule artifacts
    a2_fit = clone(model_lookup["A2"].estimator).fit(X, y)
    a2_rules_path = _extract_a2_rules(a2_fit, dirs["tables"] / "A2_tree_rules.txt")
    exported.append(a2_rules_path)
    manifest.add("A2_tree_rules", "table", "step1_static", "V1_stratified", "A2", {}, "main.py::_extract_a2_rules", a2_rules_path)

    a1_fit = clone(model_lookup["A1"].estimator).fit(X, y)
    op_rules_path = _operator_rules_from_a1(a1_fit, dirs["tables"] / "operator_rules_of_thumb.md")
    exported.append(op_rules_path)
    manifest.add("operator_rules_of_thumb", "table", "step1_static", "V1_stratified", "A1", {}, "main.py::_operator_rules_from_a1", op_rules_path)

    # Counterfactual examples
    feature_bounds = derive_feature_bounds(df, numeric_features)
    risky = np.where(p_train >= thresholds["tau_cost"])[0]
    cf_rows = []
    for idx in risky[: min(25, len(risky))]:
        cf = minimal_counterfactual(final_model, X.iloc[idx], feature_bounds, threshold=thresholds["tau_cost"], random_state=random_state + int(idx))
        cf["sample_idx"] = int(idx)
        cf_rows.append(cf)
    cf_path = dirs["tables"] / "counterfactual_examples.csv"
    pd.DataFrame(cf_rows).to_csv(cf_path, index=False)
    exported.append(cf_path)
    manifest.add("counterfactual_examples", "table", "step1_static", "V1_stratified", chosen_code, {}, "src/eval.py::minimal_counterfactual", cf_path)

    # Step 4 deployment artifacts
    model_path = dirs["model"] / "model.joblib"
    prep_path = dirs["model"] / "preprocessor.joblib"
    cal_obj_path = dirs["model"] / "calibrator.joblib"
    joblib.dump(final_model, model_path)
    prep_obj = getattr(final_model, "named_steps", {}).get("prep", None) if hasattr(final_model, "named_steps") else None
    joblib.dump(prep_obj, prep_path)
    joblib.dump(final_model if "CalibratedClassifierCV" in type(final_model).__name__ else None, cal_obj_path)
    for p, name in [(model_path, "model_artifact"), (prep_path, "preprocessor_artifact"), (cal_obj_path, "calibrator_artifact")]:
        exported.append(p)
        manifest.add(name, "model", "step4_deploy", "NA", chosen_code, chosen_model.config, "main.py::run_pipeline", p)

    cfg = {
        "model_version": "v1.1.0",
        "calibration_version": chosen_model.config.get("calibration_method", "none"),
        "thresholds": thresholds,
        "feature_bounds": {k: [float(v[0]), float(v[1])] for k, v in feature_bounds.items()},
        "features": model_cols,
        "model_code": chosen_code,
    }
    cfg_path = dirs["model"] / "config.yaml"
    _write_yaml(cfg_path, cfg)
    exported.append(cfg_path)
    manifest.add("model_config", "model", "step4_deploy", "NA", chosen_code, cfg, "main.py::run_pipeline", cfg_path)

    card_path = dirs["model"] / "model_card.md"
    _model_card(card_path, chosen_model.model_name, cfg["calibration_version"], thresholds, final_metrics)
    exported.append(card_path)
    manifest.add("model_card", "model", "step4_deploy", "NA", chosen_code, {}, "main.py::_model_card", card_path)

    dep_rows = [{"latency_ms_per_request": latency_map[chosen_code], "throughput_samples_per_sec": float(1000.0 / max(latency_map[chosen_code], 1e-9)), "artifact_footprint_mb": sum(f.stat().st_size for f in dirs["model"].glob("*.joblib")) / (1024.0 * 1024.0)}]
    dep_path = dirs["tables"] / "deployment_metrics.csv"
    pd.DataFrame(dep_rows).to_csv(dep_path, index=False)
    exported.append(dep_path)
    manifest.add("deployment_metrics", "table", "step4_deploy", "NA", chosen_code, {}, "main.py::run_pipeline", dep_path)

    for p in _copy_api_files(Path.cwd(), dirs["api"]):
        exported.append(p)
        manifest.add(f"api_{p.stem}", "software", "step4_deploy", "NA", chosen_code, {}, "main.py::_copy_api_files", p)

    for p in _copy_tests(Path.cwd(), dirs["tests"]):
        exported.append(p)
        manifest.add(f"tests_{p.stem}", "software", "step4_deploy", "NA", chosen_code, {}, "main.py::_copy_tests", p)

    # Step 5 monitoring and retraining policy
    stream_num = simulate_stream(df[numeric_features], drift_strength=0.12, random_state=random_state)
    stream_x = X.copy()
    for c in numeric_features:
        stream_x[c] = stream_num[c]

    psi_df = psi_table(df[numeric_features], stream_num[numeric_features], numeric_features)
    ks_df = ks_table(df[numeric_features], stream_num[numeric_features], numeric_features)
    p_stream = final_model.predict_proba(stream_x)[:, 1]
    y_stream = y.copy()
    pred_stream = (p_stream >= thresholds["tau_cost"]).astype(int)
    errors = (pred_stream != y_stream).astype(int)
    concept_df = concept_drift_scan(errors, p_stream)
    perf_roll = rolling_performance(y_stream, p_stream, window=max(30, len(y_stream) // 5), threshold=thresholds["tau_cost"])
    retrain_decision = retrain_trigger_policy(psi_df, ks_df, concept_df, new_sample_count=len(stream_x), min_new_samples=max(100, len(stream_x) // 3))

    psi_path = dirs["monitoring"] / "psi.csv"
    ks_path = dirs["monitoring"] / "ks.csv"
    concept_path = dirs["monitoring"] / "concept_drift.csv"
    perf_path = dirs["monitoring"] / "rolling_performance.csv"
    policy_path = dirs["monitoring"] / "retrain_policy.json"
    psi_df.to_csv(psi_path, index=False)
    ks_df.to_csv(ks_path, index=False)
    concept_df.to_csv(concept_path)
    perf_roll.to_csv(perf_path)
    policy_path.write_text(json.dumps(retrain_decision, indent=2), encoding="utf-8")

    for p, name in [(psi_path, "psi"), (ks_path, "ks"), (concept_path, "concept_drift"), (perf_path, "rolling_performance"), (policy_path, "retrain_policy")]:
        exported.append(p)
        manifest.add(name, "monitoring", "step5_monitor", "NA", chosen_code, {}, "src/monitoring.py", p)

    drift_time = perf_roll.join(concept_df[["PH_alarm", "DDM_alarm", "ADWIN_alarm"]], how="left").fillna(0)
    dm_png, dm_pdf = plot_drift_monitoring(drift_time, dirs["figures"] / "drift_monitoring", ["PR_AUC", "Recall", "FNR", "PH_alarm", "DDM_alarm", "ADWIN_alarm"])
    for p in [dm_png, dm_pdf]:
        exported.append(p)
        manifest.add("drift_monitoring", "figure", "step5_monitor", "NA", chosen_code, {}, "src/plots.py::plot_drift_monitoring", p)

    # Problem formulation + results discussion docs
    problem_path = build_problem_formulation_markdown(
        output_path=dirs["root"] / "OOS_GR1_problem_formulation.md",
        summary={
            "best_model": f"{chosen_code} {chosen_model.model_name}",
            "best_calibration": cfg["calibration_version"],
            "tau_f1": f"{thresholds['tau_f1']:.4f}",
            "tau_hr": f"{thresholds['tau_hr']:.4f}",
            "tau_cost": f"{thresholds['tau_cost']:.4f}",
            "notes": "Full execution completed with tier-by-tier comparisons.",
        },
        table_paths=[str(p) for p in sorted(dirs["tables"].glob("*.csv"))],
        figure_paths=[str(p) for p in sorted(dirs["figures"].glob("*.png"))],
    )
    exported.append(problem_path)
    manifest.add("problem_formulation", "document", "all", "all", "ALL", {}, "src/report_problem.py::build_problem_formulation_markdown", problem_path)

    tier_summary = (
        results_df.groupby("tier", as_index=False)[["PR_AUC", "FNR", "ECE", "MSE", "RMSE", "R2", "CompositeScore"]]
        .mean()
        .sort_values("CompositeScore", ascending=False)
    )
    rec = {
        "model_name": f"{chosen_code} {chosen_model.model_name}",
        "calibration": cfg["calibration_version"],
        "tau_f1": f"{thresholds['tau_f1']:.4f}",
        "tau_hr": f"{thresholds['tau_hr']:.4f}",
        "tau_cost": f"{thresholds['tau_cost']:.4f}",
    }
    manifest_path = dirs["root"] / "results_manifest.json"
    manifest.write(manifest_path)
    exported.append(manifest_path)

    results_doc = build_results_discussion_markdown(
        output_path=dirs["root"] / "OOS_GR1_results_discussion.md",
        results_df=results_df,
        tier_summary_df=tier_summary,
        recommendation=rec,
        manifest_records=manifest.records,
    )
    exported.append(results_doc)
    manifest.add("results_discussion", "document", "all", "all", "ALL", {}, "src/report_results.py::build_results_discussion_markdown", results_doc)

    # rewrite manifest with final document record included
    manifest.write(manifest_path)

    print("\nBest Deployment Recommendation")
    print(f"- Model: {rec['model_name']}")
    print(f"- Calibration: {rec['calibration']}")
    print(f"- Thresholds: tau_F1={rec['tau_f1']}, tau_HR={rec['tau_hr']}, tau_cost={rec['tau_cost']}")

    return exported


def scaffold_only(output_root: Path) -> List[Path]:
    dirs = _ensure_output_dirs(output_root)
    manifest = ManifestRegistry()
    exported: List[Path] = []

    print_ascii_flowchart()
    for p in plot_flowchart_figure(dirs["figures"]):
        exported.append(p)
        manifest.add("flowchart", "figure", "all", "all", "ALL", {}, "src/plots.py::plot_flowchart_figure", p)

    problem = build_problem_formulation_markdown(
        output_path=dirs["root"] / "OOS_GR1_problem_formulation.md",
        summary={"notes": "Scaffold generated. Awaiting dataset execution."},
    )
    exported.append(problem)
    manifest.add("problem_formulation", "document", "all", "all", "ALL", {}, "src/report_problem.py::build_problem_formulation_markdown", problem)

    placeholder_results = dirs["root"] / "OOS_GR1_results_discussion.md"
    placeholder_results.write_text(
        "# OOS GR1 Results & Discussion\n\nPending execution with attached CSV.\n",
        encoding="utf-8",
    )
    exported.append(placeholder_results)
    manifest.add("results_discussion", "document", "all", "all", "ALL", {}, "src/report_results.py::build_results_discussion_markdown", placeholder_results)

    for p in _copy_api_files(Path.cwd(), dirs["api"]):
        exported.append(p)
        manifest.add(f"api_{p.stem}", "software", "step4_deploy", "NA", "ALL", {}, "main.py::_copy_api_files", p)

    for p in _copy_tests(Path.cwd(), dirs["tests"]):
        exported.append(p)
        manifest.add(f"tests_{p.stem}", "software", "step4_deploy", "NA", "ALL", {}, "main.py::_copy_tests", p)

    readme = dirs["model"] / "README.txt"
    readme.write_text("Model artifacts will be exported after training with CSV input.\n", encoding="utf-8")
    exported.append(readme)
    manifest.add("model_placeholder", "model", "step4_deploy", "NA", "ALL", {}, "main.py::scaffold_only", readme)

    manifest_path = dirs["root"] / "results_manifest.json"
    manifest.write(manifest_path)
    exported.append(manifest_path)
    return exported


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GR1 Out-of-Step ML + Physics + Deployment Pipeline")
    p.add_argument("--data", type=str, default="data/raw/outofstep_tag_ikss_H_Sgn.csv")
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_root = Path(args.output_dir)

    if data_path.exists():
        print(f"Data found at {data_path}. Running full pipeline...")
        exported = run_pipeline(data_path, output_root, random_state=args.seed)
    else:
        print(f"Data not found at {data_path}. Generating scaffold/template only...")
        exported = scaffold_only(output_root)

    print("\n=== EXPORTED ARTIFACTS ===")
    for p in sorted(set(exported)):
        print(str(p))
    print("==========================")


if __name__ == "__main__":
    main()
