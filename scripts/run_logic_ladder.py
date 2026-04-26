from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import derive_feature_bounds, resolve_engineered_feature_columns
from src.outofstep_ml.benchmark.runner import (
    _apply_calibrator,
    _class_metrics,
    _prob_metrics,
    _select_best_calibration,
    _shifted_tests,
    make_strict_split,
)
from src.outofstep_ml.data.loaders import load_validated_dataset
from src.outofstep_ml.deployment.monitor import feature_drift_report
from src.outofstep_ml.evaluation.counterfactual_eval import evaluate_counterfactual_stability_correction
from src.outofstep_ml.evaluation.threshold_policy_compare import compare_threshold_policies
from src.outofstep_ml.models.baselines import build_baseline_ladder
from src.outofstep_ml.models.thresholds import optimize_thresholds
from src.outofstep_ml.utils.io import ensure_dir, load_yaml, save_json
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


def _write_md(df: pd.DataFrame, path: Path) -> None:
    try:
        path.write_text(df.to_markdown(index=False), encoding="utf-8")
    except Exception:
        # Keep pipeline resilient if markdown conversion is unavailable.
        pass


def _fit_stage(
    *,
    stage_id: str,
    stage_name: str,
    stage_logic: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    c_fn: float,
    c_fp: float,
    min_recall: float,
    use_calibration: bool,
    use_cost_threshold: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model = clone(estimator)
    model.fit(X_train, y_train)

    p_val_raw = np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
    calib_name, calib_obj = "none", None
    if use_calibration:
        calib_name, calib_obj, _ = _select_best_calibration(y_val, p_val_raw)
    p_val = np.clip(_apply_calibrator(calib_obj, calib_name, p_val_raw), 1e-6, 1 - 1e-6)
    th = optimize_thresholds(y_val, p_val, c_fn=c_fn, c_fp=c_fp, min_recall=min_recall)

    p_test_raw = np.clip(model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)
    p_test = np.clip(_apply_calibrator(calib_obj, calib_name, p_test_raw), 1e-6, 1 - 1e-6)
    tau_used = float(th["tau_cost"]) if use_cost_threshold else 0.5
    y_hat = (p_test >= tau_used).astype(int)

    row = {
        "stage_id": stage_id,
        "stage_name": stage_name,
        "stage_logic": stage_logic,
        "calibration": calib_name,
        "threshold_policy": "tau_cost" if use_cost_threshold else "tau_default",
        "tau_used": tau_used,
        **_prob_metrics(y_test, p_test),
        **_class_metrics(y_test, y_hat),
        "tau_cost": float(th["tau_cost"]),
        "tau_f1": float(th["tau_f1"]),
        "tau_hr": float(th["tau_hr"]),
    }
    artifacts = {
        "model": model,
        "calib_name": calib_name,
        "calib_obj": calib_obj,
        "thresholds": th,
        "p_test": p_test,
    }
    return row, artifacts


def _combo_score(row: pd.Series) -> float:
    pr = float(row.get("PR_AUC", np.nan))
    fnr = float(row.get("FNR", np.nan))
    ece = float(row.get("ECE", np.nan))
    if not np.isfinite(pr):
        return float("nan")
    # Higher is better: reward discrimination, penalize missed-instability and miscalibration.
    return float(pr - 0.35 * (fnr if np.isfinite(fnr) else 0.0) - 0.10 * (ece if np.isfinite(ece) else 0.0))


def _build_combo_defs(
    *,
    raw_ladder: Dict[str, object],
    full_ladder: Dict[str, object],
    Xr_train: pd.DataFrame,
    Xr_val: pd.DataFrame,
    Xr_test: pd.DataFrame,
    Xf_train: pd.DataFrame,
    Xf_val: pd.DataFrame,
    Xf_test: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Build non-sequential capability combinations automatically.
    Capability keys:
    1 = raw baseline (always present)
    2 = engineered physics ratios
    3 = monotonic priors
    4 = imbalance-aware + cost-sensitive thresholding
    5 = calibration
    """
    defs: List[Dict[str, Any]] = []
    optional_caps = [2, 3, 4, 5]

    for r in range(0, len(optional_caps) + 1):
        for subset in combinations(optional_caps, r):
            caps = set(subset)
            # Dependency constraints:
            # monotonic/imbalance implementations are defined on engineered feature space.
            if 3 in caps and 2 not in caps:
                continue
            if 4 in caps and 2 not in caps:
                continue

            logic_digits = [1] + sorted(list(caps))
            combo_logic = "+".join(str(d) for d in logic_digits)
            combo_id = "C" + "".join(str(d) for d in logic_digits)

            uses_full_space = bool({2, 3, 4} & caps)
            X_train, X_val, X_test = (Xf_train, Xf_val, Xf_test) if uses_full_space else (Xr_train, Xr_val, Xr_test)

            if 3 in caps:
                estimator = full_ladder["C1_monotonic"]
            elif 4 in caps:
                estimator = full_ladder["C2_physics_logit"]
            elif 2 in caps:
                estimator = full_ladder["A1_logistic"]
            else:
                estimator = raw_ladder["A1_logistic"]

            parts = ["raw"]
            if 2 in caps:
                parts.append("engineered")
            if 3 in caps:
                parts.append("monotonic")
            if 4 in caps:
                parts.append("imbalance+cost")
            if 5 in caps:
                parts.append("calibrated")

            defs.append(
                {
                    "combo_id": combo_id,
                    "combo_logic": combo_logic,
                    "combo_name": " + ".join(parts),
                    "estimator": estimator,
                    "X_train": X_train,
                    "X_val": X_val,
                    "X_test": X_test,
                    "use_calibration": bool(5 in caps),
                    "use_cost_threshold": bool(4 in caps),
                }
            )

    # Stable ordering by number of capabilities then lexical.
    defs = sorted(defs, key=lambda d: (len(d["combo_logic"].split("+")), d["combo_logic"]))
    return defs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to logic-ladder YAML config.")
    args = ap.parse_args()

    cfg = _merge_cfg(Path(args.config))
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    out_root = Path(ensure_dir(cfg.get("outputs", {}).get("root", "results/logic_ladder")))
    out_tables = Path(ensure_dir(cfg.get("outputs", {}).get("table_dir", out_root / "tables")))
    out_splits = Path(ensure_dir(cfg.get("outputs", {}).get("split_dir", out_root / "splits")))

    df, audit = load_validated_dataset(
        cfg.get("data", {}).get("path"),
        include_logs=bool(cfg.get("features", {}).get("include_logs", True)),
    )
    save_json(out_tables / "logic_ladder_data_audit.json", audit.to_dict())

    y = df["Out_of_step"].astype(int).values
    raw_numeric = [c for c in ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"] if c in df.columns]
    engineered = resolve_engineered_feature_columns(df)
    full_numeric = [c for c in (raw_numeric + engineered) if c in df.columns]
    categorical = ["GenName"] if "GenName" in df.columns else []

    X_raw = df[raw_numeric + categorical].copy()
    X_full = df[full_numeric + categorical].copy()

    strict_cfg = cfg.get("strict_eval", {})
    split = make_strict_split(
        df=df,
        y=y,
        seed=seed,
        strategy=str(strict_cfg.get("strategy", "stratified")),
        test_size=float(strict_cfg.get("test_size", 0.2)),
        val_size=float(strict_cfg.get("val_size", 0.2)),
        group_cols=["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"],
        leave_feature="Sgn_eff_MVA",
    )
    split_frame = pd.DataFrame(
        {
            "index": np.concatenate([split.train_idx, split.val_idx, split.test_idx]),
            "subset": ["train"] * len(split.train_idx) + ["val"] * len(split.val_idx) + ["test"] * len(split.test_idx),
            "seed": seed,
            "strategy": split.strategy,
        }
    )
    split_frame.to_csv(out_splits / "logic_ladder_split.csv", index=False)

    Xr_train, Xr_val, Xr_test = X_raw.iloc[split.train_idx], X_raw.iloc[split.val_idx], X_raw.iloc[split.test_idx]
    Xf_train, Xf_val, Xf_test = X_full.iloc[split.train_idx], X_full.iloc[split.val_idx], X_full.iloc[split.test_idx]
    y_train, y_val, y_test = y[split.train_idx], y[split.val_idx], y[split.test_idx]

    c_fn = float(cfg.get("thresholds", {}).get("c_fn", 10.0))
    c_fp = float(cfg.get("thresholds", {}).get("c_fp", 1.0))
    min_recall = float(cfg.get("thresholds", {}).get("min_recall", 0.95))

    raw_ladder = build_baseline_ladder(raw_numeric, categorical, random_state=seed)
    full_ladder = build_baseline_ladder(full_numeric, categorical, random_state=seed)

    stage_defs = [
        {
            "stage_id": "S1",
            "stage_name": "Raw Baseline",
            "stage_logic": "Raw features only (T, I, S, H).",
            "estimator": raw_ladder["A1_logistic"],
            "X_train": Xr_train,
            "X_val": Xr_val,
            "X_test": Xr_test,
            "use_calibration": False,
            "use_cost_threshold": False,
        },
        {
            "stage_id": "S2",
            "stage_name": "Add Physics Ratios",
            "stage_logic": "Add invH, S_over_H, S_over_I, I_over_H.",
            "estimator": full_ladder["A1_logistic"],
            "X_train": Xf_train,
            "X_val": Xf_val,
            "X_test": Xf_test,
            "use_calibration": False,
            "use_cost_threshold": False,
        },
        {
            "stage_id": "S3",
            "stage_name": "Add Monotonic Priors",
            "stage_logic": "Monotonic constraints: df/dH<=0, df/dI<=0, df/dS>=0.",
            "estimator": full_ladder["C1_monotonic"],
            "X_train": Xf_train,
            "X_val": Xf_val,
            "X_test": Xf_test,
            "use_calibration": False,
            "use_cost_threshold": False,
        },
        {
            "stage_id": "S4",
            "stage_name": "Add Imbalance + Cost Threshold",
            "stage_logic": "Physics-aware imbalance handling + tau_cost policy.",
            "estimator": full_ladder["C2_physics_logit"],
            "X_train": Xf_train,
            "X_val": Xf_val,
            "X_test": Xf_test,
            "use_calibration": False,
            "use_cost_threshold": True,
        },
        {
            "stage_id": "S5",
            "stage_name": "Full Stack (Calibrated)",
            "stage_logic": "Calibration + reliability + cost thresholding.",
            "estimator": full_ladder["C2_physics_logit"],
            "X_train": Xf_train,
            "X_val": Xf_val,
            "X_test": Xf_test,
            "use_calibration": True,
            "use_cost_threshold": True,
        },
    ]

    stage_rows: List[Dict[str, Any]] = []
    stage_artifacts: Dict[str, Dict[str, Any]] = {}
    for stage in stage_defs:
        row, artifacts = _fit_stage(
            stage_id=stage["stage_id"],
            stage_name=stage["stage_name"],
            stage_logic=stage["stage_logic"],
            estimator=stage["estimator"],
            X_train=stage["X_train"],
            y_train=y_train,
            X_val=stage["X_val"],
            y_val=y_val,
            X_test=stage["X_test"],
            y_test=y_test,
            c_fn=c_fn,
            c_fp=c_fp,
            min_recall=min_recall,
            use_calibration=bool(stage["use_calibration"]),
            use_cost_threshold=bool(stage["use_cost_threshold"]),
        )
        stage_rows.append(row)
        stage_artifacts[stage["stage_id"]] = artifacts

    stage_df = pd.DataFrame(stage_rows)
    stage_df.to_csv(out_tables / "logic_ladder_comparison.csv", index=False)
    _write_md(stage_df, out_tables / "logic_ladder_comparison.md")

    # Advanced scenario outputs from final full-stack stage.
    final_art = stage_artifacts["S5"]
    final_model = final_art["model"]
    final_calib_name = final_art["calib_name"]
    final_calib_obj = final_art["calib_obj"]
    final_tau = float(final_art["thresholds"]["tau_cost"])

    noise_cfg = cfg.get("robustness", {}).get(
        "noise",
        {"Tag_rate": 0.01, "Ikssmin_kA": 0.02, "Sgn_eff_MVA": 0.02, "H_s": 0.01},
    )
    missing_feature = str(cfg.get("robustness", {}).get("missing_feature", "Sgn_eff_MVA"))
    shifted = _shifted_tests(Xf_test, y_test, noise_cfg=noise_cfg, missing_feature=missing_feature)

    clean_pr_auc = float(stage_df.loc[stage_df["stage_id"] == "S5", "PR_AUC"].iloc[0])
    robust_rows: List[Dict[str, Any]] = []
    for shift_name, (Xs, ys) in shifted.items():
        p_shift = np.clip(_apply_calibrator(final_calib_obj, final_calib_name, final_model.predict_proba(Xs)[:, 1]), 1e-6, 1 - 1e-6)
        y_hat_shift = (p_shift >= final_tau).astype(int)
        prob_m = _prob_metrics(ys, p_shift)
        cls_m = _class_metrics(ys, y_hat_shift)
        robust_rows.append(
            {
                "shift": shift_name,
                **prob_m,
                **cls_m,
                "PR_AUC_drop_vs_clean": clean_pr_auc - float(prob_m["PR_AUC"]) if np.isfinite(prob_m["PR_AUC"]) else np.nan,
            }
        )
    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(out_tables / "logic_ladder_robustness.csv", index=False)
    _write_md(robust_df, out_tables / "logic_ladder_robustness.md")

    th_df = compare_threshold_policies(
        y_true=y_test,
        y_prob=np.asarray(final_art["p_test"]),
        thresholds=final_art["thresholds"],
        c_fn=c_fn,
        c_fp=c_fp,
    )
    th_df.to_csv(out_tables / "logic_ladder_threshold_policies.csv", index=False)
    _write_md(th_df, out_tables / "logic_ladder_threshold_policies.md")

    feature_bounds = derive_feature_bounds(df, full_numeric)
    cf_summary, cf_detail = evaluate_counterfactual_stability_correction(
        model=final_model,
        X=Xf_test,
        feature_bounds=feature_bounds,
        stable_threshold=final_tau,
        max_examples=int(cfg.get("counterfactual", {}).get("max_examples", 100)),
    )
    cf_summary.to_csv(out_tables / "logic_ladder_counterfactual_summary.csv", index=False)
    cf_detail.to_csv(out_tables / "logic_ladder_counterfactual_details.csv", index=False)
    _write_md(cf_summary, out_tables / "logic_ladder_counterfactual_summary.md")

    drift_current = shifted["noisy"][0] if "noisy" in shifted else Xf_test
    drift_reports = feature_drift_report(
        train_df=Xf_train,
        current_df=drift_current,
        features=full_numeric,
    )
    drift_reports["psi"].to_csv(out_tables / "logic_ladder_drift_psi.csv", index=False)
    drift_reports["ks"].to_csv(out_tables / "logic_ladder_drift_ks.csv", index=False)

    # S1..S9 compact cumulative scenario comparison:
    # S1-S5 are model-building stages; S6-S8 are operational scenarios;
    # S9 is compact all-in-one summary including previous scenario signals.
    scenario_rows: List[Dict[str, Any]] = []
    for _, r in stage_df.iterrows():
        scenario_rows.append(
            {
                "scenario_id": r["stage_id"],
                "scenario_name": r["stage_name"],
                "cumulative_logic": r["stage_logic"],
                "PR_AUC": r["PR_AUC"],
                "ROC_AUC": r["ROC_AUC"],
                "Recall": r["Recall"],
                "FNR": r["FNR"],
                "ECE": r["ECE"],
                "Brier": r["Brier"],
                "calibration": r["calibration"],
                "threshold_policy": r["threshold_policy"],
                "tau_used": r["tau_used"],
                "robustness_worst_pr_auc_drop": np.nan,
                "robustness_mean_pr_auc_drop": np.nan,
                "counterfactual_success_rate": np.nan,
                "drift_max_psi": np.nan,
                "drift_max_ks": np.nan,
            }
        )

    robust_drops = robust_df["PR_AUC_drop_vs_clean"].dropna() if "PR_AUC_drop_vs_clean" in robust_df.columns else pd.Series(dtype=float)
    robust_worst_drop = float(robust_drops.max()) if len(robust_drops) else np.nan
    robust_mean_drop = float(robust_drops.mean()) if len(robust_drops) else np.nan

    cf_success = float(cf_summary["success_rate"].iloc[0]) if ("success_rate" in cf_summary.columns and len(cf_summary) > 0) else np.nan

    psi_col = "PSI" if "PSI" in drift_reports["psi"].columns else ("psi" if "psi" in drift_reports["psi"].columns else None)
    ks_col = "KS" if "KS" in drift_reports["ks"].columns else ("ks" if "ks" in drift_reports["ks"].columns else None)
    drift_max_psi = float(drift_reports["psi"][psi_col].max()) if psi_col else np.nan
    drift_max_ks = float(drift_reports["ks"][ks_col].max()) if ks_col else np.nan

    scenario_rows.append(
        {
            "scenario_id": "S6",
            "scenario_name": "Robustness Scenario",
            "cumulative_logic": "Evaluate noisy, missing-feature, unseen-regime, and group-shift stress tests on S5.",
            "PR_AUC": np.nan,
            "ROC_AUC": np.nan,
            "Recall": np.nan,
            "FNR": np.nan,
            "ECE": np.nan,
            "Brier": np.nan,
            "calibration": final_calib_name,
            "threshold_policy": "tau_cost",
            "tau_used": final_tau,
            "robustness_worst_pr_auc_drop": robust_worst_drop,
            "robustness_mean_pr_auc_drop": robust_mean_drop,
            "counterfactual_success_rate": np.nan,
            "drift_max_psi": np.nan,
            "drift_max_ks": np.nan,
        }
    )
    scenario_rows.append(
        {
            "scenario_id": "S7",
            "scenario_name": "Counterfactual Scenario",
            "cumulative_logic": "Evaluate minimal feature changes needed to push predicted risk below stable threshold.",
            "PR_AUC": np.nan,
            "ROC_AUC": np.nan,
            "Recall": np.nan,
            "FNR": np.nan,
            "ECE": np.nan,
            "Brier": np.nan,
            "calibration": final_calib_name,
            "threshold_policy": "tau_cost",
            "tau_used": final_tau,
            "robustness_worst_pr_auc_drop": np.nan,
            "robustness_mean_pr_auc_drop": np.nan,
            "counterfactual_success_rate": cf_success,
            "drift_max_psi": np.nan,
            "drift_max_ks": np.nan,
        }
    )
    scenario_rows.append(
        {
            "scenario_id": "S8",
            "scenario_name": "Deployment + Drift Scenario",
            "cumulative_logic": "Evaluate feature drift monitoring (PSI, KS) on shifted current data relative to train reference.",
            "PR_AUC": np.nan,
            "ROC_AUC": np.nan,
            "Recall": np.nan,
            "FNR": np.nan,
            "ECE": np.nan,
            "Brier": np.nan,
            "calibration": final_calib_name,
            "threshold_policy": "tau_cost",
            "tau_used": final_tau,
            "robustness_worst_pr_auc_drop": np.nan,
            "robustness_mean_pr_auc_drop": np.nan,
            "counterfactual_success_rate": np.nan,
            "drift_max_psi": drift_max_psi,
            "drift_max_ks": drift_max_ks,
        }
    )

    s5_row = next((row for row in scenario_rows if row["scenario_id"] == "S5"), None)
    scenario_rows.append(
        {
            "scenario_id": "S9",
            "scenario_name": "Compact Final (All Previous Included)",
            "cumulative_logic": "S1+S2+S3+S4+S5 plus robustness (S6), counterfactual support (S7), and drift monitoring (S8).",
            "PR_AUC": s5_row["PR_AUC"] if s5_row else np.nan,
            "ROC_AUC": s5_row["ROC_AUC"] if s5_row else np.nan,
            "Recall": s5_row["Recall"] if s5_row else np.nan,
            "FNR": s5_row["FNR"] if s5_row else np.nan,
            "ECE": s5_row["ECE"] if s5_row else np.nan,
            "Brier": s5_row["Brier"] if s5_row else np.nan,
            "calibration": s5_row["calibration"] if s5_row else final_calib_name,
            "threshold_policy": s5_row["threshold_policy"] if s5_row else "tau_cost",
            "tau_used": s5_row["tau_used"] if s5_row else final_tau,
            "robustness_worst_pr_auc_drop": robust_worst_drop,
            "robustness_mean_pr_auc_drop": robust_mean_drop,
            "counterfactual_success_rate": cf_success,
            "drift_max_psi": drift_max_psi,
            "drift_max_ks": drift_max_ks,
        }
    )

    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(out_tables / "logic_ladder_scenario_comparison.csv", index=False)
    _write_md(scenario_df, out_tables / "logic_ladder_scenario_comparison.md")

    # Combination grid (non-sequential) for "best approach" selection.
    combo_defs = _build_combo_defs(
        raw_ladder=raw_ladder,
        full_ladder=full_ladder,
        Xr_train=Xr_train,
        Xr_val=Xr_val,
        Xr_test=Xr_test,
        Xf_train=Xf_train,
        Xf_val=Xf_val,
        Xf_test=Xf_test,
    )

    combo_rows: List[Dict[str, Any]] = []
    for combo in combo_defs:
        row, _ = _fit_stage(
            stage_id=combo["combo_id"],
            stage_name=combo["combo_name"],
            stage_logic=combo["combo_logic"],
            estimator=combo["estimator"],
            X_train=combo["X_train"],
            y_train=y_train,
            X_val=combo["X_val"],
            y_val=y_val,
            X_test=combo["X_test"],
            y_test=y_test,
            c_fn=c_fn,
            c_fp=c_fp,
            min_recall=min_recall,
            use_calibration=bool(combo["use_calibration"]),
            use_cost_threshold=bool(combo["use_cost_threshold"]),
        )
        row["combo_logic"] = combo["combo_logic"]
        row["combo_name"] = combo["combo_name"]
        combo_rows.append(row)

    combo_df = pd.DataFrame(combo_rows)
    combo_df["score_predictive"] = combo_df["PR_AUC"]
    combo_df["score_safety"] = combo_df["Recall"] - combo_df["FNR"]
    combo_df["score_calibrated"] = combo_df["PR_AUC"] - 0.10 * combo_df["ECE"]
    combo_df["composite_score"] = combo_df.apply(_combo_score, axis=1)
    combo_df["rank_predictive"] = combo_df["score_predictive"].rank(method="min", ascending=False).astype(int)
    combo_df["rank_safety"] = combo_df["score_safety"].rank(method="min", ascending=False).astype(int)
    combo_df["rank_calibrated"] = combo_df["score_calibrated"].rank(method="min", ascending=False).astype(int)
    combo_df["rank_composite"] = combo_df["composite_score"].rank(method="min", ascending=False).astype(int)
    combo_df = combo_df.sort_values(["rank_composite", "rank_predictive", "stage_id"], ascending=[True, True, True]).reset_index(drop=True)
    combo_df["rank"] = np.arange(1, len(combo_df) + 1)
    cols = [
        "rank",
        "stage_id",
        "combo_logic",
        "combo_name",
        "PR_AUC",
        "ROC_AUC",
        "Precision",
        "Recall",
        "F1",
        "FNR",
        "ECE",
        "Brier",
        "threshold_policy",
        "calibration",
        "tau_used",
        "score_predictive",
        "score_safety",
        "score_calibrated",
        "composite_score",
        "rank_predictive",
        "rank_safety",
        "rank_calibrated",
        "rank_composite",
    ]
    combo_df = combo_df[cols]
    combo_df.to_csv(out_tables / "logic_ladder_combination_comparison.csv", index=False)
    _write_md(combo_df, out_tables / "logic_ladder_combination_comparison.md")
    best_combo = {}
    if len(combo_df):
        best_combo = {
            "best_composite": combo_df.sort_values("rank_composite").iloc[0].to_dict(),
            "best_predictive": combo_df.sort_values("rank_predictive").iloc[0].to_dict(),
            "best_safety": combo_df.sort_values("rank_safety").iloc[0].to_dict(),
            "best_calibrated": combo_df.sort_values("rank_calibrated").iloc[0].to_dict(),
        }
    save_json(out_tables / "logic_ladder_best_combination.json", best_combo)

    summary = {
        "data_path": cfg.get("data", {}).get("path"),
        "stages_table": str(out_tables / "logic_ladder_comparison.csv"),
        "scenario_comparison_table": str(out_tables / "logic_ladder_scenario_comparison.csv"),
        "combination_table": str(out_tables / "logic_ladder_combination_comparison.csv"),
        "best_combination": str(out_tables / "logic_ladder_best_combination.json"),
        "robustness_table": str(out_tables / "logic_ladder_robustness.csv"),
        "threshold_table": str(out_tables / "logic_ladder_threshold_policies.csv"),
        "counterfactual_summary": str(out_tables / "logic_ladder_counterfactual_summary.csv"),
        "drift_psi": str(out_tables / "logic_ladder_drift_psi.csv"),
        "drift_ks": str(out_tables / "logic_ladder_drift_ks.csv"),
        "split": str(out_splits / "logic_ladder_split.csv"),
    }
    save_json(out_tables / "logic_ladder_manifest.json", summary)

    print("Logic ladder completed.")
    for k, v in summary.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
