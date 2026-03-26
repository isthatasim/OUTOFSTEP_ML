from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.outofstep_ml.benchmark.model_zoo import build_benchmark_model_specs
from src.outofstep_ml.benchmark.runner import (
    _apply_calibrator,
    _class_metrics,
    _fit_with_budget,
    _prob_metrics,
    _select_best_calibration,
    _shifted_tests,
    make_strict_split,
)
from src.outofstep_ml.data.loaders import load_validated_dataset
from src.outofstep_ml.evaluation.counterfactual_eval import evaluate_counterfactual_stability_correction
from src.outofstep_ml.evaluation.imbalance_ablation import (
    build_imbalance_ablation_models,
    evaluate_imbalance_ablation,
)
from src.outofstep_ml.evaluation.monotonic_checks import (
    aggregate_monotonic_checks,
    monotonic_consistency_check,
)
from src.outofstep_ml.evaluation.scenario_heatmaps import generate_interaction_heatmaps
from src.outofstep_ml.evaluation.scenario_migrations import (
    default_migration_scenarios,
    evaluate_migration_curves,
)
from src.outofstep_ml.evaluation.threshold_policy_compare import (
    aggregate_threshold_policy_tables,
    compare_threshold_policies,
)
from src.outofstep_ml.features.physics_features import monotonic_prior_directions
from src.outofstep_ml.models.thresholds import optimize_thresholds
from src.outofstep_ml.utils.io import ensure_dir, save_json
from src.outofstep_ml.utils.seed import set_global_seed


def _save_split_artifacts(
    out_splits: Path,
    split,
    seed: int,
    strategy: str,
) -> Dict[str, str]:
    csv_path = out_splits / f"scenario_split_seed{seed}_{strategy}.csv"
    npz_path = out_splits / f"scenario_split_seed{seed}_{strategy}.npz"
    split_frame = pd.DataFrame(
        {
            "index": np.concatenate([split.train_idx, split.val_idx, split.test_idx]),
            "subset": ["train"] * len(split.train_idx) + ["val"] * len(split.val_idx) + ["test"] * len(split.test_idx),
            "seed": seed,
            "strategy": strategy,
        }
    )
    split_frame.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, train_idx=split.train_idx, val_idx=split.val_idx, test_idx=split.test_idx)
    return {"split_csv": str(csv_path), "split_npz": str(npz_path)}


def _resolve_main_spec(specs, model_id: str):
    candidates = [s for s in specs if s.model_id == model_id]
    if not candidates:
        available = [s.model_id for s in specs]
        raise ValueError(f"Unknown main model_id={model_id}. Available={available}")
    spec = candidates[0]
    if not spec.available or spec.estimator is None:
        raise RuntimeError(f"Requested model {model_id} is unavailable: {spec.skip_reason}")
    return spec


def run_static_q1_scenarios(cfg: Dict[str, Any]) -> Dict[str, str]:
    seed0 = int(cfg.get("seed", 42))
    set_global_seed(seed0)

    out_root = Path(ensure_dir(cfg.get("outputs", {}).get("root", "outputs/static_q1_validation")))
    out_tables = Path(ensure_dir(cfg.get("outputs", {}).get("table_dir", out_root / "tables")))
    out_figures = Path(ensure_dir(cfg.get("outputs", {}).get("figure_dir", out_root / "figures")))
    out_splits = Path(ensure_dir(cfg.get("outputs", {}).get("split_dir", out_root / "splits")))

    df, audit = load_validated_dataset(cfg.get("data", {}).get("path"), include_logs=bool(cfg.get("features", {}).get("include_logs", True)))
    save_json(out_tables / "scenario_data_audit.json", audit.to_dict())
    y = df["Out_of_step"].astype(int).values

    raw = ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"]
    eng = [c for c in ["invH", "Sgn_over_H", "Sgn_over_Ik", "Ik_over_H", "log_Sgn_eff_MVA", "log_Ikssmin_kA"] if c in df.columns]
    use_engineered = bool(cfg.get("features", {}).get("use_engineered", True))
    numeric = [c for c in (raw + eng if use_engineered else raw) if c in df.columns]
    categorical = ["GenName"] if "GenName" in df.columns else []
    X = df[numeric + categorical].copy()

    specs = build_benchmark_model_specs(numeric_features=numeric, categorical_features=categorical, random_state=seed0)
    main_model_id = str(cfg.get("scenario", {}).get("main_model_id", "proposed_physics_model"))
    main_spec = _resolve_main_spec(specs, main_model_id)

    n_seeds = int(cfg.get("scenario", {}).get("n_seeds", 3))
    seeds = [seed0 + i for i in range(n_seeds)]
    base_strategy = str(cfg.get("scenario", {}).get("base_strategy", "stratified"))
    regime_strategies = cfg.get("scenario", {}).get("regime_strategies", ["grouped", "leave_S", "leave_I"])

    strict_cfg = cfg.get("strict_eval", {})
    test_size = float(strict_cfg.get("test_size", 0.2))
    val_size = float(strict_cfg.get("val_size", 0.2))
    if not bool(strict_cfg.get("enforce_no_test_tuning", True)):
        raise ValueError("strict_eval.enforce_no_test_tuning must be true.")

    c_fn = float(cfg.get("thresholds", {}).get("c_fn", 10.0))
    c_fp = float(cfg.get("thresholds", {}).get("c_fp", 1.0))
    min_recall = float(cfg.get("thresholds", {}).get("min_recall", 0.95))
    n_trials = int(cfg.get("training_budget", {}).get("n_trials", 15))
    noise_cfg = cfg.get("robustness", {}).get("noise", {"Tag_rate": 0.01, "Ikssmin_kA": 0.02, "Sgn_eff_MVA": 0.02, "H_s": 0.01})
    missing_feature = cfg.get("robustness", {}).get("missing_feature", "Sgn_eff_MVA")

    split_rows: List[Dict[str, Any]] = []
    nominal_rows: List[Dict[str, Any]] = []
    migration_rows: List[pd.DataFrame] = []
    regime_rows: List[Dict[str, Any]] = []
    noise_rows: List[Dict[str, Any]] = []
    imbalance_rows: List[pd.DataFrame] = []
    threshold_rows: List[pd.DataFrame] = []
    monotonic_rows: List[pd.DataFrame] = []
    cf_summary_rows: List[pd.DataFrame] = []
    cf_detail_rows: List[pd.DataFrame] = []

    heatmap_done = False
    heatmap_index = pd.DataFrame()

    feature_bounds = {
        c: (float(df[c].min()), float(df[c].max()))
        for c in numeric
        if pd.api.types.is_numeric_dtype(df[c])
    }

    for seed in seeds:
        split = make_strict_split(
            df=df,
            y=y,
            seed=seed,
            strategy=base_strategy,
            test_size=test_size,
            val_size=val_size,
            group_cols=["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"],
            leave_feature="Sgn_eff_MVA",
        )
        split_info = _save_split_artifacts(out_splits, split, seed, base_strategy)
        split_rows.append({"seed": seed, "strategy": base_strategy, "n_train": len(split.train_idx), "n_val": len(split.val_idx), "n_test": len(split.test_idx), **split_info})

        X_train, y_train = X.iloc[split.train_idx], y[split.train_idx]
        X_val, y_val = X.iloc[split.val_idx], y[split.val_idx]
        X_test, y_test = X.iloc[split.test_idx], y[split.test_idx]

        model, best_params, val_score = _fit_with_budget(
            main_spec,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=n_trials,
            seed=seed,
        )
        p_val_raw = np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
        calib_name, calib_obj, _ = _select_best_calibration(y_val, p_val_raw)
        p_val = np.clip(_apply_calibrator(calib_obj, calib_name, p_val_raw), 1e-6, 1 - 1e-6)
        th = optimize_thresholds(y_val, p_val, c_fn=c_fn, c_fp=c_fp, min_recall=min_recall)

        p_test = np.clip(_apply_calibrator(calib_obj, calib_name, model.predict_proba(X_test)[:, 1]), 1e-6, 1 - 1e-6)
        pm = _prob_metrics(y_test, p_test)
        cm = _class_metrics(y_test, (p_test >= float(th["tau_cost"])).astype(int))
        nominal_rows.append(
            {
                "seed": seed,
                "strategy": base_strategy,
                "model_id": main_spec.model_id,
                "model_name": main_spec.model_name,
                "Validation_PR_AUC": float(val_score),
                "calibration": calib_name,
                "best_params": str(best_params),
                "tau_cost": float(th["tau_cost"]),
                "tau_f1": float(th["tau_f1"]),
                "tau_hr": float(th["tau_hr"]),
                **pm,
                **cm,
            }
        )

        # Scenario 9
        tp = compare_threshold_policies(y_test, p_test, thresholds=th, c_fn=c_fn, c_fp=c_fp)
        tp["seed"] = seed
        threshold_rows.append(tp)

        # Scenario 2, 3, 4
        mig = evaluate_migration_curves(
            model=model,
            calibrator_method=calib_name,
            calibrator_obj=calib_obj,
            X_test=X_test,
            y_test=y_test,
            scenarios=default_migration_scenarios(),
            apply_calibrator_fn=_apply_calibrator,
            prob_metrics_fn=_prob_metrics,
            class_metrics_fn=_class_metrics,
            thresholds=th,
            feature_bounds=feature_bounds,
        )
        mig["seed"] = seed
        migration_rows.append(mig)

        # Scenario 7
        shifted = _shifted_tests(X_test, y_test, noise_cfg=noise_cfg, missing_feature=missing_feature)
        for shift_name, (Xs, ys) in shifted.items():
            p_shift = np.clip(_apply_calibrator(calib_obj, calib_name, model.predict_proba(Xs)[:, 1]), 1e-6, 1 - 1e-6)
            y_hat_shift = (p_shift >= float(th["tau_cost"])).astype(int)
            noise_rows.append(
                {
                    "seed": seed,
                    "shift": shift_name,
                    **_prob_metrics(ys, p_shift),
                    **_class_metrics(ys, y_hat_shift),
                }
            )

        # Scenario 8
        imbalance_models = build_imbalance_ablation_models(numeric_features=numeric, categorical_features=categorical, random_state=seed)
        imb = evaluate_imbalance_ablation(
            model_variants=imbalance_models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            fit_budget_fn=_fit_with_budget,
            select_calibration_fn=_select_best_calibration,
            apply_calibration_fn=_apply_calibrator,
            optimize_thresholds_fn=optimize_thresholds,
            prob_metrics_fn=_prob_metrics,
            class_metrics_fn=_class_metrics,
            c_fn=c_fn,
            c_fp=c_fp,
            min_recall=min_recall,
        )
        imb["seed"] = seed
        imbalance_rows.append(imb)
        imbalance_rows.append(
            pd.DataFrame(
                [
                    {
                        "imbalance_mode": "existing_approach_proposed",
                        "calibration": calib_name,
                        "tau_cost": float(th["tau_cost"]),
                        "tau_f1": float(th["tau_f1"]),
                        "tau_hr": float(th["tau_hr"]),
                        "seed": seed,
                        **pm,
                        **cm,
                    }
                ]
            )
        )

        # Scenario monotonic consistency
        mono = monotonic_consistency_check(
            model=model,
            X=X_test,
            monotonic_dirs=monotonic_prior_directions(),
            step_frac=float(cfg.get("scenario", {}).get("monotonic_step_frac", 0.02)),
            n_samples=int(cfg.get("scenario", {}).get("monotonic_n_samples", 400)),
            random_state=seed,
        )
        mono["seed"] = seed
        monotonic_rows.append(mono)

        # Scenario 10
        cf_summary, cf_detail = evaluate_counterfactual_stability_correction(
            model=model,
            X=X_test,
            feature_bounds=feature_bounds,
            stable_threshold=float(th["tau_cost"]),
            max_examples=int(cfg.get("scenario", {}).get("counterfactual_max_examples", 40)),
        )
        cf_summary["seed"] = seed
        cf_summary["tau_stable"] = float(th["tau_cost"])
        cf_summary_rows.append(cf_summary)
        cf_detail["seed"] = seed
        cf_detail_rows.append(cf_detail)

        # Scenario 5
        if not heatmap_done:
            heatmap_index = generate_interaction_heatmaps(
                model=model,
                X_reference=X_train,
                output_dir=out_figures / "scenario5_heatmaps",
                threshold=float(th["tau_cost"]),
                n_grid=int(cfg.get("scenario", {}).get("heatmap_grid", 100)),
            )
            heatmap_done = True

    # Scenario 6 regime shift
    for seed in seeds:
        for strategy in regime_strategies:
            split = make_strict_split(
                df=df,
                y=y,
                seed=seed,
                strategy=strategy,
                test_size=test_size,
                val_size=val_size,
                group_cols=["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"],
                leave_feature="Sgn_eff_MVA" if strategy == "leave_S" else "Ikssmin_kA",
            )
            split_info = _save_split_artifacts(out_splits, split, seed, strategy)
            split_rows.append({"seed": seed, "strategy": strategy, "n_train": len(split.train_idx), "n_val": len(split.val_idx), "n_test": len(split.test_idx), **split_info})

            X_train, y_train = X.iloc[split.train_idx], y[split.train_idx]
            X_val, y_val = X.iloc[split.val_idx], y[split.val_idx]
            X_test, y_test = X.iloc[split.test_idx], y[split.test_idx]

            model, best_params, val_score = _fit_with_budget(
                main_spec,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_trials=n_trials,
                seed=seed,
            )
            p_val_raw = np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
            calib_name, calib_obj, _ = _select_best_calibration(y_val, p_val_raw)
            p_val = np.clip(_apply_calibrator(calib_obj, calib_name, p_val_raw), 1e-6, 1 - 1e-6)
            th = optimize_thresholds(y_val, p_val, c_fn=c_fn, c_fp=c_fp, min_recall=min_recall)
            p_test = np.clip(_apply_calibrator(calib_obj, calib_name, model.predict_proba(X_test)[:, 1]), 1e-6, 1 - 1e-6)
            regime_rows.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "model_id": main_spec.model_id,
                    "model_name": main_spec.model_name,
                    "Validation_PR_AUC": float(val_score),
                    "calibration": calib_name,
                    "best_params": str(best_params),
                    "tau_cost": float(th["tau_cost"]),
                    **_prob_metrics(y_test, p_test),
                    **_class_metrics(y_test, (p_test >= float(th["tau_cost"])).astype(int)),
                }
            )

    # Persist outputs
    nominal_df = pd.DataFrame(nominal_rows)
    migration_df = pd.concat(migration_rows, ignore_index=True) if migration_rows else pd.DataFrame()
    regime_df = pd.DataFrame(regime_rows)
    robustness_df = pd.DataFrame(noise_rows)
    imbalance_df = pd.concat(imbalance_rows, ignore_index=True) if imbalance_rows else pd.DataFrame()
    threshold_df = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    threshold_agg_df = aggregate_threshold_policy_tables(threshold_rows)
    monotonic_df = aggregate_monotonic_checks(monotonic_rows)
    cf_summary_df = pd.concat(cf_summary_rows, ignore_index=True) if cf_summary_rows else pd.DataFrame()
    cf_detail_df = pd.concat(cf_detail_rows, ignore_index=True) if cf_detail_rows else pd.DataFrame()
    splits_df = pd.DataFrame(split_rows)

    nominal_path = out_tables / "scenario1_nominal_baseline.csv"
    migration_path = out_tables / "scenario2_3_4_migrations.csv"
    heatmap_index_path = out_tables / "scenario5_heatmap_index.csv"
    regime_path = out_tables / "scenario6_regime_shift.csv"
    robustness_path = out_tables / "scenario7_noise_missing.csv"
    imbalance_path = out_tables / "scenario8_imbalance_ablation.csv"
    threshold_path = out_tables / "scenario9_threshold_policies.csv"
    threshold_agg_path = out_tables / "scenario9_threshold_policies_aggregated.csv"
    monotonic_path = out_tables / "scenario_monotonic_consistency.csv"
    cf_summary_path = out_tables / "scenario10_counterfactual_summary.csv"
    cf_detail_path = out_tables / "scenario10_counterfactual_details.csv"
    split_manifest_path = out_splits / "scenario_split_manifest.csv"

    nominal_df.to_csv(nominal_path, index=False)
    migration_df.to_csv(migration_path, index=False)
    heatmap_index.to_csv(heatmap_index_path, index=False)
    regime_df.to_csv(regime_path, index=False)
    robustness_df.to_csv(robustness_path, index=False)
    imbalance_df.to_csv(imbalance_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    threshold_agg_df.to_csv(threshold_agg_path, index=False)
    monotonic_df.to_csv(monotonic_path, index=False)
    cf_summary_df.to_csv(cf_summary_path, index=False)
    cf_detail_df.to_csv(cf_detail_path, index=False)
    splits_df.to_csv(split_manifest_path, index=False)

    strict_protocol = {
        "strict_evaluation_protocol": {
            "1_train_val_test_split": True,
            "2_fit_on_train_only": True,
            "3_tune_calibrate_threshold_on_validation_only": True,
            "4_final_test_used_once_for_unbiased_comparison": True,
            "5_shifted_tests_not_used_for_tuning": True,
            "6_split_indices_and_seeds_saved": True,
            "notes": "Scenario runner uses strict split and does not tune on final holdout.",
        },
        "seed0": seed0,
        "seeds": seeds,
        "base_strategy": base_strategy,
        "regime_strategies": regime_strategies,
    }
    strict_path = out_splits / "strict_evaluation_protocol_static_q1.json"
    save_json(strict_path, strict_protocol)

    # High-level scenario summary table
    summary_rows = []
    if len(nominal_df):
        summary_rows.append({"scenario": "1_nominal_baseline", "PR_AUC_mean": float(nominal_df["PR_AUC"].mean()), "FNR_mean": float(nominal_df["FNR"].mean()), "ECE_mean": float(nominal_df["ECE"].mean())})
    if len(regime_df):
        summary_rows.append({"scenario": "6_regime_shift", "PR_AUC_mean": float(regime_df["PR_AUC"].mean()), "FNR_mean": float(regime_df["FNR"].mean()), "ECE_mean": float(regime_df["ECE"].mean())})
    if len(robustness_df):
        d = robustness_df.groupby("shift", as_index=False).mean(numeric_only=True)
        for _, r in d.iterrows():
            summary_rows.append({"scenario": f"7_{r['shift']}", "PR_AUC_mean": float(r["PR_AUC"]), "FNR_mean": float(r["FNR"]), "ECE_mean": float(r["ECE"])})
    if len(monotonic_df):
        summary_rows.append({"scenario": "monotonic_consistency", "PR_AUC_mean": np.nan, "FNR_mean": np.nan, "ECE_mean": float(monotonic_df["violation_rate"].mean())})
    scenario_summary_path = out_tables / "scenario_overview_summary.csv"
    pd.DataFrame(summary_rows).to_csv(scenario_summary_path, index=False)

    return {
        "nominal": str(nominal_path),
        "migrations": str(migration_path),
        "heatmap_index": str(heatmap_index_path),
        "regime_shift": str(regime_path),
        "noise_missing": str(robustness_path),
        "imbalance_ablation": str(imbalance_path),
        "threshold_policies": str(threshold_path),
        "threshold_policies_aggregated": str(threshold_agg_path),
        "monotonic_checks": str(monotonic_path),
        "counterfactual_summary": str(cf_summary_path),
        "counterfactual_details": str(cf_detail_path),
        "scenario_split_manifest": str(split_manifest_path),
        "strict_protocol": str(strict_path),
        "scenario_summary": str(scenario_summary_path),
    }

