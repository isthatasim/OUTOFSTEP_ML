from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.outofstep_ml.data.loaders import load_validated_dataset
from src.outofstep_ml.data.splitters import build_groups
from src.outofstep_ml.evaluation.validation import run_validation
from src.outofstep_ml.models.baselines import build_baseline_ladder
from src.features import resolve_engineered_feature_columns
from src.outofstep_ml.utils.io import ensure_dir, load_yaml


def _merge_cfg(path: Path) -> dict:
    cfg = load_yaml(path)
    base = load_yaml(path.parent / "base.yaml")
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
    args = ap.parse_args()

    cfg = _merge_cfg(Path(args.config))
    out_tables = ensure_dir(cfg.get("outputs", {}).get("table_dir", "results/tables"))

    df, _ = load_validated_dataset(cfg.get("data", {}).get("path"), include_logs=True)
    y = df["Out_of_step"].astype(int).values

    raw = ["Tag_rate", "Ikssmin_kA", "Sgn_eff_MVA", "H_s"]
    eng = resolve_engineered_feature_columns(df)
    categorical = ["GenName"] if "GenName" in df.columns else []
    groups = build_groups(df, raw, round_decimals=2)

    ablations = [
        ("raw_only", raw, ["A1_logistic", "B1_rf"]),
        ("raw_plus_engineered", raw + eng, ["A1_logistic", "B2_gbm", "C3_hybrid"]),
        ("without_monotonic_priors", raw + eng, ["B2_gbm"]),
        ("with_monotonic_priors", raw + eng, ["C1_monotonic"]),
        ("physics_regularized", raw + eng, ["C2_physics_logit"]),
    ]

    rows = []
    protocol = [{"name": "V1_stratified", "split_mode": "stratified", "n_splits": 5, "scenario": "ablation"}]
    for name, feats, model_keys in ablations:
        feats = [c for c in feats if c in df.columns]
        X = df[feats + categorical].copy()
        ladder = build_baseline_ladder(feats, categorical, random_state=int(cfg.get("seed", 42)))
        subset = {k: ladder[k] for k in model_keys}
        res = run_validation(subset, X, y, protocol, groups, df, int(cfg.get("seed", 42)))
        if len(res):
            res["ablation"] = name
            rows.append(res)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(Path(out_tables) / "ablation_suite.csv", index=False)


if __name__ == "__main__":
    main()
