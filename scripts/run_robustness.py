from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.outofstep_ml.data.loaders import load_validated_dataset
from src.outofstep_ml.evaluation.robustness import run_missing_feature_stress, run_noise_robustness
from src.outofstep_ml.models.static_physics_model import StaticPhysicsRiskModel
from src.outofstep_ml.utils.io import ensure_dir, load_yaml


def _merge_cfg(path: Path) -> dict:
    cfg = load_yaml(path)
    ext = cfg.get("extends")
    if not ext:
        return cfg
    base = load_yaml(path.parent / "base.yaml") if ext == "base" else load_yaml(path.parent / ext)
    out = dict(base)
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-bundle", default="results/model/static_model_bundle.joblib")
    args = ap.parse_args()

    cfg = _merge_cfg(Path(args.config))
    out_tables = ensure_dir(cfg.get("outputs", {}).get("table_dir", "results/tables"))

    df, _ = load_validated_dataset(cfg.get("data", {}).get("path"), include_logs=bool(cfg.get("features", {}).get("include_logs", True)))
    model = StaticPhysicsRiskModel.load(args.model_bundle)
    X = df[model.numeric_features + model.categorical_features].copy()
    y = df["Out_of_step"].astype(int).values

    noise_cfg = cfg.get("robustness", {}).get("noise", {"Ikssmin_kA": 0.02, "Sgn_eff_MVA": 0.02, "H_s": 0.01, "Tag_rate": 0.01})
    noise_metrics = run_noise_robustness(model.estimator_, X, y, noise_cfg=noise_cfg, random_state=int(cfg.get("seed", 42)))

    missing_col = cfg.get("robustness", {}).get("missing_feature", "Sgn_eff_MVA")
    missing_metrics = run_missing_feature_stress(model.estimator_, X, y, drop_column=missing_col, random_state=int(cfg.get("seed", 42)))

    pd.DataFrame([
        {"stress_test": "noise", **noise_metrics},
        {"stress_test": f"missing_{missing_col}", **missing_metrics},
    ]).to_csv(Path(out_tables) / "robustness_suite.csv", index=False)


if __name__ == "__main__":
    main()
