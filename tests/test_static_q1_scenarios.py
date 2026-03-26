from __future__ import annotations

from pathlib import Path

from src.outofstep_ml.evaluation.scenario_static_validation import run_static_q1_scenarios
from src.outofstep_ml.utils.io import load_yaml


def test_static_q1_scenarios_smoke(tmp_path: Path):
    cfg = load_yaml("configs/static_q1_validation_smoke.yaml")
    cfg["outputs"]["root"] = str(tmp_path / "out")
    cfg["outputs"]["model_dir"] = str(tmp_path / "out" / "model")
    cfg["outputs"]["table_dir"] = str(tmp_path / "out" / "tables")
    cfg["outputs"]["figure_dir"] = str(tmp_path / "out" / "figures")
    cfg["outputs"]["split_dir"] = str(tmp_path / "out" / "splits")

    outs = run_static_q1_scenarios(cfg)
    required = [
        "nominal",
        "migrations",
        "regime_shift",
        "noise_missing",
        "imbalance_ablation",
        "threshold_policies",
        "monotonic_checks",
        "counterfactual_summary",
        "strict_protocol",
    ]
    for k in required:
        assert k in outs
        assert Path(outs[k]).exists()

