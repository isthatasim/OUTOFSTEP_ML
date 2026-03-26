from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.outofstep_ml.benchmark.runner import run_full_benchmark
from src.outofstep_ml.benchmark.tables import generate_tables
from src.outofstep_ml.evaluation.scenario_static_validation import run_static_q1_scenarios
from src.outofstep_ml.utils.io import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to static_q1_validation config.")
    ap.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="If set, skip full benchmark training and run only static scenario validation.",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    if not args.skip_benchmark:
        benchmark_outputs = run_full_benchmark(cfg)
        out_root = cfg.get("outputs", {}).get("root", "outputs/static_q1_validation")
        generate_tables(out_root)
        print(f"Benchmark complete under: {Path(out_root)}")
        print(f"Strict benchmark protocol: {benchmark_outputs.get('strict_protocol')}")

    scenario_outputs = run_static_q1_scenarios(cfg)
    print("Static Q1 scenario validation complete.")
    for k, v in scenario_outputs.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()

