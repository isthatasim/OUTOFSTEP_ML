from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.outofstep_ml.benchmark.runner import run_full_benchmark
from src.outofstep_ml.benchmark.tables import generate_tables
from src.outofstep_ml.utils.io import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outputs = run_full_benchmark(cfg)
    out_root = cfg.get("outputs", {}).get("root", "results")
    generate_tables(out_root)
    print(f"Benchmark complete. Tables generated in {Path(out_root) / 'tables'}")
    if "strict_protocol" in outputs:
        print(f"Strict evaluation protocol manifest: {outputs['strict_protocol']}")
    if "split_manifest" in outputs:
        print(f"Split manifest: {outputs['split_manifest']}")


if __name__ == "__main__":
    main()
