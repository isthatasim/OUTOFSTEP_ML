from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.outofstep_ml.benchmark.tables import generate_tables
from src.outofstep_ml.utils.io import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_root = cfg.get("outputs", {}).get("root", "results")
    outs = generate_tables(out_root)
    for k, v in outs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
