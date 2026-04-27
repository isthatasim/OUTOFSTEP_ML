from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.outofstep_ml.product import GridSyncCompatibilityService


def _load_records(args: argparse.Namespace) -> list[dict]:
    if args.json:
        return [json.loads(args.json)]
    if args.input:
        path = Path(args.input)
        if path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(obj, list):
                return [dict(x) for x in obj]
            return [dict(obj)]
        if path.suffix.lower() in {".csv", ".txt"}:
            return pd.read_csv(path).to_dict(orient="records")
    raise ValueError("Provide either --json '{...}' or --input device.csv/device.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict whether an unknown device operating point is compatible for grid synchronization.")
    ap.add_argument("--model-dir", default="outputs/product")
    ap.add_argument("--json", help="Single JSON payload with Tag_rate, Ikssmin_kA, Sgn_eff_MVA, H_s.")
    ap.add_argument("--input", help="CSV or JSON file containing one or more unknown devices.")
    ap.add_argument("--output", help="Optional CSV/JSON output path. Defaults to printing JSON.")
    args = ap.parse_args()

    service = GridSyncCompatibilityService.load(args.model_dir)
    results = service.predict_many(_load_records(args))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".json":
            out.write_text(results.to_json(orient="records", indent=2), encoding="utf-8")
        else:
            results.to_csv(out, index=False)
        print(out)
    else:
        print(results.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
