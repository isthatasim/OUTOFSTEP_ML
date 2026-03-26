from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.outofstep_ml.utils.io import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default="results/model/static_model_bundle.joblib")
    ap.add_argument("--source-config", default="results/model/inference_config.yaml")
    ap.add_argument("--target-dir", default="outputs/model")
    args = ap.parse_args()

    target = ensure_dir(args.target_dir)
    shutil.copy2(args.source_model, Path(target) / "static_model_bundle.joblib")
    shutil.copy2(args.source_config, Path(target) / "inference_config.yaml")
    print(f"Exported model artifacts to {target}")


if __name__ == "__main__":
    main()
