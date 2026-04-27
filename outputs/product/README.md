# Grid Sync Compatibility Product Artifact

This folder contains the product-facing static OOS risk screener.

## Purpose

Given an unknown device / operating-point record with:

- `Tag_rate`
- `Ikssmin_kA`
- `Sgn_eff_MVA`
- `H_s`
- optional `GenName`
- optional `DeviceId`

the product returns whether the device operating point is compatible for grid synchronization.

## Verdict Logic

- `COMPATIBLE_FOR_GRID_SYNC`: calibrated OOS risk is below the threshold and the point is inside the learned training domain.
- `NOT_COMPATIBLE_HIGH_OOS_RISK`: calibrated OOS risk is at or above the safety threshold.
- `ENGINEERING_REVIEW_REQUIRED_OUT_OF_DOMAIN`: input is outside the training operating envelope, so the product does not auto-approve synchronization.

The current safety threshold is selected from validation data using a cost-sensitive policy.

## CLI

```bash
python scripts/build_grid_sync_product.py --config configs/logic_ladder.yaml --model-policy deployment_safety --output-dir outputs/product
python scripts/predict_grid_sync.py --input outputs/product/example_in_domain_device.json
```

Single JSON payload:

```bash
python scripts/predict_grid_sync.py --json "{\"DeviceId\":\"D1\",\"Tag_rate\":1000,\"Ikssmin_kA\":9,\"Sgn_eff_MVA\":5.1,\"H_s\":2.6,\"GenName\":\"GR1\"}"
```

## API

```bash
uvicorn src.outofstep_ml.product.api:app --host 0.0.0.0 --port 8000
```

Endpoint:

```text
POST /compatibility
```

Example body:

```json
{
  "DeviceId": "D1",
  "Tag_rate": 1000,
  "Ikssmin_kA": 9,
  "Sgn_eff_MVA": 5.1,
  "H_s": 2.6,
  "GenName": "GR1"
}
```

## Main Files

- `grid_sync_bundle.joblib`: model + calibrator + threshold + feature metadata.
- `grid_sync_model_card.json`: readable product metadata and holdout context.
- `grid_sync_holdout_metrics.csv`: final holdout metrics for the exported product model.
