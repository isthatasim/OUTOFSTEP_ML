# Grid Sync Compatibility Product Guide

## Product Objective

The product converts the OOS research model into a practical compatibility screener.
Given an unknown device or operating point, it answers:

> Is this device operating point compatible for grid synchronization, or is it likely to become out-of-step?

The product uses the static operating-point model. It does **not** invent dynamic PMU data and it does **not** replace detailed transient simulation. It is a fast screening layer for planning, studies, and operational review.

## Required Input

Each device record must include:

| Field | Meaning |
|---|---|
| `Tag_rate` | disturbance / acceleration proxy |
| `Ikssmin_kA` | short-circuit strength / grid-strength proxy |
| `Sgn_eff_MVA` | device stress/loading proxy |
| `H_s` | inertia constant |
| `GenName` | optional generator or study name, default `GR1` |
| `DeviceId` | optional device identifier for reporting |

Aliases are also accepted for quick use:

| Alias | Canonical field |
|---|---|
| `T` | `Tag_rate` |
| `I` | `Ikssmin_kA` |
| `S` | `Sgn_eff_MVA` |
| `H` | `H_s` |

## Internal Features

The product computes:

| Feature | Formula | Purpose |
|---|---|---|
| `invH` | `1/H` | exposes low-inertia risk |
| `S_over_H` | `S/H` | stress relative to inertia support |
| `S_over_I` | `S/I` | stress relative to grid strength |
| `I_over_H` | `I/H` | grid strength relative to inertia |

## Product Verdicts

| Verdict | Meaning | Action |
|---|---|---|
| `COMPATIBLE_FOR_GRID_SYNC` | OOS risk is below the safety threshold and the point is in-domain | Compatible under the static screening model |
| `NOT_COMPATIBLE_HIGH_OOS_RISK` | OOS risk is above the safety threshold | Do not synchronize automatically |
| `ENGINEERING_REVIEW_REQUIRED_OUT_OF_DOMAIN` | input is outside the training operating envelope | Manual engineering review required |

The model only auto-approves synchronization when both conditions are true:

1. predicted OOS probability is below the safety threshold;
2. the operating point is inside the training domain.

## Build Product Artifact

```bash
python scripts/build_grid_sync_product.py --config configs/logic_ladder.yaml --model-policy deployment_safety --output-dir outputs/product
```

This creates:

- `outputs/product/grid_sync_bundle.joblib`
- `outputs/product/grid_sync_model_card.json`
- `outputs/product/grid_sync_holdout_metrics.csv`

## Run Product From CLI

Single device JSON:

```bash
python scripts/predict_grid_sync.py --input outputs/product/example_in_domain_device.json
```

Batch CSV:

```bash
python scripts/predict_grid_sync.py --input unknown_devices.csv --output outputs/product/unknown_device_results.csv
```

## Run Product API

Install the API dependencies first:

```bash
pip install -r requirements.txt
```

Start the service:

```bash
uvicorn src.outofstep_ml.product.api:app --host 0.0.0.0 --port 8000
```

Request:

```http
POST /compatibility
```

Example JSON body:

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

## Current Exported Product Model

Current exported model policy:

```text
deployment_safety
```

This corresponds to the conservative product stack:

```text
1 + 2 + 4 + 5
raw inputs + engineered ratios + imbalance/cost threshold + calibration
```

Current validation-selected threshold:

```text
p_oos >= 0.0720 => not compatible / high OOS risk
```

Final holdout summary from the exported product artifact:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9992 |
| PR-AUC | 0.9877 |
| Recall | 0.9983 |
| FNR | 0.0017 |
| ECE | 0.0033 |
| Brier | 0.0043 |

## Product Interpretation

The best pure predictive combination in the current case study is `1+2`, which uses raw variables and engineered physics ratios.
The product uses the safer `1+2+4+5` stack because deployment should prioritize:

- low missed-instability risk;
- calibrated risk values;
- explicit threshold policy;
- out-of-domain rejection;
- operator-readable recommendations.

Therefore, the product is intentionally conservative: it may request review even when the predicted risk is low if the operating point is outside the training envelope.
