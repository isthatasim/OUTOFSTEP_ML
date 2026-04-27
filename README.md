# OUTOFSTEP_ML

Physics-aware, imbalance-aware, deployable machine-learning framework for out-of-step (OOS) risk prediction in power systems.

This repository now supports two layers:

1. **Legacy compatible pipeline** via `main.py` (kept working for backward compatibility).
2. **Config-driven research benchmark pipeline** via `scripts/*.py` and `src/outofstep_ml/*`.

## Research Purpose

Target use case: static operating-point OOS risk screening for generator stability studies (e.g., GR1).

Core features:
- physics-aware engineered ratios (`invH`, `S_over_H`, `S_over_I`, `I_over_H`)
- monotonic physical priors (`df/dH <= 0`, `df/dI <= 0`, `df/dS >= 0`)
- imbalance-aware learning and cost-sensitive thresholding
- probability calibration and reliability assessment
- robustness under noise, missing features, and unseen regimes
- counterfactual decision support
- deployment-ready inference + drift monitoring
- optional dynamic-refinement scaffold for future transient windows

## Model Naming (Clear Comparison)

To make interpretation explicit, model names are standardized as:

- Proposed model: **PhysiScreen-OOS**
  - repository id: `Proposed Physics-Aware Model`
- Base model: **Logit-Base**
  - repository id: `Logistic Regression`
- Comparison models:
  - **SVM-RBF** (`SVM (RBF)`)
  - **RF-Base** (`Random Forest`)
  - **Boost-GBM** (`XGBoost/LightGBM/HistGB`)
  - **MonoGBM-OOS** (`monogbm_oos`)
  - **EBM-OOS** (`ebm_oos`, optional `interpret` dependency)
  - **TabPFN-OOS** (`tabpfn_oos`, optional `tabpfn` dependency)
  - **Legacy-Hybrid** (`Existing Repo Model (Hybrid)`)

## Algorithm Interpretation

- **PhysiScreen-OOS (Proposed):** physics-aware static classifier with engineered ratios, validation-set calibration, and cost-sensitive thresholding.
- **Logit-Base:** transparent linear baseline for reproducible reference.
- **SVM-RBF:** high-capacity nonlinear benchmark for raw predictive performance.
- **RF-Base / Boost-GBM:** robust tabular nonlinear baselines.
- **MonoGBM-OOS:** monotonic gradient-boosting benchmark using physical signs where supported.
- **EBM-OOS:** optional glass-box nonlinear model for operator-readable feature response curves.
- **TabPFN-OOS:** optional tabular foundation-model challenger for small-to-medium tabular classification.
- **Legacy-Hybrid:** existing repository hybrid baseline retained for backward comparison.
- **Conformal-OOS:** uncertainty wrapper that produces stable/OOS prediction sets for safety decisions.

## Latest Long-Run Results Snapshot

From `outputs/static_q1_validation_xlong/tables/`:

- Benchmark best pure predictive: **SVM-RBF**
- Best deployable/overall recommended: **PhysiScreen-OOS**
- PhysiScreen-OOS nominal scenario metrics:
  - PR-AUC: `0.9939`
  - ROC-AUC: `0.9991`
  - FNR: `0.0074`
  - ECE: `0.0018`

Recommended files for quick review:
- `outputs/static_q1_validation_xlong/tables/table1_main_performance.csv`
- `outputs/static_q1_validation_xlong/tables/q1_table_nominal_baseline.csv`
- `outputs/static_q1_validation_xlong/tables/q1_table_regime_shift.csv`
- `outputs/static_q1_validation_xlong/tables/q1_table_robustness.csv`
- `outputs/static_q1_validation_xlong/tables/q1_table_threshold_policies.csv`
- `outputs/static_q1_validation_xlong/tables/q1_table_monotonic_consistency.csv`

## Dataset Expectations

Expected columns (extra columns allowed):
- `Tag_rate`
- `Ikssmin_kA`
- `Sgn_eff_MVA`
- `H_s`
- `GenName` (optional; defaults to `GR1`)
- `Out_of_step` (binary target)

### Variable Dictionary (With Meaning)

| Symbol | Column | Unit | Meaning |
|---|---|---|---|
| \(T\) | `Tag_rate` | as provided in source dataset | disturbance/acceleration proxy |
| \(I\) | `Ikssmin_kA` | kA | short-circuit strength proxy (grid strength) |
| \(S\) | `Sgn_eff_MVA` | MVA | operating stress/loading proxy |
| \(H\) | `H_s` | s | inertia constant |
| \(y\) | `Out_of_step` | - | target class (1 = OOS, 0 = stable) |
| `GenName` | `GenName` | - | generator identifier (GR1 in this case study) |

Engineered physics-aware features:

| Feature | Formula | Interpretation |
|---|---|---|
| `invH` | \(1/H\) | inverse inertia indicator |
| `S_over_H` | \(S/H\) | stress per inertia |
| `S_over_I` | \(S/I\) | stress per grid strength |
| `I_over_H` | \(I/H\) | grid strength per inertia |

A tiny demo is included at:
- `data/sample/gr1_sample.csv`

## Repository Layout

```text
OUTOFSTEP_ML/
  configs/
  data/
    sample/
  docs/
  scripts/
  src/
    outofstep_ml/
      data/
      features/
      models/
      evaluation/
      explainability/
      deployment/
      utils/
  tests/
  results/
  outputs/
  main.py
```

## Installation

### Option A: pip

```bash
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate outofstep-ml
```

## Quickstart (New Benchmark Pipeline)

Train static models and export reproducible artifacts:

```bash
python scripts/train_static.py --config configs/model_static_physics.yaml
```

Evaluate trained static model:

```bash
python scripts/evaluate_static.py --config configs/model_static_physics.yaml
```

Run ablations:

```bash
python scripts/run_ablation.py --config configs/base.yaml
```

Run progressive "one-logic-after-another" comparison (raw -> engineered -> monotonic -> imbalance+cost -> calibrated full stack):

```bash
python scripts/run_logic_ladder.py --config configs/logic_ladder.yaml
```

## Product Mode: Grid Synchronization Compatibility

The repository now includes a product-facing interface for unknown device/operating-point data. It answers the operational question:

> Can this device operating point be synchronized with the grid without high predicted OOS risk?

Build the product artifact:

```bash
python scripts/build_grid_sync_product.py --config configs/logic_ladder.yaml --model-policy deployment_safety --output-dir outputs/product
```

Score one unknown device from JSON:

```bash
python scripts/predict_grid_sync.py --input outputs/product/example_in_domain_device.json
```

Score a batch of unknown devices:

```bash
python scripts/predict_grid_sync.py --input unknown_devices.csv --output outputs/product/unknown_device_results.csv
```

Run the API product:

```bash
uvicorn src.outofstep_ml.product.api:app --host 0.0.0.0 --port 8000
```

API endpoint:

```text
POST /compatibility
```

Example request:

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

Product verdicts:
- `COMPATIBLE_FOR_GRID_SYNC`: predicted OOS risk is below the safety threshold and the point is in-domain.
- `NOT_COMPATIBLE_HIGH_OOS_RISK`: predicted OOS risk is above the safety threshold.
- `ENGINEERING_REVIEW_REQUIRED_OUT_OF_DOMAIN`: the point is outside the training envelope, so the model does not auto-approve it.

Product outputs are stored in:
- `outputs/product/grid_sync_bundle.joblib`
- `outputs/product/grid_sync_model_card.json`
- `outputs/product/grid_sync_holdout_metrics.csv`

Detailed product guide:
- `docs/GRID_SYNC_PRODUCT_GUIDE.md`

This runner exports explicit scenario rows `S1`..`S9`, where `S9` is the compact cumulative final view:
- `S1`: raw baseline
- `S2`: + engineered physics ratios
- `S3`: + monotonic priors
- `S4`: + imbalance-aware learning + cost threshold
- `S5`: + calibration (full predictive stack)
- `S6`: robustness scenario summary
- `S7`: counterfactual scenario summary
- `S8`: deployment/drift scenario summary
- `S9`: compact final (includes S1..S8)

It also exports an automatic non-sequential combination search (all valid mixes such as `1+2+4`, `1+2+5`, `1+2+4+5`, etc.) so you can select the best capability mix:
- `results/logic_ladder/tables/logic_ladder_combination_comparison.csv`
- `results/logic_ladder/tables/logic_ladder_best_combination.json`

Ranking is reported in multiple ways:
- predictive rank (`PR-AUC`)
- safety rank (`Recall - FNR`)
- calibrated rank (`PR-AUC - 0.1*ECE`)
- composite rank (default deployment-oriented blend)

### Scenario Combination Details (`S1`..`S9`)

Capability keys used in scenario/combination analysis:
- `1`: raw baseline (`T, I, S, H`)
- `2`: engineered physics ratios (`invH, S_over_H, S_over_I, I_over_H`)
- `3`: monotonic priors (`df/dH<=0`, `df/dI<=0`, `df/dS>=0`)
- `4`: imbalance-aware learning + cost-sensitive threshold (`tau_cost`)
- `5`: calibration (isotonic/Platt chosen on validation)

Scenario logic:
- `S1 = 1`
- `S2 = 1+2`
- `S3 = 1+2+3`
- `S4 = 1+2+4`
- `S5 = 1+2+4+5` (full predictive stack)
- `S6`: robustness evaluation over `S5` (noise/missing/unseen/group shift)
- `S7`: counterfactual evaluation over `S5`
- `S8`: deployment + drift evaluation over `S5`
- `S9`: compact final summary = `S1..S8` combined

Important interpretation:
- `S1..S5` are cumulative model-building stages.
- `S6..S8` are operational evaluation overlays (not additional model layers).
- `S9` is the compact “all previous included” scenario row.

### Latest Scenario Results (Dataset_output.csv)

Source: `results/logic_ladder/tables/logic_ladder_scenario_comparison.csv`

| Scenario | PR-AUC | ROC-AUC | Recall | FNR | ECE | Notes |
|---|---:|---:|---:|---:|---:|---|
| S1 | 0.9895 | 0.9984 | 1.0000 | 0.0000 | 0.0534 | Raw baseline |
| S2 | 0.9996 | 1.0000 | 1.0000 | 0.0000 | 0.0075 | Best pure discrimination |
| S3 | 0.1477 | 0.4867 | 0.0260 | 0.9740 | 0.0193 | Monotonic-only path underperforms on this dataset |
| S4 | 0.9520 | 0.9986 | 0.9983 | 0.0017 | 0.0905 | Cost-thresholding boosts safety |
| S5 | 0.9877 | 0.9992 | 0.9983 | 0.0017 | 0.0033 | Best calibrated full-stack stage |
| S9 | 0.9877 | 0.9992 | 0.9983 | 0.0017 | 0.0033 | Compact final row incl. robustness/counterfactual/drift summaries |

Operational summaries from S6/S7/S8:
- Robustness worst PR-AUC drop: `0.0138`
- Robustness mean PR-AUC drop: `0.0048`
- Counterfactual success rate: `0.0` (needs improvement)
- Max PSI drift signal: `3.2349` (strong drift alert in synthetic shifted view)

### Latest Combination Search Results

Source: `results/logic_ladder/tables/logic_ladder_combination_comparison.csv`

Top-ranked combinations by composite score:

| Rank | Combo | Meaning | PR-AUC | Recall | FNR | ECE |
|---:|---|---|---:|---:|---:|---:|
| 1 | `1+2` | raw + engineered | 0.9996 | 1.0000 | 0.0000 | 0.0075 |
| 2 | `1+2+5` | raw + engineered + calibration | 0.9929 | 0.9983 | 0.0017 | 0.0014 |
| 3 | `1+2+4+5` | raw + engineered + imbalance/cost + calibration | 0.9877 | 0.9983 | 0.0017 | 0.0033 |
| 4 | `1` | raw only | 0.9895 | 1.0000 | 0.0000 | 0.0534 |
| 5 | `1+5` | raw + calibration | 0.9878 | 0.9619 | 0.0381 | 0.0030 |

Best-combination JSON summary:
- `results/logic_ladder/tables/logic_ladder_best_combination.json`

## Performance Matrix (How To Read It)

The project uses a multi-axis performance matrix instead of a single metric.

### Discrimination metrics
- `PR-AUC` (primary for imbalance): higher is better.
- `ROC-AUC`: higher is better.

### Classification-at-threshold metrics
- `Precision`: alarm quality.
- `Recall`: instability capture rate.
- `F1`: precision/recall balance.
- `FNR`: missed-instability rate (critical safety metric), lower is better.

### Calibration metrics
- `ECE` (Expected Calibration Error): probability reliability gap, lower is better.
- `Brier`: mean squared probability error, lower is better.

### Robustness metrics
- `PR_AUC_drop_vs_clean`: degradation under shifts (noise, missing, unseen regime); lower drop is better.
- Drift indicators: `PSI`, `KS` (higher values indicate stronger shift and retraining risk).

### Actionability metrics
- Counterfactual success rate: fraction of risky points where feasible minimal changes reach stable threshold; higher is better.

### Deployment interpretation
- For operator deployment, prioritize low `FNR`, low `ECE`, acceptable robustness drop, and actionable counterfactual behavior over PR-AUC alone.

### Deployment-Oriented Composite Score

A deployment-oriented ranking can use:

```text
Composite =
0.25 * PR-AUC
+ 0.25 * Recall
- 0.20 * FNR
- 0.15 * ECE
- 0.10 * Robustness_Drop
- 0.05 * Monotonic_Violation
```

Interpretation:
- best predictive model is not always the best deployable model
- OOS screening should prioritize low missed-instability risk (`FNR`) and calibrated probabilities
- uncertainty-aware outputs should route ambiguous cases to review or future dynamic refinement

## SOTA Research Extension

The next benchmark layer adds:
- `MonoGBM-OOS`: practical monotonic gradient boosting for static OOS screening
- `EBM-OOS`: optional glass-box model for nonlinear feature response curves
- `TabPFN-OOS`: optional tabular foundation-model challenger
- `Conformal-OOS`: uncertainty-aware prediction sets for safety-critical decisions

Detailed design note:
- `docs/SOTA_RESEARCH_EXTENSION_AND_EVALUATION_MATRIX.md`

Comprehensive interpretation guide:
- `docs/OOS_COMPREHENSIVE_INTERPRETATION_GUIDE.md`

Microsoft Word research report with embedded figures, tables, and rendered mathematical equations:
- `outputs/word/OOS_GR1_research_report.docx`
- `outputs/word/OOS_GR1_research_report_detailed.docx` (expanded paper-style version with abstract, introduction, detailed case study, statistical comparison, best-approach discussion, outcomes, and conclusion)

Regenerate the Word report:

```bash
python scripts/generate_word_report.py
python scripts/generate_word_report.py --output outputs/word/OOS_GR1_research_report_detailed.docx
```

Run robustness suite:

```bash
python scripts/run_robustness.py --config configs/base.yaml
```

Run the full strict benchmark comparison (all configured model families):

```bash
python scripts/run_full_benchmark.py --config configs/full_benchmark.yaml
python scripts/generate_figures.py --mode benchmark --config configs/full_benchmark.yaml
```

Run integrated static-Q1 validation (train + all 10 static scenarios):

```bash
python scripts/run_static_q1_validation.py --config configs/static_q1_validation.yaml
python scripts/generate_static_q1_tables.py --config configs/static_q1_validation.yaml
python scripts/generate_static_q1_figures.py --config configs/static_q1_validation.yaml
```

Long-budget integrated run:

```bash
python scripts/run_static_q1_validation.py --config configs/static_q1_validation_long.yaml
```

Extra-long integrated run:

```bash
python scripts/run_static_q1_validation.py --config configs/static_q1_validation_xlong.yaml
python scripts/generate_static_q1_tables.py --config configs/static_q1_validation_xlong.yaml
python scripts/generate_static_q1_figures.py --config configs/static_q1_validation_xlong.yaml
```

Generate publication-ready figures/tables:

```bash
python scripts/generate_figures.py --config configs/base.yaml
```

Export model bundle to deployment folder:

```bash
python scripts/export_model.py --source-model results/model/static_model_bundle.joblib --source-config results/model/inference_config.yaml --target-dir outputs/model
```

## Backward-Compatible Legacy Pipeline

The original end-to-end script remains available:

```bash
python main.py --data "data/raw/Dataset_output.csv" --output-dir outputs --seed 42
```

## Outputs

New benchmark outputs are written to:
- `results/tables/`
- `results/figures/`
- `results/model/`
- `results/splits/`

Integrated static-Q1 scenario outputs are written to:
- `outputs/static_q1_validation/tables/`
- `outputs/static_q1_validation/figures/`
- `outputs/static_q1_validation/splits/`
- long run: `outputs/static_q1_validation_long/`
- extra-long run: `outputs/static_q1_validation_xlong/`

Legacy outputs remain in:
- `outputs/`

## Reproducibility Notes

- deterministic seeds exposed in config (`seed`)
- split manifests saved to `results/splits/`
- config-driven runs for training/evaluation/robustness
- explicit threshold policy (`tau_cost`, `tau_f1`, `tau_hr`)

## Strict Evaluation Protocol (Enforced)

The full benchmark runner enforces:
1. Train/Validation/Final holdout Test splits.
2. Model fitting on Train only.
3. Hyperparameter tuning, calibration fitting, threshold selection, and early stopping decisions on Validation only.
4. Final holdout Test use only for final unbiased comparison.
5. Robustness experiments on shifted test sets without using them for fitting/tuning.
6. Split indices and seeds saved to disk (`results/splits/strict_split_*.csv`, `results/splits/split_manifest.csv`, `results/splits/strict_evaluation_protocol.json`).

## Physics-Aware Model Notes

Main static model concept:
- Stage-1 static risk model from operating features `T, I, S, H`
- engineered physics features improve boundary smoothness and extrapolation
- monotonic priors enforce physically plausible trends where possible

Loss concept:
- data loss + monotonic penalty regularization
- calibration (isotonic/sigmoid) compared post hoc

## Monitoring and Deployment

Deployment modules:
- `src/outofstep_ml/deployment/predict.py`
- `src/outofstep_ml/deployment/api_schema.py`
- `src/outofstep_ml/deployment/monitor.py`

Monitoring includes feature drift reports (PSI/KS) and score distribution summaries.

## Tests

Run tests:

```bash
pytest -q
```

Key tests include:
- feature generation
- split reproducibility
- metric correctness
- inference smoke checks

## Dynamic Refinement Roadmap

`src/outofstep_ml/models/dynamic_refinement.py` is scaffold-only unless transient-window data is provided.
Expected future input shape: `[n_samples, sequence_length, n_features]` with aligned disturbance windows.

## Citation-Ready Case Study Support

See:
- `docs/CASE_STUDY_PLAN.md`
- `docs/PR_SUMMARY.md`

