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

## Dataset Expectations

Expected columns (extra columns allowed):
- `Tag_rate`
- `Ikssmin_kA`
- `Sgn_eff_MVA`
- `H_s`
- `GenName` (optional; defaults to `GR1`)
- `Out_of_step` (binary target)

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
python main.py --data "C:/Users/masim/Downloads/outofstep_tag_ikss_H_Sgn.csv" --output-dir outputs --seed 42
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
