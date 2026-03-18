# GR1 Out-of-Step (OOS) Pipeline

Publication-grade ML + physics-aware + deployable pipeline scaffold for OOS prediction.

## Problem Framing

- Goal: predict `Out_of_step` in `{0,1}` from associated static operating parameters.
- This is a **binary classification** problem used for **risk forecasting** at operating points.
- It is not a time-series forecaster because no temporal sequence column is provided.
- Evaluation includes both classification metrics and probability-forecast metrics:
  `PR-AUC`, `ROC-AUC`, `Precision`, `Recall`, `F1`, `FNR`, `Brier`, `ECE`, `MSE`, `RMSE`, `MAE`, `R2`.

## Run

```powershell
python main.py --data "C:/Users/masim/Downloads/outofstep_tag_ikss_H_Sgn.csv" --output-dir outputs --seed 42
```

If `--data` path does not exist, scaffold/template artifacts are generated only.

## Structure

- `src/data.py` - loading, audit, cleaning, leakage checks
- `src/features.py` - engineering + monotonic priors
- `src/models.py` - tiers A/B/C + calibration
- `src/eval.py` - metrics, splits, thresholding, counterfactuals
- `src/plots.py` - paper figures + flowchart
- `src/report.py` - technical markdown document generation
- `src/api_app.py` - FastAPI prototype
- `src/monitoring.py` - PSI/KS, concept drift, policy
- `src/retrain.py` - champion/challenger retraining gate
- `tests/` - pytest suite
- `outputs/` - exported artifacts
