# GR1 Out-of-Step (OOS) Pipeline

Publication-grade ML + physics-aware + deployable pipeline scaffold for OOS prediction.

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
