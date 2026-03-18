# OOS GR1 Model Card

## Model Identity
- Model: Two-stage hybrid
- Calibration: none
- Version: v1.1.0
- Task framing: pattern-based binary classification for operating-point risk forecasting (not time-series forecasting).

## Inputs
Tag_rate, Ikssmin_kA, Sgn_eff_MVA, H_s, GenName

## Outputs
p_oos and class label under threshold policy.

## Threshold Policy
- tau_cost: 0.1470
- tau_F1: 0.7596
- tau_HR: 0.9990

## Validation Snapshot
- PR_AUC: 1.0000
- ROC_AUC: 1.0000
- FNR: 0.0000
- ECE: 0.0000
