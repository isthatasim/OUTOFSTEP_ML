# Q1 Case Study Plan (Applied Energy / IEEE TSG)

## Objective
Deliver a reproducible benchmark for static OOS risk screening with calibrated probabilities, robust validation, and decision support.

## Study Blocks
1. Data audit + schema validation
2. Feature sets: raw vs physics-engineered
3. Validation regimes: stratified, grouped, leave-level-out
4. Model ladder: LR, RF, GBM, monotonic model, physics-regularized model, hybrid model
5. Calibration: none/sigmoid/isotonic
6. Threshold policy: tau_cost, tau_F1, tau_HR
7. Robustness: noise + missing-feature + unseen regime
8. Explainability: SHAP/permutation + PDP
9. Counterfactual planning recommendations
10. Deployment + drift monitoring + retraining triggers

## Metrics
Primary: PR-AUC, FNR, ECE
Secondary: ROC-AUC, F1, Precision, Recall, Specificity, Balanced Accuracy, Brier, MSE, RMSE, MAE, R2

## Publication Outputs
- Main metrics table
- Ablation table
- Calibration table
- Robustness table
- Threshold comparison table
- ROC/PR/reliability figures
- PDP and feature importance figures
- Counterfactual examples and illustration

## Future Extension
Enable dynamic refinement model when transient windows become available.
