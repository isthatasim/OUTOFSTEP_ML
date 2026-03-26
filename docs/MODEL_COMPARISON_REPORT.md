# Model Comparison Report

## Why these models
- Classical: interpretable and strong baselines for sanity and fairness.
- Strong tabular boosting: competitive SOTA-like tabular references.
- Optional deep tabular: FT-Transformer and TabNet where runtime dependencies exist.
- Existing repository model: continuity baseline.
- Proposed physics-aware model: main contribution with physical priors + actionability.

## How models were trained
- Strict Train/Val/Test protocol.
- Hyperparameter search and calibration on validation only.
- Final test used once.
- Repeated seeds and multiple split strategies for robustness.

## Expected model strengths
- CatBoost / strong boosting: often top raw predictive performance.
- Proposed physics-aware model: stronger physical consistency, calibrated risk interpretation, and decision support.
- Existing hybrid model: practical deployment continuity.

## Operational recommendation logic
Best model is selected by multi-criteria weighting:
- predictive power
- calibration quality
- robustness under shift
- runtime efficiency
- physical consistency
- explainability
- deployment readiness

## Temporal baselines
No true transient sequence benchmark is executed unless real PMU/transient windows exist.
The codebase keeps temporal extension scaffold-ready.

## Future work
- Add real PMU-window dataset adapter and train temporal baselines.
- Extend deep tabular implementation when dependencies/GPU budget are available.
- Add uncertainty quantification for risk-aware dispatch policies.
