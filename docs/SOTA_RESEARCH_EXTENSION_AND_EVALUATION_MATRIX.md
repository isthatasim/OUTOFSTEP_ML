# SOTA Research Extension and Evaluation Matrix

## Purpose

This note upgrades the OOS project framing from a single static classifier into a layered research benchmark for safety-critical out-of-step risk screening.

The current core remains static operating-point prediction using:

```text
T = Tag_rate
I = Ikssmin_kA
S = Sgn_eff_MVA
H = H_s
```

and engineered physics ratios:

```text
invH = 1 / H
S_over_H = S / H
S_over_I = S / I
I_over_H = I / H
```

## Layered Framework

| Layer | Goal | Methods |
|---|---|---|
| Static screening | Fast operating-point risk scoring | PhysiScreen-OOS, MonoGBM-OOS, SVM-RBF, RF-Base, Boost-GBM |
| Explainable tabular benchmark | Operator-trust response curves | EBM-OOS, future GAMI-Net-OOS |
| Modern tabular challenger | SOTA small-tabular comparator | TabPFN-OOS, future FT-Transformer/SAINT |
| Safety wrapper | Reliable decisions, not just scores | calibration, cost thresholds, conformal prediction |
| Robustness | Real-world data imperfections | noise, missing data, unseen regimes, group shift |
| Operator support | Actionable mitigation guidance | physics-constrained counterfactual recourse |
| Deployment monitoring | Detect distribution change | PSI, KS, score drift, retraining triggers |
| Future dynamic refinement | PMU/transient-window refinement | TCN, CNN-GRU, Transformer, ST-GNN, physics-informed Neural ODE |

## Added Benchmark Families

| Model | Repository ID | Role |
|---|---|---|
| MonoGBM-OOS | `monogbm_oos` | practical monotonic gradient-boosting baseline |
| EBM-OOS | `ebm_oos` | optional glass-box nonlinear model if `interpret` is installed |
| TabPFN-OOS | `tabpfn_oos` | optional tabular foundation-model challenger if `tabpfn` is installed |
| Conformal-OOS | logic-ladder uncertainty output | uncertainty-aware prediction sets for safety decisions |

## Evaluation Matrix

The project should not select a model only by accuracy. OOS is safety-critical, so missed instability matters more than ordinary mistakes.

| Axis | Metric | Desired direction | Reason |
|---|---|---|---|
| Discrimination | PR-AUC | higher | primary metric under class imbalance |
| Discrimination | ROC-AUC | higher | overall separability |
| Alarm quality | Precision | higher | fewer nuisance OOS alarms |
| Instability capture | Recall | higher | more true OOS cases detected |
| Safety | FNR | lower | missed OOS cases are critical |
| Balance | F1 | higher | precision-recall compromise |
| Calibration | ECE | lower | risk probabilities match observed frequencies |
| Calibration | Brier | lower | probability error |
| Robustness | worst PR-AUC drop | lower | stability under noise/missing/shift |
| Drift | PSI / KS | lower | distribution shift warning |
| Physics consistency | monotonic violation rate | lower | learned response respects physical priors |
| Actionability | counterfactual success rate | higher | risky points can be moved below threshold |
| Deployment | inference latency | lower | online screening feasibility |
| Uncertainty | conformal coverage | near target | prediction sets cover true class |
| Uncertainty | ambiguous rate | lower after safety constraints | fewer uncertain decisions |

## Deployment Selection Logic

Recommended final selection should report:

```text
Best predictive model      = highest PR-AUC / ROC-AUC
Best safety model          = lowest FNR with high recall
Best calibrated model      = lowest ECE / Brier
Best deployable model      = low FNR + low ECE + robustness + fast inference
Best research contribution = physics-aware + calibrated + uncertainty-guided + actionable
```

For this repository, `PhysiScreen-OOS` remains the proposed deployable method, while `MonoGBM-OOS`, `EBM-OOS`, and `TabPFN-OOS` are benchmark challengers.

## Future Dynamic Extension

No dynamic or PMU-window experiment should be fabricated without real sequential data.

When available, expected dynamic input shape is:

```text
[n_samples, sequence_length, n_features]
```

Recommended dynamic models:

| Model | Best use |
|---|---|
| TCN-OOS | fast transient-window classification |
| CNN-GRU / CNN-LSTM | local transient features plus temporal memory |
| Transformer-OOS | longer-range PMU-window dependency |
| STGNN-OOS / ASTGCN-OOS | topology plus time-series stability prediction |
| Physics-informed Neural ODE | rotor-dynamics consistency |

