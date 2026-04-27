# OOS Comprehensive Interpretation Guide

## 1. What Problem Are We Solving?

The project predicts whether a generator operating point will become **out-of-step (OOS)**.

The target label is:

```text
Out_of_step = 0  -> stable
Out_of_step = 1  -> out-of-step / unstable
```

The current dataset is a **static operating-point dataset**, not a PMU time-series dataset. That means the model learns risk from operating parameters rather than from a transient waveform.

Core input variables:

| Symbol | Column | Meaning |
|---|---|---|
| T | `Tag_rate` | disturbance / acceleration proxy |
| I | `Ikssmin_kA` | grid strength / short-circuit current |
| S | `Sgn_eff_MVA` | loading or stress proxy |
| H | `H_s` | inertia constant |

The goal is not just to classify `0/1`. The goal is to build a **decision-support screening system** that can say:

```text
This operating point has high OOS risk.
This risk is calibrated.
This decision is robust enough for screening.
This point is uncertain or drifted and should be reviewed.
These changes may reduce the risk.
```

## 2. Why This Is More Than a Plain Classifier

A plain classifier would learn:

```text
[T, I, S, H] -> Out_of_step
```

This project adds power-system and deployment logic:

| Added logic | Why it matters |
|---|---|
| Physics ratios | expose stress/inertia/grid-strength relationships |
| Monotonic priors | encourage physically sensible risk trends |
| Imbalance handling | avoids ignoring rare OOS cases |
| Cost thresholding | penalizes missed instability more than false alarms |
| Calibration | makes probability values meaningful |
| Robustness tests | checks noise, missing inputs, unseen regimes |
| Counterfactuals | suggests possible corrective action |
| Drift monitoring | detects future data distribution change |
| Conformal uncertainty | identifies safe/unsafe/uncertain decisions |
| Dynamic scaffold | prepares for future PMU/transient-window data |

## 3. Engineered Physics Ratios

The model uses four physically motivated ratios:

| Feature | Formula | Interpretation |
|---|---|---|
| `invH` | 1 / H | inverse inertia; larger means lower inertia support |
| `S_over_H` | S / H | stress relative to inertia |
| `S_over_I` | S / I | stress relative to grid strength |
| `I_over_H` | I / H | grid strength relative to inertia |

These features are important because OOS behavior is usually driven by **relationships between quantities**, not only by the raw values.

Current result: adding these ratios gives the strongest pure predictive result.

```text
S2 / C12 = raw + engineered ratios
PR-AUC = 0.9996
ROC-AUC = 1.0000
Recall = 1.0000
FNR = 0.0000
ECE = 0.0075
```

## 4. Scenario Logic: S1 to S9

The project now uses explicit scenarios so the analysis is easier to follow.

| Scenario | Meaning | Cumulative? |
|---|---|---|
| S1 | raw baseline | yes |
| S2 | raw + engineered physics ratios | yes |
| S3 | S2 + monotonic priors | yes |
| S4 | S2 + imbalance handling + cost threshold | yes |
| S5 | S4 + calibration | yes |
| S6 | robustness evaluation over S5 | evaluation overlay |
| S7 | counterfactual evaluation over S5 | evaluation overlay |
| S8 | deployment/drift evaluation over S5 | evaluation overlay |
| S9 | compact final row including S1 to S8 | final summary |

Important:

```text
S1-S5 build the model step by step.
S6-S8 evaluate operational readiness.
S9 summarizes the full deployment-oriented view.
```

## 5. Latest Scenario Results

Source:

```text
results/logic_ladder/tables/logic_ladder_scenario_comparison.csv
```

| Scenario | PR-AUC | ROC-AUC | Recall | FNR | ECE | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| S1 | 0.9895 | 0.9984 | 1.0000 | 0.0000 | 0.0534 | raw baseline is already strong but less calibrated |
| S2 | 0.9996 | 1.0000 | 1.0000 | 0.0000 | 0.0075 | best pure static discrimination |
| S3 | 0.1477 | 0.4867 | 0.0260 | 0.9740 | 0.0193 | current monotonic-only path fails badly |
| S4 | 0.9520 | 0.9986 | 0.9983 | 0.0017 | 0.0905 | cost threshold reduces missed OOS but calibration is poor |
| S5 | 0.9877 | 0.9992 | 0.9983 | 0.0017 | 0.0033 | best calibrated full predictive stack |
| S9 | 0.9877 | 0.9992 | 0.9983 | 0.0017 | 0.0033 | final compact deployment summary |

### Main Interpretation

The best pure static model is:

```text
S2 / C12 = raw + engineered ratios
```

The best deployment-style scenario is:

```text
S9 = S5 predictive model + robustness + counterfactual + drift + conformal uncertainty
```

The biggest warning is:

```text
S3 underperforms, so the current monotonic implementation should not be treated as the final physics-constrained answer.
```

This does not mean monotonic physics is wrong. It means the current monotonic model is too restrictive or not flexible enough. A better next step is **MonoGBM-OOS**, a monotonic gradient-boosting model.

## 6. Combination Search Results

The project also tests non-sequential combinations, not only S1 to S9.

Capability keys:

| Key | Meaning |
|---|---|
| 1 | raw features |
| 2 | engineered physics ratios |
| 3 | monotonic priors |
| 4 | imbalance handling + cost threshold |
| 5 | calibration |

Source:

```text
results/logic_ladder/tables/logic_ladder_combination_comparison.csv
```

Top combinations:

| Rank | Combination | Meaning | PR-AUC | Recall | FNR | ECE |
|---:|---|---|---:|---:|---:|---:|
| 1 | 1+2 | raw + engineered | 0.9996 | 1.0000 | 0.0000 | 0.0075 |
| 2 | 1+2+5 | raw + engineered + calibrated | 0.9929 | 0.9983 | 0.0017 | 0.0014 |
| 3 | 1+2+4+5 | raw + engineered + imbalance/cost + calibrated | 0.9877 | 0.9983 | 0.0017 | 0.0033 |
| 4 | 1 | raw only | 0.9895 | 1.0000 | 0.0000 | 0.0534 |
| 5 | 1+5 | raw + calibrated | 0.9878 | 0.9619 | 0.0381 | 0.0030 |
| 6 | 1+2+4 | raw + engineered + imbalance/cost | 0.9520 | 0.9983 | 0.0017 | 0.0905 |

### What This Means

For pure classification quality:

```text
Choose 1+2.
```

For calibrated probability quality:

```text
Choose 1+2+5.
```

For conservative deployment with a cost-sensitive threshold:

```text
Choose 1+2+4+5.
```

For the paper narrative:

```text
Report all three, because they answer different research questions.
```

## 7. Robustness Interpretation

Source:

```text
results/logic_ladder/tables/logic_ladder_robustness.csv
```

| Shift | PR-AUC | Recall | FNR | PR-AUC drop |
|---|---:|---:|---:|---:|
| clean | 0.9877 | 0.9983 | 0.0017 | 0.0000 |
| noisy | 0.9877 | 0.9983 | 0.0017 | 0.0000 |
| missing features | 0.9877 | 0.9983 | 0.0017 | 0.0001 |
| unseen regime | 0.9739 | 1.0000 | 0.0000 | 0.0138 |
| group shift | 0.9777 | 0.9912 | 0.0088 | 0.0100 |

The model is robust to noise and missing-feature masking. The larger stress comes from unseen regimes and group shift, which is expected in operating-point screening.

## 8. Conformal Uncertainty

Source:

```text
results/logic_ladder/tables/logic_ladder_conformal_summary.csv
```

Current conformal results:

| Metric | Value |
|---|---:|
| coverage | 0.9667 |
| OOS coverage | 0.9584 |
| average set size | 0.9678 |
| ambiguous rate | 0.0000 |

Interpretation:

Conformal prediction adds a safety wrapper around probability scores. Instead of only saying:

```text
p(OOS) = 0.72
```

the system can produce a prediction set:

```text
{OOS}
{Stable}
{Stable, OOS}
```

If the set is `{Stable, OOS}`, the model is uncertain and the point should be reviewed or routed to a future dynamic PMU-window model.

## 9. Counterfactual Interpretation

Current result:

```text
counterfactual success rate = 0.0
```

This is a research opportunity, not just a failure. It means the current counterfactual search is too limited or the stability threshold/action bounds are too strict.

Recommended improvement:

```text
Replace generic counterfactual search with physics-constrained recourse.
```

Action variables should be:

| Variable | Possible action |
|---|---|
| Sgn_eff_MVA | reduce loading / redispatch / curtailment |
| H_s | add inertia or synthetic inertia |
| Ikssmin_kA | strengthen grid / topology or support change |
| Tag_rate | usually not directly controllable |

## 10. Evaluation Matrix

The evaluation matrix is the full set of criteria used to judge the model.

| Axis | Metric | Desired direction | Why it matters |
|---|---|---|---|
| Discrimination | PR-AUC | higher | best for imbalanced OOS data |
| Discrimination | ROC-AUC | higher | general separability |
| Alarm quality | Precision | higher | fewer false alarms |
| OOS capture | Recall | higher | detects more true OOS points |
| Safety | FNR | lower | missed OOS is dangerous |
| Balance | F1 | higher | precision/recall tradeoff |
| Calibration | ECE | lower | probability reliability |
| Calibration | Brier | lower | probability error |
| Robustness | PR-AUC drop | lower | survives noise/shift |
| Drift | PSI / KS | lower | detects distribution change |
| Physics | monotonic violation rate | lower | physically plausible behavior |
| Actionability | counterfactual success | higher | gives feasible mitigation |
| Deployment | inference latency | lower | supports online screening |
| Uncertainty | conformal coverage | near target | safer uncertainty estimates |

For this project, the most important metrics are:

```text
FNR, Recall, PR-AUC, ECE, robustness drop, conformal OOS coverage.
```

Accuracy alone should not be used as the main metric.

## 11. Recommended Reading Order

If you are losing the thread, follow this order:

1. Read `README.md` for project orientation.
2. Read this guide for the research story.
3. Open `logic_ladder_scenario_comparison.csv` for S1 to S9.
4. Open `logic_ladder_combination_comparison.csv` for best combinations.
5. Open `logic_ladder_robustness.csv` for shift performance.
6. Open `logic_ladder_conformal_summary.csv` for uncertainty behavior.
7. Open `SOTA_RESEARCH_EXTENSION_AND_EVALUATION_MATRIX.md` for next research upgrades.

## 12. Current Bottom Line

Best pure static screening combination:

```text
1+2 = raw + engineered physics ratios
```

Best calibrated static combination:

```text
1+2+5 = raw + engineered + calibration
```

Best conservative deployment-style full stack:

```text
1+2+4+5 = raw + engineered + imbalance/cost + calibration
```

Best compact scenario for reporting all operational layers:

```text
S9
```

Main improvement target:

```text
Replace weak monotonic implementation with MonoGBM-OOS and improve physics-constrained counterfactual recourse.
```

