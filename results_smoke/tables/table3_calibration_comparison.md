| Model                        |   Uncalibrated Brier |   Calibrated Brier |   Uncalibrated ECE |   Calibrated ECE | Best calibration   |
|:-----------------------------|---------------------:|-------------------:|-------------------:|-----------------:|:-------------------|
| Existing Repo Model (Hybrid) |            0.027199  |          0.0326171 |          0.0361104 |        0.0468743 | isotonic           |
| Logistic Regression          |            0.137795  |          0.0539609 |          0.212097  |        0.0815891 | isotonic           |
| Proposed Physics-Aware Model |            0.0977832 |          0.0632373 |          0.144525  |        0.0945128 | isotonic           |
| Random Forest                |            0.0365446 |          0.100415  |          0.0850386 |        0.111314  | isotonic           |
| SVM (RBF)                    |            0.055519  |          0.0479374 |          0.10223   |        0.0808528 | isotonic           |
| XGBoost/LightGBM/HistGB      |            0.151972  |          0.0856454 |          0.18537   |        0.0775583 | isotonic           |