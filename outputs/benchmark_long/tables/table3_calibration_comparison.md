| Model                        |   Uncalibrated Brier |   Calibrated Brier |   Uncalibrated ECE |   Calibrated ECE | Best calibration   |
|:-----------------------------|---------------------:|-------------------:|-------------------:|-----------------:|:-------------------|
| Existing Repo Model (Hybrid) |            0.0282473 |          0.0282704 |          0.0290159 |        0.0290226 | isotonic           |
| Logistic Regression          |            0.0296534 |          0.0288515 |          0.0326818 |        0.0288932 | isotonic           |
| Proposed Physics-Aware Model |            0.0477266 |          0.0316384 |          0.0836869 |        0.030693  | isotonic           |
| Random Forest                |            0.0282416 |          0.0282897 |          0.0299231 |        0.0290107 | isotonic           |
| SVM (RBF)                    |            0.0277972 |          0.0279261 |          0.0295351 |        0.0282058 | isotonic           |
| XGBoost/LightGBM/HistGB      |            0.029083  |          0.0290833 |          0.029091  |        0.0290843 | isotonic           |