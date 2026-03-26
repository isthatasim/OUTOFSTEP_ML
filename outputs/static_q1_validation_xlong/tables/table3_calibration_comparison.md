| Model                        |   Uncalibrated Brier |   Calibrated Brier |   Uncalibrated ECE |   Calibrated ECE | Best calibration   |
|:-----------------------------|---------------------:|-------------------:|-------------------:|-----------------:|:-------------------|
| Existing Repo Model (Hybrid) |            0.0396286 |          0.0396411 |          0.0406201 |        0.0405701 | isotonic           |
| Logistic Regression          |            0.0422884 |          0.0412853 |          0.0462468 |        0.0413371 | isotonic           |
| Proposed Physics-Aware Model |            0.0567602 |          0.0433608 |          0.090869  |        0.0423744 | isotonic           |
| Random Forest                |            0.0396832 |          0.039657  |          0.0415415 |        0.0405553 | isotonic           |
| SVM (RBF)                    |            0.0394946 |          0.039406  |          0.0415486 |        0.039614  | isotonic           |
| XGBoost/LightGBM/HistGB      |            0.0400596 |          0.0400594 |          0.040087  |        0.0400604 | isotonic           |