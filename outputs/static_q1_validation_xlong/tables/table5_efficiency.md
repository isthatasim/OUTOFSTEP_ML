| Model                        |   Training Time |   Inference Time |   Memory / Model Size | Suitable for Online Screening   |
|:-----------------------------|----------------:|-----------------:|----------------------:|:--------------------------------|
| Existing Repo Model (Hybrid) |        1.09236  |      0.0073331   |            0.562025   | True                            |
| Logistic Regression          |       11.9982   |      0.00153662  |            0.00635052 | True                            |
| Proposed Physics-Aware Model |       38.4003   |      0.000493781 |            0.00293255 | True                            |
| Random Forest                |       83.6218   |      0.00968435  |            0.770726   | True                            |
| SVM (RBF)                    |      121.185    |      0.0192939   |            0.0414566  | True                            |
| XGBoost/LightGBM/HistGB      |        0.667458 |      0.00380233  |            0.191084   | True                            |