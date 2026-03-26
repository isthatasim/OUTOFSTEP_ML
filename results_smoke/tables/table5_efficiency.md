| Model                        |   Training Time |   Inference Time |   Memory / Model Size | Suitable for Online Screening   |
|:-----------------------------|----------------:|-----------------:|----------------------:|:--------------------------------|
| Existing Repo Model (Hybrid) |       0.16829   |        0.219533  |            0.175357   | True                            |
| Logistic Regression          |       0.0196459 |        0.0602375 |            0.00635052 | True                            |
| Proposed Physics-Aware Model |       0.0812576 |        0.0383825 |            0.00293255 | True                            |
| Random Forest                |       0.300659  |        0.176783  |            0.132378   | True                            |
| SVM (RBF)                    |       0.017761  |        0.0757    |            0.00943279 | True                            |
| XGBoost/LightGBM/HistGB      |       0.43658   |        0.359967  |            0.109206   | True                            |