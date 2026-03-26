| Model                        |   Training Time |   Inference Time |   Memory / Model Size | Suitable for Online Screening   |
|:-----------------------------|----------------:|-----------------:|----------------------:|:--------------------------------|
| Existing Repo Model (Hybrid) |         1.33317 |      0.00923761  |            0.572581   | True                            |
| Logistic Regression          |        10.5363  |      0.00187459  |            0.00635052 | True                            |
| Proposed Physics-Aware Model |        35.1389  |      0.000593494 |            0.00293255 | True                            |
| Random Forest                |        68.8944  |      0.014267    |            0.913531   | True                            |
| SVM (RBF)                    |       111.847   |      0.024758    |            0.041342   | True                            |
| XGBoost/LightGBM/HistGB      |         0.9083  |      0.00467638  |            0.191721   | True                            |