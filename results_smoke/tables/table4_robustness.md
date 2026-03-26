| Model                        |    Clean |    Noisy |   Missing Features |   Unseen Regime |   Group Shift |   Performance Drop |
|:-----------------------------|---------:|---------:|-------------------:|----------------:|--------------:|-------------------:|
| Existing Repo Model (Hybrid) | 0.825    | 0.825    |              0.825 |             nan |           nan |          0         |
| Logistic Regression          | 0.825    | 0.825    |              0.785 |             nan |           nan |          0.02      |
| Proposed Physics-Aware Model | 0.825    | 0.825    |              0.825 |             nan |           nan |          0         |
| Random Forest                | 0.665    | 0.558333 |              0.825 |             nan |           nan |         -0.0266667 |
| SVM (RBF)                    | 0.825    | 0.825    |              0.785 |             nan |           nan |          0.02      |
| XGBoost/LightGBM/HistGB      | 0.390812 | 0.390812 |              0.14  |             nan |           nan |          0.125406  |