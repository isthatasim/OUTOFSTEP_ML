from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring import concept_drift_scan, psi_table, retrain_trigger_policy


def test_psi_and_policy():
    ref = pd.DataFrame({"a": np.random.normal(0, 1, 500), "b": np.random.normal(1, 1, 500)})
    cur = pd.DataFrame({"a": np.random.normal(1.5, 1, 500), "b": np.random.normal(1, 1, 500)})
    psi = psi_table(ref, cur, ["a", "b"])
    assert "PSI" in psi.columns

    errors = np.random.binomial(1, 0.2, 500)
    probs = np.random.uniform(0, 1, 500)
    concept = concept_drift_scan(errors, probs)
    decision = retrain_trigger_policy(psi, pd.DataFrame(columns=["KS_pvalue"]), concept, new_sample_count=500)
    assert "trigger_retrain" in decision
