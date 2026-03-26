from __future__ import annotations

from typing import Dict

import numpy as np

from src.eval import evaluate_probabilities


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, c_fn: float = 10.0, c_fp: float = 1.0) -> Dict[str, float]:
    return evaluate_probabilities(y_true, y_prob, c_fn=c_fn, c_fp=c_fp)
