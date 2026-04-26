from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 1.0
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / max(n, 1))
    return float(np.quantile(scores, q_level, method="higher"))


def binary_conformal_prediction_sets(
    y_calib: np.ndarray,
    p_calib: np.ndarray,
    p_test: np.ndarray,
    alpha: float = 0.10,
    class_conditional: bool = True,
) -> Dict[str, object]:
    """
    Split-conformal prediction sets for binary OOS risk.
    Classes are encoded as 0=stable and 1=OOS.
    """
    y_calib = np.asarray(y_calib).astype(int)
    p_calib = np.asarray(p_calib, dtype=float)
    p_test = np.asarray(p_test, dtype=float)

    probs_cal = np.column_stack([1.0 - p_calib, p_calib])
    probs_test = np.column_stack([1.0 - p_test, p_test])
    true_scores = 1.0 - probs_cal[np.arange(len(y_calib)), y_calib]

    global_q = _quantile(true_scores, alpha=alpha)
    q_by_class = {0: global_q, 1: global_q}
    if class_conditional:
        for cls in [0, 1]:
            mask = y_calib == cls
            q_by_class[cls] = _quantile(true_scores[mask], alpha=alpha) if mask.any() else global_q

    sets: List[List[int]] = []
    for row in probs_test:
        pred_set = [cls for cls in [0, 1] if (1.0 - row[cls]) <= q_by_class[cls]]
        sets.append(pred_set)

    return {
        "sets": sets,
        "alpha": float(alpha),
        "q_global": float(global_q),
        "q_stable": float(q_by_class[0]),
        "q_oos": float(q_by_class[1]),
        "class_conditional": bool(class_conditional),
    }


def summarize_conformal_sets(y_true: np.ndarray, conformal: Dict[str, object]) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    sets = conformal["sets"]
    contains = np.asarray([int(y in s) for y, s in zip(y_true, sets)], dtype=float)
    sizes = np.asarray([len(s) for s in sets], dtype=float)
    singleton = sizes == 1
    ambiguous = sizes == 2
    empty = sizes == 0
    oos_mask = y_true == 1

    rows = [
        {
            "alpha": float(conformal["alpha"]),
            "class_conditional": bool(conformal["class_conditional"]),
            "coverage": float(contains.mean()) if len(contains) else np.nan,
            "oos_coverage": float(contains[oos_mask].mean()) if oos_mask.any() else np.nan,
            "average_set_size": float(sizes.mean()) if len(sizes) else np.nan,
            "singleton_rate": float(singleton.mean()) if len(singleton) else np.nan,
            "ambiguous_rate": float(ambiguous.mean()) if len(ambiguous) else np.nan,
            "empty_rate": float(empty.mean()) if len(empty) else np.nan,
            "q_global": float(conformal["q_global"]),
            "q_stable": float(conformal["q_stable"]),
            "q_oos": float(conformal["q_oos"]),
        }
    ]
    return pd.DataFrame(rows)

