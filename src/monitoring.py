from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover
    ks_2samp = None


def _psi_single(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    qs = np.quantile(ref, np.linspace(0, 1, bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf

    ref_hist = np.histogram(ref, bins=qs)[0].astype(float)
    cur_hist = np.histogram(cur, bins=qs)[0].astype(float)

    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1.0), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1.0), 1e-6, None)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def psi_table(reference_df: pd.DataFrame, current_df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows = []
    for c in features:
        if c not in reference_df.columns or c not in current_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(reference_df[c]):
            continue
        rows.append({"feature": c, "PSI": _psi_single(reference_df[c].values, current_df[c].values)})
    return pd.DataFrame(rows)


def ks_table(reference_df: pd.DataFrame, current_df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows = []
    for c in features:
        if c not in reference_df.columns or c not in current_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(reference_df[c]):
            continue
        if ks_2samp is None:
            stat, p = np.nan, np.nan
        else:
            stat, p = ks_2samp(reference_df[c].values, current_df[c].values)
        rows.append({"feature": c, "KS_stat": float(stat), "KS_pvalue": float(p)})
    return pd.DataFrame(rows)


def rolling_performance(y_true: np.ndarray, y_prob: np.ndarray, window: int = 200, threshold: float = 0.5) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    out = []
    for t in range(window, len(y_true) + 1):
        yt = y_true[t - window : t]
        yp = y_prob[t - window : t]
        yb = (yp >= threshold).astype(int)

        fn = np.sum((yt == 1) & (yb == 0))
        tp = np.sum((yt == 1) & (yb == 1))
        fnr = fn / max(fn + tp, 1)

        out.append(
            {
                "t": t,
                "PR_AUC": float(average_precision_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan,
                "Recall": float(recall_score(yt, yb, zero_division=0)),
                "FNR": float(fnr),
            }
        )
    if not out:
        return pd.DataFrame(columns=["t", "PR_AUC", "Recall", "FNR"])
    return pd.DataFrame(out).set_index("t")


@dataclass
class DriftAlert:
    detector: str
    triggered: bool
    t: int
    value: float


class PageHinkley:
    def __init__(self, delta: float = 0.005, lamb: float = 50.0):
        self.delta = delta
        self.lamb = lamb
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0
        self.t = 0

    def update(self, x: float) -> bool:
        self.t += 1
        self.mean += (x - self.mean) / self.t
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        return (self.cum - self.min_cum) > self.lamb


class DDM:
    def __init__(self):
        self.n = 1
        self.p = 1.0
        self.s = 0.0
        self.p_min = np.inf
        self.s_min = np.inf

    def update(self, error: int) -> bool:
        self.n += 1
        self.p += (error - self.p) / self.n
        self.s = np.sqrt(self.p * (1 - self.p) / self.n)
        if self.p + self.s <= self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s
        return self.p + self.s > self.p_min + 3 * self.s_min


class ADWINLite:
    """Simplified ADWIN-like detector using split-window mean difference."""

    def __init__(self, min_window: int = 40, delta: float = 0.002):
        self.min_window = min_window
        self.delta = delta
        self.window: List[float] = []

    def update(self, x: float) -> bool:
        self.window.append(float(x))
        if len(self.window) < 2 * self.min_window:
            return False
        w = np.array(self.window, dtype=float)
        mid = len(w) // 2
        w0, w1 = w[:mid], w[mid:]

        m0, m1 = float(w0.mean()), float(w1.mean())
        eps = np.sqrt(2.0 * np.log(2.0 / self.delta) * (1.0 / len(w0) + 1.0 / len(w1)))
        if abs(m0 - m1) > eps:
            self.window = list(w1)
            return True
        if len(self.window) > 800:
            self.window = self.window[-400:]
        return False


def concept_drift_scan(errors: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    ph = PageHinkley()
    ddm = DDM()
    adw = ADWINLite()
    rows = []

    for t, (e, p) in enumerate(zip(errors, probs), start=1):
        ph_alarm = ph.update(float(e))
        ddm_alarm = ddm.update(int(e))
        adw_alarm = adw.update(float(e))
        rows.append(
            {
                "t": t,
                "PH_alarm": int(ph_alarm),
                "DDM_alarm": int(ddm_alarm),
                "ADWIN_alarm": int(adw_alarm),
                "error": float(e),
                "prob": float(p),
            }
        )
    return pd.DataFrame(rows).set_index("t")


def retrain_trigger_policy(
    psi_df: pd.DataFrame,
    ks_df: pd.DataFrame,
    concept_df: pd.DataFrame,
    new_sample_count: int,
    psi_threshold: float = 0.2,
    ks_pvalue_threshold: float = 0.01,
    min_new_samples: int = 200,
) -> Dict[str, int]:
    psi_alarm = int((psi_df["PSI"] > psi_threshold).any()) if len(psi_df) else 0
    ks_alarm = int((ks_df["KS_pvalue"] < ks_pvalue_threshold).any()) if len(ks_df) else 0
    concept_alarm = int(
        ((concept_df["PH_alarm"] + concept_df["DDM_alarm"] + concept_df["ADWIN_alarm"]) > 0).any()
    ) if len(concept_df) else 0

    trigger = int((psi_alarm or ks_alarm or concept_alarm) and (new_sample_count >= min_new_samples))
    return {
        "psi_alarm": psi_alarm,
        "ks_alarm": ks_alarm,
        "concept_alarm": concept_alarm,
        "new_sample_count": int(new_sample_count),
        "trigger_retrain": trigger,
    }


def simulate_stream(reference_df: pd.DataFrame, drift_strength: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    stream = reference_df.copy().reset_index(drop=True)
    for c in stream.columns:
        if pd.api.types.is_numeric_dtype(stream[c]):
            shift = drift_strength * np.std(stream[c].values)
            stream[c] = stream[c].astype(float) + rng.normal(0.0, shift, size=len(stream))
    return stream
