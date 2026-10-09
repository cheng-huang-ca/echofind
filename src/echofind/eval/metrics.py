"""Detection metrics at a fixed false-alarm rate, scored per object and per frame.

The unit a rescue crew sees is an alarm on a frame, so the metrics are object-level:

- A target object is found at threshold t when any candidate matched to it scores >= t.
  Objects that the proposal stage never produced a candidate for keep score -inf: a later
  rung cannot recover a miss of the front end, and recall reports that honestly.
- A false alarm is any candidate not matched to a target object that scores >= t.
  False alarms per frame (FAPF) = false alarms / frames scored.

recall_at_fapf(t*) uses the threshold that gives the target FAPF on the same data (a point on the
FROC curve): it measures ranking quality and is what the ladder compares. transfer() instead
applies a threshold chosen on other data and reports the recall and FAPF actually realised, which
is what a deployed device would see.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def threshold_at_fapf(neg_scores: np.ndarray, n_frames: int, fapf: float) -> float:
    """Lowest threshold t with (#neg >= t) / n_frames <= fapf.

    With the negatives sorted in descending order s_(1) >= s_(2) >= ..., at most
    k = floor(fapf * n_frames) may pass, so t is just above s_(k+1).
    """
    s = np.sort(np.asarray(neg_scores, dtype=float)[np.isfinite(neg_scores)])[::-1]
    k = int(np.floor(fapf * n_frames + 1e-9))
    if k >= len(s):
        return -np.inf
    return float(np.nextafter(s[k], np.inf))


def recall_at_threshold(obj_scores: np.ndarray, t: float) -> float:
    o = np.asarray(obj_scores, dtype=float)
    return float(np.mean(o >= t)) if len(o) else np.nan


def fapf_at_threshold(neg_scores: np.ndarray, n_frames: int, t: float) -> float:
    return float(np.sum(np.asarray(neg_scores) >= t) / max(n_frames, 1))


def recall_at_fapf(obj_scores, neg_scores, n_frames: int, fapf: float) -> float:
    """Object recall at the threshold giving `fapf` false alarms per frame on the same data."""
    return recall_at_threshold(obj_scores, threshold_at_fapf(neg_scores, n_frames, fapf))


def froc(obj_scores, neg_scores, n_frames: int, fapf_grid: np.ndarray) -> np.ndarray:
    """Recall at each false-alarm-per-frame level of the grid (the FROC curve)."""
    return np.array([recall_at_fapf(obj_scores, neg_scores, n_frames, f) for f in fapf_grid])


def object_pr_auc(obj_scores, neg_scores) -> float:
    """Average precision with each target object as one positive (best candidate's score) and
    each unmatched candidate as one negative. Missed objects (-inf) count as never retrieved."""
    o = np.asarray(obj_scores, dtype=float)
    n = np.asarray(neg_scores, dtype=float)
    lo = np.nanmin(np.r_[o[np.isfinite(o)], n[np.isfinite(n)], 0.0]) - 1.0
    y = np.r_[np.ones(len(o)), np.zeros(len(n))]
    s = np.r_[np.where(np.isfinite(o), o, lo), np.where(np.isfinite(n), n, lo)]
    return float(average_precision_score(y, s)) if len(o) else np.nan


def f1_at_threshold(obj_scores, neg_scores, t: float) -> float:
    """F1 with TP = found objects, FN = missed objects, FP = false-alarm candidates."""
    o = np.asarray(obj_scores)
    tp = np.sum(o >= t)
    fn = len(o) - tp
    fp = np.sum(np.asarray(neg_scores) >= t)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def ece(prob: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b / n) |mean(y_b) - mean(p_b)| over equal-width probability bins."""
    p = np.clip(np.asarray(prob, dtype=float), 0, 1)
    y = np.asarray(y, dtype=float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)
