"""Cluster bootstrap for object-level detection metrics.

Frames from one session are near-duplicates, so frames are not independent draws. The bootstrap
therefore resamples whole clusters (sessions) with replacement and recomputes the metric on the
union of the drawn clusters (Field and Welsh, 2007; Davison and Hinkley, 1997, section 3.8).
For a paired comparison of two rungs, both are scored on the same resampled clusters, so the
interval is for the difference itself and shared cluster difficulty cancels.

Inputs are per-cluster pieces: for each cluster, the object scores, the false-alarm candidate
scores and the number of frames. Percentile intervals are reported.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .metrics import recall_at_fapf


@dataclass
class Clustered:
    """Scores split by cluster: obj[c], neg[c] are arrays, frames[c] an int."""
    obj: list[np.ndarray]
    neg: list[np.ndarray]
    frames: np.ndarray

    @classmethod
    def from_frame(cls, clusters_obj: Sequence, obj_scores, clusters_neg: Sequence, neg_scores,
                   frames_per_cluster: dict) -> Clustered:
        keys = sorted(frames_per_cluster)
        co, cn = np.asarray(clusters_obj), np.asarray(clusters_neg)
        os_, ns = np.asarray(obj_scores, dtype=float), np.asarray(neg_scores, dtype=float)
        return cls([os_[co == k] for k in keys], [ns[cn == k] for k in keys],
                   np.array([frames_per_cluster[k] for k in keys]))

    def take(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        o = np.concatenate([self.obj[i] for i in idx]) if len(idx) else np.array([])
        n = np.concatenate([self.neg[i] for i in idx]) if len(idx) else np.array([])
        return o, n, int(self.frames[idx].sum())


Metric = Callable[[np.ndarray, np.ndarray, int], float]


def recall_metric(fapf: float) -> Metric:
    return lambda o, n, f: recall_at_fapf(o, n, f, fapf)


def cluster_bootstrap(data: Clustered, metric: Metric, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and (1 - alpha) percentile interval of metric(obj, neg, frames)."""
    rng = np.random.default_rng(seed)
    k = len(data.frames)
    point = metric(*data.take(np.arange(k)))
    vals = []
    for _ in range(n_boot):
        v = metric(*data.take(rng.integers(0, k, k)))
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if vals else (np.nan, np.nan)
    return float(point), float(lo), float(hi)


def paired_cluster_bootstrap(a: Clustered, b: Clustered, metric: Metric, n_boot: int = 2000,
                             seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Interval for metric(b) - metric(a) with the same cluster draws for both rungs.

    a and b must list the same clusters in the same order (same test data, different scores).
    """
    if len(a.frames) != len(b.frames) or np.any(a.frames != b.frames):
        raise ValueError("paired bootstrap needs the same clusters for both rungs")
    rng = np.random.default_rng(seed)
    k = len(a.frames)
    full = np.arange(k)
    point = metric(*b.take(full)) - metric(*a.take(full))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, k, k)
        v = metric(*b.take(idx)) - metric(*a.take(idx))
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if vals else (np.nan, np.nan)
    return float(point), float(lo), float(hi)
