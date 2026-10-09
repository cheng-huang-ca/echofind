"""Operating point from mission cost (docs/PLAN.md, W4).

Expected time to clear one search area with a detector at threshold t:

    E[T](t) = T_scan + N_fa(t) * T_check + P_miss(t) * T_research

- N_fa(t) = FAPF(t) * F, the false alarms in one scan of F frames; each costs a check
  (reposition, look again, or send a diver).
- P_miss(t) = (1 - r(t))^k, with r(t) the single-look object recall and k the number of
  independent looks at the body during the scan. k = 1 is the conservative case; k > 1
  assumes looks fail independently, which is optimistic because the same pose and range
  repeat (multi-look fusion, R5, is what would earn k > 1).
- T_research is the time lost when the body is missed: a second search by other means.

The threshold that minimises E[T] trades checks against misses; when T_research is large the
optimum moves to more false alarms. All times are inputs, not facts: see
reports/decisions/301-mission-cost-parameters.md for the values used and why.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MissionCost:
    frames_per_scan: int = 30      # one handheld sweep
    t_scan_s: float = 60.0
    t_check_s: float = 180.0       # per false alarm
    t_research_s: float = 1800.0   # per missed body
    looks: int = 1


def p_miss(recall: np.ndarray, looks: int) -> np.ndarray:
    """P_miss = (1 - r)^k for k independent looks."""
    return (1.0 - np.asarray(recall, dtype=float)) ** looks


def expected_time(fapf: np.ndarray, recall: np.ndarray, c: MissionCost) -> np.ndarray:
    """E[T] = T_scan + FAPF * F * T_check + (1 - r)^k * T_research, in seconds."""
    fapf = np.asarray(fapf, dtype=float)
    return (c.t_scan_s + fapf * c.frames_per_scan * c.t_check_s
            + p_miss(recall, c.looks) * c.t_research_s)


def operating_curve(obj_scores: np.ndarray, neg_scores: np.ndarray, n_frames: int,
                    n_points: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thresholds with the FAPF and object recall each gives (candidate thresholds are
    quantiles of the finite scores, plus +inf, which raises no alarm at all)."""
    o = np.asarray(obj_scores, dtype=float)
    n = np.asarray(neg_scores, dtype=float)
    finite = np.r_[o[np.isfinite(o)], n[np.isfinite(n)]]
    thr = np.unique(np.r_[np.quantile(finite, np.linspace(0, 1, n_points)), np.inf])
    ns = np.sort(n)
    fa = (len(ns) - np.searchsorted(ns, thr, side="left")) / n_frames
    os_ = np.sort(o)
    rec = (len(os_) - np.searchsorted(os_, thr, side="left")) / max(len(o), 1)
    return thr, fa, rec


def best_threshold(obj_scores, neg_scores, n_frames: int, c: MissionCost) -> dict:
    """Threshold minimising E[T], with the FAPF, recall and E[T] it gives."""
    thr, fa, rec = operating_curve(obj_scores, neg_scores, n_frames)
    et = expected_time(fa, rec, c)
    i = int(np.argmin(et))
    return {"threshold": float(thr[i]), "fapf": float(fa[i]), "recall": float(rec[i]),
            "expected_time_s": float(et[i]), "fa_per_scan": float(fa[i] * c.frames_per_scan)}
