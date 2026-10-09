"""Object-level metrics at a fixed false-alarm rate, cluster bootstrap and group splits."""

import numpy as np
import pandas as pd
import pytest

from echofind.eval import splits as sp
from echofind.eval.bootstrap import (
    Clustered,
    cluster_bootstrap,
    paired_cluster_bootstrap,
    recall_metric,
)
from echofind.eval.metrics import (
    ece,
    f1_at_threshold,
    fapf_at_threshold,
    object_pr_auc,
    recall_at_fapf,
    threshold_at_fapf,
)


def test_threshold_at_fapf_allows_floor_of_rate_times_frames():
    """FAPF(t) = #(neg >= t) / frames <= target, with k = floor(target * frames) passing."""
    neg = np.arange(100, dtype=float)              # 0..99
    t = threshold_at_fapf(neg, n_frames=40, fapf=0.1)   # k = 4 may pass: 99, 98, 97, 96
    assert fapf_at_threshold(neg, 40, t) == pytest.approx(0.1)
    assert 95 < t <= 96


def test_recall_at_fapf_counts_missed_objects():
    """Recall = found / all objects, and objects with score -inf (never proposed) are misses."""
    obj = np.array([10.0, 5.0, -np.inf, 0.5])
    neg = np.array([1.0, 2.0, 3.0])
    # 1 false alarm allowed in 10 frames: threshold just above 2, so objects 10 and 5 are found
    assert recall_at_fapf(obj, neg, 10, 0.1) == pytest.approx(0.5)


def test_f1_and_pr_auc_extremes():
    """F1 = 2TP / (2TP + FP + FN); AP = 1 for a perfect ranking."""
    obj, neg = np.array([3.0, 4.0]), np.array([1.0, 2.0])
    assert f1_at_threshold(obj, neg, 2.5) == 1.0
    assert f1_at_threshold(obj, neg, 1.5) == pytest.approx(4 / 5)
    assert object_pr_auc(obj, neg) == pytest.approx(1.0)


def test_ece_of_perfectly_calibrated_bins_is_zero_and_known_offset():
    """ECE = sum_b (n_b/n) |acc_b - conf_b|; constant p = 0.8 with 50% positives gives 0.3."""
    p = np.full(1000, 0.8)
    y = np.r_[np.ones(500), np.zeros(500)]
    assert ece(p, y) == pytest.approx(0.3)
    assert ece(np.r_[np.full(10, 0.05), np.full(10, 0.95)], np.r_[np.zeros(10), np.ones(10)]) \
        == pytest.approx(0.05)


def _clusters(rng, shift=0.0):
    obj = [rng.normal(3 + shift, 1, 20) for _ in range(6)]
    neg = [rng.normal(0, 1, 200) for _ in range(6)]
    return Clustered(obj, neg, np.full(6, 50))


def test_cluster_bootstrap_interval_contains_point():
    rng = np.random.default_rng(0)
    pt, lo, hi = cluster_bootstrap(_clusters(rng), recall_metric(0.05), 300, 0)
    assert lo <= pt <= hi and 0 <= lo and hi <= 1


def test_paired_bootstrap_identical_scores_gives_zero_and_detects_shift():
    """Paired difference of a rung with itself is exactly 0; a clear shift excludes 0."""
    rng = np.random.default_rng(1)
    a = _clusters(rng)
    d, lo, hi = paired_cluster_bootstrap(a, a, recall_metric(0.05), 200, 0)
    assert d == lo == hi == 0
    b = Clustered([o + 2.0 for o in a.obj], a.neg, a.frames)
    d, lo, hi = paired_cluster_bootstrap(a, b, recall_metric(0.05), 200, 0)
    assert d > 0 and lo > 0


def test_group_splits_are_disjoint_and_check_catches_leaks():
    fr = pd.DataFrame({"grp": list("aabbcc"), "x": range(6)})
    for fold in sp.leave_one_group_out(fr, "grp"):
        sp.check_disjoint(fr, fold, "grp")
    bad = ("bad", np.array([1, 1, 0, 0, 0, 0], bool), np.array([0, 1, 1, 0, 0, 0], bool))
    with pytest.raises(ValueError):
        sp.check_disjoint(fr, bad, "grp")
