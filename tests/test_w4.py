"""Mission-cost operating point, OOD score and input QA checks (W4)."""

import numpy as np
import pandas as pd
import pytest

from echofind.eval.cost import MissionCost, best_threshold, expected_time, operating_curve
from echofind.eval.ood import MahalanobisOOD, flag_threshold
from echofind.eval.qa import frame_qa


def test_expected_time_formula():
    """E[T] = T_scan + FAPF * F * T_check + (1 - r)^k * T_research."""
    c = MissionCost(frames_per_scan=30, t_scan_s=60, t_check_s=180, t_research_s=1800, looks=2)
    assert expected_time(0.1, 0.5, c) == pytest.approx(60 + 0.1 * 30 * 180 + 0.25 * 1800)


def test_operating_curve_and_optimum():
    """FAPF and recall fall as the threshold rises; with free checks the optimum takes every
    object (recall 1), with free misses it raises no false alarm."""
    rng = np.random.default_rng(0)
    obj, neg = rng.normal(2, 1, 200), rng.normal(0, 1, 5000)
    thr, fa, rec = operating_curve(obj, neg, 1000)
    assert np.all(np.diff(fa) <= 0) and np.all(np.diff(rec) <= 0)
    assert np.isinf(thr[-1]) and fa[-1] == 0 and rec[-1] == 0
    assert best_threshold(obj, neg, 1000, MissionCost(t_check_s=0))["recall"] == 1.0
    assert best_threshold(obj, neg, 1000, MissionCost(t_research_s=0))["fapf"] == 0


def test_mahalanobis_ood_flags_shifted_frames():
    """D^2 of standard-normal features has mean ~ dimension; a 3-sigma mean shift is flagged."""
    rng = np.random.default_rng(1)
    cols = ["a", "b", "c"]
    train = pd.DataFrame(rng.normal(size=(5000, 3)), columns=cols)
    m = MahalanobisOOD(cols).fit(train)
    assert m.candidate_score(train).mean() == pytest.approx(3, rel=0.1)
    ind = pd.DataFrame(rng.normal(size=(400, 3)), columns=cols)
    shift = pd.DataFrame(rng.normal(3, 1, size=(400, 3)), columns=cols)
    f_in = m.frame_score(ind, np.repeat(np.arange(40), 10))
    f_sh = m.frame_score(shift, np.repeat(np.arange(40), 10))
    t = flag_threshold(f_in, 0.05)
    assert np.mean(f_in > t) <= 0.05 and np.mean(f_sh > t) > 0.9


def test_qa_separates_edge_padding_from_dropped_rows():
    a = np.full((100, 50), 0.2, np.float32)
    a[97:] = 0                                     # trailing export padding: not a failure
    q = frame_qa(a)
    assert q["edge_pad"] == pytest.approx(0.03) and q["dropped_rows"] == 0 and q["qa_pass"]
    a[40:45] = 0                                   # interior drop of 5% of rows
    a[:10, :5] = 1.0                               # 50 of 5,000 samples saturated = 1%
    q = frame_qa(a)
    assert q["dropped_rows"] == pytest.approx(0.05) and q["fail_dropped_rows"]
    assert q["fail_saturation"] and not q["qa_pass"]
