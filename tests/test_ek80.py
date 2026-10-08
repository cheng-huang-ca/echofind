"""Bottom detection and the impulse-noise test on synthetic echograms (no EK80 files needed)."""

import numpy as np

from echofind.io import ek80


def synthetic(n_ping=40, n=2000, dr=0.02, bottom_m=30.0, seed=0):
    rng = np.random.default_rng(seed)
    r = np.arange(n) * dr
    sv = -90 + 3 * rng.standard_normal((n_ping, n))
    j = int(bottom_m / dr)
    sv[:, j:j + 5] = -10
    return sv, r, j


def test_bottom_survives_interference_spikes():
    """A spike stronger than the seabed in some pings must not move the seabed track."""
    sv, r, j = synthetic()
    sv[5, 600:700] = 0  # 2 m interference burst at 12 m, louder than the seabed
    sv[20, 900:1000] = 0
    b = ek80.bottom_index(sv, r)
    assert np.all(np.abs(b - j) <= 5)


def test_impulse_mask_flags_single_ping_bursts_only():
    """Ryan et al. (2015) two-sided test: a one-ping burst is flagged, the seabed is not."""
    sv, r, j = synthetic()
    sv[7, 600:700] = -20
    sv[3, 1700:] = np.nan  # a shorter record must not look like a burst in its neighbours
    m = ek80.impulse_mask(sv, 0.02)
    assert m[7, 620:680].all()
    assert not m[:, j:j + 5].any()
    assert m.mean() < 0.01
