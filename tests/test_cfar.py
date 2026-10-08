"""CFAR false-alarm rates and detection probabilities against closed forms (H3)."""

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.special import gammaln

from echofind.dsp.cfar import (
    CFARConfig,
    cfar_detect,
    pd_swerling1,
    pfa_ca,
    pfa_go,
    pfa_os,
    pfa_so,
    scale_factor,
)
from echofind.dsp.detection_theory import pd_swerling0, required_snr_db
from echofind.dsp.detection_theory import pd_swerling1 as pd_ideal_sw1


def test_ca_scale_factor_closed_form():
    """CA-CFAR: alpha = N (Pfa^(-1/N) - 1) inverts Pfa = (1 + alpha/N)^-N."""
    for n in (8, 16, 32):
        cfg = CFARConfig("ca", n_ref=n, pfa=1e-4)
        assert pfa_ca(scale_factor(cfg), n) == pytest.approx(1e-4, rel=1e-10)


@pytest.mark.parametrize("n,t", [(4, 1.3), (8, 0.7), (16, 0.4)])
def test_go_so_closed_forms_match_quadrature(n, t):
    """Hansen-Sawyers GO/SO Pfa = E[exp(-t max/min(S1, S2))], S ~ Gamma(n), by quadrature."""
    f, F = stats.gamma(n).pdf, stats.gamma(n).cdf
    go = integrate.quad(lambda y: np.exp(-t * y) * 2 * F(y) * f(y), 0, np.inf)[0]
    so = integrate.quad(lambda y: np.exp(-t * y) * 2 * (1 - F(y)) * f(y), 0, np.inf)[0]
    assert pfa_go(t, n) == pytest.approx(go, rel=1e-8)
    assert pfa_so(t, n) == pytest.approx(so, rel=1e-8)


def test_os_closed_form_matches_quadrature():
    """Rohling OS-CFAR Pfa = prod (N-i)/(N-i+alpha) = E[exp(-alpha X_(k))] over the k-th
    order-statistic density N!/((k-1)!(N-k)!) F^(k-1) (1-F)^(N-k) f."""
    N, k, a = 24, 18, 2.5
    logc = gammaln(N + 1) - gammaln(k) - gammaln(N - k + 1)
    dens = lambda x: np.exp(logc + (k - 1) * np.log1p(-np.exp(-x)) - (N - k + 1) * x)  # noqa
    q = integrate.quad(lambda x: np.exp(-a * x) * dens(x), 0, np.inf)[0]
    assert pfa_os(a, N, k) == pytest.approx(q, rel=1e-8)


@pytest.mark.parametrize("kind", ["ca", "go", "so", "os"])
def test_empirical_pfa_in_exponential_noise(kind):
    """H3 baseline: in homogeneous exponential noise each detector holds its design Pfa
    (binomial count within 5 sigma), and is invariant to the noise level."""
    rng = np.random.default_rng(10)
    cfg = CFARConfig(kind, n_ref=24, n_guard=2, pfa=1e-3)
    x = rng.exponential(1.0, size=(100, 20_000))
    det, thr = cfar_detect(x, cfg)
    n = np.isfinite(thr).sum()
    k = det.sum()
    sd = np.sqrt(n * cfg.pfa * (1 - cfg.pfa))
    assert abs(k - n * cfg.pfa) < 5 * sd
    det_scaled, _ = cfar_detect(1e6 * x[:5], cfg)
    np.testing.assert_array_equal(det_scaled, det[:5])


@pytest.mark.parametrize("kind", ["ca", "os"])
def test_swerling1_pd_with_cfar(kind):
    """Swerling-1 CFAR: Pd = Pfa(scale / (1 + S)); for CA, (1 + alpha/(N(1+S)))^-N."""
    rng = np.random.default_rng(11)
    cfg = CFARConfig(kind, n_ref=16, n_guard=0, pfa=1e-3)
    snr = 10 ** (15 / 10)
    m = 40_000
    win = rng.exponential(1.0, size=(m, 17))
    win[:, 8] = rng.exponential(1 + snr, size=m)
    det, _ = cfar_detect(win, cfg)
    pd_mc = det[:, 8].mean()
    assert pd_mc == pytest.approx(pd_swerling1(cfg, snr)[0], abs=4 * np.sqrt(0.25 / m))
    if kind == "ca":
        a = scale_factor(cfg)
        assert pd_swerling1(cfg, snr)[0] == pytest.approx((1 + a / (16 * (1 + snr))) ** -16)


def test_cfar_loss_vanishes_with_many_reference_cells():
    """CA-CFAR Swerling-1 Pd tends to the ideal Pfa^(1/(1+S)) as N grows."""
    snr = 10 ** (13 / 10)
    ideal = pd_ideal_sw1(snr, 1e-4)
    gaps = [ideal - pd_swerling1(CFARConfig("ca", n_ref=n, pfa=1e-4), snr)[0]
            for n in (8, 32, 512)]
    assert gaps[0] > gaps[1] > gaps[2] > 0 and gaps[2] < 0.005


def test_ideal_detector_pd_formulas():
    """Swerling 0: Pd = Q1(sqrt(2S), sqrt(-2 ln Pfa)); Swerling 1: Pd = Pfa^(1/(1+S));
    checked by Monte Carlo at S = 10 dB, and DT inverts both."""
    rng = np.random.default_rng(12)
    S, pfa, m = 10.0, 1e-3, 200_000
    T = -np.log(pfa)
    n = (rng.standard_normal(m) + 1j * rng.standard_normal(m)) / np.sqrt(2)
    pd0 = np.mean(np.abs(np.sqrt(S) + n) ** 2 > T)
    a1 = np.sqrt(S) * (rng.standard_normal(m) + 1j * rng.standard_normal(m)) / np.sqrt(2)
    pd1 = np.mean(np.abs(a1 + n) ** 2 > T)
    assert pd0 == pytest.approx(pd_swerling0(S, pfa), abs=0.005)
    assert pd1 == pytest.approx(pd_ideal_sw1(S, pfa), abs=0.005)
    for model, fn in (("swerling0", pd_swerling0), ("swerling1", pd_ideal_sw1)):
        dt = required_snr_db(0.9, pfa, model)
        assert fn(10 ** (dt / 10), pfa) == pytest.approx(0.9, abs=1e-6)


def test_go_cfar_limits_false_alarms_at_clutter_edge():
    """H3: at a +15 dB clutter step, CA-CFAR's threshold is dragged down by the quiet side and
    over-reports on the loud side; GO-CFAR takes the louder half-window and holds Pfa closer."""
    rng = np.random.default_rng(13)
    L, edge, pfa = 400, 200, 1e-3
    lvl = np.where(np.arange(L) < edge, 1.0, 10 ** 1.5)
    x = rng.exponential(1.0, size=(20_000, L)) * lvl
    excess = {}
    for kind in ("ca", "go"):
        det, _ = cfar_detect(x, CFARConfig(kind, n_ref=24, n_guard=2, pfa=pfa))
        near = slice(edge, edge + 14)               # loud cells with quiet cells in the window
        excess[kind] = det[:, near].mean() / pfa
    assert excess["ca"] > 5
    assert excess["go"] < excess["ca"] / 3
