"""Checks of the clutter models in echofind.features.clutter against closed forms."""

import numpy as np
from scipy import integrate

from echofind.features import clutter


def test_k_pdf_integrates_to_one():
    """K amplitude pdf (Jakeman and Pusey) has unit area for heavy and light tails."""
    for nu in (0.5, 2.0, 20.0):
        area, _ = integrate.quad(lambda a, nu=nu: np.exp(clutter.k_logpdf(np.array(a), nu, 1.0)),
                                 0, np.inf, limit=200)
        assert abs(area - 1) < 1e-4


def test_k_tends_to_rayleigh_for_large_nu():
    """As nu -> infinity the K pdf approaches Rayleigh with s2 = mu / 2."""
    a = np.linspace(0.05, 3, 50)
    k = clutter.k_logpdf(a, 100.0, 1.0)
    r = clutter.rayleigh_logpdf(a, 0.5)
    assert np.max(np.abs(np.exp(k) - np.exp(r))) < 0.02


def test_rayleigh_mle_closed_form_and_aic_choice():
    """Rayleigh MLE s2 = mean(a^2)/2; on Rayleigh data AIC should not favour K strongly."""
    rng = np.random.default_rng(1)
    a = rng.rayleigh(scale=np.sqrt(0.5), size=20000)
    fits = {f.model: f for f in clutter.fit_all(a)}
    assert abs(fits["rayleigh"].params["sigma2"] - 0.5) < 0.02
    assert fits["k"].params["nu"] > 10
    assert fits["k"].aic > fits["rayleigh"].aic - 4


def test_k_fit_recovers_shape_and_beats_rayleigh():
    """Compound K samples (gamma texture x exponential speckle) give nu back and a lower AIC."""
    rng = np.random.default_rng(2)
    a = clutter.sample_k(20000, nu=1.5, mu=2.0, rng=rng)
    fits = {f.model: f for f in clutter.fit_all(a)}
    assert 1.2 < fits["k"].params["nu"] < 1.9
    assert fits["k"].aic < fits["rayleigh"].aic - 100
    assert clutter.scintillation_index(a) > 1.5  # SI = 1 + 2/nu for K intensity
