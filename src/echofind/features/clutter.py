"""Envelope (amplitude) statistics of sonar background: Rayleigh, Weibull and K fits.

All three models are for the amplitude a = |echo| >= 0 of a background cell. Rayleigh is the
fully developed speckle case (many scatterers, complex-Gaussian echo). Weibull and K have a
shape parameter that lets the tail grow heavier when few strong scatterers dominate, which is
what makes a fixed threshold fail near the bottom (hypothesis H1 in docs/PLAN.md).

Model comparison uses AIC = 2k - 2 ln L on the same samples, so lower is better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

NU_MAX = 100.0  # K shape above this is indistinguishable from Rayleigh here


@dataclass
class Fit:
    model: str
    params: dict[str, float]
    loglik: float
    k: int

    @property
    def aic(self) -> float:
        return 2 * self.k - 2 * self.loglik


def rayleigh_logpdf(a: np.ndarray, sigma2: float) -> np.ndarray:
    """Rayleigh: p(a) = (a / s2) exp(-a^2 / (2 s2))."""
    return np.log(a) - np.log(sigma2) - a**2 / (2 * sigma2)


def fit_rayleigh(a: np.ndarray) -> Fit:
    """Closed-form MLE: s2 = mean(a^2) / 2."""
    sigma2 = float(np.mean(a**2) / 2)
    return Fit("rayleigh", {"sigma2": sigma2}, float(rayleigh_logpdf(a, sigma2).sum()), 1)


def fit_weibull(a: np.ndarray) -> Fit:
    """Two-parameter Weibull (location fixed at 0): p(a) = (c/l)(a/l)^(c-1) exp(-(a/l)^c)."""
    c, _, scale = stats.weibull_min.fit(a, floc=0)
    ll = float(stats.weibull_min.logpdf(a, c, 0, scale).sum())
    return Fit("weibull", {"shape": float(c), "scale": float(scale)}, ll, 2)


def k_logpdf(a: np.ndarray, nu: float, mu: float) -> np.ndarray:
    """K-distributed amplitude with shape nu and mean intensity mu = E[a^2] (Jakeman and Pusey):

    p(a) = 4 / Gamma(nu) * (nu/mu)^((nu+1)/2) * a^nu * K_{nu-1}(2 a sqrt(nu/mu)).

    It tends to Rayleigh as nu -> infinity; small nu means a heavy tail. Uses the scaled Bessel
    function kve, log K_v(z) = log kve(v, z) - z, to stay finite for large z. Valid up to
    nu of about 100 (kve overflows beyond); fit_k caps nu there, where K is Rayleigh in practice.
    """
    b = np.sqrt(nu / mu)
    z = 2 * a * b
    return (np.log(4) - special.gammaln(nu) + (nu + 1) * np.log(b) + nu * np.log(a)
            + np.log(special.kve(nu - 1, z)) - z)


def fit_k(a: np.ndarray) -> Fit:
    """MLE of (nu, mu) for the K amplitude pdf, starting from the method-of-moments nu.

    Moment start: for K intensity I, E[I^2]/E[I]^2 = 2 (1 + 1/nu).
    """
    mu0 = float(np.mean(a**2))
    ratio = float(np.mean(a**4) / mu0**2)
    nu0 = 1 / (ratio / 2 - 1) if ratio > 2.05 else 40.0
    nu0 = float(np.clip(nu0, 0.1, 40.0))  # start below NU_MAX

    def nll(p):
        nu, mu = np.exp(p)
        if nu > NU_MAX:
            return 1e300
        val = -k_logpdf(a, nu, mu).sum()
        return val if np.isfinite(val) else 1e300

    res = optimize.minimize(nll, np.log([nu0, mu0]), method="Nelder-Mead",
                            options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 2000})
    nu, mu = np.exp(res.x)
    return Fit("k", {"nu": float(nu), "mu": float(mu)}, float(-res.fun), 2)


def fit_all(a: np.ndarray, max_samples: int = 20000, seed: int = 0) -> list[Fit]:
    """Fit all three models to positive amplitudes (subsampled with a fixed seed)."""
    a = np.asarray(a, dtype=float).ravel()
    a = a[np.isfinite(a) & (a > 0)]
    if a.size > max_samples:
        a = np.random.default_rng(seed).choice(a, max_samples, replace=False)
    return [fit_rayleigh(a), fit_weibull(a), fit_k(a)]


def scintillation_index(a: np.ndarray) -> float:
    """SI = var(I) / E[I]^2 with I = a^2: 1 for Rayleigh amplitudes, 1 + 2/nu for K."""
    i = np.asarray(a, dtype=float).ravel() ** 2
    return float(i.var() / i.mean() ** 2)


def sample_k(n: int, nu: float, mu: float, rng: np.random.Generator) -> np.ndarray:
    """K amplitudes as compound speckle: I = texture * exponential, texture ~ Gamma(nu, mu/nu)."""
    texture = rng.gamma(nu, mu / nu, n)
    return np.sqrt(texture * rng.exponential(1.0, n))
