"""Closed-form detection probabilities for a single look with a square-law detector.

Noise power is normalised to 1 (|n|^2 ~ Exp(1)), SNR S is the signal-to-noise power ratio after
the matched filter, and a fixed threshold T = -ln Pfa gives the stated false-alarm rate.

- Swerling 0 (steady target): Pd = Q1(sqrt(2S), sqrt(2T)) (Marcum Q), computed as the survival
  function of a non-central chi-square with 2 degrees of freedom and non-centrality 2S at 2T.
- Swerling 1 (Rayleigh-fluctuating target): Pd = Pfa^(1 / (1 + S)).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2


def threshold_for_pfa(pfa: float) -> float:
    """Fixed threshold on |x|^2 for unit noise power: T = -ln Pfa."""
    return float(-np.log(pfa))


def pd_swerling0(snr_linear, pfa: float) -> np.ndarray:
    """Pd = Q1(sqrt(2S), sqrt(-2 ln Pfa)) for a steady target in complex Gaussian noise."""
    s = np.asarray(snr_linear, dtype=float)
    return ncx2.sf(2 * threshold_for_pfa(pfa), df=2, nc=2 * s)


def pd_swerling1(snr_linear, pfa: float) -> np.ndarray:
    """Pd = Pfa^(1/(1+S)) for a Rayleigh-fluctuating target."""
    s = np.asarray(snr_linear, dtype=float)
    return pfa ** (1 / (1 + s))


def required_snr_db(pd: float, pfa: float, model: str = "swerling1") -> float:
    """SNR (dB) at which the ideal detector reaches `pd` (the detection threshold DT)."""
    if model == "swerling1":
        return float(10 * np.log10(np.log(pfa) / np.log(pd) - 1))
    f = lambda x: pd_swerling0(10 ** (x / 10), pfa) - pd  # noqa: E731
    return float(brentq(f, -20, 40))
