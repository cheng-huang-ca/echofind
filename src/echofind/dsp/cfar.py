"""Constant false-alarm rate (CFAR) detectors on square-law (power) samples.

Each detector compares the cell under test (CUT) with a threshold scaled from N reference
cells, n = N/2 on each side, beyond G guard cells per side. In exponential (Rayleigh-envelope)
noise the false-alarm probability depends only on the scale factor, not the noise level:

- CA-CFAR:  T = alpha * mean(all N cells),  Pfa = (1 + alpha/N)^-N,  alpha = N (Pfa^(-1/N) - 1)
- SO-CFAR:  T = t * min(S_lead, S_lag),     Pfa = 2 sum_{k=0}^{n-1} C(n-1+k, k) (2+t)^-(n+k)
- GO-CFAR:  T = t * max(S_lead, S_lag),     Pfa = 2 (1+t)^-n - Pfa_SO(t)
- OS-CFAR:  T = alpha * x_(k) (k-th smallest of N),  Pfa = prod_{i=0}^{k-1} (N-i) / (N-i+alpha)

S_lead and S_lag are sums of n cells. The SO and GO forms follow Hansen and Sawyers (1980), with
E[exp(-t max)] + E[exp(-t min)] = 2 E[exp(-t S)]. OS follows Rohling (1983).

For a Swerling-1 target of mean SNR S in the CUT, the CUT is exponential with mean 1 + S, so
Pd(S) = Pfa evaluated at the scale factor divided by (1 + S).

Oversampled input: when the matched-filter output has m samples per resolution cell, adjacent
cells are correlated and a contiguous window of N holds only about N/m independent ones, so the
closed forms above under-set the threshold (Pfa 4-6x too high at m = 4). With `stride = m` the
reference cells are taken every m samples (still N of them, spanning about N resolution cells),
which keeps them nearly independent and the closed forms valid, while every sample is still
tested; see reports/decisions/103-cfar-strided-reference.md.

Cells whose full reference window does not fit inside the profile get a NaN threshold and are
never detections.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import brentq

KINDS = ("ca", "go", "so", "os")


@dataclass(frozen=True)
class CFARConfig:
    kind: str = "ca"
    n_ref: int = 32          # N, total reference cells (n_ref/2 per side)
    n_guard: int = 2         # G, guard cells per side
    pfa: float = 1e-4
    k: int | None = None     # OS rank; default 3N/4
    stride: int = 1          # spacing of reference cells (set to samples per resolution cell)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.n_ref % 2:
            raise ValueError("n_ref must be even")

    @property
    def os_rank(self) -> int:
        return self.k if self.k is not None else int(round(0.75 * self.n_ref))



# ---------------------------------------------------------------- false-alarm closed forms
def pfa_ca(alpha: float, n_ref: int) -> float:
    """Pfa = (1 + alpha/N)^-N for CA-CFAR with T = alpha * mean."""
    return float((1 + alpha / n_ref) ** (-n_ref))


def pfa_so(t: float, n_half: int) -> float:
    """Hansen-Sawyers SO-CFAR: Pfa = 2 sum_k C(n-1+k, k) (2+t)^-(n+k), with T = t * min(sums)."""
    return float(2 * sum(comb(n_half - 1 + k, k) * (2 + t) ** (-(n_half + k))
                         for k in range(n_half)))


def pfa_go(t: float, n_half: int) -> float:
    """GO-CFAR: Pfa = 2 (1+t)^-n - Pfa_SO(t), with T = t * max(sums)."""
    return float(2 * (1 + t) ** (-n_half) - pfa_so(t, n_half))


def pfa_os(alpha: float, n_ref: int, k: int) -> float:
    """Rohling OS-CFAR: Pfa = prod_{i=0}^{k-1} (N-i)/(N-i+alpha), with T = alpha * x_(k)."""
    i = np.arange(k)
    return float(np.exp(np.sum(np.log(n_ref - i) - np.log(n_ref - i + alpha))))


def pfa_of_scale(cfg: CFARConfig, scale: float) -> float:
    """False-alarm probability of `cfg` at a given threshold scale factor."""
    n = cfg.n_ref // 2
    if cfg.kind == "ca":
        return pfa_ca(scale, cfg.n_ref)
    if cfg.kind == "so":
        return pfa_so(scale, n)
    if cfg.kind == "go":
        return pfa_go(scale, n)
    return pfa_os(scale, cfg.n_ref, cfg.os_rank)


def scale_factor(cfg: CFARConfig) -> float:
    """Threshold scale factor that gives cfg.pfa in exponential noise."""
    if cfg.kind == "ca":
        return cfg.n_ref * (cfg.pfa ** (-1 / cfg.n_ref) - 1)
    f = lambda s: np.log(pfa_of_scale(cfg, s)) - np.log(cfg.pfa)  # noqa: E731
    hi = 1.0
    while f(hi) > 0:
        hi *= 2
    return float(brentq(f, 1e-12, hi, xtol=1e-12, rtol=1e-12))


def pd_swerling1(cfg: CFARConfig, snr_linear: np.ndarray | float) -> np.ndarray:
    """Pd for a Swerling-1 target in exponential noise: Pfa(scale / (1 + S))."""
    s = scale_factor(cfg)
    snr = np.atleast_1d(np.asarray(snr_linear, dtype=float))
    return np.array([pfa_of_scale(cfg, s / (1 + x)) for x in snr])


# ---------------------------------------------------------------- detectors
def _offsets(n: int, g: int, m: int) -> np.ndarray:
    """Offsets of the lag-side reference cells (positive numbers; lead side is the mirror)."""
    return g + 1 + m * np.arange(n)


def _reach(n: int, g: int, m: int) -> int:
    return int(_offsets(n, g, m)[-1])


def _sums(p: np.ndarray, n: int, g: int, m: int = 1):
    """Lag (before) and lead (after) reference sums for every cell; NaN where they do not fit."""
    L = p.shape[-1]
    lag = np.full(p.shape, np.nan)
    lead = np.full(p.shape, np.nan)
    R = _reach(n, g, m)
    if L <= 2 * R:
        return lag, lead
    mid = slice(R, L - R)
    lag[..., mid] = 0.0
    lead[..., mid] = 0.0
    for o in _offsets(n, g, m):
        lag[..., mid] += p[..., R - o : L - R - o]
        lead[..., mid] += p[..., R + o : L - R + o]
    return lag, lead


def _os_stat(p: np.ndarray, n: int, g: int, k: int, m: int = 1) -> np.ndarray:
    L = p.shape[-1]
    R = _reach(n, g, m)
    w = 2 * R + 1
    out = np.full(p.shape, np.nan)
    if L < w:
        return out
    off = _offsets(n, g, m)
    keep = np.r_[R - off[::-1], R + off]
    flat = p.reshape(-1, L)
    out_flat = out.reshape(-1, L)
    for row in range(flat.shape[0]):
        win = sliding_window_view(flat[row], w)[:, keep]
        out_flat[row, R : L - R] = np.partition(win, k - 1, axis=-1)[:, k - 1]
    return out


def cfar_threshold(power: np.ndarray, cfg: CFARConfig) -> np.ndarray:
    """CFAR threshold for every cell along the last axis (NaN where the window does not fit)."""
    p = np.asarray(power, dtype=float)
    n, g = cfg.n_ref // 2, cfg.n_guard
    s = scale_factor(cfg)
    if cfg.kind == "os":
        return s * _os_stat(p, n, g, cfg.os_rank, cfg.stride)
    lag, lead = _sums(p, n, g, cfg.stride)
    if cfg.kind == "ca":
        return s * (lag + lead) / cfg.n_ref
    if cfg.kind == "go":
        return s * np.fmax(lag, lead)
    return s * np.fmin(lag, lead)


def cfar_detect(power: np.ndarray, cfg: CFARConfig) -> tuple[np.ndarray, np.ndarray]:
    """Boolean detections and thresholds along the last axis."""
    thr = cfar_threshold(power, cfg)
    with np.errstate(invalid="ignore"):
        det = np.asarray(power) > thr
    return det & np.isfinite(thr), thr
