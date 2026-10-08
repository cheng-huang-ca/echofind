"""Matched filtering (pulse compression) and resolution measurement.

The matched filter for a known pulse s(t) is h(t) = s*(-t); its peak output SNR is E/N0 for
complex baseband white noise of density N0 (2E/N0 for a real passband signal), independent of
the pulse shape. Relative to the raw echo in the signal band B, that is a gain of B*T.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def matched_filter(x: np.ndarray, replica: np.ndarray, axis: int = -1,
                   normalize: bool = True) -> np.ndarray:
    """Correlate x with the replica along `axis`, aligned so the output index equals the
    echo's start sample (an echo starting at sample k peaks at sample k).

    With normalize=True the replica is scaled to unit energy (sum |s|^2 = 1), so white noise of
    variance sigma^2 per sample keeps variance sigma^2 after filtering.
    """
    h = np.conj(replica[::-1])
    if normalize:
        h = h / np.sqrt(np.sum(np.abs(replica) ** 2))
    x = np.moveaxis(np.asarray(x), axis, -1)
    shape = (1,) * (x.ndim - 1) + (len(h),)
    y = fftconvolve(x, h.reshape(shape), mode="full", axes=-1)
    y = y[..., len(h) - 1 : len(h) - 1 + x.shape[-1]]
    return np.moveaxis(y, -1, axis)


def compression_gain_theory_db(bandwidth: float, duration: float) -> float:
    """Pulse-compression gain 10 log10(B T) in dB."""
    return float(10 * np.log10(bandwidth * duration))


def lfm_resolution_theory(bandwidth: float, c: float = 1500.0) -> float:
    """Nominal LFM range resolution c / (2B) in metres."""
    return c / (2 * bandwidth)


def cw_resolution_theory(duration: float, c: float = 1500.0) -> float:
    """Nominal CW range resolution c T / 2 in metres."""
    return c * duration / 2


def width_at_level(y: np.ndarray, level_db: float = -3.0, upsample: int = 16) -> float:
    """Width in samples of the main lobe of |y| at `level_db` below its peak (power dB).

    The envelope is interpolated linearly between samples after FFT upsampling of y.
    """
    y = np.asarray(y)
    if upsample > 1:
        from scipy.signal import resample

        y = resample(y, len(y) * upsample)
    p = np.abs(y) ** 2
    k = int(np.argmax(p))
    thr = p[k] * 10 ** (level_db / 10)
    left = k
    while left > 0 and p[left] > thr:
        left -= 1
    right = k
    while right < len(p) - 1 and p[right] > thr:
        right += 1
    # linear interpolation at both crossings
    lx = left + (thr - p[left]) / (p[left + 1] - p[left]) if p[left] <= thr else left
    rx = right - (thr - p[right]) / (p[right - 1] - p[right]) if p[right] <= thr else right
    return float((rx - lx) / upsample)
