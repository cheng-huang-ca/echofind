"""Band-pass filtering and IQ (complex baseband) demodulation of real passband echoes."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(x: np.ndarray, fs: float, f_lo: float, f_hi: float, order: int = 6,
             axis: int = -1) -> np.ndarray:
    """Zero-phase Butterworth band-pass (forward-backward, so group delay is zero)."""
    sos = butter(order, [f_lo, f_hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=axis)


def iq_demodulate(x: np.ndarray, fs: float, fc: float, bandwidth: float, decimate: int = 1,
                  order: int = 6, t0: float = 0.0, axis: int = -1) -> np.ndarray:
    """Complex baseband of a real passband signal.

    Mixing Re{s e^{j w t}} with 2 e^{-j w t} gives s + s* e^{-2 j w t}; a zero-phase low-pass at
    about 0.6 B removes the 2 fc image and returns s. The result is then decimated by `decimate`
    (keep fs/decimate above B).
    """
    x = np.moveaxis(np.asarray(x, dtype=float), axis, -1)
    t = t0 + np.arange(x.shape[-1]) / fs
    mixed = 2 * x * np.exp(-2j * np.pi * fc * t)
    cutoff = min(0.6 * bandwidth, 0.45 * fs, 0.9 * fc)
    sos = butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    bb = sosfiltfilt(sos, mixed, axis=-1)[..., ::decimate]
    return np.moveaxis(bb, -1, axis)
