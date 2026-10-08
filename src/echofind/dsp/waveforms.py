"""Transmit waveforms: CW and LFM pulses, as complex baseband or real passband.

Baseband convention: s(t) = a(t) exp(j phi(t)) with the carrier removed, so the passband pulse
is Re{s(t) exp(j 2 pi fc t)}. An LFM baseband sweep runs from -B/2 to +B/2 about the carrier.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import windows


def pulse_time(duration: float, fs: float) -> np.ndarray:
    """Sample times 0 .. duration (exclusive) at rate fs."""
    n = max(int(round(duration * fs)), 1)
    return np.arange(n) / fs


def _taper(n: int, taper: str | None, alpha: float = 0.1) -> np.ndarray:
    if taper is None or taper == "rect":
        return np.ones(n)
    if taper == "tukey":
        return windows.tukey(n, alpha)
    if taper == "hann":
        return windows.hann(n)
    raise ValueError(f"unknown taper {taper!r}")


def cw_pulse(duration: float, fs: float, taper: str | None = None) -> np.ndarray:
    """Complex baseband CW pulse of the given duration (unit amplitude)."""
    t = pulse_time(duration, fs)
    return _taper(len(t), taper).astype(complex)


def lfm_pulse(bandwidth: float, duration: float, fs: float, taper: str | None = None) -> np.ndarray:
    """Complex baseband LFM pulse sweeping -B/2 .. +B/2 over the duration (unit amplitude).

    Instantaneous frequency f(t) = -B/2 + (B/T) t, so phi(t) = 2 pi (-B/2 t + B t^2 / (2T)).
    """
    t = pulse_time(duration, fs)
    k = bandwidth / duration
    phase = 2 * np.pi * (-bandwidth / 2 * t + 0.5 * k * t**2)
    return _taper(len(t), taper) * np.exp(1j * phase)


def to_passband(baseband: np.ndarray, fs: float, fc: float, t0: float = 0.0) -> np.ndarray:
    """Real passband signal Re{s(t) exp(j 2 pi fc t)}; fs must exceed 2 (fc + B/2)."""
    t = t0 + np.arange(len(baseband)) / fs
    return np.real(baseband * np.exp(2j * np.pi * fc * t))


def pulse_energy(s: np.ndarray, fs: float) -> float:
    """Pulse energy E = integral |s(t)|^2 dt."""
    return float(np.sum(np.abs(s) ** 2) / fs)
