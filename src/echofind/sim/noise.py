"""Ambient noise for the simulator: white Gaussian, or real MBARI hydrophone recordings.

MBARI Pacific Sound 256 kHz files are 24-bit mono WAV with no published sensitivity, so levels
are dB re full scale. `mbari_baseband` brings a segment to complex baseband about fc and scales
it so its *floor* matches a requested spectrum level NL (dB re 1 uPa^2/Hz). The floor is
estimated from the median of |n|^2 (median / ln 2 is the mean of an exponential), so clicks,
ship passages and other transients stay as excess above the Gaussian-equivalent floor instead
of being averaged into it. Pass `sensitivity_db` to use a real calibration instead.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from ..dsp.demod import iq_demodulate
from .clutter import complex_gaussian


def white_noise(n: int, nl_db: float, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Complex baseband white noise with spectrum level NL: variance 10^(NL/10) * fs."""
    return np.sqrt(10 ** (nl_db / 10) * fs) * complex_gaussian(n, rng)


def read_wav_segment(path: str | Path, start_s: float, duration_s: float) -> tuple[int, np.ndarray]:
    """Read part of a PCM WAV (8/16/24/32-bit) as float in units of full scale (+/-1)."""
    with wave.open(str(path), "rb") as w:
        fs, width, nch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        w.setpos(int(start_s * fs))
        raw = w.readframes(int(duration_s * fs))
    if width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
             | (b[:, 2].astype(np.int32) << 16))
        x = np.where(x >= 1 << 23, x - (1 << 24), x).astype(float) / (1 << 23)
    else:
        dt = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
        x = np.frombuffer(raw, dtype=dt).astype(float)
        x = (x - 128) / 128 if width == 1 else x / float(np.iinfo(dt).max)
    if nch > 1:
        x = x.reshape(-1, nch)[:, 0]
    return fs, x


def mbari_baseband(path: str | Path, start_s: float, n: int, fs_out: float, fc: float,
                   bandwidth: float, nl_db: float | None = None,
                   sensitivity_db: float | None = None) -> np.ndarray:
    """n samples of real hydrophone noise as complex baseband about fc at rate fs_out.

    Scaling: if sensitivity_db (dB re full scale per uPa) is given, levels are absolute;
    otherwise the median-based floor is set to spectrum level nl_db.
    """
    fs_in = 256_000
    dur = n / fs_out + 0.02
    fs_in, x = read_wav_segment(path, start_s, dur)
    if fc + bandwidth / 2 >= fs_in / 2:
        raise ValueError(f"fc + B/2 = {fc + bandwidth / 2:.0f} Hz is above Nyquist {fs_in / 2}")
    x = x - x.mean()  # the recordings carry a large DC offset
    bb = iq_demodulate(x, fs_in, fc, bandwidth)
    # rational resampling to fs_out
    from fractions import Fraction

    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    bb = resample_poly(bb, fr.numerator, fr.denominator)
    m = int(0.01 * fs_out)  # drop filter edge
    bb = bb[m : m + n]
    if len(bb) < n:
        raise ValueError("segment too short")
    if sensitivity_db is not None:
        return bb / 10 ** (sensitivity_db / 20)
    if nl_db is None:
        raise ValueError("give nl_db or sensitivity_db")
    floor = np.median(np.abs(bb) ** 2) / np.log(2)
    # the IQ filter passes ~1.2 B of the band, so the floor variance maps to NL over fs_out:
    # rescale so the in-band density equals NL, i.e. variance = NL * fs_out * (in-band fraction)
    target = 10 ** (nl_db / 10) * min(1.2 * bandwidth, fs_out)
    return bb * np.sqrt(target / floor)
