"""Range from travel time and time-varying gain (TVG).

Point targets need 40 log10 r + 2 alpha r (two-way spherical spreading and absorption), volume
backscatter 20 log10 r + 2 alpha r, and seabed backscatter at grazing incidence falls near
30 log10 r. alpha is the absorption coefficient in dB/m.
"""

from __future__ import annotations

import numpy as np


def range_from_time(t: np.ndarray | float, c: float = 1500.0) -> np.ndarray | float:
    """Range r = c t / 2 for two-way travel time t."""
    return c * np.asarray(t) / 2


def range_axis(n: int, fs: float, c: float = 1500.0, t0: float = 0.0) -> np.ndarray:
    """Range of each of n samples at rate fs, with sample 0 at time t0."""
    return range_from_time(t0 + np.arange(n) / fs, c)


def tvg_gain_db(r: np.ndarray, alpha_db_per_m: float, spreading: float = 40.0,
                r_min: float = 1.0) -> np.ndarray:
    """TVG in dB: spreading log10(r) + 2 alpha r, with r clipped below at r_min."""
    r = np.maximum(np.asarray(r, dtype=float), r_min)
    return spreading * np.log10(r) + 2 * alpha_db_per_m * r


def apply_tvg(power: np.ndarray, r: np.ndarray, alpha_db_per_m: float, spreading: float = 40.0,
              r_min: float = 1.0, axis: int = -1) -> np.ndarray:
    """Multiply a power (|x|^2) profile by the TVG gain along `axis`."""
    g = 10 ** (tvg_gain_db(r, alpha_db_per_m, spreading, r_min) / 10)
    shape = [1] * np.ndim(power)
    shape[axis] = -1
    return power * g.reshape(shape)
