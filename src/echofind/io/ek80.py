"""Read Simrad EK80 .raw files (NOAA HB2305) into calibrated Sv and pre-TVG level.

On HB2305 every file holds complex FM samples for ES38 (34-45 kHz), ES70 (45-90 kHz) and ES200
(160-260 kHz) in Beam_group1, and a CW 18 kHz channel (ES18) in Beam_group2. echopype does the
pulse compression and calibration; this module only picks channels and adds the level before
time-varied gain (TVG), which is what the receiver actually sees.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", module="echopype")


def open_ek80(path: str | Path):
    import echopype as ep

    return ep.open_raw(str(path), sonar_model="EK80")


def channel_label(channel: str) -> str:
    """'WBT 400503-15 ES70-7C_ES' -> 'ES70'."""
    return channel.split()[-1].split("-")[0].split("_")[0]


def compute_sv(ed, waveform: str = "BB") -> xr.Dataset:
    """Calibrated Sv (dB re 1 m^-1) for FM channels (waveform='BB') or the 18 kHz CW channel.

    Adds `level_pre_tvg` = Sv - 20 log10 r - 2 alpha r, i.e. Sv with the TVG removed (relative
    dB; constant calibration terms are kept). Ranges under 1 m are dropped (near field).
    """
    import echopype as ep

    has_complex = "backscatter_i" in ed["Sonar/Beam_group2"] if waveform == "CW" else True
    encode = "complex" if (waveform == "BB" or has_complex) else "power"
    ds = ep.calibrate.compute_Sv(ed, waveform_mode=waveform, encode_mode=encode)
    ds = ds.assign_coords(label=("channel", [channel_label(c) for c in ds.channel.values]))
    r = ds["echo_range"]
    alpha = ds["sound_absorption"]
    tvg = 20 * np.log10(r.where(r >= 1.0)) + 2 * alpha * r
    ds["tvg_db"] = tvg
    ds["level_pre_tvg"] = ds["Sv"] - tvg
    return ds


def fm_bandwidth(ed) -> dict[str, float]:
    """Transmit bandwidth (Hz) per FM channel label, from Beam_group1."""
    b = ed["Sonar/Beam_group1"]
    return {channel_label(str(c)): abs(float(b.transmit_frequency_stop.sel(channel=c).max())
                                       - float(b.transmit_frequency_start.sel(channel=c).max()))
            for c in b.channel.values}


def impulse_mask(sv: np.ndarray, dr: float, threshold_db: float = 10.0,
                 smooth_m: float = 0.5) -> np.ndarray:
    """True where a ping exceeds both neighbouring pings by threshold_db at the same range
    (impulsive noise test of Ryan et al. 2015, ICES J. Mar. Sci. 72(8)), after a smooth_m
    running mean in range so a single fish or the seabed does not trip it."""
    k = max(1, int(round(smooth_m / dr)))
    lin = 10 ** (sv / 10)  # NaN (beyond a ping's record) stays NaN and never compares True
    kern = np.ones(k) / k
    sm = np.apply_along_axis(lambda x: np.convolve(x, kern, mode="same"), 1, lin)
    sm_db = 10 * np.log10(sm)
    mask = np.zeros(sv.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        mask[1:-1] = ((sm_db[1:-1] - sm_db[:-2] > threshold_db)
                      & (sm_db[1:-1] - sm_db[2:] > threshold_db))
    return mask


def bottom_index(sv: np.ndarray, r: np.ndarray, min_range: float = 5.0,
                 max_range: float = 150.0, max_jump_m: float = 1.5) -> np.ndarray:
    """Per-ping bottom sample, robust to interference spikes and fish.

    Pick the strongest sample between min_range and max_range in each ping (the cap keeps
    TVG-amplified noise at long range from winning; HB2305 works on a 30-60 m shelf), take a
    9-ping running median of those ranges as the seabed track, and re-pick any ping more than
    max_jump_m off the track as its strongest sample within +-1 m of the track.
    Returns -1 for an all-NaN ping.
    `sv` is (ping, sample) in dB and `r` is the range in m, (sample,).
    """
    from scipy.ndimage import median_filter

    r = np.asarray(r)
    dr = float(np.nanmedian(np.diff(r)))
    inside = (r >= min_range) & (r <= max_range)
    x = np.where(np.isfinite(sv) & inside[None, :], sv, -np.inf)
    ok = np.isfinite(x).any(axis=1)
    pick = np.where(ok, np.argmax(x, axis=1), -1)
    rp = np.where(pick >= 0, r[np.clip(pick, 0, None)], np.nan)
    track = median_filter(np.nan_to_num(rp, nan=np.nanmedian(rp)), size=9, mode="nearest")
    w = int(round(1.0 / dr))
    for i in np.where(ok & (np.abs(rp - track) > max_jump_m))[0]:
        c = int(np.searchsorted(r, track[i]))
        lo, hi = max(c - w, 0), min(c + w, sv.shape[1])
        pick[i] = lo + int(np.argmax(x[i, lo:hi]))
    return pick
