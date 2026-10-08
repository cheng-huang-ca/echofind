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


def bottom_index(sv: np.ndarray, r: np.ndarray, min_range: float = 5.0,
                 threshold_db: float = -40.0) -> np.ndarray:
    """Per-ping bottom sample: the first sample beyond min_range whose Sv exceeds threshold_db,
    refined to the local maximum within the next 1 m. Returns -1 where nothing qualifies.

    `sv` is (ping, sample) in dB and `r` is the matching range in m (ping, sample) or (sample,).
    """
    r2 = np.broadcast_to(r, sv.shape)
    ok = (r2 >= min_range) & (sv > threshold_db)
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), -1)
    dr = np.nanmedian(np.diff(r2, axis=1))
    win = max(1, int(round(1.0 / dr)))
    out = first.copy()
    for i, j in enumerate(first):
        if j >= 0:
            seg = np.nan_to_num(sv[i, j:j + win], nan=-999)
            out[i] = j + int(np.argmax(seg))
    return out
