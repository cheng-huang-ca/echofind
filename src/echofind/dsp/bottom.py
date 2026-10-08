"""Bottom tracking on echograms (pings x range).

The bottom is an extended echo: once the beam meets the seabed, the return stays high for metres
of range (down-looking: the footprint and sub-bottom tail; forward-looking: grazing
reverberation). A body or a fish is compact. So the tracker looks for persistence, not peaks:

1. Average power into cells of `cell_m` and convert to dB.
2. Forward running median over `persist_m`: med[i] = median(db[i : i + w]). A compact target
   shorter than persist_m / 2 cannot raise this median, so it cannot capture the track.
3. The bottom is where med peaks (beyond r_min), if it stands `min_db` above the profile's 10th
   percentile; its leading edge is the first cell within `backstep_m` before the peak where med
   rises above peak - `edge_db`, shifted by w/2 to undo the forward window's lead.
4. A running median of width `median_pings` across pings removes single-ping outliers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import medfilt


@dataclass(frozen=True)
class BottomConfig:
    r_min: float = 1.0           # ignore the ring-down zone (m)
    cell_m: float = 0.05         # averaging cell
    persist_m: float = 3.0       # bottom must stay high over this range
    min_db: float = 10.0         # persistent level above the 10th-percentile floor
    edge_db: float = 6.0         # leading edge: first cell above peak median - edge_db
    backstep_m: float = 5.0      # search window before the peak for the leading edge
    median_pings: int = 5        # odd; 1 disables cross-ping smoothing


def track_bottom(power: np.ndarray, r: np.ndarray, cfg: BottomConfig | None = None
                 ) -> np.ndarray:
    """Bottom range per ping (NaN where none is found). `power` is pings x range, TVG applied."""
    cfg = cfg or BottomConfig()
    p = np.atleast_2d(np.asarray(power, dtype=float))
    dr = float(np.median(np.diff(r)))
    m = max(int(round(cfg.cell_m / dr)), 1)
    nc = p.shape[-1] // m
    cells = p[:, : nc * m].reshape(p.shape[0], nc, m).mean(-1)
    rc = r[: nc * m : m] + (m - 1) * dr / 2
    db = 10 * np.log10(cells + 1e-30)
    w = max(int(round(cfg.persist_m / cfg.cell_m)), 1)
    start = int(np.searchsorted(rc, cfg.r_min))
    back = max(int(round(cfg.backstep_m / cfg.cell_m)), 1)
    out = np.full(p.shape[0], np.nan)
    if nc - start < w:
        return out
    for i, row in enumerate(db):
        med = np.median(sliding_window_view(row[start:], w), axis=-1)
        k = int(np.argmax(med))
        floor = np.percentile(row[start:], 10)
        if med[k] - floor < cfg.min_db:
            continue
        lo = max(k - back, 0)
        above = np.nonzero(med[lo : k + 1] > med[k] - cfg.edge_db)[0]
        edge = start + lo + above[0] + w // 2
        out[i] = rc[min(edge, nc - 1)]
    if cfg.median_pings > 1 and np.isfinite(out).sum() >= cfg.median_pings:
        filled = np.where(np.isfinite(out), out, np.nanmedian(out))
        sm = medfilt(filled, cfg.median_pings)
        out = np.where(np.isfinite(out), sm, np.nan)
    return out
