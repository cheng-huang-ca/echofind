"""Turn a CFAR detection mask into a candidate list (range, bearing, level, extent).

Detections are grouped into connected components over (beam, range); each component becomes one
candidate described at its peak cell. Cells within `bottom_guard_m` of, or beyond, the tracked
bottom are dropped first, because the bottom echo is not a candidate (a body lying on the bottom
shows as a bump just above the bottom edge, which the guard is sized to keep).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

COLUMNS = ["beam", "bearing_deg", "range_m", "level_db", "snr_db", "range_extent_m",
           "beam_extent", "n_cells"]


def candidates(power: np.ndarray, det: np.ndarray, thr: np.ndarray, r: np.ndarray,
               bearing_deg: np.ndarray | None = None, bottom_m: np.ndarray | None = None,
               bottom_guard_m: float = 0.3, scale_factor: float = 1.0,
               min_snr_db: float | None = None) -> pd.DataFrame:
    """Candidate table from beams x range arrays of power, detections and thresholds.

    snr_db is the peak power over the CFAR noise estimate (threshold / scale_factor).
    min_snr_db drops weak components: CFAR finds echoes, and in real water most of them are
    plankton, bubbles or fish well below a body's echo (see reports/w2_frontend_design.md).
    """
    p = np.atleast_2d(power)
    d = np.atleast_2d(det).copy()
    t = np.atleast_2d(thr)
    if bottom_m is not None:
        b = np.broadcast_to(np.asarray(bottom_m, dtype=float), (p.shape[0],))
        cut = np.where(np.isfinite(b), b - bottom_guard_m, np.inf)
        d &= r[None, :] < cut[:, None]
    if bearing_deg is None:
        bearing_deg = np.zeros(p.shape[0])
    lab, n = ndimage.label(d, structure=np.ones((3, 3)))
    if n == 0:
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    dr = float(np.median(np.diff(r))) if len(r) > 1 else 0.0
    for sl, idx in zip(ndimage.find_objects(lab), range(1, n + 1), strict=True):
        m = lab[sl] == idx
        sub = np.where(m, p[sl], -np.inf)
        bi, ri = np.unravel_index(np.argmax(sub), sub.shape)
        bi += sl[0].start
        ri += sl[1].start
        noise = t[bi, ri] / scale_factor
        rows.append({
            "beam": int(bi),
            "bearing_deg": float(bearing_deg[bi]),
            "range_m": float(r[ri]),
            "level_db": float(10 * np.log10(p[bi, ri])),
            "snr_db": float(10 * np.log10(p[bi, ri] / noise)),
            "range_extent_m": float((sl[1].stop - sl[1].start) * dr),
            "beam_extent": int(sl[0].stop - sl[0].start),
            "n_cells": int(m.sum()),
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    if min_snr_db is not None:
        df = df[df.snr_db >= min_snr_db]
    return df.sort_values(["beam", "range_m"], ignore_index=True)
