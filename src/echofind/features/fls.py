"""Candidates and physics features from forward-looking sonar frames (UATD, W3 rungs R0 to R2).

Pipeline per frame (image rows = range from the sonar outward, columns = beams):

1. Power p = a^2 from the 8-bit amplitude, with a floor of half a grey level so log is finite.
2. Block-average to cells of about 4 cm in range and 1 Gemini beam step (720 kHz) or 2 (1200 kHz).
   The raw rows are about 1 cm apart, far finer than the 1.7 m target, and averaging speckle
   cuts the CFAR false-alarm count (the cells are no longer exponential, so pfa is a knob here,
   not a guarantee; see reports/decisions/203-fls-proposals.md).
3. Range normalisation q = p / b(r), with b(r) the median power across beams at that range row,
   smoothed along range. This is a data-driven TVG: it removes spreading loss and the near-range
   reverberation ramp, and it puts every frame on the same scale (the "normalised" view for H5).
4. OS-CFAR along range in every beam (decision 105), then connected components over
   (range, beam) after a one-cell closing. Each component is one candidate, gated on SNR and
   size. R0 adds a physical size gate on top (models/ladder.py).

Features come in two views: level features from q (normalised) and from p (raw), plus shape
features that do not depend on the view. Absolute range and bearing are deliberately not
features: in UATD the targets sit at a few fixed ranges per session, so they would be shortcuts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage, stats

from echofind.dsp.cfar import CFARConfig, cfar_threshold, scale_factor

FLOOR = (0.5 / 255.0) ** 2
SHADOW_SCAN_M = 3.0  # how far behind a candidate the shadow-length scan looks
SHADOW_GAP = 3       # cells the shadow may start after the candidate's far edge


OS_CFAR = CFARConfig("os", n_ref=32, n_guard=4, pfa=1e-3)


@dataclass(frozen=True)
class ProposalConfig:
    cell_range_m: float = 0.04       # block-average target in range
    beams_per_cell_1200: int = 2
    beams_per_cell_720: int = 1
    bg_smooth_cells: int = 9         # range smoothing of the background row median
    cfar: CFARConfig = OS_CFAR
    min_snr_db: float = 8.0
    min_cells: int = 3
    max_per_frame: int = 60          # keep the strongest components by SNR


DEFAULT = ProposalConfig()


@dataclass
class Grid:
    """Downsampled frame: power, normalised power, and cell geometry."""
    p: np.ndarray        # (range cells, beam cells) raw power
    q: np.ndarray        # range-normalised power
    fr: int              # raw rows per range cell
    fb: int              # raw beams per beam cell
    dr: float            # m per range cell
    dth: float           # rad per beam cell
    r: np.ndarray        # range (m) at each range-cell centre


def _block_mean(x: np.ndarray, fr: int, fb: int) -> np.ndarray:
    h, w = (x.shape[0] // fr) * fr, (x.shape[1] // fb) * fb
    return x[:h, :w].reshape(h // fr, fr, w // fb, fb).mean(axis=(1, 3))


def make_grid(a: np.ndarray, range_m: float, azimuth_deg: float, freq_khz: int,
              cfg: ProposalConfig = DEFAULT) -> Grid:
    dr_raw = range_m / a.shape[0]
    fr = max(1, int(round(cfg.cell_range_m / dr_raw)))
    fb = cfg.beams_per_cell_1200 if freq_khz >= 1000 else cfg.beams_per_cell_720
    p = _block_mean(a.astype(np.float64) ** 2 + FLOOR, fr, fb)
    bg = np.median(p, axis=1)
    bg = ndimage.uniform_filter1d(bg, cfg.bg_smooth_cells, mode="nearest")
    q = p / np.maximum(bg, FLOOR)[:, None]
    dr = dr_raw * fr
    dth = np.deg2rad(azimuth_deg) / a.shape[1] * fb
    return Grid(p, q, fr, fb, dr, dth, (np.arange(p.shape[0]) + 0.5) * dr)


def propose(g: Grid, cfg: ProposalConfig = DEFAULT) -> pd.DataFrame:
    """Candidate components with their bounding boxes (grid and raw-pixel coordinates)."""
    thr = cfar_threshold(g.q.T, cfg.cfar).T          # CFAR runs along the last axis: range
    with np.errstate(invalid="ignore"):
        det = (g.q > thr) & np.isfinite(thr)
    det = ndimage.binary_closing(det, np.ones((3, 3))) | det
    lab, n = ndimage.label(det, np.ones((3, 3)))
    if n == 0:
        return pd.DataFrame()
    s = scale_factor(cfg.cfar)
    noise = np.where(np.isfinite(thr), thr / s, np.nan)
    rows = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == i
        ncell = int(m.sum())
        if ncell < cfg.min_cells:
            continue
        sub = np.where(m, g.q[sl], -np.inf)
        pr, pb = np.unravel_index(np.argmax(sub), sub.shape)
        pr += sl[0].start
        pb += sl[1].start
        nz = noise[pr, pb]
        if not np.isfinite(nz):
            nz = np.nanmedian(noise[sl]) if np.isfinite(noise[sl]).any() else 1.0
        snr = 10 * np.log10(g.q[pr, pb] / nz)
        if snr < cfg.min_snr_db:
            continue
        rows.append({"label_id": i, "r0": sl[0].start, "r1": sl[0].stop, "b0": sl[1].start,
                     "b1": sl[1].stop, "peak_r": pr, "peak_b": pb, "snr_db": float(snr),
                     "n_cells": ncell})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).nlargest(cfg.max_per_frame, "snr_db").reset_index(drop=True)
    # raw-pixel coordinates (x = beam column, y = range row), for matching to the XML boxes
    df["x0"], df["x1"] = df.b0 * g.fb, df.b1 * g.fb
    df["y0"], df["y1"] = df.r0 * g.fr, df.r1 * g.fr
    df["px"], df["py"] = (df.peak_b + 0.5) * g.fb, (df.peak_r + 0.5) * g.fr
    df.attrs["labels"] = lab
    return df


def match(cands: pd.DataFrame, objects: pd.DataFrame, pad: float = 0.1) -> pd.DataFrame:
    """Assign each candidate to the smallest ground-truth box (padded by `pad` of its size)
    that contains the candidate's peak. Adds obj_idx (-1 if none) and cls ('background')."""
    c = cands.copy()
    c["obj_idx"], c["cls"] = -1, "background"
    if c.empty or objects.empty:
        return c
    ob = objects
    x0 = np.minimum(ob.xmin, ob.xmax).to_numpy(float)
    x1 = np.maximum(ob.xmin, ob.xmax).to_numpy(float)
    y0 = np.minimum(ob.ymin, ob.ymax).to_numpy(float)
    y1 = np.maximum(ob.ymin, ob.ymax).to_numpy(float)
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - pad * w, x1 + pad * w, y0 - pad * h, y1 + pad * h
    area = (x1 - x0) * (y1 - y0)
    px, py = c.px.to_numpy()[:, None], c.py.to_numpy()[:, None]
    inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
    a = np.where(inside, area[None, :], np.inf)
    best = np.argmin(a, axis=1)
    hit = np.isfinite(a[np.arange(len(c)), best])
    c.loc[hit, "obj_idx"] = ob.index.to_numpy()[best[hit]]
    c.loc[hit, "cls"] = ob.cls.to_numpy()[best[hit]]
    return c


# ----------------------------------------------------------------------------- features
R1_FEATURES = ["peak_db", "snr_db", "range_ext_m", "cross_ext_m", "n_hl", "hl_spacing_m",
               "shadow_depth_db", "shadow_len_m", "contrast_db", "speckle_c"]
LEVEL_FEATURES = ["peak_db", "contrast_db", "shadow_depth_db", "shadow_len_m", "mean_db",
                  "p50_db", "p90_db", "energy_db", "bg_db", "bg_std_db", "front_contrast_db",
                  "front_back_db", "shadow_min_db", "shadow_frac", "contrast2_db"]
SHAPE_FEATURES = ["snr_db", "range_ext_m", "cross_ext_m", "n_hl", "hl_spacing_m", "speckle_c",
                  "area_m2", "aspect", "fill", "n_cells", "skew", "kurt", "hl2_rel_db",
                  "hl_energy_frac", "range_w3_m", "range_w6_m", "cross_w3_m", "cross_w6_m",
                  "grad_mean", "entropy", "orient", "ecc", "peak_offset"]
R2_FEATURES = SHAPE_FEATURES + LEVEL_FEATURES  # 38, from either the normalised or raw view


def _db(x):
    return 10 * np.log10(np.maximum(x, FLOOR))


def _widths(profile_db: np.ndarray, step: float) -> tuple[float, float]:
    """Width (m) of the contiguous run around the maximum within 3 and 6 dB of it."""
    k = int(np.argmax(profile_db))
    out = []
    for drop in (3.0, 6.0):
        ok = profile_db >= profile_db[k] - drop
        lo = k
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        hi = k
        while hi < len(ok) - 1 and ok[hi + 1]:
            hi += 1
        out.append((hi - lo + 1) * step)
    return out[0], out[1]


def _level(img: np.ndarray, m: np.ndarray, sl, ring_sl, shadow_sl, front_sl, ring2_sl,
           behind_sl) -> dict:
    """Level features of one candidate on one view (power image)."""
    box = img[sl]
    inside = box[m]
    peak = inside.max()
    ring = img[ring_sl].copy()
    rm = np.ones(ring.shape, bool)
    rm[sl[0].start - ring_sl[0].start: sl[0].stop - ring_sl[0].start,
       sl[1].start - ring_sl[1].start: sl[1].stop - ring_sl[1].start] = False
    bgv = ring[rm] if rm.any() else ring.ravel()
    bg = np.median(bgv)  # robust: the ring may hold a neighbouring highlight
    ring2 = img[ring2_sl]
    sh = img[shadow_sl]
    fr = img[front_sl]
    # median over range rows: the target's own range sidelobes spill into the first rows
    sh_mean = np.median(sh.mean(axis=1)) if sh.size else bg
    fr_mean = np.median(fr.mean(axis=1)) if fr.size else bg
    # shadow length: consecutive range cells behind the target whose beam-mean is 3 dB below
    # bg, starting within SHADOW_GAP cells of the target's far edge
    run = 0
    behind = img[behind_sl]
    if behind.size:
        below = _db(behind.mean(axis=1)) < _db(bg) - 3
        start = int(np.argmax(below[:SHADOW_GAP])) if below[:SHADOW_GAP].any() else len(below)
        while start + run < len(below) and below[start + run]:
            run += 1
    box_db = _db(box)
    return {
        "peak_db": _db(peak), "contrast_db": _db(inside.mean()) - _db(bg),
        "shadow_depth_db": _db(bg) - _db(sh_mean), "shadow_len_m": float(run),
        "mean_db": float(box_db.mean()), "p50_db": float(np.percentile(box_db, 50)),
        "p90_db": float(np.percentile(box_db, 90)), "energy_db": _db(inside.sum()),
        "bg_db": _db(bg), "bg_std_db": float(np.std(_db(bgv))),
        "front_contrast_db": _db(fr_mean) - _db(bg), "front_back_db": _db(fr_mean) - _db(sh_mean),
        "shadow_min_db": float(_db(sh.mean(axis=1)).min()) - _db(bg) if sh.size else 0.0,
        "shadow_frac": float(np.mean(_db(sh) < _db(bg) - 3)) if sh.size else 0.0,
        "contrast2_db": _db(inside.mean()) - _db(ring2.mean()),
    }


def features(g: Grid, cands: pd.DataFrame, ring: int = 6, ring2: int = 15) -> pd.DataFrame:
    """R2 feature table (normalised view, plus `<name>_raw` level features from raw power)."""
    if cands.empty:
        return cands
    lab = cands.attrs["labels"]
    H, W = g.q.shape
    rows = []
    for c in cands.itertuples():
        sl = (slice(c.r0, c.r1), slice(c.b0, c.b1))
        m = lab[sl] == c.label_id
        nr, nb = c.r1 - c.r0, c.b1 - c.b0
        rc = g.r[min(int(c.peak_r), H - 1)]
        dx = rc * g.dth                                   # cross-range metres per beam cell
        ring_sl = (slice(max(c.r0 - ring, 0), min(c.r1 + ring, H)),
                   slice(max(c.b0 - ring, 0), min(c.b1 + ring, W)))
        ring2_sl = (slice(max(c.r0 - ring2, 0), min(c.r1 + ring2, H)),
                    slice(max(c.b0 - ring2, 0), min(c.b1 + ring2, W)))
        L = max(2 * nr, 5)
        shadow_sl = (slice(c.r1, min(c.r1 + L, H)), slice(c.b0, c.b1))
        behind_sl = (slice(c.r1, min(c.r1 + int(SHADOW_SCAN_M / g.dr), H)), slice(c.b0, c.b1))
        front_sl = (slice(max(c.r0 - L, 0), c.r0), slice(c.b0, c.b1))

        qbox_db = _db(g.q[sl])
        inside = g.q[sl][m]
        pk = qbox_db.max()
        # highlights: 3x3 local maxima within 6 dB of the peak, inside the component
        mx = ndimage.maximum_filter(qbox_db, size=3)
        hl = np.argwhere((qbox_db == mx) & (qbox_db >= pk - 6) & m)
        hl_m = hl * np.array([g.dr, dx])
        spacing = float(np.max(np.linalg.norm(hl_m[:, None] - hl_m[None], axis=-1))) \
            if len(hl) > 1 else 0.0
        hl_vals = np.sort(qbox_db[hl[:, 0], hl[:, 1]])[::-1] if len(hl) else np.array([pk])
        rr, bb = np.nonzero(m)
        ys, xs = rr * g.dr, bb * dx
        cov = np.cov(np.vstack([ys, xs])) if len(rr) > 2 else np.eye(2) * 1e-6
        ev, evec = np.linalg.eigh(cov + 1e-12 * np.eye(2))
        orient = float(np.degrees(np.arctan2(abs(evec[1, 1]), abs(evec[0, 1]))))
        ecc = float(np.sqrt(1 - ev[0] / ev[1])) if ev[1] > 0 else 0.0
        prof_r = qbox_db.max(axis=1)
        prof_b = qbox_db.max(axis=0)
        rw3, rw6 = _widths(prof_r, g.dr)
        cw3, cw6 = _widths(prof_b, dx)
        padded = np.pad(qbox_db, 1, mode="edge")         # 1-cell-wide boxes have no gradient
        gy, gx = (d[1:-1, 1:-1] for d in np.gradient(padded))
        hist, _ = (np.histogram(qbox_db, bins=16) if np.ptp(qbox_db) > 1e-6
                   else (np.array([qbox_db.size]), None))   # a clipped, flat box
        pdist = hist[hist > 0] / hist.sum()
        cen = np.array([rr.mean(), bb.mean()])
        pk_off = np.linalg.norm((np.array([c.peak_r - c.r0, c.peak_b - c.b0]) - cen)
                                * [g.dr, dx]) / max(nr * g.dr, nb * dx, 1e-6)
        range_ext, cross_ext = nr * g.dr, nb * dx
        flat_ok = np.ptp(qbox_db) > 1e-3 * max(abs(qbox_db).max(), 1.0)
        f = {
            "range_ext_m": range_ext, "cross_ext_m": cross_ext,
            "n_hl": len(hl), "hl_spacing_m": spacing,
            "speckle_c": float(inside.std() / inside.mean()),
            "area_m2": float(m.sum() * g.dr * dx), "aspect": range_ext / max(cross_ext, 1e-6),
            "fill": float(m.mean()),
            "skew": float(stats.skew(qbox_db, axis=None)) if flat_ok else 0.0,
            "kurt": float(stats.kurtosis(qbox_db, axis=None)) if flat_ok else 0.0,
            "hl2_rel_db": float(hl_vals[1] - hl_vals[0]) if len(hl_vals) > 1 else -20.0,
            "hl_energy_frac": float(np.sum(10 ** (hl_vals / 10)) / np.sum(inside)),
            "range_w3_m": rw3, "range_w6_m": rw6, "cross_w3_m": cw3, "cross_w6_m": cw6,
            "grad_mean": float(np.mean(np.hypot(gy, gx))),
            "entropy": float(-np.sum(pdist * np.log2(pdist))),
            "orient": orient, "ecc": ecc, "peak_offset": float(pk_off),
        }
        lv = _level(g.q, m, sl, ring_sl, shadow_sl, front_sl, ring2_sl, behind_sl)
        lv["shadow_len_m"] *= g.dr
        f.update(lv)
        lr = _level(g.p, m, sl, ring_sl, shadow_sl, front_sl, ring2_sl, behind_sl)
        lr["shadow_len_m"] *= g.dr
        f.update({f"{k}_raw": v for k, v in lr.items()})
        rows.append(f)
    return pd.concat([cands.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def frame_candidates(a: np.ndarray, frame: pd.Series, objects: pd.DataFrame,
                     cfg: ProposalConfig = DEFAULT) -> tuple[pd.DataFrame, Grid]:
    """Proposals, ground-truth matching and features for one frame."""
    g = make_grid(a, frame.range_m, frame.azimuth_deg, frame.freq_khz, cfg)
    c = propose(g, cfg)
    if c.empty:
        return c, g
    labels = c.attrs["labels"]
    c = match(c, objects)
    c.attrs["labels"] = labels
    c = features(g, c)
    c["range_m"] = g.r[np.minimum(c.peak_r.to_numpy(), len(g.r) - 1)]
    return c, g
