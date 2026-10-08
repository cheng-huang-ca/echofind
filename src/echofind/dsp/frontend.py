"""The front end as one call: passband or IQ echoes in, range profiles and candidates out.

Order of blocks (see reports/w2_frontend_design.md):
  band-pass -> IQ demodulation -> matched filter -> |.|^2 -> CFAR (on un-TVG'd power)
  -> TVG (for levels and bottom tracking) -> bottom track -> candidate list.

CFAR runs before TVG on purpose: in the noise-limited regime the matched-filter noise is flat in
range, and CFAR adapts locally anyway; TVG would only add a range slope inside the window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bottom import BottomConfig, track_bottom
from .candidates import candidates
from .cfar import CFARConfig, cfar_detect, scale_factor
from .demod import bandpass, iq_demodulate
from .matched_filter import matched_filter
from .tvg import apply_tvg, range_axis


@dataclass(frozen=True)
class FrontEndConfig:
    c: float = 1500.0
    alpha_db_per_m: float = 0.0
    tvg_spreading: float = 40.0
    cfar: CFARConfig = field(default_factory=CFARConfig)
    bottom: BottomConfig | None = field(default_factory=BottomConfig)
    bottom_guard_m: float = 0.3
    # down-looking echosounders: nothing beyond the bottom is in the water column, so drop it.
    # Forward-looking or tilted beams: the seabed beyond the bottom-entry range is still search
    # area (a body lies there), so keep it and let CFAR work against the reverberation.
    exclude_beyond_bottom: bool = False
    min_snr_db: float | None = None     # post-CFAR gate on candidate SNR


@dataclass
class FrontEndResult:
    r: np.ndarray            # range axis (m)
    iq: np.ndarray           # beams x range complex baseband (before matched filter)
    mf: np.ndarray           # matched-filter output, complex
    power: np.ndarray        # |mf|^2
    tvg_db: np.ndarray       # 10 log10 of TVG-corrected power
    det: np.ndarray          # CFAR detections
    thr: np.ndarray          # CFAR thresholds
    bottom_m: np.ndarray     # bottom range per beam/ping (NaN if none)
    candidates: pd.DataFrame


def demodulate(x: np.ndarray, fs: float, fc: float, bandwidth: float, decimate: int = 1,
               axis: int = -1) -> np.ndarray:
    """Band-pass about fc +/- 0.6 B, then IQ-demodulate and decimate."""
    lo, hi = max(fc - 0.6 * bandwidth, 1.0), min(fc + 0.6 * bandwidth, 0.49 * fs)
    xb = bandpass(x, fs, lo, hi, axis=axis)
    return iq_demodulate(xb, fs, fc, bandwidth, decimate=decimate, axis=axis)


def process_iq(iq: np.ndarray, fs: float, replica: np.ndarray,
               cfg: FrontEndConfig | None = None, t0: float = 0.0,
               bearing_deg: np.ndarray | None = None) -> FrontEndResult:
    """Run matched filter, CFAR, TVG, bottom tracking and candidates on beams x range IQ."""
    cfg = cfg or FrontEndConfig()
    iq = np.atleast_2d(iq)
    r = range_axis(iq.shape[-1], fs, cfg.c, t0)
    mf = matched_filter(iq, replica)
    power = np.abs(mf) ** 2
    det, thr = cfar_detect(power, cfg.cfar)
    tvg = apply_tvg(power, r, cfg.alpha_db_per_m, cfg.tvg_spreading)
    tvg_db = 10 * np.log10(tvg + 1e-30)
    bottom = (track_bottom(tvg, r, cfg.bottom) if cfg.bottom is not None
              else np.full(iq.shape[0], np.nan))
    cands = candidates(power, det, thr, r, bearing_deg,
                       bottom if cfg.exclude_beyond_bottom else None, cfg.bottom_guard_m,
                       scale_factor(cfg.cfar), cfg.min_snr_db)
    return FrontEndResult(r, iq, mf, power, tvg_db, det, thr, bottom, cands)


def sum_adjacent_beams(x: np.ndarray, k: int, coherent: bool = False) -> np.ndarray:
    """Emulate a k-times wider beam by summing k adjacent beams (axis 0, no overlap).

    coherent=True sums complex samples (a phased sub-array, beam steered to the centre);
    coherent=False sums power, which is what a wider single-element beam sees from
    uncorrelated scatterers. Trailing beams that do not fill a group are dropped.
    """
    x = np.asarray(x)
    m = (x.shape[0] // k) * k
    g = x[:m].reshape((m // k, k) + x.shape[1:])
    return g.sum(axis=1) if coherent else (np.abs(g) ** 2 if np.iscomplexobj(g) else g).sum(1)
