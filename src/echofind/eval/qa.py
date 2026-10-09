"""Input QA checks on a sonar frame, run before the detector (W4 safety).

Each check returns a number and a pass/fail against a fixed limit. A failed check means the
detector's output is not trustworthy for that frame, and the device should say so.

- saturation: share of samples at the top code (255). Clipped echoes lose their shape and
  level, so level and extent features break. Limit 0.5% of samples.
- dead beams: share of beams that are zero over the whole range (a failed channel, or a
  blanked sector). Limit 2% of beams.
- dropped rows: share of range rows that are zero across all beams (a lost ping segment),
  counted only in the middle 90% of the range axis. UATD 720 kHz frames end in 2-3% zero
  rows of export padding (sometimes with a stray non-zero row inside it); zero rows in the
  first and last 5% are reported as `edge_pad` and not failed. Limit 2% of rows.
- low signal: the 99.9th percentile amplitude is below 4 grey levels, so nothing in the frame
  can reach the CFAR SNR gate (gain too low, transducer out of water, or wrong range setting).
"""

from __future__ import annotations

import numpy as np

LIMITS = {"saturation": 0.005, "dead_beams": 0.02, "dropped_rows": 0.02, "low_signal": 4 / 255}


EDGE = 0.05  # share of the range axis at each end where zero rows count as padding


def _rows(zrow: np.ndarray) -> dict:
    """Split all-zero rows into edge padding (first/last 5% of range) and interior drops."""
    n = len(zrow)
    e = int(np.ceil(EDGE * n))
    edge = int(zrow[:e].sum() + zrow[n - e:].sum())
    return {"dropped_rows": float(zrow[e: n - e].sum() / n), "edge_pad": edge / n}


def frame_qa(a: np.ndarray) -> dict:
    """QA metrics and pass flags for one frame (amplitude in [0, 1], rows = range)."""
    a = np.asarray(a, dtype=np.float32)
    zero = a <= 0.5 / 255
    out = {
        "saturation": float(np.mean(a >= 254.5 / 255)),
        "dead_beams": float(np.mean(zero.all(axis=0))),
        **_rows(zero.all(axis=1)),
        "p999": float(np.quantile(a, 0.999)),
    }
    out["fail_saturation"] = out["saturation"] > LIMITS["saturation"]
    out["fail_dead_beams"] = out["dead_beams"] > LIMITS["dead_beams"]
    out["fail_dropped_rows"] = out["dropped_rows"] > LIMITS["dropped_rows"]
    out["fail_low_signal"] = out["p999"] < LIMITS["low_signal"]
    out["qa_pass"] = not any(out[k] for k in out if k.startswith("fail_"))
    return out
