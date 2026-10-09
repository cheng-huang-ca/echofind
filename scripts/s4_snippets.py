"""S4: cut CNN inputs for every W3 candidate (rows aligned with uatd_candidates.parquet).

From each frame's range-normalised grid (echofind.features.fls.make_grid, the same view R2's
features use):
- snippet: 64 range cells (2.56 m at 4 cm) x 32 beam cells around the candidate peak, peak at
  range row 24 so 1.6 m behind the target (its shadow) is inside; 10 log10(q) clipped to
  [-5, 35] dB and stored as uint8 (0..255). Cells outside the frame are 0 dB (background).
- profile: 128 range cells from 32 before the peak, max over the peak beam +/- 1 (the 1-D
  echo a single-beam handheld would see), same dB scale, float16.

Writes data/processed/uatd_snips.u8 (N x 64 x 32) and uatd_profiles.f16 (N x 128) as raw
memmaps, plus uatd_snips_meta.json.

Run: uv run python scripts/s4_snippets.py [--jobs J]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import PROC  # noqa: E402

from echofind.features.fls import make_grid  # noqa: E402
from echofind.io.uatd import load_image  # noqa: E402

H, W, PRE = 64, 32, 24
PL, PPRE = 128, 32
LO, HI = -5.0, 35.0


def _db(q):
    return np.clip(10 * np.log10(np.maximum(q, 1e-12)), LO, HI)


def _one(args):
    frame, rows = args
    g = make_grid(load_image(pd.Series(frame)), frame["range_m"], frame["azimuth_deg"],
                  frame["freq_khz"])
    q = _db(g.q)
    Hq, Wq = q.shape
    pad = np.zeros((Hq + 2 * PL, Wq + 2 * W), np.float32)     # 0 dB = background
    pad[PL: PL + Hq, W: W + Wq] = q
    snips = np.empty((len(rows), H, W), np.uint8)
    profs = np.empty((len(rows), PL), np.float16)
    for i, (pr, pb) in enumerate(rows):
        r0, b0 = PL + pr - PRE, W + pb - W // 2
        s = pad[r0: r0 + H, b0: b0 + W]
        snips[i] = np.round((s - LO) / (HI - LO) * 255).astype(np.uint8)
        p0 = PL + pr - PPRE
        profs[i] = pad[p0: p0 + PL, W + pb - 1: W + pb + 2].max(axis=1)
    return snips, profs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    fr = pd.read_parquet(PROC / "uatd_frames.parquet").set_index("frame_id", drop=False)
    ca = pd.read_parquet(PROC / "uatd_candidates.parquet", columns=["frame_id", "peak_r",
                                                                    "peak_b"])
    n = len(ca)
    snips = np.lib.format.open_memmap(PROC / "uatd_snips.npy", "w+", np.uint8, (n, H, W))
    profs = np.lib.format.open_memmap(PROC / "uatd_profiles.npy", "w+", np.float16, (n, PL))
    groups = ca.groupby("frame_id").indices
    keys = list(groups)
    jobs = [(fr.loc[k].to_dict(), ca[["peak_r", "peak_b"]].to_numpy()[groups[k]]) for k in keys]
    with ProcessPoolExecutor(args.jobs) as ex:
        for k, (s, p) in zip(keys, ex.map(_one, jobs, chunksize=16), strict=True):
            snips[groups[k]] = s
            profs[groups[k]] = p
    snips.flush()
    profs.flush()
    (PROC / "uatd_snips_meta.json").write_text(json.dumps(
        {"n": n, "snip_shape": [H, W], "peak_row": PRE, "profile_len": PL, "profile_pre": PPRE,
         "db_range": [LO, HI], "source": "fls.make_grid range-normalised power"}, indent=2))
    print("snippets", snips.shape, "profiles", profs.shape)


if __name__ == "__main__":
    main()
