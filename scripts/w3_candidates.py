"""W3: build the UATD candidate table (proposals, ground-truth matching, R1/R2 features).

Reads data/raw/uatd (all three splits are pooled and re-split by group later) and writes
data/processed/uatd_candidates.parquet, data/processed/uatd_frames.parquet and
data/processed/uatd_objects.parquet, plus reports/w3/candidates_meta.json.

Run: uv run python scripts/w3_candidates.py [--limit N] [--jobs J]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import OUT, PROC, RAW, write_meta  # noqa: E402

from echofind.features.fls import ProposalConfig, frame_candidates  # noqa: E402
from echofind.io.uatd import load_image, load_index  # noqa: E402

CFG = ProposalConfig()
_OBJ: pd.DataFrame | None = None


def _init(objects: pd.DataFrame) -> None:
    global _OBJ
    _OBJ = objects


def _one(frame: dict) -> pd.DataFrame:
    f = pd.Series(frame)
    objs = _OBJ[_OBJ.frame_id == f.frame_id]
    c, _ = frame_candidates(load_image(f), f, objs, CFG)
    if c.empty:
        return c
    c.attrs = {}
    c.insert(0, "frame_id", f.frame_id)
    return c.drop(columns=["label_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    frames, objects = load_index(RAW / "uatd")
    objects.index.name = "obj_idx"
    if args.limit:
        frames = frames.sample(args.limit, random_state=0)
    PROC.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(objects,)) as ex:
        parts = list(ex.map(_one, frames.to_dict("records"), chunksize=16))
    cands = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    frames.to_parquet(PROC / "uatd_frames.parquet")
    objects.reset_index().to_parquet(PROC / "uatd_objects.parquet")
    cands.to_parquet(PROC / "uatd_candidates.parquet")
    summ = {
        "frames": len(frames), "objects": len(objects), "candidates": len(cands),
        "candidates_per_frame": round(len(cands) / len(frames), 2),
        "objects_proposed_by_class": (
            objects.assign(hit=objects.index.isin(cands.obj_idx))
            .groupby("cls").hit.mean().round(4).to_dict()),
        "seconds": round(time.time() - t, 1),
    }
    write_meta("candidates", {"proposal": asdict(CFG), **summ})
    (OUT / "candidates_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
