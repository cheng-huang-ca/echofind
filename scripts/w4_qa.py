"""W4: run the input QA checks (echofind.eval.qa) on every UATD frame.

Writes data/processed/uatd_frame_qa.parquet and reports/w4/qa_summary.csv.
Run: uv run python scripts/w4_qa.py [--jobs J]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import PROC, ROOT  # noqa: E402

from echofind.eval.qa import frame_qa  # noqa: E402
from echofind.io.uatd import load_image  # noqa: E402

OUT = ROOT / "reports" / "w4"


def _one(frame: dict) -> dict:
    return {"frame_id": frame["frame_id"], **frame_qa(load_image(pd.Series(frame)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    fr = pd.read_parquet(PROC / "uatd_frames.parquet")
    with ProcessPoolExecutor(args.jobs) as ex:
        rows = list(ex.map(_one, fr.to_dict("records"), chunksize=32))
    qa = pd.DataFrame(rows).merge(fr[["frame_id", "site", "freq_khz"]], on="frame_id")
    qa.to_parquet(PROC / "uatd_frame_qa.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    fails = [c for c in qa.columns if c.startswith("fail_")]
    s = qa.groupby(["site", "freq_khz"])[fails + ["qa_pass"]].mean()
    s.insert(0, "frames", qa.groupby(["site", "freq_khz"]).size())
    s.to_csv(OUT / "qa_summary.csv", float_format="%.4f")
    print(s.to_string())


if __name__ == "__main__":
    main()
