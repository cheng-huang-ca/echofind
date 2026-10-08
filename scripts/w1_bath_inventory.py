"""W1: inventory the Bath side-scan archive: which folders kept a target/ subdirectory.

Most labels in the archive were lost (its README), so W3 needs to know where labelled target
examples survive, per site and frequency (450 vs 990 kHz). This reads only the zip's central
directory, either from the local archive/extraction or remotely with HTTP Range requests
(about 1 MB instead of 6 GB), and writes reports/bath_inventory.csv.

Run: uv run python scripts/w1_bath_inventory.py [--remote]
"""

from __future__ import annotations

import argparse
import io
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
URL = "https://researchdata.bath.ac.uk/1467/1/sidescan_sonar_images_PRIME_CNN_training.zip"
LOCAL = REPO / "data" / "raw" / "bath"
OUT = REPO / "reports" / "bath_inventory.csv"
IMAGE = re.compile(r"\.(png|jpe?g|bmp|tiff?)$", re.I)


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP Range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def list_paths(remote: bool) -> list[str]:
    if remote:
        with zipfile.ZipFile(io.BufferedReader(HttpFile(URL), 1 << 20)) as z:
            return z.namelist()
    zips = list(LOCAL.glob("*.zip"))
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            return z.namelist()
    return [str(p.relative_to(LOCAL)) for p in LOCAL.rglob("*") if p.is_file()]


def inventory(paths: list[str]) -> pd.DataFrame:
    """One row per image folder: image count, whether it sits under a target/ directory,
    and the frequency and survey tokens found in its path."""
    counts: Counter[tuple[str, bool]] = Counter()
    for p in paths:
        if not IMAGE.search(p):
            continue
        parts = PurePosixPath(p).parts
        is_target = any(s.lower() in {"target", "targets"} for s in parts[:-1])
        survey = "/".join(s for s in parts[:-1] if s.lower() not in {"target", "targets"})
        counts[(survey, is_target)] += 1
    rows = []
    for (survey, is_target), n in sorted(counts.items()):
        freq = re.search(r"(450|990)", survey)
        date = re.search(r"\d{1,2}_[a-z]{3}_\d{4}", survey, re.I)
        rows.append({"folder": survey, "subset": "target" if is_target else "other",
                     "images": n, "freq_khz": int(freq.group(1)) if freq else None,
                     "survey_date": date.group(0) if date else None})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true", help="read the zip directory over HTTP")
    args = ap.parse_args()
    df = inventory(list_paths(args.remote))
    df.to_csv(OUT, index=False)
    with_t = df[df.subset == "target"]
    print(f"{len(df)} folder rows, {df.images.sum()} images; "
          f"{with_t.folder.nunique()} folders keep target/ ({with_t.images.sum()} images)")


if __name__ == "__main__":
    main()
