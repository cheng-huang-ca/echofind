"""S4: cache frozen ImageNet ResNet-50 features (2048-d, float16) for every candidate snippet.

Resumable: progress is saved every chunk in uatd_resnet50.progress, so a killed run continues
where it stopped. Writes data/processed/uatd_resnet50.npy (N x 2048, float16).

Run: uv run python scripts/s4_resnet_features.py [--batch 256]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import PROC  # noqa: E402

from echofind.models.cnn import ResNet50Features, snippet_batch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=8192)
    args = ap.parse_args()
    torch.set_num_threads(6)
    snips = np.load(PROC / "uatd_snips.npy", mmap_mode="r")
    n = len(snips)
    path, prog = PROC / "uatd_resnet50.npy", PROC / "uatd_resnet50.progress"
    feats = (np.load(path, mmap_mode="r+") if path.exists()
             else np.lib.format.open_memmap(path, "w+", np.float16, (n, 2048)))
    done = int(prog.read_text()) if prog.exists() else 0
    model = ResNet50Features()
    t0 = time.time()
    for c0 in range(done, n, args.chunk):
        c1 = min(n, c0 + args.chunk)
        for s in range(c0, c1, args.batch):
            idx = np.arange(s, min(c1, s + args.batch))
            feats[idx] = model(snippet_batch(snips, idx)).numpy().astype(np.float16)
        feats.flush()
        prog.write_text(str(c1))
        rate = (c1 - done) / (time.time() - t0)
        print(f"{c1}/{n}  {rate:.0f} snippets/s  eta {(n - c1) / rate / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
