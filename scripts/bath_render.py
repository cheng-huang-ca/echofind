"""Bath labelling, step 2-3 support: render 990 kHz legs for review at square 2 cm pixels.

Each leg (reports/w3/bath_legs.csv, 990 kHz, labels "target" and "unlisted") is resampled from
its raw grid (2,047 columns over the slant range; rows spaced by along-track length / rows) to
2 cm x 2 cm pixels, contrast-stretched with one fixed rule (1st to 99.5th percentile), and drawn
with a 1 m grid in metres, so a reviewer can give a box as across/along-track metres. One fixed
display rule for every leg, no model output on the image (protocol step 3).

Writes data/interim/bath_review/<leg_id>.png and data/interim/bath_review/index.csv.
Run: uv run python scripts/bath_render.py [--overlay labels.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import RAW, ROOT  # noqa: E402

from echofind import viz  # noqa: E402

BATH = RAW / "bath" / "sidescan_sonar_images_PRIME_CNN_training"
OUT = ROOT / "data" / "interim" / "bath_review"
CELL = 0.02
plt = viz.plt


def leg_path(r) -> Path:
    if r.label == "target":
        sub = "target" if r.deployment == "main" else "target/extra_target_images_alt_deployment"
    else:
        sub = str(r.freq_khz)
    p = BATH / r.site / r.date / r.run / sub / r.file
    if not p.exists() and r.label == "target":            # target legs also kept under 990/
        p = BATH / r.site / r.date / r.run / "990" / r.file
    return p


def square(r) -> np.ndarray:
    a = np.asarray(Image.open(leg_path(r)).convert("L"), dtype=np.float32)
    H, W = a.shape
    im = Image.fromarray(a.astype(np.uint8)).resize(
        (int(round(r.range_m / CELL)), int(round(r.along_m / CELL))), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def render(r, leg_id: str, boxes: pd.DataFrame | None = None) -> Path:
    b = square(r)
    h, w = b.shape
    fig, ax = plt.subplots(figsize=(w / 100 + 0.8, h / 100 + 0.6))
    ax.imshow(b, cmap="gray", extent=[0, r.range_m, r.along_m, 0],
              vmin=np.percentile(b, 1), vmax=np.percentile(b, 99.5))
    ax.set_xticks(np.arange(0, r.range_m + 0.01, 1))
    ax.set_yticks(np.arange(0, r.along_m + 0.01, 1))
    ax.grid(color="#e34948", lw=0.3, alpha=0.45)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{leg_id}  {r.site} {r.date} {r.run} {r.side} leg {r.leg} ({r.label})",
                 fontsize=8)
    if boxes is not None:
        for b_ in boxes.itertuples():
            ax.add_patch(plt.Rectangle((b_.x0_m, b_.y0_m), b_.x1_m - b_.x0_m, b_.y1_m - b_.y0_m,
                                       fill=False, ec="#1baf7a", lw=1.5))
            ax.text(b_.x0_m, b_.y0_m - 0.15, b_.grade, color="#1baf7a", fontsize=8)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{leg_id}{'_boxed' if boxes is not None else ''}.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def legs() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "reports" / "w3" / "bath_legs.csv")
    df = df[(df.freq_khz == 990) & df.label.isin(["target", "unlisted"])].copy()
    order = {"target": 0, "unlisted": 1}
    df = df.sort_values(["label", "site", "date", "run", "deployment", "side", "leg"],
                        key=lambda s: s.map(order) if s.name == "label" else s)
    df["leg_id"] = [f"L{i:03d}" for i in range(len(df))]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", help="labels CSV: draw boxes on the legs that have them")
    args = ap.parse_args()
    v = viz.setup()
    _ = v
    df = legs()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "index.csv", index=False)
    lab = pd.read_csv(args.overlay) if args.overlay else None
    for r in df.itertuples():
        if lab is not None:
            bx = lab[(lab.leg_id == r.leg_id) & lab.x0_m.notna()]
            if len(bx):
                render(r, r.leg_id, bx)
        else:
            render(r, r.leg_id)
    print(len(df), "legs rendered to", OUT)


if __name__ == "__main__":
    main()
