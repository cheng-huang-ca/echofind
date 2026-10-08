"""Shared helpers for W2 scripts: output paths, run metadata and figure style."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "w2"
FIG = ROOT / "reports" / "figures"
RAW = ROOT / "data" / "raw"

# categorical slots in fixed order (dataviz reference palette, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 5, "font.size": 10,
        "legend.frameon": False, "figure.dpi": 110,
    })


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def manifest_hash() -> str:
    m = RAW / "MANIFEST.json"
    return hashlib.sha256(m.read_bytes()).hexdigest()[:16] if m.exists() else "none"


def write_meta(name: str, config: dict):
    """Record config, git commit and data manifest hash next to the results."""
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"script": name, "git_commit": git_commit(), "data_manifest": manifest_hash(),
            "config": config}
    (OUT / f"{name}_meta.json").write_text(json.dumps(meta, indent=2, default=str))


def savefig(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
