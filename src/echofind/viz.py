"""Shared matplotlib style for report figures: one validated categorical order, quiet axes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated palette; see the dataviz reference).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SURFACE = "#fcfcfb"
# One colour per sonar channel, used in every figure.
CHANNEL_COLOR = {"ES18": SERIES[6], "ES38": SERIES[0], "ES70": SERIES[1], "ES200": SERIES[2]}
CHANNEL_ORDER = ["ES18", "ES38", "ES70", "ES200"]
SEASON_COLOR = {"Jan": SERIES[0], "Apr": SERIES[2], "Jul": SERIES[1], "Sep": SERIES[6]}
MODEL_COLOR = {"rayleigh": SERIES[0], "weibull": SERIES[1], "k": SERIES[2]}
ECHOGRAM_CMAP = "Blues"  # sequential, one hue: darker = stronger


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 2.0, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
