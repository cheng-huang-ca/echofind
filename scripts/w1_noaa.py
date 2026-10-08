"""W1 stage 1: reduce every NOAA HB2305 file to small cached products for the atlas.

For each file and channel: mean level-vs-range profiles before and after TVG, a far-range noise
estimate, per-ping bottom range and bottom peak Sv, a decimated echogram, and normalized envelope
samples in three windows (open water, near bottom, seabed) for the clutter fits.

Run: uv run python scripts/w1_noaa.py   (about 5 minutes, peak memory about 4 GB)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from echofind.io import ek80

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "noaa"
OUT = REPO / "data" / "interim" / "w1" / "noaa"
C = 1500.0
ECHO_MAX_M = 60.0  # echogram depth cut (the shelf here is 30 to 60 m)
WINDOWS = {  # metres relative to the detected bottom (negative = above it), or absolute range
    "open_water": ("range", 8.0, -5.0),
    "near_bottom": ("bottom", -1.5, -0.2),
    "seabed": ("bottom", 0.0, 1.0),
}
RNG = np.random.default_rng(0)


def resolution_m(ds, i: int) -> float:
    """Range resolution: c/(2B) after pulse compression for FM, c*tau/2 for CW."""
    if "transmit_frequency_start" in ds and str(ds.label.values[i]) != "ES18":
        b = abs(float(ds.transmit_frequency_stop.isel(channel=i).max())
                - float(ds.transmit_frequency_start.isel(channel=i).max()))
        if b > 0:
            return C / (2 * b)
    return C * 1.024e-3 / 2


def profile(r: np.ndarray, x_db: np.ndarray, step: float = 0.1, rmax: float | None = None):
    """Mean over pings in the linear domain, binned to `step` metres; returns (r_bin, dB)."""
    lin = np.nanmean(10 ** (x_db / 10), axis=0)
    good = np.isfinite(r) & np.isfinite(lin) & (r >= 1.0)
    r, lin = r[good], lin[good]
    edges = np.arange(1.0, (rmax or r.max()) + step, step)
    idx = np.digitize(r, edges) - 1
    keep = (idx >= 0) & (idx < len(edges) - 1)
    s = np.bincount(idx[keep], lin[keep], len(edges) - 1)
    n = np.bincount(idx[keep], minlength=len(edges) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return edges[:-1] + step / 2, 10 * np.log10(s / n)


def envelope_samples(sv: np.ndarray, r: np.ndarray, bot: np.ndarray, dec: int, kind: str,
                     lo: float, hi: float, max_n: int = 20000) -> np.ndarray:
    """Amplitudes in a window, normalized by the local mean intensity, one sample per cell.

    The local mean is taken over all pings and +-0.25 m in range, so the TVG and the range trend
    drop out and what remains is the fluctuation statistic the clutter models describe.
    """
    dr = float(np.nanmedian(np.diff(r)))
    inten = 10 ** (sv / 10)
    rows = []
    if kind == "range":
        j0 = int(np.searchsorted(r, lo))
        j1 = int(np.nanmin(np.where(bot >= 0, bot, np.nan)) + hi / dr)
        if j1 - j0 < 10:
            return np.empty(0)
        block = inten[:, j0:j1]
    else:
        k0, k1 = int(round(lo / dr)), int(round(hi / dr))
        if k1 - k0 < 2:
            return np.empty(0)
        good = bot >= 0
        idx = bot[good][:, None] + np.arange(k0, k1)[None, :]
        idx = np.clip(idx, 0, inten.shape[1] - 1)
        block = np.take_along_axis(inten[good], idx, axis=1)
    half = max(1, int(round(0.25 / dr)))
    colmean = np.nanmean(block, axis=0)
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    local = np.convolve(np.nan_to_num(colmean), kernel, mode="same")
    a = np.sqrt(block / local[None, :])[:, ::dec]
    rows = a[np.isfinite(a) & (a > 0)]
    if rows.size > max_n:
        rows = RNG.choice(rows, max_n, replace=False)
    return rows.astype(np.float32)


def process(path: Path) -> list[dict]:
    ed = ek80.open_ek80(path)
    lat = float(ed["Platform"].latitude.mean())
    lon = float(ed["Platform"].longitude.mean())
    stem = path.stem
    rows = []
    for waveform in ("BB", "CW"):
        ds = ek80.compute_sv(ed, waveform)
        times = ds.ping_time.values
        for i, lab in enumerate(ds.label.values):
            lab = str(lab)
            sv = ds.Sv.isel(channel=i).values
            pre = ds.level_pre_tvg.isel(channel=i).values
            r = ds.echo_range.isel(channel=i).values
            r = r[0] if r.ndim == 2 else r
            valid = np.isfinite(r) & np.isfinite(np.nanmean(sv, axis=0))
            last = int(np.where(valid)[0].max()) + 1
            sv, pre, r = sv[:, :last], pre[:, :last], r[:last]
            dr = float(np.nanmedian(np.diff(r)))
            res = resolution_m(ds, i)
            dec = max(1, int(round(res / dr)))
            bot = ek80.bottom_index(sv, r)
            rb = np.where(bot >= 0, r[np.clip(bot, 0, None)], np.nan)
            bpeak = np.where(bot >= 0, sv[np.arange(len(bot)), np.clip(bot, 0, None)], np.nan)
            rp, pre_prof = profile(r, pre)
            _, sv_prof = profile(r, sv)
            _, tvg_prof = profile(r, ds.tvg_db.isel(channel=i).values[:, :last]
                                  if ds.tvg_db.isel(channel=i).ndim == 2
                                  else ds.tvg_db.isel(channel=i).values[None, :last])
            # Noise: median pre-TVG level over the farthest 15% of the recorded range, where
            # bottom multiples have decayed (checked on the profiles in the atlas figure).
            far = rp > 0.85 * np.nanmax(rp)
            noise_db = float(np.nanmedian(pre_prof[far]))
            bot_pre = np.where(bot >= 0, pre[np.arange(len(bot)), np.clip(bot, 0, None)], np.nan)
            # Water column above the bottom, below the near field and surface bubbles.
            wc = (r >= 8.0) & (r <= np.nanmin(rb) - 3)
            wc_sv = float(10 * np.log10(np.nanmean(10 ** (sv[:, wc] / 10)))) if wc.any() else np.nan
            np.savez_compressed(
                OUT / f"profile_{stem}_{lab}.npz", r=rp, pre=pre_prof, sv=sv_prof, tvg=tvg_prof,
                ping_time=times.astype("datetime64[ms]").astype(np.int64), bottom_r=rb,
                bottom_sv=bpeak)
            if stem.startswith("ComplexSamples-D20230724-T132531") or stem == "D20230726-T231007":
                cut = r <= ECHO_MAX_M
                step = max(1, int(round(0.05 / dr)))
                eg = sv[:, cut]
                n = eg.shape[1] // step * step
                eg = 10 * np.log10(np.nanmean(10 ** (eg[:, :n] / 10)
                                              .reshape(eg.shape[0], -1, step), axis=2))
                np.savez_compressed(OUT / f"echogram_{stem}_{lab}.npz", sv=eg.astype(np.float32),
                                    r=r[cut][:n].reshape(-1, step).mean(1),
                                    ping_time=times.astype("datetime64[ms]").astype(np.int64))
            env = {}
            for w, (kind, lo, hi) in WINDOWS.items():
                env[w] = envelope_samples(sv, r, bot, dec, kind, lo, hi)
            np.savez_compressed(OUT / f"envelope_{stem}_{lab}.npz", **env)
            rows.append({
                "file": stem, "channel": lab, "waveform": waveform, "lat": lat, "lon": lon,
                "start": str(times[0])[:19], "pings": len(times),
                "ping_interval_s": float(np.median(np.diff(times)) / np.timedelta64(1, "s"))
                if len(times) > 1 else np.nan,
                "max_range_m": float(np.nanmax(r)), "sample_m": dr, "resolution_m": res,
                "bottom_range_m": float(np.nanmedian(rb)),
                "bottom_found": float(np.mean(bot >= 0)),
                "bottom_sv_db": float(10 * np.log10(np.nanmean(10 ** (bpeak / 10)))),
                "bottom_sv_std_db": float(np.nanstd(bpeak)),
                "noise_pre_tvg_db": noise_db,
                "bottom_snr_db": float(np.nanmedian(bot_pre) - noise_db),
                "water_sv_db": wc_sv,
                "n_env": {w: int(v.size) for w, v in env.items()},
            })
        del ds
    del ed
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(RAW.glob("*.raw")):
        print("processing", path.name, flush=True)
        rows += process(path)
    df = pd.DataFrame(rows)
    df["n_env"] = df["n_env"].map(json.dumps)
    df.to_csv(OUT / "summary.csv", index=False)
    print("wrote", OUT / "summary.csv", len(df), "rows")


if __name__ == "__main__":
    main()
