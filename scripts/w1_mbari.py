"""W1 stage 2: reduce the four MBARI MARS 256 kHz files (one 10-minute file per season, 2024).

Per file: a Welch PSD over the whole 10 minutes, a 1-second spectrogram, band levels per second,
impulse counts per second (clicks and snapping shrimp), persistent narrowband tones, and envelope
samples of the 60 to 120 kHz band for the clutter fits.

Levels are in dB re full scale (FS), because no sensitivity is published for the 256 kHz files
(see reports/decisions/001-mbari-levels-re-full-scale.md).

Run: uv run python scripts/w1_mbari.py   (about 3 minutes, peak memory about 3 GB)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "mbari"
OUT = REPO / "data" / "interim" / "w1" / "mbari"
SEASON = {"01": "Jan", "04": "Apr", "07": "Jul", "09": "Sep"}
BANDS = {  # name: (lo Hz, hi Hz) and the source each band mostly reflects
    "ship 50-1000 Hz": (50, 1000),
    "wind/rain 2-20 kHz": (2000, 20000),
    "clicks/shrimp 20-120 kHz": (20000, 120000),
}
SONAR_BAND = (60000, 120000)  # the top of what this recorder sees; nearest to a handheld sonar


def read_fs(path: Path) -> tuple[int, np.ndarray]:
    """24-bit PCM -> float32 in [-1, 1)."""
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    scale = 2.0 ** 31 if x.dtype == np.int32 else float(np.iinfo(x.dtype).max + 1)
    return fs, (x / scale).astype(np.float32)


def tones(f: np.ndarray, psd_db: np.ndarray, min_prom_db: float = 8.0) -> pd.DataFrame:
    """Narrow peaks standing min_prom_db above a 2 kHz running median of the PSD."""
    df = f[1] - f[0]
    width = int(2000 / df) | 1
    base = signal.medfilt(psd_db, width)
    ex = psd_db - base
    pk, prop = signal.find_peaks(ex, height=min_prom_db, width=(None, max(2, int(200 / df))))
    return pd.DataFrame({"freq_hz": f[pk], "excess_db": prop["peak_heights"],
                         "width_hz": prop["widths"] * df})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary, tone_rows = [], []
    for path in sorted(RAW.glob("*.wav")):
        season = SEASON[path.stem.split("_")[1][4:6]]
        print("processing", path.name, season, flush=True)
        fs, x = read_fs(path)
        x = x - x.mean()
        f, pxx = signal.welch(x, fs, nperseg=16384, noverlap=8192, window="hann")
        psd_db = 10 * np.log10(pxx + 1e-30)
        # 1 s spectrogram: Welch inside each second (4096-point, 62.5 Hz bins).
        n_sec = len(x) // fs
        frames = x[: n_sec * fs].reshape(n_sec, fs)
        sxx = []
        for k in range(0, n_sec, 30):  # chunks keep Welch's segment copies small
            fspec, s = signal.welch(frames[k:k + 30], fs, nperseg=4096, noverlap=2048, axis=-1)
            sxx.append(s)
        sxx = np.concatenate(sxx)
        spec_db = (10 * np.log10(sxx + 1e-30)).astype(np.float32)
        band_db = {}
        for name, (lo, hi) in BANDS.items():
            sel = (fspec >= lo) & (fspec < hi)
            band_db[name] = 10 * np.log10(sxx[:, sel].sum(1) * (fspec[1] - fspec[0]))
        # Impulses: 1 ms energy of the 20-120 kHz band, counted when 12 dB over the
        # median of its second (a click or snap stands well above the diffuse background).
        sos = signal.butter(6, [20000, 120000], btype="band", fs=fs, output="sos")
        ms = fs // 1000
        e = np.concatenate([  # 60 s chunks bound memory; edge effects last < 1 ms
            (signal.sosfiltfilt(sos, frames[k:k + 60].ravel()).astype(np.float32) ** 2)
            .reshape(-1, 1000, ms).mean(-1) for k in range(0, n_sec, 60)])
        e_db = 10 * np.log10(e + 1e-30)
        clicks = (e_db > np.median(e_db, axis=1, keepdims=True) + 12).sum(1)
        # Envelope of the sonar band for the noise clutter fit (one sample per 1/B cell).
        sos2 = signal.butter(6, SONAR_BAND, btype="band", fs=fs, output="sos")
        env = np.abs(signal.hilbert(signal.sosfiltfilt(sos2, x[: 60 * fs])))
        step = int(fs / (SONAR_BAND[1] - SONAR_BAND[0])) + 1
        env = env[::step]
        env = env / np.sqrt(np.mean(env**2))
        rng = np.random.default_rng(0)
        env = rng.choice(env, 20000, replace=False).astype(np.float32)
        np.savez_compressed(OUT / f"{season}.npz", f=f, psd_db=psd_db, fspec=fspec,
                            spec_db=spec_db[:, ::2], fspec_dec=fspec[::2], clicks=clicks,
                            env_sonar_band=env,
                            **{f"band_{i}": v for i, v in enumerate(band_db.values())})
        t = tones(f, psd_db)
        t["season"] = season
        tone_rows.append(t)
        row = {"season": season, "file": path.name, "fs": fs, "seconds": n_sec,
               "clicks_per_s_median": float(np.median(clicks)),
               "clicks_per_s_p95": float(np.percentile(clicks, 95)),
               "n_tones": len(t)}
        for name, v in band_db.items():
            row[f"{name} median dBFS"] = float(np.median(v))
            row[f"{name} p95-p5 dB"] = float(np.percentile(v, 95) - np.percentile(v, 5))
        summary.append(row)
        del x, frames, sxx
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    pd.concat(tone_rows).to_csv(OUT / "tones.csv", index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
