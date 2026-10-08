"""W1 stage 3: atlas figures and tables from the cached products in data/interim/w1.

Writes reports/figures/w1/*.png, reports/w1_clutter_fits.csv and reports/w1_variability.csv.
Run after scripts/w1_noaa.py and scripts/w1_mbari.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import special

from echofind import viz
from echofind.features import clutter

REPO = Path(__file__).resolve().parents[1]
NOAA = REPO / "data" / "interim" / "w1" / "noaa"
MBARI = REPO / "data" / "interim" / "w1" / "mbari"
FIG = REPO / "reports" / "figures" / "w1"
REP = REPO / "reports"
ECHO_FILE = "ComplexSamples-D20230724-T132531"
LONG_FILE = "D20230722-T131829"
PING_FILE = "D20230726-T231007"
CLUTTER_FILE = "D20230726-T231007"
WINDOW_LABEL = {"open_water": "Open water (8 m to bottom - 5 m)",
                "near_bottom": "Near bottom (1.5 to 0.2 m above)",
                "seabed": "Seabed echo (0 to 1 m below)"}

viz.setup()


def short(stem: str) -> str:
    """'D20230722-T131829' -> '22 Jul 13:18'; ComplexSamples files get a 'BB' prefix."""
    s = stem.replace("ComplexSamples-", "")
    tag = "BB " if stem.startswith("Complex") else ""
    return f"{tag}{s[7:9]} Jul {s[11:13]}:{s[13:15]}"


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def ccdf(a: np.ndarray):
    a = np.sort(a)
    return a, 1 - np.arange(len(a)) / len(a)


def model_ccdf(fit: clutter.Fit, a: np.ndarray) -> np.ndarray:
    """Closed-form exceedance P(A > a): Rayleigh exp(-a^2/2s2); Weibull exp(-(a/l)^c);
    K 2/Gamma(nu) (b a)^nu K_nu(2 b a) with b = sqrt(nu/mu)."""
    p = fit.params
    if fit.model == "rayleigh":
        return np.exp(-(a**2) / (2 * p["sigma2"]))
    if fit.model == "weibull":
        return np.exp(-((a / p["scale"]) ** p["shape"]))
    b = np.sqrt(p["nu"] / p["mu"])
    z = 2 * b * a
    return np.exp(np.log(2) - special.gammaln(p["nu"]) + p["nu"] * np.log(b * a)
                  + np.log(special.kve(p["nu"], z)) - z)


# ---------------------------------------------------------------- NOAA figures

def fig_track(summary: pd.DataFrame):
    s = summary.groupby("file").first().reset_index()
    bb = summary[summary.channel == "ES38"].set_index("file")["bottom_range_m"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(s.lon, s.lat, s=40, color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=1.5,
               zorder=3)
    # Files within ~3 km share one label so the text does not collide.
    s = s.sort_values("lon").reset_index(drop=True)
    cluster, k = [], 0
    for i in range(len(s)):
        if i and np.hypot(s.lon[i] - s.lon[i - 1], s.lat[i] - s.lat[i - 1]) > 0.06:
            k += 1
        cluster.append(k)
    s["cluster"] = cluster
    for k, g in s.groupby("cluster"):
        text = "\n".join(f"{short(f)}  ({bb.get(f, np.nan):.0f} m)" for f in sorted(g.file))
        below = k % 2 == 1  # alternate above and below so neighbouring blocks do not overlap
        ax.annotate(text, (g.lon.mean(), g.lat.mean()), xytext=(0, -10 if below else 10),
                    textcoords="offset points", fontsize=7.5, color=viz.INK2, ha="center",
                    va="top" if below else "bottom")
    ax.set_xlim(s.lon.min() - 0.15, s.lon.max() + 0.15)
    ax.set_ylim(s.lat.min() - 0.12, s.lat.max() + 0.08)
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title("F1. NOAA HB2305 files used: positions and seabed range below the transducer")
    ax.set_aspect(1 / np.cos(np.deg2rad(41.2)))
    return viz.save(fig, FIG / "f01_noaa_track.png")


def fig_echograms():
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 8.2), sharex=True, constrained_layout=True)
    for ax, lab in zip(axes, viz.CHANNEL_ORDER, strict=True):
        e = load_npz(NOAA / f"echogram_{ECHO_FILE}_{lab}.npz")
        p = load_npz(NOAA / f"profile_{ECHO_FILE}_{lab}.npz")
        t = (e["ping_time"] - e["ping_time"][0]) / 1000
        im = ax.pcolormesh(t, e["r"], e["sv"].T, cmap=viz.ECHOGRAM_CMAP, vmin=-90, vmax=-20,
                           shading="auto", rasterized=True)
        ax.plot(t, p["bottom_r"], color=viz.SERIES[1], lw=1, label="detected bottom")
        ax.set_ylim(45, 0)
        ax.set_ylabel("Range (m)")
        kind = "CW 18 kHz" if lab == "ES18" else {"ES38": "FM 34-45 kHz", "ES70": "FM 45-90 kHz",
                                                    "ES200": "FM 160-260 kHz"}[lab]
        ax.set_title(f"{lab}  ({kind})", fontsize=9)
        ax.grid(False)
    axes[0].legend(loc="lower left", fontsize=7.5)
    axes[-1].set_xlabel("Time since first ping (s)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="Sv (dB re 1 m$^{-1}$)")
    fig.suptitle("F2. Echograms per frequency, 24 Jul 13:25 (broadband file, 163 pings)",
                 x=0.02, ha="left", fontweight="bold", fontsize=10)
    return viz.save(fig, FIG / "f02_echograms.png")


def fig_tvg(summary: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for lab in viz.CHANNEL_ORDER:
        p = load_npz(NOAA / f"profile_{LONG_FILE}_{lab}.npz")
        ok = np.isfinite(p["pre"]) & (p["pre"] > -200)  # empty bins and the record's last edge
        p = {k: (v[ok] if v.shape == ok.shape else v) for k, v in p.items()}
        c = viz.CHANNEL_COLOR[lab]
        a1.plot(p["r"], p["pre"], color=c, lw=1.2, label=lab)
        a2.plot(p["r"], p["sv"], color=c, lw=1.2, label=lab)
        noise = summary.query("file == @LONG_FILE and channel == @lab").noise_pre_tvg_db.iloc[0]
        a1.axhline(noise, color=c, lw=0.8, ls=":")
        a2.plot(p["r"], noise + p["tvg"], color=c, lw=0.9, ls="--")
    a1.set_xscale("log")
    a1.set_xlim(1, 500)
    a1.set_xlabel("Range (m)")
    a2.set_xlabel("Range (m)")
    a1.set_ylabel("Level before TVG (dB, uncalibrated offset)")
    a2.set_ylabel("Sv after TVG (dB re 1 m$^{-1}$)")
    a1.set_title("Before TVG: level falls with range until it meets the noise (dotted)")
    a1.set_ylim(-165, -30)
    a2.set_title("After TVG: noise rises as 20 log r + 2 alpha r (dashed)")
    a2.set_ylim(-110, 10)
    a1.legend(ncol=4, fontsize=7.5, loc="upper right")
    fig.suptitle(f"F3. Mean level vs range before and after TVG, {short(LONG_FILE)} "
                 "(19 pings, 500 m record; seabed at ~39 m, multiples beyond)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return viz.save(fig, FIG / "f03_tvg_noise.png")


def fig_noise_snr(summary: pd.DataFrame):
    summary = summary[summary.max_range_m > 400]  # noise needs the 500 m records (decision 002)
    files = sorted(summary.file.unique())
    x = {f: i for i, f in enumerate(files)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0), sharex=True)
    for k, lab in enumerate(viz.CHANNEL_ORDER):
        d = summary[summary.channel == lab]
        off = (k - 1.5) * 0.12
        kw = dict(color=viz.CHANNEL_COLOR[lab], s=30, edgecolor=viz.SURFACE, linewidth=1, label=lab)
        a1.scatter([x[f] + off for f in d.file], d.noise_pre_tvg_db, **kw)
        a2.scatter([x[f] + off for f in d.file], d.bottom_snr_db, **kw)
    for a in (a1, a2):
        a.set_xticks(range(len(files)), [short(f) for f in files], rotation=60, ha="right",
                     fontsize=7.5)
    a1.set_ylabel("Noise level before TVG (dB, uncalibrated)")
    a2.set_ylabel("Seabed echo SNR (dB)")
    a1.set_title("Far-range noise floor per file and channel")
    a2.set_title("Seabed peak over noise, before TVG")
    a1.legend(ncol=4, fontsize=7.5)
    fig.suptitle("F4. Noise floor and seabed SNR across the seven 500 m records "
                 "(median over pings, farthest 15% of range)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return viz.save(fig, FIG / "f04_noise_snr.png")


def fig_ping(summary: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [3, 2]})
    bins = np.arange(-16, 10.5, 1.0)
    for lab in viz.CHANNEL_ORDER:
        p = load_npz(NOAA / f"profile_{PING_FILE}_{lab}.npz")
        dev = p["bottom_sv"] - 10 * np.log10(np.nanmean(10 ** (p["bottom_sv"] / 10)))
        a1.hist(dev, bins=bins, histtype="step", lw=1.8, color=viz.CHANNEL_COLOR[lab],
                density=True, label=f"{lab}  sd {np.nanstd(dev):.1f} dB")
        if lab == "ES38":
            t = (p["ping_time"] - p["ping_time"][0]) / 1000
            a2.plot(t, p["bottom_r"], color=viz.INK2, lw=1.2)
    # Rayleigh fading: 10 log10 of an exponential intensity with unit mean.
    x = np.linspace(-16, 10, 300)
    i = 10 ** (x / 10)
    a1.plot(x, i * np.exp(-i) * np.log(10) / 10, color=viz.MUTED, lw=1.2, ls="--",
            label="Rayleigh fading (sd 5.6 dB)")
    a1.set_xlabel("Seabed peak Sv minus its mean (dB)")
    a1.set_ylabel("Density")
    a1.legend(fontsize=7.5, loc="upper left")
    a1.set_title("Per-ping spread of the seabed peak")
    a2.set_ylabel("Seabed range (m)")
    a2.invert_yaxis()
    a2.set_xlabel("Time since first ping (s)")
    a2.set_title("Seabed track (ES38)")
    fig.suptitle(f"F5. Ping-to-ping fluctuation of the seabed echo, {short(PING_FILE)} "
                 "(184 pings, 1 ping/s): narrower than Rayleigh fading",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return viz.save(fig, FIG / "f05_ping_fluctuation.png")


def all_clutter_fits(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (f, lab), _ in summary.groupby(["file", "channel"]):
        env = load_npz(NOAA / f"envelope_{f}_{lab}.npz")
        for w, a in env.items():
            if a.size < 500:
                continue
            fits = {x.model: x for x in clutter.fit_all(a)}
            best = min(fits.values(), key=lambda x: x.aic)
            rows.append({"source": "noaa", "file": f, "channel": lab, "window": w, "n": a.size,
                         "si": clutter.scintillation_index(a),
                         "k_nu": fits["k"].params["nu"],
                         "weibull_shape": fits["weibull"].params["shape"],
                         **{f"aic_{m}": x.aic for m, x in fits.items()},
                         "best": best.model,
                         "delta_aic_rayleigh": fits["rayleigh"].aic - best.aic})
    return pd.DataFrame(rows)


def fig_clutter_ccdf():
    labs = ["ES38", "ES70", "ES200"]
    fig, axes = plt.subplots(len(labs), 3, figsize=(10, 8), sharex=True, sharey=True)
    for i, lab in enumerate(labs):
        env = load_npz(NOAA / f"envelope_{CLUTTER_FILE}_{lab}.npz")
        for j, w in enumerate(WINDOW_LABEL):
            ax = axes[i, j]
            a = env[w].astype(float)
            fits = clutter.fit_all(a)
            xs, ps = ccdf(a)
            ax.plot(20 * np.log10(xs), ps, color=viz.INK, lw=2.4, label="data")
            grid = np.linspace(xs[0], xs[-1] * 1.2, 400)
            for ft in fits:
                ax.plot(20 * np.log10(grid), model_ccdf(ft, grid), color=viz.MODEL_COLOR[ft.model],
                        lw=1.3, ls="--", label=f"{ft.model} (AIC {ft.aic - fits[0].aic:+.0f})")
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 1.2)
            ax.set_xlim(-25, 20)
            nu = [f for f in fits if f.model == "k"][0].params["nu"]
            si = clutter.scintillation_index(a)
            ax.text(0.03, 0.05, f"{lab}  K nu = {nu:.2g}\nSI = {si:.2f}",
                    transform=ax.transAxes, fontsize=7.5, color=viz.INK2)
            ax.legend(fontsize=6.5, loc="upper right")
            if i == 0:
                ax.set_title(WINDOW_LABEL[w], fontsize=9)
            if j == 0:
                ax.set_ylabel("P(A > a)")
            if i == len(labs) - 1:
                ax.set_xlabel("Normalized amplitude a (dB re local rms)")
    fig.suptitle("F6. Envelope exceedance with Rayleigh, Weibull and K fits, "
                 f"{short(CLUTTER_FILE)} "
                 "(AIC relative to Rayleigh; lower is better)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return viz.save(fig, FIG / "f06_clutter_ccdf.png")


def fig_clutter_summary(fits: pd.DataFrame):
    order = list(WINDOW_LABEL)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0))
    for k, lab in enumerate(viz.CHANNEL_ORDER):
        d = fits[(fits.channel == lab) & (fits.source == "noaa")]
        xs = [order.index(w) + (k - 1.5) * 0.15 for w in d.window]
        kw = dict(color=viz.CHANNEL_COLOR[lab], s=24, edgecolor=viz.SURFACE, linewidth=1, label=lab)
        a1.scatter(xs, d.k_nu, **kw)
        a2.scatter(xs, d.si, **kw)
    for a in (a1, a2):
        a.set_xticks(range(3), ["open water", "near bottom", "seabed"])
    a1.set_yscale("log")
    a1.axhline(clutter.NU_MAX, color=viz.MUTED, lw=0.8, ls=":")
    a1.text(2.4, clutter.NU_MAX * 0.8, "Rayleigh limit (cap)", fontsize=7, ha="right",
            va="top", color=viz.INK2)
    a1.set_ylabel("K shape nu (small = heavy tail)")
    a2.set_yscale("log")
    a2.axhline(1, color=viz.MUTED, lw=0.8, ls=":")
    a2.set_ylabel("Scintillation index (1 = Rayleigh)")
    a1.set_title("K-distribution shape per file and channel")
    a2.set_title("Scintillation index per file and channel")
    a1.legend(ncol=2, fontsize=7.5, loc="upper right", bbox_to_anchor=(1, 0.93))
    fig.suptitle("F7. H1 across all NOAA files: Rayleigh fails at FM resolution in every window; "
                 "only 18 kHz CW (0.77 m cells) nears it in open water",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return viz.save(fig, FIG / "f07_clutter_summary.png")


# ---------------------------------------------------------------- MBARI figures

def mbari():
    return {s: load_npz(MBARI / f"{s}.npz") for s in ["Jan", "Apr", "Jul", "Sep"]}


def fig_psd(m):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for s, d in m.items():
        sel = d["f"] >= 10
        ax.plot(d["f"][sel], d["psd_db"][sel], color=viz.SEASON_COLOR[s], lw=1.2, label=f"{s} 2024")
    ax.set_xscale("log")
    ax.set_xlim(10, 128000)
    for lo, hi, name in [(50, 1000, "ship"), (2000, 20000, "wind / rain"),
                         (20000, 120000, "clicks, shrimp")]:
        ax.axvspan(lo, hi, color=viz.GRID, alpha=0.35, lw=0)
        ax.text(np.sqrt(lo * hi), 0.97, name, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=7.5, color=viz.INK2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB re FS$^2$/Hz)")
    ax.legend(ncol=4, fontsize=8, loc="lower left")
    ax.axvline(100000, color=viz.MUTED, lw=0.8, ls=":")
    ax.text(98000, -195, "anti-alias roll-off", ha="right", fontsize=7, color=viz.INK2)
    ax.set_title("F8. MBARI MARS (891 m depth) ambient noise, Welch PSD of one 10-min file per "
                 "season")
    return viz.save(fig, FIG / "f08_mbari_psd.png")


def fig_spectrograms(m):
    """Each spectrogram is whitened by its own median spectrum, so events stand out from the
    steep spectral slope."""
    fig, axes = plt.subplots(4, 1, figsize=(8, 8.4), sharex=True, constrained_layout=True)
    for ax, (s, d) in zip(axes, m.items(), strict=True):
        f = d["fspec_dec"]
        sel = (f >= 50) & (f <= 100000)
        spec = d["spec_db"][:, sel]
        t = np.arange(spec.shape[0]) / 60
        im = ax.pcolormesh(t, f[sel] / 1000, (spec - np.median(spec, axis=0)).T,
                           cmap=viz.ECHOGRAM_CMAP, vmin=0, vmax=15, shading="auto",
                           rasterized=True)
        ax.set_yscale("log")
        ax.set_ylabel("kHz")
        ax.set_title(f"{s} 2024", fontsize=9)
        ax.grid(False)
    axes[-1].set_xlabel("Minutes into file")
    fig.colorbar(im, ax=axes, shrink=0.6, label="dB above the file's median spectrum")
    fig.suptitle("F9. Whitened spectrograms (1 s frames): transients, song and vessel signatures",
                 x=0.02, ha="left", fontweight="bold", fontsize=10)
    return viz.save(fig, FIG / "f09_mbari_spectrograms.png")


def fig_bands(m):
    names = ["ship 50-1000 Hz", "wind/rain 2-20 kHz", "clicks/shrimp 20-120 kHz"]
    fig, axes = plt.subplots(4, 1, figsize=(8, 7.4), sharex=True)
    for s, d in m.items():
        t = np.arange(len(d["clicks"])) / 60
        for i in range(3):
            axes[i].plot(t, d[f"band_{i}"], color=viz.SEASON_COLOR[s], lw=1.0, label=s)
        axes[3].plot(t, d["clicks"], color=viz.SEASON_COLOR[s], lw=1.0, label=s)
    for i, n in enumerate(names):
        axes[i].set_ylabel("dB re FS$^2$")
        axes[i].set_title(f"Band level, {n}", fontsize=9)
    axes[3].set_ylabel("count per s")
    axes[3].set_title("Impulses per second (1 ms frames 12 dB over the second's median)",
                      fontsize=9)
    axes[3].set_xlabel("Minutes into file")
    axes[0].legend(ncol=4, fontsize=7.5, loc="upper right")
    fig.suptitle("F10. Band levels and impulse rate per second: ship, wind or rain, and "
                 "biological transients", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return viz.save(fig, FIG / "f10_mbari_bands.png")


def mbari_fits(m) -> pd.DataFrame:
    rows = []
    for s, d in m.items():
        a = d["env_sonar_band"].astype(float)
        fits = {x.model: x for x in clutter.fit_all(a)}
        best = min(fits.values(), key=lambda x: x.aic)
        rows.append({"source": "mbari", "file": s, "channel": "60-120 kHz", "window": "noise",
                     "n": a.size, "si": clutter.scintillation_index(a),
                     "k_nu": fits["k"].params["nu"],
                     "weibull_shape": fits["weibull"].params["shape"],
                     **{f"aic_{k}": x.aic for k, x in fits.items()}, "best": best.model,
                     "delta_aic_rayleigh": fits["rayleigh"].aic - best.aic})
    return pd.DataFrame(rows)


def fig_noise_env(m):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s, d in m.items():
        xs, ps = ccdf(d["env_sonar_band"].astype(float))
        ax.plot(20 * np.log10(xs), ps, color=viz.SEASON_COLOR[s], lw=1.6, label=s)
    g = np.linspace(0.01, 6, 400)
    ax.plot(20 * np.log10(g), np.exp(-g**2), color=viz.INK2, lw=1.2, ls="--",
            label="Rayleigh (Gaussian noise)")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.2)
    ax.set_xlim(-25, 20)
    ax.set_xlabel("Envelope amplitude (dB re rms)")
    ax.set_ylabel("P(A > a)")
    ax.legend(fontsize=8)
    ax.set_title("F11. Ambient-noise envelope, 60-120 kHz band (first minute of each file)")
    return viz.save(fig, FIG / "f11_mbari_noise_envelope.png")


# ---------------------------------------------------------------- variability matrix

def variability(summary: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    nb = fits[(fits.source == "noaa") & (fits.window == "near_bottom")]
    nb = nb.set_index(["file", "channel"]).k_nu
    rows = []
    for f, d in summary.groupby("file"):
        row = {"file": short(f), "lat": d.lat.iloc[0], "seabed_range_m":
               d[d.channel == "ES38"].bottom_range_m.iloc[0]}
        for lab in ["ES18", "ES38", "ES70", "ES200"]:
            r = d[d.channel == lab].iloc[0]
            row[f"{lab} noise dB"] = r.noise_pre_tvg_db
            row[f"{lab} seabed SNR dB"] = r.bottom_snr_db
            row[f"{lab} seabed Sv dB"] = r.bottom_sv_db
            row[f"{lab} seabed sd dB"] = r.bottom_sv_std_db
            row[f"{lab} water Sv dB"] = r.water_sv_db
            row[f"{lab} near-bottom K nu"] = nb.get((f, lab), np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("file")


def fig_variability(v: pd.DataFrame):
    cols = [c for c in v.columns if c not in ("file", "lat") and v[c].notna().any()]
    z = v[cols].astype(float)
    zz = ((z - z.mean()) / z.std(ddof=0).replace(0, 1)).to_numpy()
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(v) + 2.6))
    im = ax.imshow(zz, cmap="RdBu_r", norm=TwoSlopeNorm(0, -2.5, 2.5), aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            val = z.iat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.0f}" if abs(val) >= 10 else f"{val:.1f}", ha="center",
                        va="center", fontsize=6.5,
                        color=viz.SURFACE if abs(zz[i, j]) > 1.6 else viz.INK)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(v)), v.file, fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="z-score within column")
    ax.set_title("F12. Variability matrix: NOAA files x (frequency, metric). Cell text is the "
                 "value; colour is its z-score in the column")
    return viz.save(fig, FIG / "f12_variability_matrix.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(NOAA / "summary.csv")
    summary["n_env"] = summary["n_env"].map(json.loads)
    m = mbari()
    fits = pd.concat([all_clutter_fits(summary), mbari_fits(m)], ignore_index=True)
    fits.round(4).to_csv(REP / "w1_clutter_fits.csv", index=False)
    v = variability(summary, fits)
    v.round(2).to_csv(REP / "w1_variability.csv", index=False)
    for fn in (lambda: fig_track(summary), fig_echograms, lambda: fig_tvg(summary),
               lambda: fig_noise_snr(summary), lambda: fig_ping(summary), fig_clutter_ccdf,
               lambda: fig_clutter_summary(fits), lambda: fig_psd(m),
               lambda: fig_spectrograms(m), lambda: fig_bands(m), lambda: fig_noise_env(m),
               lambda: fig_variability(v)):
        print("wrote", fn().name, flush=True)


if __name__ == "__main__":
    main()
