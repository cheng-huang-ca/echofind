"""Pd-vs-SNR curves: Monte Carlo through the matched filter and detectors vs closed form.

Configurations (LFM, B = 20 kHz, T = 1 ms, fc = 80 kHz, Pfa = 1e-4):
  white-x4-contig   white noise, MF output at 4 B, contiguous reference cells (naive)
  white-x4          white noise, MF output at 4 B, reference cells every 4 samples (front end)
  white-x1          white noise, MF output at B (one sample per resolution cell)
  mbari-x4          real MBARI hydrophone noise (four seasons, 70-90 kHz band), as white-x4
Detectors: ideal fixed threshold (noise floor known), CA-, GO-, OS-CFAR with N = 32.
Targets: Swerling 0 (steady, random phase) and Swerling 1 (Rayleigh-fluctuating).

SNR is the matched-filter output SNR, |a|^2 E / N0, relative to the Gaussian-equivalent floor
(median |n|^2 / ln 2), so real-noise transients count as excess noise, not as floor.

Writes reports/w2/pd_vs_snr.csv, reports/w2/pfa_empirical.csv, reports/w2/snr_at_pd90.csv and
figures w2_pd_vs_snr.png, w2_pd_real_noise.png, w2_mbari_noise_tails.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from w2_common import INK2, OUT, RAW, C, savefig, setup_style, write_meta  # noqa: E402

from echofind.dsp.cfar import CFARConfig, cfar_detect, cfar_threshold  # noqa: E402
from echofind.dsp.cfar import pd_swerling1 as pd_cfar_sw1  # noqa: E402
from echofind.dsp.detection_theory import pd_swerling0, pd_swerling1  # noqa: E402
from echofind.dsp.matched_filter import matched_filter  # noqa: E402
from echofind.dsp.waveforms import lfm_pulse  # noqa: E402
from echofind.sim.noise import mbari_baseband  # noqa: E402

SEED = 20261008
B, T, FC, PFA = 20e3, 1e-3, 80e3, 1e-4
L, I0, HALF = 2048, 1024, 80          # row length, target cell, half-window for thresholds
N_TRIALS = 4000
SNR_DB = np.arange(0, 25, 1.0)
DETS = ("ideal", "ca", "go", "os")
MBARI_START_S, MBARI_DUR_S = 60.0, 30.0


def cfg_for(kind: str, over: int, stride: int) -> CFARConfig:
    return CFARConfig(kind, n_ref=32, n_guard=over, pfa=PFA, stride=stride)


def floor_normalise(mf_rows: np.ndarray) -> np.ndarray:
    """Scale so the Gaussian-equivalent floor median(|n|^2)/ln 2 is 1."""
    return mf_rows / np.sqrt(np.median(np.abs(mf_rows) ** 2) / np.log(2))


def white_rows(n_rows: int, over: int, rng) -> np.ndarray:
    rep = lfm_pulse(B, T, 4 * B)
    n = L * (4 // over) + len(rep)      # pad so every kept output sample sees a full pulse
    x = (rng.standard_normal((n_rows, n)) + 1j * rng.standard_normal((n_rows, n))) / np.sqrt(2)
    mf = matched_filter(x, rep)[:, : L * (4 // over) : 4 // over]
    return floor_normalise(mf)


def mbari_rows(over: int) -> tuple[np.ndarray, list[str]]:
    rows, labels = [], []
    rep = lfm_pulse(B, T, 4 * B)
    for f in sorted((RAW / "mbari").glob("MARS_*.wav")):
        n = int(MBARI_DUR_S * 4 * B)
        bb = mbari_baseband(f, MBARI_START_S, n, 4 * B, FC, B, nl_db=0.0)
        mf = matched_filter(bb, rep)[len(rep):][:: 4 // over]
        k = len(mf) // L
        r = floor_normalise(mf[: k * L].reshape(k, L))
        rows.append(r)
        labels += [f.stem.split("_")[1]] * k
    return np.concatenate(rows), labels


def target_response(over: int) -> np.ndarray:
    """MF output of a unit-amplitude echo whose peak lands on cell I0, normalised to peak 1."""
    rep = lfm_pulse(B, T, 4 * B)
    x = np.zeros(L * (4 // over) + len(rep), complex)
    x[I0 * (4 // over) : I0 * (4 // over) + len(rep)] = rep
    g = matched_filter(x, rep)[:: 4 // over][:L]
    return g / g[I0]


def detect_at_cell(rows: np.ndarray, over: int, stride: int) -> dict[str, np.ndarray]:
    """Detection decision at cell I0 for each detector (thresholds from a local window)."""
    win = rows[:, I0 - HALF : I0 + HALF + 1]
    p = np.abs(win) ** 2
    out = {"ideal": p[:, HALF] > -np.log(PFA)}
    for kind in DETS[1:]:
        thr = cfar_threshold(p, cfg_for(kind, over, stride))[:, HALF]
        out[kind] = p[:, HALF] > thr
    return out


def empirical_pfa(rows: np.ndarray, over: int, stride: int) -> dict[str, float]:
    p = np.abs(rows) ** 2
    res = {"ideal": float(np.mean(p > -np.log(PFA)))}
    for kind in DETS[1:]:
        det, thr = cfar_detect(p, cfg_for(kind, over, stride))
        res[kind] = float(det.sum() / np.isfinite(thr).sum())
    return res


def sweep(noise: np.ndarray, over: int, stride: int, rng, label: str) -> list[dict]:
    g = target_response(over)
    recs = []
    for sw in (0, 1):
        for s_db in SNR_DB:
            s = 10 ** (s_db / 10)
            n = noise.shape[0]
            if sw == 0:
                a = np.sqrt(s) * np.exp(2j * np.pi * rng.random(n))
            else:
                a = np.sqrt(s) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
            d = detect_at_cell(noise + a[:, None] * g[None, :], over, stride)
            for kind, v in d.items():
                recs.append({"config": label, "swerling": sw, "detector": kind,
                             "snr_db": s_db, "pd": float(v.mean()), "n": n})
    return recs


def theory_rows() -> list[dict]:
    recs = []
    s = 10 ** (SNR_DB / 10)
    for sw, fn in ((0, pd_swerling0), (1, pd_swerling1)):
        for s_db, v in zip(SNR_DB, fn(s, PFA), strict=True):
            recs.append({"config": "theory", "swerling": sw, "detector": "ideal",
                         "snr_db": s_db, "pd": float(v), "n": 0})
    for kind in DETS[1:]:
        for n_eff, lab in ((32, "theory"),):
            cfg = CFARConfig(kind, n_ref=n_eff, pfa=PFA)
            for s_db, v in zip(SNR_DB, pd_cfar_sw1(cfg, s), strict=True):
                recs.append({"config": lab, "swerling": 1, "detector": kind,
                             "snr_db": s_db, "pd": float(v), "n": 0})
    return recs


def snr_at(df: pd.DataFrame, pd_target: float = 0.9) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(["config", "swerling", "detector"]):
        g = g.sort_values("snr_db")
        y = np.maximum.accumulate(g.pd.to_numpy())
        v = float(np.interp(pd_target, y, g.snr_db)) if y[-1] >= pd_target else np.nan
        out.append(dict(zip(["config", "swerling", "detector"], key, strict=True),
                        snr_db_at_pd90=v))
    return pd.DataFrame(out)


def main():
    rng = np.random.default_rng(SEED)
    setup_style()
    import matplotlib.pyplot as plt

    recs, pfa_recs = theory_rows(), []
    for over, stride, label in ((4, 1, "white-x4-contig"), (4, 4, "white-x4"), (1, 1, "white-x1")):
        rows = white_rows(N_TRIALS, over, rng)
        recs += sweep(rows, over, stride, rng, label)
        pfa_recs.append({"config": label, "season": "all", **empirical_pfa(rows, over, stride)})
    have_mbari = any((RAW / "mbari").glob("MARS_*.wav"))
    if have_mbari:
        rows, seasons = mbari_rows(4)
        recs += sweep(rows, 4, 4, rng, "mbari-x4")
        for season in sorted(set(seasons)):
            m = np.array(seasons) == season
            pfa_recs.append({"config": "mbari-x4", "season": season,
                             **empirical_pfa(rows[m], 4, 4)})
        pfa_recs.append({"config": "mbari-x4", "season": "all", **empirical_pfa(rows, 4, 4)})

    df = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "pd_vs_snr.csv", index=False)
    pfa = pd.DataFrame(pfa_recs)
    pfa.to_csv(OUT / "pfa_empirical.csv", index=False)
    s90 = snr_at(df)
    s90.to_csv(OUT / "snr_at_pd90.csv", index=False)
    write_meta("w2_pd_curves", {"seed": SEED, "B": B, "T": T, "fc": FC, "pfa": PFA, "L": L,
                                "n_trials": N_TRIALS,
                                "cfar": "N=32, G=oversample, OS k=24, stride per config",
                                "mbari": [MBARI_START_S, MBARI_DUR_S]})

    # ---- figure 1: white noise, oversampled front end, both Swerling cases
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, sw in zip(axes, (0, 1), strict=True):
        th = df[(df.config == "theory") & (df.swerling == sw) & (df.detector == "ideal")]
        ax.plot(th.snr_db, th.pd, color=INK2, lw=1.2, ls="--", label="ideal, theory")
        for i, kind in enumerate(DETS):
            g = df[(df.config == "white-x4") & (df.swerling == sw) & (df.detector == kind)]
            ax.plot(g.snr_db, g.pd, color=C[i], marker="o", ms=3, label=f"{kind}, Monte Carlo")
        if sw == 1:
            g = df[(df.config == "theory") & (df.detector == "ca")]
            ax.plot(g.snr_db, g.pd, color=C[1], lw=1.2, ls=":", label="ca, theory (N = 32)")
        ax.set_title(f"Swerling {sw}, white noise, MF at 4B, strided reference", fontsize=10)
        ax.set_xlabel("matched-filter output SNR (dB)")
    axes[0].set_ylabel(f"Pd at Pfa = {PFA:g}")
    axes[1].legend(loc="lower right", fontsize=8)
    savefig(fig, "w2_pd_vs_snr.png")

    if have_mbari:
        # ---- figure 2: real noise vs white, Swerling 1
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, kind in enumerate(DETS):
            w = df[(df.config == "white-x4") & (df.swerling == 1) & (df.detector == kind)]
            m = df[(df.config == "mbari-x4") & (df.swerling == 1) & (df.detector == kind)]
            ax.plot(w.snr_db, w.pd, color=C[i], lw=1.2, ls="--")
            ax.plot(m.snr_db, m.pd, color=C[i], marker="o", ms=3, label=kind)
        ax.set_title("Swerling 1: real MBARI noise (solid) vs white (dashed)", fontsize=10)
        ax.set_xlabel("SNR over the Gaussian-equivalent floor (dB)")
        ax.set_ylabel(f"Pd at design Pfa = {PFA:g}")
        ax.legend(loc="lower right", fontsize=8)
        savefig(fig, "w2_pd_real_noise.png")

        # ---- figure 3: exceedance of normalised MF noise power, per season
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.linspace(0, 20, 200)
        ax.semilogy(x, np.exp(-x), color=INK2, lw=1.2, ls="--", label="exponential (Gaussian)")
        for i, season in enumerate(sorted(set(seasons))):
            p = np.sort((np.abs(rows[np.array(seasons) == season]) ** 2).ravel())
            ex = 1 - np.searchsorted(p, x) / len(p)
            ax.semilogy(x, np.maximum(ex, 1e-9), color=C[i], label=season)
        ax.axvline(-np.log(PFA), color=GRID_LINE, lw=1)
        ax.set_ylim(1e-7, 1.5)
        ax.set_xlabel("MF noise power / Gaussian-equivalent floor")
        ax.set_ylabel("P(power > x)")
        ax.set_title("MBARI 70-90 kHz noise after the matched filter", fontsize=10)
        ax.legend(fontsize=8)
        savefig(fig, "w2_mbari_noise_tails.png")

    print(pfa.round(6).to_string(index=False))
    print(s90.pivot_table(index=["config", "swerling"], columns="detector",
                          values="snr_db_at_pd90").round(2).to_string())


GRID_LINE = "#a8a7a2"

if __name__ == "__main__":
    main()
