"""Simulator sweeps for H2 and H3, and Pd vs range for a body through the full front end.

H2  pulse-compression gain and range resolution over a B x T grid, through the passband chain
    (band-pass, IQ demodulation, matched filter), vs 10 log10(B T) and 0.886 c/(2B).
H3  CFAR at clutter edges (false-alarm excess vs distance to a +15 dB step), with an
    interfering target (masking), and in K-distributed clutter (Pfa vs shape nu).
Pd vs range: body on a sand bottom and 1 m above it, fresh vs salt water, 80 vs 200 kHz,
    simulator -> front end (OS-CFAR, strided reference) -> candidate within 0.3 m of truth.

Writes reports/w2/{h2_grid,h3_edge,h3_masking,h3_k_clutter,pd_vs_range}.csv and figures
w2_h2_gain_resolution.png, w2_h3_edge.png, w2_pd_vs_range.png.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from w2_common import C, INK2, OUT, savefig, setup_style, write_meta  # noqa: E402

from echofind.dsp.cfar import CFARConfig, cfar_detect  # noqa: E402
from echofind.dsp.frontend import FrontEndConfig, demodulate, process_iq  # noqa: E402
from echofind.dsp.matched_filter import (  # noqa: E402
    compression_gain_theory_db,
    lfm_resolution_theory,
    matched_filter,
    width_at_level,
)
from echofind.dsp.waveforms import lfm_pulse, to_passband  # noqa: E402
from echofind.sim import FRESH, SALT, Scene, Sonar, simulate_ping  # noqa: E402
from echofind.sim.acoustics import mf_snr_db  # noqa: E402
from echofind.sim.clutter import k_clutter  # noqa: E402

SEED = 20261009
KINDS = ("ca", "go", "so", "os")


# ---------------------------------------------------------------- H2
def h2_grid(rng) -> pd.DataFrame:
    recs = []
    c = 1500.0
    for B in (5e3, 10e3, 20e3, 40e3):
        for T in (0.5e-3, 1e-3, 2e-3, 5e-3):
            fc = 120e3
            fs = 600e3
            dec = int(fs // (4 * B))
            s_pb = to_passband(lfm_pulse(B, T, fs), fs, fc)
            n = int(0.03 * fs) + len(s_pb)
            sig = np.zeros(n)
            sig[n // 3 : n // 3 + len(s_pb)] = s_pb
            noise = rng.standard_normal(int(0.4 * fs))
            snr_in = 0.5 / (np.var(noise) * 2 * B / fs)
            rep = lfm_pulse(B, T, fs / dec)
            y = matched_filter(demodulate(sig, fs, fc, B, dec), rep)
            peak = np.abs(y).max() ** 2
            m = len(rep) + 50
            n_out = matched_filter(demodulate(noise, fs, fc, B, dec), rep)[m:-m]
            gain = 10 * np.log10(peak / np.mean(np.abs(n_out) ** 2) / snr_in)
            width = width_at_level(y) / (fs / dec) * c / 2
            recs.append({"B_hz": B, "T_s": T, "BT": B * T,
                         "gain_db": gain, "gain_theory_db": compression_gain_theory_db(B, T),
                         "res_3db_m": width, "res_theory_m": 0.886 * lfm_resolution_theory(B, c),
                         "res_nominal_m": lfm_resolution_theory(B, c)})
    df = pd.DataFrame(recs)
    df["gain_err_db"] = df.gain_db - df.gain_theory_db
    df["res_err_rel"] = df.res_3db_m / df.res_theory_m - 1
    df["pass"] = (df.gain_err_db.abs() < 1.0) & (df.res_err_rel.abs() < 0.10)
    return df


# ---------------------------------------------------------------- H3
def h3_edge(rng, step_db=15.0, n_rows=40_000, pfa=1e-3) -> pd.DataFrame:
    L, edge = 200, 100
    lvl = np.where(np.arange(L) < edge, 1.0, 10 ** (step_db / 10))
    x = rng.exponential(1.0, size=(n_rows, L)) * lvl
    recs = []
    for kind in KINDS:
        det, thr = cfar_detect(x, CFARConfig(kind, n_ref=24, n_guard=2, pfa=pfa))
        rate = det.mean(0) / pfa
        for i in range(L):
            if np.isfinite(thr[0, i]):
                recs.append({"detector": kind, "offset_cells": i - edge, "pfa_ratio": rate[i]})
    return pd.DataFrame(recs)


def h3_masking(rng, n_rows=20_000, pfa=1e-3, snr_db=15.0) -> pd.DataFrame:
    """Swerling-1 target in cell 0, an equal interferer at +sep cells (inside the window)."""
    recs = []
    L, c0 = 80, 40
    s = 10 ** (snr_db / 10)
    for sep in (0, 4, 8):
        x = rng.exponential(1.0, size=(n_rows, L))
        x[:, c0] = rng.exponential(1 + s, size=n_rows)
        if sep:
            x[:, c0 + sep] = rng.exponential(1 + s, size=n_rows)
        for kind in KINDS:
            det, _ = cfar_detect(x, CFARConfig(kind, n_ref=24, n_guard=2, pfa=pfa))
            recs.append({"detector": kind, "interferer_offset": sep if sep else None,
                         "pd": det[:, c0].mean()})
    return pd.DataFrame(recs)


def h3_k_clutter(rng, pfa=1e-3) -> pd.DataFrame:
    """Texture correlated over 50 cells (patchy seabed, longer than the CFAR window) or
    independent per cell (spiky clutter at the resolution scale)."""
    recs = []
    for corr in (50, 1):
        for nu in (0.5, 1.0, 2.0, 5.0, 20.0, np.inf):
            p = np.abs(k_clutter((400, 5000), nu, rng, corr_cells=corr)) ** 2
            for kind in ("ideal",) + KINDS:
                if kind == "ideal":
                    rate = np.mean(p > -np.log(pfa) * p.mean())
                else:
                    det, thr = cfar_detect(p, CFARConfig(kind, n_ref=24, n_guard=2, pfa=pfa))
                    rate = det.sum() / np.isfinite(thr).sum()
                recs.append({"texture_corr_cells": corr, "nu": nu, "detector": kind,
                             "pfa_ratio": rate / pfa})
    return pd.DataFrame(recs)


# ---------------------------------------------------------------- Pd vs range
RANGES = (5, 10, 15, 20, 25, 30, 40, 50, 60)
N_PINGS = 100
CASES = {
    "fresh-80k": (FRESH, Sonar(fc=80e3, bandwidth=20e3)),
    "salt-80k": (SALT, Sonar(fc=80e3, bandwidth=20e3)),
    "salt-200k": (SALT, Sonar(fc=200e3, bandwidth=40e3)),
}
PLACEMENT = {"on-bottom": 0.15, "1m-above": 1.0}


def _pd_task(args):
    case, place, rng_m, seed = args
    water, son = CASES[case]
    son = replace(son, max_range_m=70.0)
    rng = np.random.default_rng(seed)
    sc = Scene(water=water, water_depth_m=5.0, target_range_m=float(rng_m),
               target_height_m=PLACEMENT[place])
    over = son.oversample
    cfg = FrontEndConfig(c=water.c, alpha_db_per_m=water.alpha(son.fc),
                         cfar=CFARConfig("os", n_ref=32, n_guard=2 * over, pfa=1e-4, stride=over))
    hits, fa, snr = 0, 0, []
    for _ in range(N_PINGS):
        sc_i = replace(sc, target_aspect_deg=float(rng.uniform(0, 180)))
        p = simulate_ping(son, sc_i, rng)
        res = process_iq(p.iq, p.fs, p.replica, cfg)
        d = np.abs(res.candidates.range_m.to_numpy() - p.truth["target_range_m"])
        hits += bool((d < 0.3 + p.truth["target_extent_m"]).any())
        fa += int((d >= 2.0).sum())
        k = int(np.argmin(np.abs(p.r - p.truth["target_range_m"])))
        snr.append(10 * np.log10(np.abs(matched_filter(p.target, p.replica)[k]) ** 2
                                 / np.mean(np.abs(matched_filter(p.noise + p.reverb,
                                                                 p.replica)[k - 40 : k + 40])
                                           ** 2)))
    slant = np.hypot(rng_m, 5.0 - PLACEMENT[place] - son.depth_m)
    return {"case": case, "placement": place, "range_m": rng_m, "pd": hits / N_PINGS,
            "false_alarms_per_ping": fa / N_PINGS, "median_snr_db": float(np.median(snr)),
            "sonar_eq_snr_db": float(mf_snr_db(son.sl_db, -20.0, slant, water.alpha(son.fc),
                                               40.0, 0.0, son.duration)),
            "n_pings": N_PINGS}


def pd_vs_range(seed) -> pd.DataFrame:
    ss = np.random.SeedSequence(seed)
    tasks = [(c, p, r, int(s.generate_state(1)[0]))
             for (c, p, r), s in zip(
                 [(c, p, r) for c in CASES for p in PLACEMENT for r in RANGES],
                 ss.spawn(len(CASES) * len(PLACEMENT) * len(RANGES)), strict=True)]
    with ProcessPoolExecutor(4) as ex:
        return pd.DataFrame(list(ex.map(_pd_task, tasks)))


def main():
    rng = np.random.default_rng(SEED)
    setup_style()
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    h2 = h2_grid(rng)
    h2.to_csv(OUT / "h2_grid.csv", index=False)
    edge = h3_edge(rng)
    edge.to_csv(OUT / "h3_edge.csv", index=False)
    mask = h3_masking(rng)
    mask.to_csv(OUT / "h3_masking.csv", index=False)
    kc = h3_k_clutter(rng)
    kc.to_csv(OUT / "h3_k_clutter.csv", index=False)
    pr = pd_vs_range(SEED)
    pr.to_csv(OUT / "pd_vs_range.csv", index=False)
    write_meta("w2_sim_sweeps", {"seed": SEED, "h2": "fc 120 kHz, fs 600 kHz, MF at 4B",
                                 "h3": "N=24, G=2, Pfa=1e-3, +15 dB step",
                                 "pd_vs_range": {"ranges": RANGES, "n_pings": N_PINGS,
                                                 "depth_m": 5.0, "sl_db": 200, "nl_db": 40,
                                                 "ts_db": -20, "cfar": "OS N=32 stride 4"}})

    # H2 figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    lim = [h2.gain_theory_db.min() - 2, h2.gain_theory_db.max() + 2]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--", label="theory 10 log10(BT)")
    ax.fill_between(lim, np.array(lim) - 1, np.array(lim) + 1, color=C[0], alpha=0.12,
                    lw=0, label="+/- 1 dB pass band")
    ax.plot(h2.gain_theory_db, h2.gain_db, "o", color=C[0], label="measured (B x T grid)")
    ax.set_xlabel("theory gain (dB)")
    ax.set_ylabel("measured gain (dB)")
    ax.set_title("H2: pulse-compression gain", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    for i, B in enumerate(sorted(h2.B_hz.unique())):
        g = h2[h2.B_hz == B]
        ax.plot(g.T_s * 1e3, g.res_3db_m * 100, "o-", color=C[i], label=f"B = {B / 1e3:g} kHz")
        ax.axhline(g.res_theory_m.iloc[0] * 100, color=C[i], lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pulse length T (ms)")
    ax.set_ylabel("-3 dB range resolution (cm)")
    ax.set_title("H2: resolution vs 0.886 c/(2B) (dotted)", fontsize=10)
    ax.legend(fontsize=8)
    savefig(fig, "w2_h2_gain_resolution.png")

    # H3 figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for i, kind in enumerate(KINDS):
        g = edge[edge.detector == kind]
        ax.semilogy(g.offset_cells, np.maximum(g.pfa_ratio, 1e-2), color=C[i], label=kind)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-30, 30)
    ax.set_xlabel("cell offset from +15 dB clutter edge")
    ax.set_ylabel("false-alarm rate / design Pfa")
    ax.set_title("H3: CFAR at a clutter edge", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    for corr, ls, mk in ((50, "-", "o"), (1, "--", "s")):
        for i, kind in enumerate(("ideal",) + KINDS):
            g = kc[(kc.detector == kind) & (kc.texture_corr_cells == corr)]
            nu = np.where(np.isfinite(g.nu), g.nu, 100.0)
            ax.loglog(nu, g.pfa_ratio, ls, marker=mk, ms=4, color=([INK2] + C)[i],
                      label=kind if corr == 50 else None)
    ax.set_xlabel("K shape nu (100 = Rayleigh)")
    ax.set_ylabel("false-alarm rate / design Pfa")
    ax.set_title("H3: K clutter; solid patchy (50 cells), dashed spiky (1 cell)", fontsize=10)
    ax.legend(fontsize=8)
    savefig(fig, "w2_h3_edge.png")

    # Pd vs range
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, place in zip(axes, PLACEMENT, strict=True):
        for i, case in enumerate(CASES):
            g = pr[(pr.case == case) & (pr.placement == place)]
            ax.plot(g.range_m, g.pd, "o-", color=C[i], label=case)
        ax.set_title(f"Body {place.replace('-', ' ')}, 5 m deep, OS-CFAR Pfa 1e-4", fontsize=10)
        ax.set_xlabel("horizontal range (m)")
    axes[0].set_ylabel("Pd (candidate within 0.3 m)")
    axes[1].legend(fontsize=8)
    savefig(fig, "w2_pd_vs_range.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, case in enumerate(CASES):
        g = pr[(pr.case == case) & (pr.placement == "on-bottom")]
        ax.plot(g.range_m, g.median_snr_db, "o-", color=C[i], label=f"{case}, simulated SINR")
        ax.plot(g.range_m, g.sonar_eq_snr_db, ls="--", lw=1.2, color=C[i])
    ax.set_ylim(-10, 85)
    ax.set_xlabel("horizontal range (m)")
    ax.set_ylabel("matched-filter SNR (dB)")
    ax.set_title("Body on bottom: reverberation-limited SINR (solid) vs\n"
                 "noise-limited sonar equation, on axis (dashed)", fontsize=10)
    ax.legend(fontsize=8)
    savefig(fig, "w2_sinr_vs_range.png")

    print(h2[["B_hz", "T_s", "gain_err_db", "res_err_rel", "pass"]].round(3).to_string())
    print(mask.round(3).to_string())
    print(kc.pivot_table(index=["texture_corr_cells", "nu"], columns="detector",
                         values="pfa_ratio").round(2))
    print(pr.round(2).to_string())


if __name__ == "__main__":
    main()
