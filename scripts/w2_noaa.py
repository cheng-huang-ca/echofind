"""The front end on real EK80 broadband pings (NOAA HB2305).

1. Parity: echofind.dsp.matched_filter vs echopype's EK80 pulse compression, using echopype's
   transmit replica (tapered chirp through the WBT and PC filters), per channel.
2. Pulse-compression gain on the bottom echo: peak-to-floor ratio after vs before the matched
   filter, against 10 log10(B T) for the nominal sweep.
3. Bottom tracking and CFAR (CA, GO, OS) on the ES70 channel: detections per ping in the water
   column and in the 3 m above the bottom (H3 on real bottom transitions), and an echogram.

echopype is read here directly; the project reader (echofind.io.ek80) is W1's.
Writes reports/w2/noaa_parity.csv, noaa_gain.csv, noaa_cfar.csv; figure w2_noaa_echogram.png.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from w2_common import INK2, OUT, RAW, savefig, setup_style, write_meta  # noqa: E402

from echofind.dsp.bottom import BottomConfig, track_bottom  # noqa: E402
from echofind.dsp.candidates import candidates  # noqa: E402
from echofind.dsp.cfar import CFARConfig, cfar_detect, scale_factor  # noqa: E402
from echofind.dsp.matched_filter import compression_gain_theory_db, matched_filter  # noqa: E402
from echofind.dsp.tvg import apply_tvg  # noqa: E402
from echofind.sim.acoustics import absorption_fg  # noqa: E402

warnings.filterwarnings("ignore")
FILES = ("D20230726-T231007.raw", "D20230722-T131829.raw")
ECHOGRAM_FILE = "D20230726-T231007.raw"
R_MIN = 10.0
SNR_GATE_DB = 20.0


def load(path: Path):
    import echopype as ep
    from echopype.calibrate.ek80_complex import get_filter_coeff, get_transmit_signal

    ed = ep.open_raw(str(path), sonar_model="EK80")
    beam = ed["Sonar/Beam_group1"]
    vend = ed["Vendor_specific"]
    fs_rx = vend["receiver_sampling_frequency"]
    coeff = get_filter_coeff(vend)
    chirp, _ = get_transmit_signal(beam, coeff, "BB", fs_rx)
    env = ed["Environment"]
    c = float(np.nanmean(env["sound_speed_indicative"].values))
    return ed, beam, chirp, c


def env_alpha(env, f_hz: float) -> float:
    """Francois-Garrison absorption from the file's temperature and salinity (defaults if
    the Environment group does not carry them)."""
    def get(name, default):
        return float(np.nanmean(env[name].values)) if name in env else default
    return float(absorption_fg(f_hz, get("temperature", 12.0), get("salinity", 33.0),
                               get("depth", 50.0), get("acidity", 8.0)))


def channel_data(beam, ch):
    b = beam.sel(channel=ch)
    x = (b["backscatter_r"].values + 1j * b["backscatter_i"].values)   # ping, range, sector
    ok = ~np.all(np.isnan(x[..., 0]), axis=0)
    x = x[:, ok, :]
    dt = float(np.unique(b["sample_interval"].values[~np.isnan(b["sample_interval"].values)])[0])
    f0 = float(np.nanmax(b["transmit_frequency_start"].values))
    f1 = float(np.nanmax(b["transmit_frequency_stop"].values))
    T = float(np.nanmax(b["transmit_duration_nominal"].values))
    return np.nan_to_num(x), dt, abs(f1 - f0), T


def parity(beam, chirp, label):
    from echopype.calibrate.ek80_complex import compress_pulse

    recs = []
    for ch in beam["channel"].values:
        sub = beam.sel(channel=[ch]).isel(ping_time=slice(0, 3))
        bs = sub["backscatter_r"] + 1j * sub["backscatter_i"]
        pc = compress_pulse(bs, chirp).sel(channel=ch).transpose("ping_time", "range_sample",
                                                                  "beam").values
        x = np.nan_to_num(bs.sel(channel=ch).transpose("ping_time", "range_sample",
                                                       "beam").values)
        ours = matched_filter(x, np.asarray(chirp[str(ch)]), axis=1, normalize=False)
        ok = np.isfinite(pc)
        err = np.max(np.abs(ours[ok] - pc[ok])) / np.max(np.abs(pc[ok]))
        recs.append({"file": label, "channel": str(ch), "max_rel_err": float(err),
                     "replica_len": len(chirp[str(ch)])})
    return recs


def gain_on_bottom(x, rep, dt, B, T, label, ch):
    """Peak (bottom) power over the median power of the last 10% of samples, before vs after."""
    raw = np.abs(x.mean(-1)) ** 2
    mf = np.abs(matched_filter(x.mean(-1), rep, normalize=True)) ** 2
    tail = slice(int(0.9 * raw.shape[1]), raw.shape[1] - len(rep))
    g = []
    for i in range(raw.shape[0]):
        k0 = int(np.argmax(raw[i, int(R_MIN / 750 / dt):])) + int(R_MIN / 750 / dt)
        before = raw[i, k0] / np.median(raw[i, tail])
        win = slice(max(k0 - len(rep), 0), k0 + len(rep))
        after = mf[i, win].max() / np.median(mf[i, tail])
        g.append(10 * np.log10(after / before))
    return {"file": label, "channel": str(ch), "B_hz": B, "T_s": T,
            "gain_theory_db": compression_gain_theory_db(B, T),
            "gain_bottom_median_db": float(np.median(g)),
            "gain_bottom_iqr_db": float(np.subtract(*np.percentile(g, [75, 25]))),
            "n_pings": len(g)}


def main():
    setup_style()
    import matplotlib.pyplot as plt

    par, gains, cf = [], [], []
    for fname in FILES:
        path = RAW / "noaa" / fname
        if not path.exists():
            print("missing", path)
            continue
        ed, beam, chirp, c = load(path)
        label = path.stem
        par += parity(beam, chirp, label)
        for ch in beam["channel"].values:
            x, dt, B, T = channel_data(beam, ch)
            gains.append(gain_on_bottom(x, np.asarray(chirp[str(ch)]), dt, B, T, label, ch))
        # ES70 front end
        ch70 = [c_ for c_ in beam["channel"].values if "ES70" in str(c_)][0]
        x, dt, B, T = channel_data(beam, ch70)
        rep = np.asarray(chirp[str(ch70)])
        mf = matched_filter(x.mean(-1), rep)
        p = np.abs(mf) ** 2
        r = c * np.arange(p.shape[1]) * dt / 2
        alpha = env_alpha(ed["Environment"], 70e3)
        tvg = apply_tvg(p, r, alpha, 20.0)
        bottom = track_bottom(tvg, r, BottomConfig(r_min=R_MIN, cell_m=0.05, persist_m=0.3,
                                                   min_db=15.0, median_pings=5))
        stride = max(int(round(1 / (dt * B))), 1)
        res_cfg = {}
        for kind in ("ca", "go", "os"):
            cfg = CFARConfig(kind, n_ref=32, n_guard=4, pfa=1e-4, stride=stride)
            det, thr = cfar_detect(p, cfg)
            res_cfg[kind] = (det, thr, cfg)
            near = (r[None, :] >= bottom[:, None] - 3.0) & (r[None, :] < bottom[:, None] - 0.3)
            col = (r[None, :] >= R_MIN) & (r[None, :] < bottom[:, None] - 3.0)
            cf.append({"file": label, "detector": kind, "pings": p.shape[0],
                       "bottom_found_frac": float(np.isfinite(bottom).mean()),
                       "bottom_median_m": float(np.nanmedian(bottom)),
                       "cell_rate_water_column": float((det & col).sum() / max(col.sum(), 1)),
                       "cell_rate_near_bottom": float((det & near).sum() / max(near.sum(), 1))})
        if fname == ECHOGRAM_FILE:
            det, thr, cfg = res_cfg["os"]
            cands = candidates(p, det, thr, r, None, bottom, 0.3, scale_factor(cfg))
            cands = cands[cands.range_m > R_MIN]
            cands.to_csv(OUT / "noaa_candidates_es70.csv", index=False)
            gate_rows = [{"min_snr_db": g, "candidates_per_ping": float(
                (cands.snr_db >= g).sum() / p.shape[0])} for g in (0, 10, 15, 20, 25, 30)]
            pd.DataFrame(gate_rows).to_csv(OUT / "noaa_candidate_gate.csv", index=False)
            print(pd.DataFrame(gate_rows).round(2).to_string())
            n_all = len(cands)
            cands = cands[cands.snr_db >= SNR_GATE_DB]
            fig, ax = plt.subplots(figsize=(10, 4.5))
            db = 10 * np.log10(tvg.T + 1e-30)
            vmax = np.percentile(db, 99.9)
            rmax = min(np.nanmax(bottom) + 20, r[-1]) if np.isfinite(bottom).any() else r[-1]
            im = ax.imshow(db, aspect="auto", origin="upper", cmap="Blues",
                           extent=[0, p.shape[0], r[-1], 0], vmin=vmax - 60, vmax=vmax)
            ax.plot(np.arange(p.shape[0]) + 0.5, bottom, color="#eb6834", lw=1.5,
                    label="bottom track")
            ax.plot(cands.beam + 0.5, cands.range_m, "o", ms=3, mfc="none", color="#1baf7a",
                    label=f"OS-CFAR candidates, SNR >= {SNR_GATE_DB:g} dB "
                    f"({len(cands)} of {n_all})")
            ax.set_ylim(rmax, 0)
            ax.set_xlabel("ping")
            ax.set_ylabel("range (m)")
            ax.set_title(f"ES70 45-90 kHz FM, {label}: matched filter + 20 log r TVG "
                         "(dB, uncalibrated)", fontsize=10)
            ax.legend(loc="lower left", fontsize=8, labelcolor=INK2)
            fig.colorbar(im, ax=ax, label="relative level (dB)")
            savefig(fig, "w2_noaa_echogram.png")
        del ed

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(par).to_csv(OUT / "noaa_parity.csv", index=False)
    pd.DataFrame(gains).to_csv(OUT / "noaa_gain.csv", index=False)
    pd.DataFrame(cf).to_csv(OUT / "noaa_cfar.csv", index=False)
    write_meta("w2_noaa", {"files": FILES, "r_min": R_MIN, "cfar": "N=32 G=4 Pfa=1e-4",
                           "bottom": "persist 0.3 m, cell 0.05 m, min 15 dB", "tvg": "20 log r"})
    print(pd.DataFrame(par).to_string())
    print(pd.DataFrame(gains).round(2).to_string())
    print(pd.DataFrame(cf).round(5).to_string())


if __name__ == "__main__":
    main()
