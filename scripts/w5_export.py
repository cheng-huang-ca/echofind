"""W5: export everything the C++ edge build needs, and the accuracy side of the Pareto plot.

1. Filter coefficients: the band-pass and low-pass SOS that echofind.dsp.demodulate designs,
   for two sonar settings (80 kHz / 20 kHz LFM and 200 kHz / 40 kHz LFM, 1 ms; W2 defaults).
2. Benchmark pings: a 50 m ping per setting from echofind.sim (body at 30 m, bottom
   reverberation, noise), upsampled and mixed to real passband. Written under data/processed/edge
   (regenerable, not committed) with an ef_bench config file.
3. Golden vectors: short (20 m, 80 kHz) cases for demodulation, matched filter, CA/GO/OS-CFAR
   and TVG, with the Python outputs, in edge/tests/golden (small, committed).
4. Models: R0, R1 and R2 at 25, 50, 100 and 300 trees, trained on all lake candidates, compiled
   to edge/generated/models.c (echofind.models.export_c). Each variant's cross-session recall at
   0.05 FAPF (W3 protocol) goes to reports/w5/model_accuracy.csv for the Pareto plot.

Run: uv run python scripts/w5_export.py [--boot 300]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import PROC, ROOT, git_commit, manifest_hash  # noqa: E402
from w3_ladder import BODY_GROUPS, FAPF, SEED, load, run_fold, summarise  # noqa: E402

from echofind.dsp.cfar import CFARConfig, cfar_threshold  # noqa: E402
from echofind.dsp.frontend import demodulate  # noqa: E402
from echofind.dsp.matched_filter import matched_filter  # noqa: E402
from echofind.dsp.tvg import apply_tvg, range_axis  # noqa: E402
from echofind.features.fls import R1_FEATURES, R2_FEATURES  # noqa: E402
from echofind.models.export_c import lgbm_to_c, logistic_to_c, r0_to_c, write_c  # noqa: E402
from echofind.models.ladder import SIZE_GATE_M, R1Logistic, R2Trees  # noqa: E402
from echofind.sim.scene import Scene, Sonar, simulate_ping  # noqa: E402

EDGE = ROOT / "edge"
GOLD = EDGE / "tests" / "golden"
GEN = EDGE / "generated"
BENCH = PROC / "edge"
OUT = ROOT / "reports" / "w5"
KINDS = {"ca": 0, "go": 1, "so": 2, "os": 3}
SETTINGS = {  # name: (fc, B, decimate): baseband fs = 4B, passband fs = decimate * 4B
    "80k": (80e3, 20e3, 5),
    "200k": (200e3, 40e3, 6),
}
TREES = (25, 50, 100, 300)


def design_sos(fc, bw, fs):
    """The filters echofind.dsp.frontend.demodulate and iq_demodulate design internally."""
    lo, hi = max(fc - 0.6 * bw, 1.0), min(fc + 0.6 * bw, 0.49 * fs)
    bp = butter(6, [lo, hi], btype="bandpass", fs=fs, output="sos")
    lp = butter(6, min(0.6 * bw, 0.45 * fs, 0.9 * fc), btype="lowpass", fs=fs, output="sos")
    return bp, lp


def passband_ping(fc, bw, dec, max_range, seed=0):
    """Simulated baseband ping, upsampled by `dec` and mixed to a real passband signal."""
    son = Sonar(fc=fc, bandwidth=bw, max_range_m=max_range)
    ping = simulate_ping(son, Scene(target_range_m=min(30.0, 0.6 * max_range)),
                         np.random.default_rng(seed))
    fs = son.fs * dec
    up = resample_poly(ping.iq, dec, 1)
    t = np.arange(len(up)) / fs
    x = np.real(up * np.exp(2j * np.pi * fc * t))
    s = np.max(np.abs(x))
    return x / s, ping.replica, fs, son.fs, ping.truth["c"], ping.truth["alpha_db_per_m"]


def save(path: Path, a) -> str:
    a = np.asarray(a)
    if np.iscomplexobj(a):
        a = np.stack([a.real, a.imag], axis=-1)
    a.astype("<f8").tofile(path)
    return path.name


def golden() -> None:
    GOLD.mkdir(parents=True, exist_ok=True)
    fc, bw, dec = SETTINGS["80k"]
    x, rep, fs, fs_bb, c, alpha = passband_ping(fc, bw, dec, 20.0, seed=1)
    bp, lp = design_sos(fc, bw, fs)
    iq = demodulate(x, fs, fc, bw, decimate=dec)
    mf = matched_filter(iq, rep)
    p = np.abs(mf) ** 2
    p = p / p.max()
    lines = ["# name kind args... expected_file (written by scripts/w5_export.py)"]
    files = {"x": save(GOLD / "x_80k.bin", x), "bp": save(GOLD / "bp_80k.bin", bp),
             "lp": save(GOLD / "lp_80k.bin", lp), "iq": save(GOLD / "iq_80k.bin", iq),
             "rep": save(GOLD / "rep_80k.bin", rep), "mf": save(GOLD / "mf_80k.bin", mf / np.abs(
                 mf).max()), "p": save(GOLD / "p_80k.bin", p)}
    # the matched filter is linear: compare against the normalised output with normalised input
    save(GOLD / "iqn_80k.bin", iq / np.abs(mf).max())
    lines.append(f"demod_80k demod {fs!r} {fc!r} {dec} 0.0 {files['x']} {files['bp']} "
                 f"{files['lp']} {files['iq']}")
    lines.append(f"mf_80k mf iqn_80k.bin {files['rep']} {files['mf']}")
    stride = int(round(fs_bb / bw))
    for kind in ("ca", "go", "os"):
        for st, g in ((1, 2), (stride, 2 * stride)):
            cfg = CFARConfig(kind, n_ref=32, n_guard=g, pfa=1e-4, stride=st)
            thr = cfar_threshold(p, cfg)
            f = save(GOLD / f"thr_{kind}_s{st}.bin", thr)
            lines.append(f"cfar_{kind}_stride{st} cfar {KINDS[kind]} 32 {g} -1 {st} 0.0001 "
                         f"{files['p']} {f}")
    r = range_axis(len(p), fs_bb, c)
    tv = apply_tvg(p, r, alpha, 40.0)
    lines.append(f"tvg_80k tvg {fs_bb!r} {c!r} 0.0 {alpha!r} 40.0 1.0 {files['p']} "
                 f"{save(GOLD / 'tvg_80k.bin', tv / tv.max())}")
    # TVG output is scaled to unit peak; the C++ side is compared on the same scale, so feed it
    # p scaled by 1/max(tv) as well
    save(GOLD / "p_tvgin_80k.bin", p / tv.max())
    lines[-1] = lines[-1].replace(files["p"], "p_tvgin_80k.bin")
    (GOLD / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def bench_inputs(ca: pd.DataFrame) -> None:
    BENCH.mkdir(parents=True, exist_ok=True)
    rows = ca[R2_FEATURES].sample(1000, random_state=SEED).to_numpy(dtype=float)
    feat = np.zeros((len(rows), 64))
    feat[:, : rows.shape[1]] = np.nan_to_num(rows)
    save(BENCH / "features.bin", feat)
    for name, (fc, bw, dec) in SETTINGS.items():
        x, rep, fs, fs_bb, c, alpha = passband_ping(fc, bw, dec, 50.0)
        bp, lp = design_sos(fc, bw, fs)
        stride = int(round(fs_bb / bw))
        cfg = {"fs": repr(fs), "fc": repr(fc), "decimate": dec, "c": repr(c),
               "alpha": repr(alpha), "cfar_kind": 3, "n_ref": 32, "n_guard": 2 * stride,
               "stride": stride, "pfa": 1e-4, "min_snr_db": 20.0, "cands_per_ping": 20,
               "deploy_model": "r2_t300",
               "ping": save(BENCH / f"ping_{name}.bin", x),
               "bp": save(BENCH / f"bp_{name}.bin", bp),
               "lp": save(BENCH / f"lp_{name}.bin", lp),
               "replica": save(BENCH / f"rep_{name}.bin", rep), "features": "features.bin"}
        (BENCH / f"bench_{name}.txt").write_text(
            "\n".join(f"{k} {v}" for k, v in cfg.items()) + "\n", encoding="utf-8")


def models(fr, ob, ca, boot, accuracy: bool = True) -> None:
    lake_c = ca[ca.site == "lake"]
    y = lake_c.y.to_numpy()
    specs = {"r0": ("R0", {}), "r1": ("R1", {})}
    for t in TREES:
        specs[f"r2_t{t}"] = ("R2", {"params": {**R2Trees().params, "n_estimators": t}})
    cms = [r0_to_c("r0", SIZE_GATE_M, list(R2_FEATURES))]
    r1 = R1Logistic(seed=SEED).fit(lake_c, y)
    cms.append(logistic_to_c(r1.model, "r1", list(R1_FEATURES), list(R2_FEATURES)))
    boosters = {}
    for t in TREES:
        m = R2Trees(seed=SEED, params={**R2Trees().params, "n_estimators": t}).fit(lake_c, y)
        boosters[t] = m
        cms.append(lgbm_to_c(m.model.booster_, f"r2_t{t}", list(R2_FEATURES)))
    GEN.mkdir(parents=True, exist_ok=True)
    write_c(cms, GEN / "models.c", GEN / "models.json")
    # reference scores for the parity test: 5,000 lake candidates, Python double precision
    ref = lake_c.sample(5000, random_state=SEED)
    out = {"r0": np.where((np.maximum(ref.range_ext_m, ref.cross_ext_m) >= SIZE_GATE_M[0])
                          & (np.maximum(ref.range_ext_m, ref.cross_ext_m) <= SIZE_GATE_M[1]),
                          ref.snr_db, -np.inf),
           "r1": r1.score(ref)}
    for t, m in boosters.items():
        out[f"r2_t{t}"] = m.score(ref)
    ref_df = pd.DataFrame(ref[R2_FEATURES].to_numpy(), columns=R2_FEATURES)
    for k, v in out.items():
        ref_df[f"score_{k}"] = v
    ref_df["y"] = ref.y.to_numpy()
    ref_df.to_parquet(BENCH / "model_reference.parquet")
    # small committed copy for the CI parity test: 300 rows, features then one score per model
    small = ref_df.iloc[:300]
    save(GOLD / "models_x.bin", small[R2_FEATURES].to_numpy(dtype=float))
    save(GOLD / "models_scores.bin", small[[f"score_{k}" for k in out]].to_numpy(dtype=float))

    if not accuracy:
        return
    # accuracy for the Pareto plot: cross-session recall at 0.05 FAPF (W3 protocol)
    lake = fr[fr.site == "lake"]
    rows = []
    for name, spec in specs.items():
        s_all = np.full(len(ca), np.nan)
        t_all = np.full(len(ca), np.nan)
        for g in BODY_GROUPS:
            te_f = set(lake.frame_id[lake.grp == g])
            tr_f = set(lake.frame_id[lake.grp != g])
            te, s, _, thr = run_fold(spec, ca, tr_f, te_f, fr)
            s_all[te], t_all[te] = s, thr
        te = ~np.isnan(s_all)
        tef = set(ca.frame_id[te]) | set(lake.frame_id[lake.grp.isin(BODY_GROUPS)])
        fm = fr.loc[sorted(tef)]
        row, _, _ = summarise("cross-session", "pooled", name, ob[ob.frame_id.isin(tef)],
                              ca[te], s_all[te], None, t_all[te], fm, boot)
        rows.append(row)
        print(name, round(row["recall_at_op"], 3), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "model_accuracy.csv", index=False, float_format="%.4f")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--no-accuracy", action="store_true", help="only regenerate the C models")
    args = ap.parse_args()
    fr, ob, ca = load()
    golden()
    bench_inputs(ca)
    models(fr, ob, ca, args.boot, accuracy=not args.no_accuracy)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "export_meta.json").write_text(json.dumps({
        "script": "w5_export", "git_commit": git_commit(), "data_manifest": manifest_hash(),
        "config": {"settings": SETTINGS, "trees": TREES, "fapf": FAPF, "seed": SEED,
                   "boot": args.boot}}, indent=2))


if __name__ == "__main__":
    main()
