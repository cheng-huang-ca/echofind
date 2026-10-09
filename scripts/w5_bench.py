"""W5: benchmark the edge build against the 68 ms budget, and measure what the export costs.

Needs `scripts/w5_export.py` and a built edge/ (cmake -S edge -B edge/build && cmake --build
edge/build). Writes reports/w5/{benchmark.csv, bench_*.json, float32_cost.csv, machine.json,
sizes.csv} and reports/figures/w5/{pareto,stages}.png.

1. C++ per-ping latency (ef_bench, one pinned core, float32 and float64) for a 50 m ping at
   80 kHz and 200 kHz, with p50/p95/max per stage and peak memory.
2. The same stages in Python (echofind.dsp), for the speed-up.
3. Float32 export cost: cross-session folds (W3 protocol) scored on double features and on
   float32-rounded features (what the float C model sees; identical branches by construction,
   tests/test_edge_parity.py), compared with a paired cluster bootstrap.
4. Pareto: cross-session recall at 0.05 FAPF (reports/w5/model_accuracy.csv) against the
   classifier's share of the per-ping budget.

Run: uv run python scripts/w5_bench.py [--pings 500] [--core 2]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import PROC, ROOT, git_commit  # noqa: E402
from w3_ladder import BODY_GROUPS, FAPF, SEED, load, object_scores, summarise  # noqa: E402

from echofind import viz  # noqa: E402
from echofind.dsp.cfar import CFARConfig, cfar_threshold, scale_factor  # noqa: E402
from echofind.dsp.frontend import demodulate  # noqa: E402
from echofind.dsp.matched_filter import matched_filter  # noqa: E402
from echofind.dsp.tvg import apply_tvg, range_axis  # noqa: E402
from echofind.eval.bootstrap import paired_cluster_bootstrap, recall_metric  # noqa: E402
from echofind.features.fls import R2_FEATURES  # noqa: E402
from echofind.models.ladder import R2Trees  # noqa: E402

BUILD = ROOT / "edge" / "build"
BENCH = PROC / "edge"
OUT = ROOT / "reports" / "w5"
FIG = ROOT / "reports" / "figures" / "w5"
BUDGET_MS = 2 * 50 / 1480 * 1e3      # back-to-back pings at the 50 m setting
STAGES = ["demodulate", "matched_filter", "power_os_cfar", "candidates_tvg", "classifier"]
plt = viz.plt


def exe(name: str) -> Path:
    for p in (BUILD / f"{name}.exe", BUILD / name):
        if p.exists():
            return p
    raise SystemExit(f"{name} not built: cmake -S edge -B edge/build && cmake --build edge/build")


def machine() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version()}
    if sys.platform == "win32":
        ps = ("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,"
              "NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json")
        try:
            info.update(json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                                            ps], text=True)))
        except Exception as e:  # noqa: BLE001
            info["cpu_error"] = str(e)
    else:
        info["cpu"] = platform.processor()
    return info


def c_bench(cfg: str, pings: int, core: int, f64: bool) -> dict:
    cmd = [str(exe("ef_bench")), str(BENCH / f"bench_{cfg}.txt"), "--pings", str(pings),
           "--core", str(core)] + (["--f64"] if f64 else [])
    return json.loads(subprocess.check_output(cmd, text=True))


def py_bench(cfg: str, reps: int = 15) -> dict:
    """The same stages in Python (median of `reps` runs, microseconds)."""
    kv = dict(line.split() for line in (BENCH / f"bench_{cfg}.txt").read_text().splitlines())
    fs, fc, dec, c = float(kv["fs"]), float(kv["fc"]), int(kv["decimate"]), float(kv["c"])
    bw = float({"80k": 20e3, "200k": 40e3}[cfg])
    x = np.fromfile(BENCH / kv["ping"], "<f8")
    rd = np.fromfile(BENCH / kv["replica"], "<f8")
    rep = rd[0::2] + 1j * rd[1::2]
    cfg_os = CFARConfig("os", n_ref=32, n_guard=int(kv["n_guard"]), pfa=1e-4,
                        stride=int(kv["stride"]))
    feats = np.fromfile(BENCH / "features.bin", "<f8").reshape(-1, 64)[:20, : len(R2_FEATURES)]
    lake = pd.read_parquet(PROC / "uatd_candidates.parquet", columns=R2_FEATURES + ["cls"]).sample(
        40000, random_state=SEED)
    model = R2Trees(seed=SEED).fit(lake, (lake.cls == "human body").to_numpy().astype(int))
    X = pd.DataFrame(feats, columns=R2_FEATURES)
    t = {k: [] for k in STAGES}
    for _ in range(reps):
        a = time.perf_counter()
        iq = demodulate(x, fs, fc, bw, decimate=dec)
        b = time.perf_counter()
        y = matched_filter(iq, rep)
        d = time.perf_counter()
        p = np.abs(y) ** 2
        thr = cfar_threshold(p, cfg_os)
        e = time.perf_counter()
        r = range_axis(len(p), fs / dec, c)
        apply_tvg(p, r, float(kv["alpha"]))
        _ = np.flatnonzero(p > np.nan_to_num(thr, nan=np.inf)), scale_factor(cfg_os)
        f = time.perf_counter()
        model.score(X)
        g = time.perf_counter()
        for k, v in zip(STAGES, (b - a, d - b, e - d, f - e, g - f), strict=True):
            t[k].append(v * 1e6)
    return {k: float(np.median(v)) for k, v in t.items()}


def float32_cost(boot: int) -> pd.DataFrame:
    """Cross-session recall at 0.05 FAPF, double vs float32-rounded features, same models."""
    fr, ob, ca = load()
    lake = fr[fr.site == "lake"]
    s64 = np.full(len(ca), np.nan)
    s32 = np.full(len(ca), np.nan)
    for g in BODY_GROUPS:
        tr = ca.frame_id.isin(lake.frame_id[lake.grp != g]).to_numpy()
        te = ca.frame_id.isin(lake.frame_id[lake.grp == g]).to_numpy()
        m = R2Trees(seed=SEED).fit(ca[tr], ca.y.to_numpy()[tr])
        s64[te] = m.score(ca[te])
        X32 = ca.loc[te, R2_FEATURES].astype(np.float32).astype(np.float64)
        s32[te] = m.score(X32)
    te = ~np.isnan(s64)
    tef = set(lake.frame_id[lake.grp.isin(BODY_GROUPS)])
    fm = fr.loc[sorted(tef)]
    ob_t = ob[ob.frame_id.isin(tef)]
    rows, cls = [], {}
    for name, s in (("float64", s64[te]), ("float32", s32[te])):
        row, cl, _ = summarise("cross-session", "pooled", f"R2 {name}", ob_t, ca[te], s, None,
                               np.full(te.sum(), np.inf), fm, boot)
        rows.append({k: row[k] for k in ("rung", "recall_at_op", "recall_lo", "recall_hi",
                                         "pr_auc")})
        cls[name] = cl
    d, lo, hi = paired_cluster_bootstrap(cls["float64"], cls["float32"], recall_metric(FAPF), boot,
                                         SEED)
    neg = ca.y.to_numpy()[te] == 0
    t64 = np.sort(s64[te][neg])[::-1][int(FAPF * len(fm))]
    flips = np.mean((s64[te] > t64) != (s32[te] > t64))
    rows.append({"rung": "float32 - float64", "recall_at_op": d, "recall_lo": lo, "recall_hi": hi,
                 "pr_auc": np.nan, "decision_flip_rate": float(flips),
                 "max_abs_score_diff": float(np.nanmax(np.abs(s64[te] - s32[te])))})
    _ = object_scores
    return pd.DataFrame(rows)


def sizes() -> pd.DataFrame:
    rows = []
    for f in ("ef_bench.exe", "ef_bench", "libef_edge.dll", "libef_edge.so"):
        p = BUILD / f
        if p.exists():
            rows.append({"artifact": f, "bytes": p.stat().st_size})
    for p in BUILD.rglob("models.c.o*"):
        rows.append({"artifact": f"object {p.parent.name}/{p.name}", "bytes": p.stat().st_size})
    meta = json.loads((ROOT / "edge" / "generated" / "models.json").read_text())
    for m in meta:
        rows.append({"artifact": f"model {m['name']} ({m.get('trees', '-')} trees, "
                     f"{m.get('leaves', '-')} leaves)", "bytes": np.nan})
    return pd.DataFrame(rows)


def figures(bench: pd.DataFrame, acc: pd.DataFrame) -> None:
    viz.setup()
    # stage breakdown, p95, float32 C++ vs Python
    fig, ax = plt.subplots(figsize=(8, 3.4))
    d = bench[bench.precision.isin(["float32", "python"])]
    labels = [f"{r.setting} {r.precision}" for r in d.itertuples()]
    left = np.zeros(len(d))
    for i, s in enumerate(STAGES):
        v = d[f"{s}_p95_us"].to_numpy() / 1e3
        ax.barh(labels, v, left=left, color=viz.SERIES[i], label=s.replace("_", " "))
        left += v
    ax.axvline(BUDGET_MS, color=viz.SERIES[7], ls="--", lw=1.2)
    ax.text(BUDGET_MS, len(d) - 0.5, " 68 ms budget", color=viz.SERIES[7], fontsize=8,
            va="bottom", ha="right")
    ax.set_xlim(0, BUDGET_MS * 1.08)
    ax.set_xlabel("Time per 50 m ping (ms; C++: p95 per stage; Python: median per stage)")
    ax.legend(fontsize=7, ncol=3, loc="lower right", framealpha=0.9)
    ax.set_title("Per-ping latency by stage, one pinned core")
    viz.save(fig, FIG / "stages.png")
    plt.close(fig)

    # Pareto: recall vs classifier share of the budget (20 candidates per ping, float32 C)
    b = bench[(bench.setting == "200k") & (bench.precision == "float32")].iloc[0]
    fe = b.total_p95_us - b.classifier_p95_us
    pts = []
    for r in acc.itertuples():
        t = b[f"model_{r.rung}_p95_us"]
        pts.append((r.rung, (fe + t) / 1e3, t / 1e3, r.recall_at_op, r.recall_lo, r.recall_hi))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, total, cls_ms, rec, lo, hi in pts:
        col = viz.SERIES[0] if name == "r0" else viz.SERIES[1] if name == "r1" else viz.SERIES[2]
        for ax, x in ((axes[0], cls_ms * 1e3), (axes[1], total)):
            ax.errorbar(x, rec, yerr=[[rec - lo], [hi - rec]], fmt="o", color=col, ms=5,
                        capsize=2, lw=1)
            ax.annotate(name, (x, rec), textcoords="offset points", xytext=(5, 4), fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Classifier time per ping, 20 candidates (µs, p95)")
    axes[0].set_ylabel("Cross-session recall at 0.05 FAPF")
    axes[0].set_title("Accuracy vs classifier cost")
    axes[1].axvline(BUDGET_MS, color=viz.SERIES[7], ls="--", lw=1.2)
    axes[1].set_xlim(0, BUDGET_MS * 1.1)
    axes[1].set_xlabel("Whole chain per 50 m ping at 200 kHz (ms, p95; dashed = 68 ms)")
    axes[1].set_title("Accuracy vs end-to-end latency")
    viz.save(fig, FIG / "pareto.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pings", type=int, default=500)
    ap.add_argument("--core", type=int, default=2)
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    golden = subprocess.run([str(exe("ef_golden")), str(ROOT / "edge" / "tests" / "golden")],
                            capture_output=True, text=True)
    (OUT / "golden.txt").write_text(golden.stdout)
    print(golden.stdout.strip().splitlines()[-1])
    mach = machine()
    (OUT / "machine.json").write_text(json.dumps(mach, indent=2))
    rows = []
    for cfg in ("80k", "200k"):
        for f64 in (False, True):
            j = c_bench(cfg, args.pings, args.core, f64)
            (OUT / f"bench_{cfg}_{j['precision']}.json").write_text(json.dumps(j, indent=1))
            row = {"setting": cfg, "precision": j["precision"], "pings": j["pings"],
                   "samples_passband": j["samples_passband"],
                   "samples_baseband": j["samples_baseband"], "peak_rss_mb": j["peak_rss_mb"],
                   "mean_cfar_candidates": j["mean_cfar_candidates"]}
            for s in STAGES + ["total"]:
                row[f"{s}_p50_us"], row[f"{s}_p95_us"], row[f"{s}_max_us"] = j[f"{s}_us"]
            row["ca_cfar_extra_p95_us"] = j["ca_cfar_extra_us"][1]
            for k, v in j.items():
                if k.startswith("model_"):
                    row[k.replace("_us", "_p95_us")] = v[1]
            row["budget_share_p95"] = row["total_p95_us"] / (BUDGET_MS * 1e3)
            rows.append(row)
        py = py_bench(cfg)
        rows.append({"setting": cfg, "precision": "python",
                     **{f"{s}_p95_us": v for s, v in py.items()},
                     "total_p95_us": sum(py.values()),
                     "budget_share_p95": sum(py.values()) / (BUDGET_MS * 1e3)})
        print(cfg, "done", flush=True)
    bench = pd.DataFrame(rows)
    bench.to_csv(OUT / "benchmark.csv", index=False, float_format="%.2f")
    cost = float32_cost(args.boot)
    cost.to_csv(OUT / "float32_cost.csv", index=False, float_format="%.5f")
    sz = sizes()
    sz.to_csv(OUT / "sizes.csv", index=False)
    acc = pd.read_csv(OUT / "model_accuracy.csv")
    figures(bench, acc)
    (OUT / "bench_meta.json").write_text(json.dumps(
        {"script": "w5_bench", "git_commit": git_commit(), "budget_ms": BUDGET_MS,
         "pings": args.pings, "core": args.core}, indent=2))
    cols = ["setting", "precision"] + [f"{s}_p95_us" for s in STAGES] + ["total_p95_us",
                                                                          "budget_share_p95",
                                                                          "peak_rss_mb"]
    print(bench[cols].to_string(index=False, float_format="%.1f"))
    print(cost.to_string(index=False))
    print(sz.to_string(index=False))


if __name__ == "__main__":
    main()
