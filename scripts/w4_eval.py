"""W4: field-honest evaluation of the W3 ladder (per condition, calibration, mission cost,
threshold recalibration from body-free frames, OOD rescan score, QA).

Inputs: data/processed/uatd_*.parquet from scripts/w3_candidates.py, w3_ladder.py and
w4_qa.py. Outputs: reports/w4/*.csv and reports/figures/w4/*.png.

Run: uv run python scripts/w4_eval.py [--boot 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import PROC, ROOT, git_commit, manifest_hash  # noqa: E402
from w3_ladder import FAPF, load, object_scores  # noqa: E402

from echofind import viz  # noqa: E402
from echofind.eval.bootstrap import Clustered, cluster_bootstrap  # noqa: E402
from echofind.eval.cost import (  # noqa: E402
    MissionCost,
    best_threshold,
    expected_time,
    operating_curve,
)
from echofind.eval.metrics import ece, threshold_at_fapf  # noqa: E402
from echofind.eval.ood import MahalanobisOOD, flag_threshold  # noqa: E402
from echofind.features.fls import R2_FEATURES  # noqa: E402

OUT = ROOT / "reports" / "w4"
FIG = ROOT / "reports" / "figures" / "w4"
SEED = 0
BANDS = [(0, 6), (6, 9), (9, 12), (12, 30)]
FRAMES_PER_SCAN = MissionCost().frames_per_scan
plt = viz.plt
RUNG_COLOR = {"R0": viz.SERIES[0], "R1": viz.SERIES[1], "R2": viz.SERIES[2]}

# (label, OOF column prefix, frame filter); filter picks the test frames of that evaluation
EVALS = [
    ("cross-session", "cross-session", lambda f: (f.site == "lake")
     & f.grp.isin(["720k_r10", "720k_r15", "1200k_r10", "1200k_r15"])),
    ("720k->1200k", "cross-frequency", lambda f: (f.site == "lake") & (f.freq_khz == 1200)),
    ("1200k->720k", "cross-frequency", lambda f: (f.site == "lake") & (f.freq_khz == 720)),
]


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


class Ctx:
    """Data for one evaluation: test frames, their candidates, body objects and scores."""

    def __init__(self, fr, ob, ca, oof, label, prefix, filt, rung):
        self.fm = fr[filt(fr)]
        m = ca.frame_id.isin(self.fm.frame_id).to_numpy()
        self.ca = ca[m]
        self.ob = ob[ob.frame_id.isin(self.fm.frame_id)].copy()
        self.s = oof[f"{prefix}:{rung}"].to_numpy()[m]
        self.thr_t = oof[f"{prefix}:{rung}:thr"].to_numpy()[m]
        self.neg = self.ca.y.to_numpy() == 0
        self.o = object_scores(self.ob, self.ca, self.s)
        self.thr_op = threshold_at_fapf(self.s[self.neg], len(self.fm), FAPF)
        self.label, self.rung = label, rung


def body_range(ob: pd.DataFrame, fr: pd.DataFrame) -> np.ndarray:
    f = fr.loc[ob.frame_id]
    yc = (ob.ymin.to_numpy() + ob.ymax.to_numpy()) / 2
    return yc / f.height.to_numpy() * f.range_m.to_numpy()


# ------------------------------------------------------------------ A. per range band
def by_range(ctxs, fr, boot) -> pd.DataFrame:
    rows = []
    for c in ctxs:
        rng_o = body_range(c.ob, fr)
        rng_c = c.ca.range_m.to_numpy()
        fpc = c.fm.groupby("session").size().to_dict()
        for lo, hi in BANDS:
            mo = (rng_o >= lo) & (rng_o < hi)
            mn = c.neg & (rng_c >= lo) & (rng_c < hi)
            row = {"eval": c.label, "rung": c.rung, "band_m": f"{lo}-{hi}", "bodies": int(mo.sum()),
                   "fa_per_frame": float(np.sum(c.s[mn] >= c.thr_op) / len(c.fm))}
            row["body_sessions"] = int(c.ob.session[mo].nunique())
            if mo.sum():
                cl = Clustered.from_frame(c.ob.session[mo], c.o[mo], c.ca.session[mn], c.s[mn],
                                          fpc)
                t = c.thr_op
                pt, lo_, hi_ = cluster_bootstrap(cl, lambda o, n, f, t=t: float(np.mean(o >= t))
                                                 if len(o) else np.nan, boot, SEED)
                if row["body_sessions"] < 3:     # too few clusters for an interval
                    lo_, hi_ = np.nan, np.nan
                row.update(recall=pt, recall_lo=lo_, recall_hi=hi_)
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B. calibration
def calibration(ctxs) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, curves = [], []
    for c in ctxs:
        if c.rung == "R0":
            continue
        p, y = sigmoid(c.s), c.ca.y.to_numpy()
        hi = p >= 0.05
        rows.append({"eval": c.label, "rung": c.rung, "ece_all": ece(p, y),
                     "ece_p_ge_0.05": ece(p[hi], y[hi]) if hi.any() else np.nan,
                     "share_p_ge_0.05": float(hi.mean()),
                     "mean_p_body": float(p[y == 1].mean()), "base_rate": float(y.mean())})
        edges = np.r_[0, np.logspace(-3, 0, 13)]
        b = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
        for k in range(len(edges) - 1):
            m = b == k
            if m.sum() >= 20:
                curves.append({"eval": c.label, "rung": c.rung, "p_mean": p[m].mean(),
                               "observed": y[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows), pd.DataFrame(curves)


# ------------------------------------------------------------------ C. mission cost
def mission(ctxs) -> pd.DataFrame:
    rows = []
    grid = [(tc, tr, k) for tc in (60, 180, 600) for tr in (600, 1800, 7200) for k in (1, 3)]
    for c in ctxs:
        thr, fa, rec = operating_curve(c.o, c.s[c.neg], len(c.fm))
        i05 = int(np.argmin(np.abs(fa - FAPF)))
        for tc, tr, k in grid:
            mc = MissionCost(t_check_s=tc, t_research_s=tr, looks=k)
            b = best_threshold(c.o, c.s[c.neg], len(c.fm), mc)
            et05 = float(expected_time(fa[i05], rec[i05], mc))
            rows.append({"eval": c.label, "rung": c.rung, "t_check_s": tc, "t_research_s": tr,
                         "looks": k, **{f"opt_{kk}": v for kk, v in b.items()},
                         "et_at_0.05_s": et05, "et_no_detector_s": 60.0 + tr,
                         "saving_vs_0.05_s": et05 - b["expected_time_s"]})
    return pd.DataFrame(rows)


def mission_figure(ctxs) -> None:
    c = next(x for x in ctxs if x.label == "cross-session" and x.rung == "R2")
    thr, fa, rec = operating_curve(c.o, c.s[c.neg], len(c.fm))
    ok = (fa > 1e-3) & (fa <= 2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for j, (tr, k) in enumerate([(1800, 1), (7200, 1), (1800, 3)]):
        for tc, ls in ((60, ":"), (180, "-"), (600, "--")):
            et = expected_time(fa, rec, MissionCost(t_check_s=tc, t_research_s=tr, looks=k))
            ax = axes[0] if k == 1 else axes[1]
            ax.plot(fa[ok], et[ok] / 60, ls=ls, color=viz.SERIES[j],
                    label=f"re-search {tr // 60} min, check {tc // 60} min")
            i = np.argmin(np.where(ok, et, np.inf))
            ax.plot(fa[i], et[i] / 60, "o", color=viz.SERIES[j], ms=4)
    for ax, t in zip(axes, ("One look per scan (k = 1)", "Three independent looks (k = 3)"),
                     strict=True):
        ax.set_xscale("log")
        ax.axvline(FAPF, color=viz.MUTED, lw=1, ls="--")
        ax.set_xlabel("False alarms per frame (30 frames per scan)")
        ax.set_title(t)
    axes[0].set_ylabel("Expected time to clear area (min)")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Mission cost, R2 cross-session: dots mark the optimum; dashed line = 0.05 FAPF",
                 x=0.01, ha="left", fontsize=10)
    viz.save(fig, FIG / "mission_cost.png")
    plt.close(fig)


# ------------------------------------------------------------------ D. recalibration
def recalibration(fr, ob, ca, oof, reps=200) -> pd.DataFrame:
    """Set the 0.05 FAPF threshold from n body-free frames of the target domain, then score the
    remaining target frames. Body-free frames need no target, so a crew can record them."""
    rng = np.random.default_rng(SEED)
    rows = []
    body_frames = set(ob.frame_id)
    setups = [(lab, pre, filt, rung) for lab, pre, filt in EVALS[1:] for rung in ("R1", "R2")]
    setups += [("lake->sea", "sea-fa", lambda f: f.site == "sea", r) for r in ("R1", "R2")]
    for lab, pre, filt, rung in setups:
        c = Ctx(fr, ob, ca, oof, lab, pre, filt, rung)
        free = np.array([f for f in c.fm.frame_id if f not in body_frames])
        cand_f = c.ca.frame_id.to_numpy()
        base = {"eval": lab, "rung": rung, "body_free_frames": len(free)}
        tr = c.thr_t[0]
        rows.append({**base, "n_cal": 0, "fapf_med": np.mean(c.s[c.neg] >= tr) * c.neg.sum()
                     / len(c.fm), "fapf_p10": np.nan, "fapf_p90": np.nan,
                     "recall_med": float(np.mean(c.o >= tr)) if len(c.o) else np.nan})
        for n in (10, 25, 50, 100, 200, 400):
            if n >= len(free) * 0.8:
                continue
            fa_v, rec_v = [], []
            for _ in range(reps):
                cal = set(rng.choice(free, n, replace=False))
                is_cal = np.fromiter((f in cal for f in cand_f), bool, len(cand_f))
                t = threshold_at_fapf(c.s[c.neg & is_cal], n, FAPF)
                ev = ~is_cal
                n_ev = len(c.fm) - n
                fa_v.append(np.sum(c.s[c.neg & ev] >= t) / n_ev)
                if len(c.o):
                    rec_v.append(np.mean(c.o >= t))
            rows.append({**base, "n_cal": n, "fapf_med": np.median(fa_v),
                         "fapf_p10": np.quantile(fa_v, 0.1), "fapf_p90": np.quantile(fa_v, 0.9),
                         "recall_med": np.median(rec_v) if rec_v else np.nan})
    return pd.DataFrame(rows)


def recal_figure(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, ((lab, rung), d) in enumerate(df.groupby(["eval", "rung"], sort=False)):
        d = d.sort_values("n_cal")
        x = d.n_cal.replace(0, 3)
        ax.plot(x, d.fapf_med, marker="o", ms=3, color=viz.SERIES[i], label=f"{lab} {rung}")
        ax.fill_between(x, d.fapf_p10, d.fapf_p90, color=viz.SERIES[i], alpha=0.12, lw=0)
    ax.axhline(FAPF, color=viz.MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([3, 10, 25, 50, 100, 200, 400])
    ax.set_xticklabels(["source\nthreshold", "10", "25", "50", "100", "200", "400"])
    ax.set_xlabel("Body-free frames from the new condition used to set the threshold")
    ax.set_ylabel("Realised false alarms per frame")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Re-setting the threshold from body-free frames (median, 10–90% band)")
    viz.save(fig, FIG / "recalibration.png")
    plt.close(fig)


# ------------------------------------------------------------------ E. OOD
def ood(fr, ob, ca, oof) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows, err = [], []
    lake = fr[fr.site == "lake"]
    for src in (720, 1200):
        hold = f"{src}k_r15"
        train_f = lake[(lake.freq_khz == src) & (lake.grp != hold)].frame_id.to_numpy()
        rng.shuffle(train_f)
        fit_f, cal_f = set(train_f[::2]), set(train_f[1::2])
        cfit = ca[ca.frame_id.isin(fit_f)]
        model = MahalanobisOOD(R2_FEATURES).fit(cfit.sample(min(60000, len(cfit)),
                                                           random_state=SEED))
        fs = model.frame_score(ca, ca.frame_id.to_numpy())
        thr = flag_threshold(fs[fs.index.isin(cal_f)])
        other = 1200 if src == 720 else 720
        tests = {
            "in-dist, new session": lake[lake.grp == hold].frame_id,
            "new frequency": lake[lake.freq_khz == other].frame_id,
            "new site, same frequency": fr[(fr.site == "sea") & (fr.freq_khz == src)].frame_id,
            "new site and frequency": fr[(fr.site == "sea") & (fr.freq_khz == other)].frame_id,
        }
        ref = fs.reindex(tests["in-dist, new session"]).dropna()
        for name, fids in tests.items():
            v = fs.reindex(fids).dropna()
            rows.append({"source": f"lake {src} kHz", "test": name, "frames": len(v),
                         "flag_rate": float(np.mean(v > thr)),
                         "auroc_vs_new_session": np.nan if name.startswith("in-dist") else
                         float(roc_auc_score(np.r_[np.zeros(len(ref)), np.ones(len(v))],
                                             np.r_[ref.values, v.values]))})
        # do flags line up with errors? cross-frequency misses and false alarms on target freq
        c = Ctx(fr, ob, ca, oof, f"{src}k->{other}k", "cross-frequency",
                lambda f, o=other: (f.site == "lake") & (f.freq_khz == o), "R2")
        flagged = set(fs.index[fs > thr])
        fo = c.ob.frame_id.isin(flagged).to_numpy()
        fc = c.ca.frame_id.isin(flagged).to_numpy()
        t = c.thr_t[0]
        for state, mo, mc in (("flagged", fo, fc), ("not flagged", ~fo, ~fc)):
            nfr = c.fm.frame_id.isin(flagged).sum() if state == "flagged" else \
                (~c.fm.frame_id.isin(flagged)).sum()
            err.append({"eval": c.label, "frames": state, "n_frames": int(nfr),
                        "bodies": int(mo.sum()),
                        "recall_transfer": float(np.mean(c.o[mo] >= t)) if mo.any() else np.nan,
                        "fapf_transfer": float(np.sum(c.s[c.neg & mc] >= t) / max(nfr, 1))})
    # sea false alarms of the all-lake model: flagged by an all-lake OOD model?
    lf = lake.frame_id.to_numpy().copy()
    rng.shuffle(lf)
    fit_f, cal_f = set(lf[::2]), set(lf[1::2])
    cfit = ca[ca.frame_id.isin(fit_f)]
    model = MahalanobisOOD(R2_FEATURES).fit(cfit.sample(60000, random_state=SEED))
    d2 = model.candidate_score(ca)
    cal_m = ca.frame_id.isin(cal_f).to_numpy()
    ct = flag_threshold(d2[cal_m])
    sea = (ca.site == "sea").to_numpy()
    s, t = oof["sea-fa:R2"].to_numpy(), oof["sea-fa:R2:thr"].to_numpy()
    fa = sea & (s >= t)
    for cls in ["plane", "background", "rov", "all"]:
        m = fa & ((ca.cls == cls).to_numpy() if cls != "all" else True)
        err.append({"eval": "lake->sea R2 false alarms", "frames": f"class {cls}",
                    "n_frames": int(m.sum()), "bodies": 0,
                    "share_candidates_flagged": float(np.mean(d2[m] > ct)) if m.any() else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(err)


def calib_figure(curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, lab in zip(axes, ["cross-session", "720k->1200k", "1200k->720k"], strict=True):
        for rung in ("R1", "R2"):
            d = curves[(curves["eval"] == lab) & (curves.rung == rung)]
            ax.plot(d.p_mean, d.observed, marker="o", ms=3, color=RUNG_COLOR[rung], label=rung)
        ax.plot([1e-3, 1], [1e-3, 1], color=viz.MUTED, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(lab)
        ax.set_xlabel("Predicted P(body) per candidate")
    axes[0].set_ylabel("Observed body share")
    axes[0].legend()
    fig.suptitle("Reliability (log bins); dashed = perfect calibration", x=0.01, ha="left",
                 fontsize=10)
    viz.save(fig, FIG / "reliability.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    viz.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    fr, ob, ca = load()
    oof = pd.read_parquet(PROC / "uatd_oof_scores.parquet")
    ctxs = [Ctx(fr, ob, ca, oof, lab, pre, filt, r) for lab, pre, filt in EVALS
            for r in ("R0", "R1", "R2")]

    br = by_range(ctxs, fr, args.boot)
    br.to_csv(OUT / "recall_by_range.csv", index=False, float_format="%.4f")
    print(br.to_string(index=False, float_format="%.3f"))

    cal, curves = calibration(ctxs)
    cal.to_csv(OUT / "calibration.csv", index=False, float_format="%.4f")
    curves.to_csv(OUT / "reliability_curves.csv", index=False, float_format="%.5f")
    calib_figure(curves)
    print(cal.to_string(index=False, float_format="%.4f"))

    ms = mission(ctxs)
    ms.to_csv(OUT / "mission_cost.csv", index=False, float_format="%.4f")
    mission_figure(ctxs)
    print(ms[(ms.t_check_s == 180) & (ms.t_research_s == 1800)].to_string(
        index=False, float_format="%.3f"))

    rc = recalibration(fr, ob, ca, oof)
    rc.to_csv(OUT / "recalibration.csv", index=False, float_format="%.4f")
    recal_figure(rc)
    print(rc.to_string(index=False, float_format="%.4f"))

    oo, oe = ood(fr, ob, ca, oof)
    oo.to_csv(OUT / "ood_flags.csv", index=False, float_format="%.4f")
    oe.to_csv(OUT / "ood_vs_errors.csv", index=False, float_format="%.4f")
    print(oo.to_string(index=False, float_format="%.3f"))
    print(oe.to_string(index=False, float_format="%.3f"))

    (OUT / "eval_meta.json").write_text(json.dumps({
        "script": "w4_eval", "git_commit": git_commit(), "data_manifest": manifest_hash(),
        "config": {"fapf": FAPF, "boot": args.boot, "seed": SEED, "bands_m": BANDS,
                   "frames_per_scan": FRAMES_PER_SCAN}}, indent=2))


if __name__ == "__main__":
    main()
