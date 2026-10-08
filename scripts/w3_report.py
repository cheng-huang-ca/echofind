"""W3: figures and tables for the ladder checkpoint (FROC, SHAP, failure gallery, confusers).

Inputs: reports/w3/{ladder_results,froc}.csv and data/processed/uatd_*.parquet from
scripts/w3_candidates.py and scripts/w3_ladder.py.
Outputs: reports/figures/w3/*.png, reports/w3/{shap_importance,confusers_cross_session,
ladder_table}.csv/.md.

Run: uv run python scripts/w3_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import FIG, OUT, PROC  # noqa: E402
from w3_ladder import GROUPS, load  # noqa: E402

from echofind import viz  # noqa: E402
from echofind.features.fls import R2_FEATURES  # noqa: E402
from echofind.io.uatd import load_image  # noqa: E402
from echofind.models.ladder import R2Trees  # noqa: E402

plt = viz.plt
RUNG_COLOR = {"R0": viz.SERIES[0], "R1": viz.SERIES[1], "R2": viz.SERIES[2]}
SPLIT_TITLE = {"random": "Random frames (leaky)", "cross-session": "Cross-session (lake)",
               "720k->1200k": "720 → 1200 kHz", "1200k->720k": "1200 → 720 kHz"}


def froc_figure(fr: pd.DataFrame) -> None:
    panels = [("random", "pooled"), ("cross-session", "pooled"),
              ("cross-frequency", "720k->1200k"), ("cross-frequency", "1200k->720k")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, (split, fold) in zip(axes, panels, strict=True):
        for rung, col in RUNG_COLOR.items():
            d = fr[(fr.split == split) & (fr.fold == fold) & (fr.rung == rung)]
            ax.plot(d.fapf, d.recall, marker="o", ms=3, color=col, label=rung)
        ax.axvline(0.05, color=viz.MUTED, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_title(SPLIT_TITLE.get(fold if split == "cross-frequency" else split, split))
        ax.set_xlabel("False alarms per frame")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Body recall (object level)")
    axes[0].legend(loc="upper left")
    fig.suptitle("FROC by split; dashed line = operating point 0.05 FAPF", x=0.01, ha="left",
                 fontsize=10)
    viz.save(fig, FIG / "froc.png")
    plt.close(fig)


def shap_figure(ca: pd.DataFrame) -> pd.DataFrame:
    import shap

    lake = ca[ca.site == "lake"]
    m = R2Trees(seed=0).fit(lake, lake.y.to_numpy())
    rng = np.random.default_rng(0)
    pos = np.flatnonzero(lake.y.to_numpy() == 1)
    neg = np.flatnonzero(lake.y.to_numpy() == 0)
    idx = np.r_[rng.choice(pos, min(1500, len(pos)), replace=False),
                rng.choice(neg, 3000, replace=False)]
    X = lake.iloc[idx][R2_FEATURES]
    sv = shap.TreeExplainer(m.model).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({"feature": R2_FEATURES, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    gmap = {f: g for g, fs in GROUPS.items() for f in fs}
    imp["group"] = imp.feature.map(gmap)
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(OUT / "shap_importance.csv", index=False, float_format="%.4f")
    top = imp.head(15)[::-1]
    gcol = {g: viz.SERIES[i] for i, g in enumerate(GROUPS)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].barh(top.feature, top.mean_abs_shap, color=[gcol[g] for g in top.group])
    axes[0].set_xlabel("mean |SHAP| (log-odds)")
    axes[0].set_title("R2: top 15 features (all lake candidates)")
    gs = imp.groupby("group").mean_abs_shap.sum().sort_values()
    axes[1].barh(gs.index, gs.values, color=[gcol[g] for g in gs.index])
    axes[1].set_xlabel("sum of mean |SHAP|")
    axes[1].set_title("By feature group")
    viz.save(fig, FIG / "shap.png")
    plt.close(fig)
    return imp


def crop(frame: pd.Series, cx: float, cy: float, half_m: float = 1.5) -> np.ndarray:
    a = load_image(frame)
    dr = frame.range_m / frame.height
    hy = int(half_m / dr)
    r = max(cy * dr, 1.0)
    hx = int(half_m / (r * np.deg2rad(frame.azimuth_deg) / frame.width))
    y0, x0 = int(cy) - hy, int(cx) - hx
    out = np.full((2 * hy, 2 * hx), np.nan)
    ys, xs = slice(max(y0, 0), min(y0 + 2 * hy, a.shape[0])), slice(max(x0, 0),
                                                                     min(x0 + 2 * hx, a.shape[1]))
    out[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0] = a[ys, xs]
    return 20 * np.log10(np.maximum(out, 0.5 / 255))


def gallery(frames: pd.DataFrame, ca: pd.DataFrame, oof: pd.DataFrame, n: int = 6) -> None:
    s, thr = oof["cross-session:R2"], oof["cross-session:R2:thr"]
    tested = s.notna()
    fa = ca[tested & (ca.y == 0)].assign(score=s[tested & (ca.y == 0)])
    top_fa = fa.sort_values("score", ascending=False).drop_duplicates("frame_id").head(n)
    body = ca[tested & (ca.y == 1)].assign(score=s[tested & (ca.y == 1)])
    best = body.sort_values("score", ascending=False).drop_duplicates("obj_idx")
    missed = best.sort_values("score").head(n)
    hits = best.head(n)
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 7.2))
    rows = [("Highest-scoring false alarms", top_fa), ("Missed bodies (lowest best score)", missed),
            ("Found bodies (highest score)", hits)]
    for i, (title, d) in enumerate(rows):
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if j >= len(d):
                ax.axis("off")
                continue
            c = d.iloc[j]
            f = frames.loc[c.frame_id]
            ax.imshow(crop(f, c.px, c.py), cmap="gray", vmin=-50, vmax=-5, aspect="auto",
                      origin="upper")
            passed = "above" if c.score >= thr[c.name] else "below"
            ax.set_title(f"{c.cls}\n{f.freq_khz} kHz, {c.range_m:.1f} m, {passed} thr",
                         fontsize=7, loc="center")
        axes[i, 0].annotate(title, (0, 1.32), xycoords="axes fraction", fontsize=9,
                            fontweight="bold", ha="left")
    fig.subplots_adjust(hspace=0.6)
    fig.suptitle("R2 cross-session failure gallery: 3 m × 3 m crops, range increases downward, "
                 "20 log10 amplitude (−50 to −5 dB)", x=0.01, ha="left", fontsize=9)
    viz.save(fig, FIG / "failure_gallery.png")
    plt.close(fig)


def confusers(ca: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """False alarms per class at the transferred threshold, cross-session, per rung."""
    out = []
    for rung in ("R0", "R1", "R2"):
        s, t = oof[f"cross-session:{rung}"], oof[f"cross-session:{rung}:thr"]
        m = s.notna() & (ca.y == 0)
        passed = s[m] >= t[m]
        c = ca[m].assign(passed=passed).groupby("cls").agg(candidates=("passed", "size"),
                                                           false_alarms=("passed", "sum"))
        c["fa_rate"] = c.false_alarms / c.candidates
        c["rung"] = rung
        out.append(c.reset_index())
    df = pd.concat(out)
    df.to_csv(OUT / "confusers_cross_session.csv", index=False, float_format="%.4f")
    return df


def near_duplicates(frames: pd.DataFrame, ca: pd.DataFrame) -> pd.DataFrame:
    """H4 trace: distance from each test body candidate to its nearest training body candidate
    (standardised R2 features), under the random and the cross-session split."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from w3_ladder import BODY_GROUPS, SEED

    from echofind.eval import splits as sp

    lake = frames[frames.site == "lake"]
    body = ca[(ca.y == 1) & (ca.site == "lake")]
    X = StandardScaler().fit_transform(body[R2_FEATURES].to_numpy())
    fid = body.frame_id.to_numpy()
    out = []
    for split, folds in (("random", sp.random_frames(lake, 5, SEED)),
                         ("cross-session", sp.leave_one_group_out(lake, "grp", BODY_GROUPS))):
        d_all = []
        for _, tr, te in folds:
            trm, tem = np.isin(fid, lake.frame_id[tr]), np.isin(fid, lake.frame_id[te])
            if tem.sum() == 0 or trm.sum() == 0:
                continue
            d, _ = NearestNeighbors(n_neighbors=1).fit(X[trm]).kneighbors(X[tem])
            d_all.append(d[:, 0])
        d = np.concatenate(d_all)
        out.append({"split": split, "test_body_candidates": len(d),
                    "nn_dist_median": np.median(d), "nn_dist_p10": np.percentile(d, 10),
                    "share_within_0.5": np.mean(d < 0.5)})
    df = pd.DataFrame(out)
    df.to_csv(OUT / "h4_near_duplicates.csv", index=False, float_format="%.4f")
    return df


def per_group(frames: pd.DataFrame, ca: pd.DataFrame, ob: pd.DataFrame,
              oof: pd.DataFrame) -> pd.DataFrame:
    """Cross-session recall at the operating point for each held-out group separately."""
    from w3_ladder import BODY_GROUPS, FAPF, object_scores

    from echofind.eval.metrics import recall_at_fapf

    out = []
    for g in BODY_GROUPS:
        fm = frames[(frames.site == "lake") & (frames.grp == g)]
        m = ca.frame_id.isin(fm.frame_id).to_numpy()
        ob_g = ob[ob.frame_id.isin(fm.frame_id)]
        row = {"held_out": g, "frames": len(fm), "bodies": len(ob_g)}
        for rung in ("R0", "R1", "R2"):
            s = oof[f"cross-session:{rung}"].to_numpy()[m]
            o = object_scores(ob_g, ca[m], s)
            row[rung] = recall_at_fapf(o, s[ca.y.to_numpy()[m] == 0], len(fm), FAPF)
        out.append(row)
    df = pd.DataFrame(out)
    df.to_csv(OUT / "cross_session_by_group.csv", index=False, float_format="%.4f")
    return df


def table() -> None:
    r = pd.read_csv(OUT / "ladder_results.csv")
    r = r[r.rung.isin(["R0", "R1", "R2", "R2-raw"])]
    lines = ["| Split | Fold | Rung | Bodies | Recall @ 0.05 FAPF [95% CI] | PR-AUC | "
             "Transferred: recall / FAPF | F1 | ECE |", "|" + " --- |" * 9]
    for x in r.itertuples():
        ece = "" if pd.isna(x.ece) else f"{x.ece:.3f}"
        lines.append(f"| {x.split} | {x.fold} | {x.rung} | {x.bodies} | {x.recall_at_op:.2f} "
                     f"[{x.recall_lo:.2f}, {x.recall_hi:.2f}] | {x.pr_auc:.2f} | "
                     f"{x.recall_transfer:.2f} / {x.fapf_transfer:.3f} | {x.f1_transfer:.2f} | "
                     f"{ece} |")
    (OUT / "ladder_table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    viz.setup()
    frames, ob, ca = load()
    oof = pd.read_parquet(PROC / "uatd_oof_scores.parquet")
    froc_figure(pd.read_csv(OUT / "froc.csv"))
    table()
    print(per_group(frames, ca, ob, oof).to_string(index=False))
    print(near_duplicates(frames, ca).to_string(index=False))
    print(confusers(ca, oof).to_string(index=False))
    print(shap_figure(ca).head(12).to_string(index=False))
    gallery(frames, ca, oof)


if __name__ == "__main__":
    main()
