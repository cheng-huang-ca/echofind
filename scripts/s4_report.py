"""S4: figures and diagnostics for the deep rungs.

1. Ladder figure: recall at 0.05 FAPF with 95% CI for R0-R2 (W3) and R3-R4 (S4) on
   cross-session and both cross-frequency directions.
2. H5: does physics normalisation (R2 vs R2-raw) narrow the cross-frequency gap more than a
   bigger model (R3/R4 vs R2)?
3. Shortcut check: many UATD frames have a fixed-range seam, a row where the level across all
   beams drops by 6 dB or more and stays down (export padding or a range gate). If the CNN gain
   came from that seam rather than the target, it would sit in candidates with a seam. Recall
   and false alarms are split by "seam in snippet" for R2 and the CNN rungs. Natural
   reverberation decay also passes a 6 dB step test, so this over-counts seams.

Run: uv run python scripts/s4_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import PROC, ROOT  # noqa: E402
from w3_ladder import BODY_GROUPS, FAPF, load, object_scores  # noqa: E402

from echofind import viz  # noqa: E402
from echofind.eval.metrics import threshold_at_fapf  # noqa: E402

OUT = ROOT / "reports" / "s4"
FIG = ROOT / "reports" / "figures" / "s4"
plt = viz.plt
ORDER = ["R0", "R1", "R2", "R3a", "R3b", "R4a", "R4b"]


def ladder_figure() -> pd.DataFrame:
    w3 = pd.read_csv(ROOT / "reports" / "w3" / "ladder_results.csv")
    s4 = pd.read_csv(OUT / "deep_results.csv")
    r = pd.concat([w3[w3.rung.isin(ORDER)], s4])
    r = r[r.split.isin(["cross-session", "cross-frequency"])]
    panels = [("cross-session", "pooled", "Cross-session (lake)"),
              ("cross-frequency", "720k->1200k", "720 → 1,200 kHz"),
              ("cross-frequency", "1200k->720k", "1,200 → 720 kHz")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (split, fold, title) in zip(axes, panels, strict=True):
        d = r[(r.split == split) & (r.fold == fold)].set_index("rung").reindex(ORDER).dropna(
            subset=["recall_at_op"])
        x = np.arange(len(d))
        cols = [viz.SERIES[0] if k in ("R0", "R1", "R2") else viz.SERIES[2] for k in d.index]
        ax.bar(x, d.recall_at_op, color=cols, width=0.6)
        ax.errorbar(x, d.recall_at_op, yerr=[d.recall_at_op - d.recall_lo,
                                             d.recall_hi - d.recall_at_op],
                    fmt="none", ecolor=viz.INK2, capsize=3, lw=1)
        ax.set_xticks(x, d.index)
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Body recall at 0.05 FAPF (95% CI)")
    fig.suptitle("The ladder: W3 rungs (blue) and S4 deep rungs (green)", x=0.01, ha="left",
                 fontsize=10)
    FIG.mkdir(parents=True, exist_ok=True)
    viz.save(fig, FIG / "ladder.png")
    plt.close(fig)
    r.to_csv(OUT / "ladder_all.csv", index=False, float_format="%.4f")
    return r


def h5(r: pd.DataFrame) -> pd.DataFrame:
    w3 = pd.read_csv(ROOT / "reports" / "w3" / "ladder_results.csv")
    rows = []
    for fold in ("720k->1200k", "1200k->720k"):
        g = lambda k, f=fold: float(pd.concat([w3, r]).query(  # noqa: E731
            "split == 'cross-frequency' and fold == @f and rung == @k").recall_at_op.iloc[0])
        row = {"fold": fold, "R2": g("R2"), "R2-raw": g("R2-raw"),
               "normalisation_gain": g("R2") - g("R2-raw")}
        for k in ("R3b", "R4a", "R4b"):
            try:
                row[f"{k}_gain_over_R2"] = g(k) - g("R2")
            except IndexError:
                pass
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "h5.csv", index=False, float_format="%.4f")
    return df


def _frame_seam(frame: dict) -> tuple[str, int, float]:
    """Range row (grid cells) of the largest persistent drop of the across-beam median level,
    and its size in dB. A seam is a drop of 6 dB or more that stays down for 16 cells."""
    from echofind.features.fls import make_grid
    from echofind.io.uatd import load_image

    g = make_grid(load_image(pd.Series(frame)), frame["range_m"], frame["azimuth_deg"],
                  frame["freq_khz"])
    # raw power: the normalised view divides each row by its median, which hides a full seam
    m = np.convolve(np.median(10 * np.log10(np.maximum(g.p, 1e-12)), axis=1), np.ones(3) / 3,
                    mode="same")
    best_r, best_d = -1, 0.0
    for r in range(16, len(m) - 16):
        d = m[r - 16: r].mean() - m[r: r + 16].mean()
        if d > best_d:
            best_r, best_d = r, d
    return frame["frame_id"], best_r, float(best_d)


def frame_seams(fr: pd.DataFrame, jobs: int = 4) -> pd.DataFrame:
    path = PROC / "uatd_frame_seam.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(jobs) as ex:
        rows = list(ex.map(_frame_seam, fr.reset_index(drop=True).to_dict("records"),
                           chunksize=32))
    df = pd.DataFrame(rows, columns=["frame_id", "seam_row", "seam_drop_db"])
    df.to_parquet(path)
    return df


def seam_check(fr, ob, ca) -> pd.DataFrame:
    oof = pd.read_parquet(PROC / "uatd_oof_scores.parquet")
    deep = pd.read_parquet(PROC / "uatd_oof_deep.parquet")
    lake = fr[(fr.site == "lake") & fr.grp.isin(BODY_GROUPS)]
    fs = frame_seams(fr).set_index("frame_id")
    m = ca.frame_id.isin(lake.frame_id).to_numpy()
    idx = np.flatnonzero(m)
    c = ca.iloc[idx]
    srow = fs.seam_row.reindex(c.frame_id).to_numpy()
    sdrop = fs.seam_drop_db.reindex(c.frame_id).to_numpy()
    rel = srow - c.peak_r.to_numpy()          # seam position relative to the snippet peak row
    seam = (sdrop >= 6) & (rel >= -8) & (rel < 40)
    neg = c.y.to_numpy() == 0
    ob_l = ob[ob.frame_id.isin(lake.frame_id)]
    # an object "has a seam" if its best-scoring R2 candidate's snippet has one
    rows = []
    for rung in ["R2", "R3a", "R3b", "R4a", "R4b"]:
        src = oof if rung == "R2" else deep
        col = f"cross-session:{rung}"
        if col not in src:
            continue
        s = src[col].to_numpy()[idx]
        t = threshold_at_fapf(s[neg], len(lake), FAPF)
        best = pd.DataFrame({"obj": c.obj_idx.to_numpy(), "s": s, "seam": seam})
        best = best[best.obj >= 0].sort_values("s").groupby("obj").last()
        o = object_scores(ob_l, c, s)
        oseam = ob_l.obj_idx.map(best.seam).fillna(False).to_numpy().astype(bool)
        for flag in (True, False):
            nm = neg & (seam == flag)
            rows.append({"rung": rung, "seam_in_snippet": flag,
                         "bodies": int((oseam == flag).sum()),
                         "recall": float(np.mean(o[oseam == flag] >= t)),
                         "negatives": int(nm.sum()),
                         "fa_rate_per_candidate": float(np.mean(s[nm] >= t))})
    df = pd.DataFrame(rows)
    df["share_of_negatives_with_seam"] = float(seam[neg].mean())
    df["share_of_body_candidates_with_seam"] = float(seam[~neg].mean())
    df.to_csv(OUT / "seam_check.csv", index=False, float_format="%.4f")
    return df


def main() -> None:
    viz.setup()
    r = ladder_figure()
    print(r[["split", "fold", "rung", "recall_at_op", "recall_lo", "recall_hi"]].to_string(
        index=False, float_format="%.3f"))
    print(h5(r).to_string(index=False, float_format="%.3f"))
    fr, ob, ca = load()
    print(seam_check(fr, ob, ca).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
