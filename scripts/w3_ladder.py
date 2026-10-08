"""W3: run ladder rungs R0 to R2 over the UATD splits and write the checkpoint results table.

Inputs: data/processed/uatd_{frames,objects,candidates}.parquet (scripts/w3_candidates.py).
Outputs (reports/w3/): ladder_results.csv, ladder_paired.csv, froc.csv, sea_false_alarms.csv,
and data/processed/uatd_oof_scores.parquet for the gallery and feature pages.

Splits (reports/decisions/201 and 203):
- random: 5-fold over lake frames. Leaks near-duplicate frames; kept to show H4.
- cross-session: leave one (frequency, range band) group out, over the four lake groups that
  hold bodies; every other lake frame always trains.
- cross-frequency: train on lake 720 kHz, test on lake 1200 kHz, and the reverse.
- sea (false alarms only): train on all lake frames, test on sea frames, which have no bodies.

Operating point: 0.05 false alarms per frame (FAPF), fixed before results (decision 203).

Run: uv run python scripts/w3_ladder.py [--boot 1000]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import OUT, PROC, ROOT, git_commit, manifest_hash, write_meta  # noqa: E402

from echofind.eval import splits as sp  # noqa: E402
from echofind.eval.bootstrap import (  # noqa: E402
    Clustered,
    cluster_bootstrap,
    paired_cluster_bootstrap,
    recall_metric,
)
from echofind.eval.metrics import (  # noqa: E402
    ece,
    froc,
    object_pr_auc,
    threshold_at_fapf,
)
from echofind.features.fls import R2_FEATURES  # noqa: E402
from echofind.models.ladder import make, raw_view  # noqa: E402

FAPF = 0.05
FAPF_GRID = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
BODY = "human body"
SEED = 0
BODY_GROUPS = ["720k_r10", "720k_r15", "1200k_r10", "1200k_r15"]

# feature groups for the drop-one-group ablation (H6)
GROUPS = {
    "shadow": ["shadow_depth_db", "shadow_len_m", "shadow_min_db", "shadow_frac",
               "front_back_db"],
    "extent": ["range_ext_m", "cross_ext_m", "area_m2", "aspect", "fill", "n_cells",
               "range_w3_m", "range_w6_m", "cross_w3_m", "cross_w6_m", "orient", "ecc"],
    "highlights": ["n_hl", "hl_spacing_m", "hl2_rel_db", "hl_energy_frac", "peak_offset"],
    "level": ["peak_db", "snr_db", "contrast_db", "contrast2_db", "mean_db", "p50_db", "p90_db",
              "energy_db", "bg_db", "front_contrast_db"],
    "texture": ["speckle_c", "skew", "kurt", "grad_mean", "entropy", "bg_std_db"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(R2_FEATURES)


def rung_specs() -> dict[str, tuple[str, dict]]:
    specs = {"R0": ("R0", {}), "R1": ("R1", {}), "R2": ("R2", {}),
             "R2-raw": ("R2", {"features": raw_view(R2_FEATURES)})}
    for gname, cols in GROUPS.items():
        specs[f"R2-no-{gname}"] = ("R2", {"features": [c for c in R2_FEATURES if c not in cols]})
    return specs


def load():
    fr = pd.read_parquet(PROC / "uatd_frames.parquet")
    ob = pd.read_parquet(PROC / "uatd_objects.parquet")
    ca = pd.read_parquet(PROC / "uatd_candidates.parquet")
    fr["rband"] = (fr.range_m / 5).round().astype(int) * 5
    fr["grp"] = fr.freq_khz.astype(str) + "k_r" + fr.rband.astype(str)
    fr = fr.set_index("frame_id", drop=False)
    ca["y"] = (ca.cls == BODY).astype(int)
    for c in ("site", "freq_khz", "grp", "session"):
        ca[c] = fr.loc[ca.frame_id, c].to_numpy()
    ob = ob[ob.cls == BODY].copy()
    for c in ("site", "freq_khz", "grp", "session"):
        ob[c] = fr.loc[ob.frame_id, c].to_numpy()
    return fr, ob, ca


def inner_threshold(spec, ca_tr: pd.DataFrame, n_frames: int) -> float:
    """Threshold at FAPF from out-of-fold scores on the training side (grouped by session)."""
    name, kw = spec
    if name == "R0":
        s = make(name, **kw).score(ca_tr)
    else:
        s = np.full(len(ca_tr), np.nan)
        for tr, te in GroupKFold(3).split(ca_tr, groups=ca_tr.session):
            m = make(name, seed=SEED, **kw).fit(ca_tr.iloc[tr], ca_tr.y.to_numpy()[tr])
            s[te] = m.score(ca_tr.iloc[te])
    return threshold_at_fapf(s[ca_tr.y.to_numpy() == 0], n_frames, FAPF)


def run_fold(spec, ca, tr_frames, te_frames, fr):
    """Fit on training frames; score test candidates; return scores, probs and threshold."""
    name, kw = spec
    tr = ca.frame_id.isin(tr_frames).to_numpy()
    te = ca.frame_id.isin(te_frames).to_numpy()
    model = make(name, seed=SEED, **kw).fit(ca[tr], ca.y.to_numpy()[tr])
    thr = inner_threshold(spec, ca[tr], len(tr_frames))
    return te, model.score(ca[te]), model.prob(ca[te]), thr


def object_scores(ob_te: pd.DataFrame, ca_te: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    """Best candidate score per body object (-inf if the proposal stage missed it)."""
    hit = ca_te.obj_idx.to_numpy()
    best = pd.Series(s).groupby(hit).max()
    return ob_te.obj_idx.map(best).fillna(-np.inf).to_numpy()


def summarise(split, fold, rung, ob_te, ca_te, s, p, thr_c, frames_te, boot):
    neg = ca_te.y.to_numpy() == 0
    o = object_scores(ob_te, ca_te, s)
    # realised performance with each candidate's own fold threshold
    passed = s >= thr_c
    found = pd.Series(passed).groupby(ca_te.obj_idx.to_numpy()).any()
    found_o = ob_te.obj_idx.map(found).fillna(False).to_numpy()
    n_frames = len(frames_te)
    fa_real = float(passed[neg].sum() / n_frames)
    rec_real = float(found_o.mean()) if len(o) else np.nan
    tp, fn, fp = found_o.sum(), (~found_o).sum(), passed[neg].sum()
    f1_real = float(2 * tp / max(2 * tp + fp + fn, 1))
    fpc = frames_te.groupby("session").size().to_dict()
    cl = Clustered.from_frame(ob_te.session, o, ca_te.session[neg], s[neg], fpc)
    pt, lo, hi = cluster_bootstrap(cl, recall_metric(FAPF), boot, SEED) if len(o) else (
        np.nan, np.nan, np.nan)
    row = {"split": split, "fold": fold, "rung": rung, "frames": n_frames, "bodies": len(o),
           "negatives": int(neg.sum()), "clusters": len(fpc),
           "recall_at_op": pt, "recall_lo": lo, "recall_hi": hi,
           "pr_auc": object_pr_auc(o, s[neg]) if len(o) else np.nan,
           "recall_transfer": rec_real, "fapf_transfer": fa_real, "f1_transfer": f1_real,
           "ece": ece(p, ca_te.y.to_numpy()) if p is not None else np.nan}
    return row, cl, o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--rungs", nargs="*", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    fr, ob, ca = load()
    lake = fr[fr.site == "lake"]
    sea = fr[fr.site == "sea"]

    folds: dict[str, list] = {
        "random": [(n, lake.frame_id[tr], lake.frame_id[te])
                   for n, tr, te in sp.random_frames(lake, 5, SEED)],
        "cross-session": [], "cross-frequency": [],
        "sea-fa": [("lake->sea", lake.frame_id, sea.frame_id)],
    }
    for fold in sp.leave_one_group_out(lake, "grp", BODY_GROUPS):
        sp.check_disjoint(lake, fold, "grp")
        n, tr, te = fold
        folds["cross-session"].append((n, lake.frame_id[tr], lake.frame_id[te]))
    for a, b in ((720, 1200), (1200, 720)):
        n, tr, te = sp.transfer(lake, "freq_khz", a, b)
        folds["cross-frequency"].append((f"{a}k->{b}k", lake.frame_id[tr], lake.frame_id[te]))

    specs = rung_specs()
    run_specs = {
        "random": ["R0", "R1", "R2"],
        "cross-session": list(specs),
        "cross-frequency": ["R0", "R1", "R2", "R2-raw"],
        "sea-fa": ["R0", "R1", "R2"],
    }
    if args.rungs:
        run_specs = {k: [r for r in v if r in args.rungs] for k, v in run_specs.items()}

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("w3-ladder")
    meta = write_meta("ladder", {"fapf": FAPF, "boot": args.boot, "seed": SEED,
                                 "body_groups": BODY_GROUPS, "run_specs": run_specs})
    rows, paired, froc_rows, sea_rows = [], [], [], []
    oof = ca[["frame_id", "obj_idx", "cls", "y", "session", "grp", "freq_khz", "site"]].copy()

    for split, fl in folds.items():
        # evaluation units: pooled over folds for random / cross-session, per fold otherwise
        pooled = split in ("random", "cross-session")
        store: dict[str, dict] = {}
        for rung in run_specs[split]:
            spec = specs[rung]
            s_all = np.full(len(ca), np.nan)
            p_all = np.full(len(ca), np.nan)
            t_all = np.full(len(ca), np.nan)
            for fname, trf, tef in fl:
                te, s, p, thr = run_fold(spec, ca, set(trf), set(tef), fr)
                s_all[te], t_all[te] = s, thr
                if p is not None:
                    p_all[te] = p
                if not pooled:
                    fm = fr.loc[list(tef)]
                    ca_te = ca[te]
                    if split == "sea-fa":
                        n_frames = len(fm)
                        negs = ca_te.cls.to_numpy()
                        passed = s >= thr
                        sea_rows.append({"rung": rung, "fapf_transfer": passed.sum() / n_frames,
                                         "frames": n_frames, "threshold": thr,
                                         **{f"fa_{c}": int(passed[negs == c].sum())
                                            for c in sorted(set(negs))}})
                        continue
                    ob_te = ob[ob.frame_id.isin(tef)]
                    row, cl, o = summarise(split, fname, rung, ob_te, ca_te, s,
                                           p if p is not None else None, thr, fm, args.boot)
                    rows.append(row)
                    store.setdefault(fname, {})[rung] = cl
                    neg = ca_te.y.to_numpy() == 0
                    froc_rows += [{"split": split, "fold": fname, "rung": rung, "fapf": f,
                                   "recall": r} for f, r in
                                  zip(FAPF_GRID, froc(o, s[neg], len(fm), FAPF_GRID), strict=True)]
            if pooled:
                te = ~np.isnan(s_all)
                tef = set(ca.frame_id[te]) | set().union(*[set(x[2]) for x in fl])
                fm = fr.loc[sorted(tef)]
                ob_te = ob[ob.frame_id.isin(tef)]
                ca_te = ca[te]
                has_p = not np.all(np.isnan(p_all[te]))
                row, cl, o = summarise(split, "pooled", rung, ob_te, ca_te, s_all[te],
                                       p_all[te] if has_p else None, t_all[te], fm, args.boot)
                rows.append(row)
                store.setdefault("pooled", {})[rung] = cl
                neg = ca_te.y.to_numpy() == 0
                froc_rows += [{"split": split, "fold": "pooled", "rung": rung, "fapf": f,
                               "recall": r} for f, r in
                              zip(FAPF_GRID, froc(o, s_all[te][neg], len(fm), FAPF_GRID),
                                  strict=True)]
            if rung in ("R0", "R1", "R2", "R2-raw"):
                oof[f"{split}:{rung}"] = s_all
                oof[f"{split}:{rung}:thr"] = t_all
            with mlflow.start_run(run_name=f"{split}/{rung}"):
                mlflow.set_tags({"split": split, "rung": rung, "git_commit": git_commit(),
                                 "data_manifest": manifest_hash()})
                mlflow.log_params({"fapf": FAPF, "seed": SEED, **{
                    k: str(v)[:250] for k, v in spec[1].items()}, "model": spec[0]})
                for r in [x for x in rows if x["split"] == split and x["rung"] == rung]:
                    for k in ("recall_at_op", "recall_lo", "recall_hi", "pr_auc",
                              "recall_transfer", "fapf_transfer", "f1_transfer", "ece"):
                        if np.isfinite(r[k]):
                            mlflow.log_metric(f"{r['fold']}.{k}".replace(">", "_"), r[k])
            print(f"{split:16s} {rung:14s} done", flush=True)

        # paired comparisons up the ladder, plus H5 and H6 contrasts
        comps = [("R0", "R1"), ("R1", "R2"), ("R0", "R2"), ("R2-raw", "R2")]
        comps += [(f"R2-no-{g}", "R2") for g in GROUPS]
        for fname, d in store.items():
            for a, b in comps:
                if a in d and b in d:
                    diff, lo, hi = paired_cluster_bootstrap(d[a], d[b], recall_metric(FAPF),
                                                            args.boot, SEED)
                    paired.append({"split": split, "fold": fname, "lower": a, "upper": b,
                                   "diff_recall": diff, "lo": lo, "hi": hi,
                                   "upper_better": bool(lo > 0)})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "ladder_results.csv", index=False, float_format="%.4f")
    pd.DataFrame(paired).to_csv(OUT / "ladder_paired.csv", index=False, float_format="%.4f")
    pd.DataFrame(froc_rows).to_csv(OUT / "froc.csv", index=False, float_format="%.4f")
    pd.DataFrame(sea_rows).to_csv(OUT / "sea_false_alarms.csv", index=False, float_format="%.4f")
    oof.to_parquet(PROC / "uatd_oof_scores.parquet")
    print(pd.DataFrame(rows)[["split", "fold", "rung", "bodies", "recall_at_op", "recall_lo",
                              "recall_hi", "pr_auc", "recall_transfer", "fapf_transfer"]]
          .to_string(index=False, float_format="%.3f"))
    print(meta["git_commit"][:8], meta["data_manifest"])


if __name__ == "__main__":
    main()
