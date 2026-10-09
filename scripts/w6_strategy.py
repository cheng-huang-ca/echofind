"""W6: evidence for the data-collection memo (coverage audit, ICC and sample size, field design).

Inputs: data/processed/uatd_*.parquet (W3), reports/w3/bath_legs.csv. Outputs (reports/w6/):
coverage.csv, coverage_uatd_bodies.csv, icc.json, power.csv, field_design.csv,
field_design_summary.csv; figures under reports/figures/w6/.

Run: uv run python scripts/w6_strategy.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import PROC, ROOT, git_commit  # noqa: E402
from w3_ladder import BODY_GROUPS, FAPF, load, object_scores  # noqa: E402

from echofind import viz  # noqa: E402
from echofind.eval.metrics import threshold_at_fapf  # noqa: E402
from echofind.eval.power import (  # noqa: E402
    design_effect,
    icc_binary,
    n_two_proportions,
    placements_per_arm,
    simulate_power,
)

OUT = ROOT / "reports" / "w6"
FIG = ROOT / "reports" / "figures" / "w6"
SEED = 0
BANDS = [(0, 6), (6, 12), (12, 25), (25, 50)]
plt = viz.plt


# ------------------------------------------------------------------ coverage audit
def coverage(fr: pd.DataFrame, ob_all: pd.DataFrame) -> pd.DataFrame:
    body = ob_all[ob_all.cls == "human body"].merge(
        fr[["frame_id", "site", "freq_khz", "range_m", "height", "session"]].reset_index(drop=True),
        on="frame_id")
    body["range_body_m"] = (body.ymin + body.ymax) / 2 / body.height * body.range_m
    body["band"] = pd.cut(body.range_body_m, [b[0] for b in BANDS] + [BANDS[-1][1]],
                          labels=[f"{a}-{b} m" for a, b in BANDS], right=False)
    t = body.groupby(["freq_khz", "band"], observed=False).agg(
        objects=("cls", "size"), sessions=("session", "nunique")).reset_index()
    t.to_csv(OUT / "coverage_uatd_bodies.csv", index=False)

    legs = pd.read_csv(ROOT / "reports" / "w3" / "bath_legs.csv")
    bath_t = legs[(legs.label == "target")]
    n_b = len(body)
    rows = [
        # dimension, level, UATD body objects, Bath target legs, what AquaEye needs
        ("target", "mannequin (one model)", n_b, len(bath_t), "have, proxy only"),
        ("target", "real human body", 0, 0, "none: ethics-approved cadaver or tissue phantom"),
        ("target", "clothing / pose varied", 0, 0, "not logged in either set"),
        ("site type", "lake (fresh)", n_b, 0, "have, one lake"),
        ("site type", "sea (salt)", 0, 0, "UATD sea frames hold no bodies"),
        ("site type", "canal / harbour (fresh-brackish)", 0,
         int(bath_t.site.isin(["bathampton_canal", "underfall_yard"]).sum()),
         "Bath, unboxed (protocol in reports/w3_bath_labelling_protocol.md)"),
        ("site type", "river with current, ice, weed bed", 0, 0, "none"),
        ("bottom", "logged bottom type", 0, 0, "neither set logs it"),
        ("depth", "logged water depth", 0, 0, "neither set logs it"),
        ("sensor", "forward-looking 720 kHz", int((body.freq_khz == 720).sum()), 0, "have"),
        ("sensor", "forward-looking 1,200 kHz", int((body.freq_khz == 1200).sum()), 0, "have"),
        ("sensor", "side-scan 450 / 990 kHz", 0, len(bath_t), "Bath, unboxed"),
        ("sensor", "AquaEye-like handheld, dual frequency, 1-D pings", 0, 0, "none"),
    ]
    for (a, b) in BANDS:
        n = int(((body.range_body_m >= a) & (body.range_body_m < b)).sum())
        rows.append(("range", f"{a}-{b} m", n, 0,
                     "have" if n > 300 else ("thin" if n else "none")))
    df = pd.DataFrame(rows, columns=["dimension", "level", "uatd_body_objects", "bath_target_legs",
                                     "status"])
    df.to_csv(OUT / "coverage.csv", index=False)
    return t


def coverage_figure(t: pd.DataFrame) -> None:
    piv = t.pivot(index="freq_khz", columns="band", values="objects").fillna(0)
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), [f"{f} kHz" for f in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = int(piv.values[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > piv.values.max() / 2 else viz.INK, fontsize=9)
    ax.grid(False)
    ax.set_xlabel("Range to body")
    ax.set_title("UATD body objects by frequency and range (one lake, one mannequin)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    viz.save(fig, FIG / "coverage.png")
    plt.close(fig)


# ------------------------------------------------------------------ ICC and sample size
def icc_from_w3(fr, ob, ca) -> dict:
    """ICC of 'body found' at the 0.05 FAPF point across looks within a session (W3 R2
    cross-session scores), as the planning value for looks within one placement."""
    oof = pd.read_parquet(PROC / "uatd_oof_scores.parquet")
    lake = fr[(fr.site == "lake") & fr.grp.isin(BODY_GROUPS)]
    m = ca.frame_id.isin(lake.frame_id).to_numpy()
    s = oof["cross-session:R2"].to_numpy()[m]
    c = ca[m]
    obl = ob[ob.frame_id.isin(lake.frame_id)]
    o = object_scores(obl, c, s)
    t = threshold_at_fapf(s[c.y.to_numpy() == 0], len(lake), FAPF)
    found = (o >= t).astype(float)
    rho = icc_binary(found, obl.session.to_numpy())
    rng = np.random.default_rng(SEED)
    sess = obl.session.to_numpy()
    keys = np.unique(sess)
    boots = []
    for _ in range(500):
        pick = rng.choice(keys, len(keys))
        idx = np.concatenate([np.flatnonzero(sess == k) for k in pick])
        lab = np.concatenate([[i] * np.sum(sess == k) for i, k in enumerate(pick)])
        boots.append(icc_binary(found[idx], lab))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    res = {"icc": rho, "icc_lo": float(lo), "icc_hi": float(hi), "sessions": int(len(keys)),
           "objects": int(len(found)), "recall": float(found.mean()),
           "note": "UATD session = one range setting at one sound speed; frames are looks"}
    (OUT / "icc.json").write_text(json.dumps(res, indent=2))
    return res


def power_table(rho: float) -> pd.DataFrame:
    rows = []
    for p0, delta, looks, r in itertools.product((0.4, 0.7), (0.10, 0.15, 0.20), (1, 10, 30),
                                                 (rho, 0.5, 0.8)):
        p1 = min(p0 + delta, 0.99)
        rows.append({"baseline_recall": p0, "difference": delta, "looks_per_placement": looks,
                     "icc": round(r, 3), "looks_independent": round(n_two_proportions(p0, p1), 1),
                     "design_effect": round(design_effect(looks, r), 2),
                     "placements_per_arm": placements_per_arm(p0, p1, looks, r)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "power.csv", index=False)
    return df


def power_figure(rho: float) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    d = np.linspace(0.08, 0.3, 40)
    for i, (r, looks) in enumerate([(rho, 10), (0.5, 10), (rho, 1)]):
        k = [placements_per_arm(0.6, 0.6 - x, looks, r) for x in d]
        looks_txt = f"{looks} look{'s' if looks > 1 else ''}"
        ax.plot(d, k, color=viz.SERIES[i], label=f"ICC {r:.2f}, {looks_txt} per placement")
    ax.axhline(48, color=viz.MUTED, ls="--", lw=1)
    ax.text(0.3, 52, "48 per arm (proposed trial, per range level)", ha="right", fontsize=8,
            color=viz.INK2)
    ax.set_yscale("log")
    ax.set_xlabel("Recall difference to detect (baseline 0.6; alpha 0.05, power 0.8)")
    ax.set_ylabel("Placements per condition")
    ax.legend(fontsize=8)
    ax.set_title("How many body placements a comparison needs")
    viz.save(fig, FIG / "power.png")
    plt.close(fig)


# ------------------------------------------------------------------ field design
SITES = [("S1", "mud"), ("S2", "sand"), ("S3", "rock"), ("S4", "weed")]
RANGES = (10, 20, 50)
POSES = ("prone", "supine", "curled")
CLOTHING = ("light", "heavy")
CONFUSERS = ("tyre", "log", "rock_pile", "debris_bag")


def field_design(seed: int = SEED) -> pd.DataFrame:
    """Randomised complete block design. Block = site (bottom type is the block's level) x day.
    Each site runs the full range x pose x clothing factorial (18 body placements) twice over
    two days, each day holding one replicate (a complete block). Each day also has 4
    confuser placements (one per type, random range) and two body-free calibration sweeps
    (start and end of day). Both frequencies record every ping, so frequency is within
    placement. Order within a day is random; labellers see placement IDs, not conditions."""
    rng = np.random.default_rng(seed)
    rows = []
    for site, bottom in SITES:
        for day in (1, 2):
            cells = [dict(kind="body", range_m=r, pose=p, clothing=c)
                     for r, p, c in itertools.product(RANGES, POSES, CLOTHING)]
            cells += [dict(kind="confuser", range_m=int(rng.choice(RANGES)), pose=t,
                           clothing="-") for t in CONFUSERS]
            order = rng.permutation(len(cells))
            seq = [dict(kind="empty_sweep", range_m=0, pose="-", clothing="-")]
            seq += [cells[i] for i in order]
            seq += [dict(kind="empty_sweep", range_m=0, pose="-", clothing="-")]
            for k, cell in enumerate(seq, 1):
                rows.append({"site": site, "bottom": bottom, "day": day, "slot": k,
                             "placement_id": f"{site}D{day}-{k:02d}", **cell,
                             "passes": 3, "frequencies": "low+high",
                             "aspect": "random heading, logged" if cell["kind"] != "empty_sweep"
                             else "-"})
    return pd.DataFrame(rows)


def main() -> None:
    viz.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    fr, ob, ca = load()
    ob_all = pd.read_parquet(PROC / "uatd_objects.parquet")
    t = coverage(fr, ob_all)
    coverage_figure(t)
    print(pd.read_csv(OUT / "coverage.csv").to_string(index=False))
    print(t.to_string(index=False))
    icc = icc_from_w3(fr, ob, ca)
    print(icc)
    pw = power_table(icc["icc"])
    power_figure(icc["icc"])
    print(pw[(pw.looks_per_placement == 10)].to_string(index=False))
    d = field_design()
    d.to_csv(OUT / "field_design.csv", index=False)
    s = d.groupby(["site", "kind"]).size().unstack(fill_value=0)
    s.to_csv(OUT / "field_design_summary.csv")
    print(s)
    # what the proposed trial can detect: 24 placements per range level (4 sites x 2 days x 3)
    k_arm = int((d[(d.kind == "body")].groupby("range_m").size()).min())
    detect = []
    for delta in (0.10, 0.15, 0.20, 0.25):
        for r in (icc["icc"], 0.5):
            detect.append({"placements_per_arm": k_arm, "difference": delta, "icc": round(r, 3),
                           "looks": 10, "power_sim": simulate_power(0.6, 0.6 - delta, k_arm, 10,
                                                                    r, n_sims=2000, seed=SEED)})
    detect = pd.DataFrame(detect)
    detect.to_csv(OUT / "trial_power.csv", index=False, float_format="%.3f")
    print(detect.to_string(index=False))
    (OUT / "strategy_meta.json").write_text(json.dumps(
        {"script": "w6_strategy", "git_commit": git_commit(), "seed": SEED}, indent=2))


if __name__ == "__main__":
    main()
