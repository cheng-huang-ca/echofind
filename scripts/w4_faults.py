"""W4: fault injection on real UATD frames, for the FMEA-lite hazard table.

Every released UATD frame passes the input QA checks, so the checks are exercised by
corrupting real frames instead. Test frames are the 720 kHz, 10 m lake group (all body frames
plus an equal number of body-free frames); R2 and the OOD model are trained on every other
lake frame, as in the cross-session split. For each fault, the script reports how often QA
fails the frame, how often the OOD score flags it, and R2's body recall and false alarms per
frame at the clean-data 0.05 FAPF threshold.

Faults (each is a physical hazard from the plan's FMEA list):
- saturation: +12 dB receiver gain, clipped at full scale (target close to the transducer,
  gain set too high).
- dropped rows: 10% of range rows zeroed in one block (lost ping segment, cable fault).
- dead beams: a 10% sector of beams zeroed (blocked or failed elements).
- wake: a bright speckle cloud over 0-40% of range in a 30% sector at +15 dB over the frame
  median (bubbles from a boat wake or a diver's exhaust).
- low gain: -20 dB receiver gain (gain set too low; weak echoes fall below the 8-bit floor).

Run: uv run python scripts/w4_faults.py [--jobs J]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import PROC, ROOT  # noqa: E402
from w3_ladder import load, object_scores  # noqa: E402

from echofind.eval.metrics import threshold_at_fapf  # noqa: E402
from echofind.eval.ood import MahalanobisOOD, flag_threshold  # noqa: E402
from echofind.eval.qa import frame_qa  # noqa: E402
from echofind.features.fls import R2_FEATURES, frame_candidates  # noqa: E402
from echofind.io.uatd import load_image  # noqa: E402
from echofind.models.ladder import R2Trees  # noqa: E402

OUT = ROOT / "reports" / "w4"
GROUP = "720k_r10"
FAULTS = ["clean", "saturation", "dropped_rows", "dead_beams", "wake", "low_gain"]
_OBJ: pd.DataFrame | None = None


def corrupt(a: np.ndarray, fault: str, rng: np.random.Generator) -> np.ndarray:
    a = a.copy()
    H, W = a.shape
    if fault == "saturation":
        a = np.minimum(a * 10 ** (12 / 20), 1.0)
    elif fault == "dropped_rows":
        r0 = rng.integers(int(0.1 * H), int(0.8 * H))
        a[r0: r0 + int(0.1 * H)] = 0
    elif fault == "dead_beams":
        b0 = rng.integers(0, int(0.9 * W))
        a[:, b0: b0 + int(0.1 * W)] = 0
    elif fault == "wake":
        b0 = rng.integers(0, int(0.7 * W))
        sl = (slice(0, int(0.4 * H)), slice(b0, b0 + int(0.3 * W)))
        lvl = max(np.median(a), 2 / 255) * 10 ** (15 / 20)
        a[sl] = np.maximum(a[sl], lvl * rng.rayleigh(1 / np.sqrt(2), a[sl].shape))
    elif fault == "low_gain":
        a = np.floor(a * 255 * 0.1) / 255
    return np.clip(np.round(a * 255) / 255, 0, 1).astype(np.float32)


def _init(objects: pd.DataFrame) -> None:
    global _OBJ
    _OBJ = objects


def _one(args) -> pd.DataFrame:
    frame, fault, seed = args
    f = pd.Series(frame)
    a = corrupt(load_image(f), fault, np.random.default_rng(seed))
    q = frame_qa(a)
    c, _ = frame_candidates(a, f, _OBJ[_OBJ.frame_id == f.frame_id])
    if c.empty:
        c = pd.DataFrame({"frame_id": [f.frame_id], "no_cand": [True]})
    else:
        c.attrs = {}
        c["frame_id"] = f.frame_id
        c["no_cand"] = False
    c["fault"] = fault
    c["qa_pass"] = q["qa_pass"]
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    fr, ob, ca = load()
    objects = pd.read_parquet(PROC / "uatd_objects.parquet").set_index("obj_idx")
    lake = fr[fr.site == "lake"]
    test = lake[lake.grp == GROUP]
    body_f = set(ob.frame_id)
    pos = test[test.frame_id.isin(body_f)]
    neg = test[~test.frame_id.isin(body_f)].sample(len(pos), random_state=0)
    test = pd.concat([pos, neg])
    train = ca[ca.frame_id.isin(lake[lake.grp != GROUP].frame_id)]
    model = R2Trees(seed=0).fit(train, train.y.to_numpy())
    ood = MahalanobisOOD(R2_FEATURES).fit(train.sample(60000, random_state=0))
    tr_frames = train.frame_id.unique()
    cal = train[train.frame_id.isin(set(tr_frames[::7]))]
    ood_thr = flag_threshold(ood.frame_score(cal, cal.frame_id.to_numpy()))

    jobs = [(f, fault, i) for fault in FAULTS for i, f in enumerate(test.to_dict("records"))]
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(objects,)) as ex:
        parts = list(ex.map(_one, jobs, chunksize=8))
    allc = pd.concat(parts, ignore_index=True)
    rows = []
    ob_t = ob[ob.frame_id.isin(test.frame_id)]
    thr = None
    for fault in FAULTS:
        d = allc[allc.fault == fault]
        frames_qa = d.groupby("frame_id").qa_pass.first()
        c = d[~d.no_cand.astype(bool)].copy()
        c["y"] = (c.cls == "human body").astype(int)
        s = model.score(c)
        fs = ood.frame_score(c, c.frame_id.to_numpy()).reindex(test.frame_id).fillna(0)
        if fault == "clean":
            thr = threshold_at_fapf(s[c.y.to_numpy() == 0], len(test), 0.05)
        o = object_scores(ob_t, c, s)
        rows.append({"fault": fault, "frames": len(test), "bodies": len(o),
                     "qa_fail_rate": float(1 - frames_qa.mean()),
                     "ood_flag_rate": float(np.mean(fs > ood_thr)),
                     "caught_rate": float(np.mean((fs > ood_thr).to_numpy()
                                                  | ~frames_qa.reindex(test.frame_id).to_numpy())),
                     "candidates_per_frame": len(c) / len(test),
                     "body_recall": float(np.mean(o >= thr)),
                     "fapf": float(np.sum(s[c.y.to_numpy() == 0] >= thr) / len(test))})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "fault_injection.csv", index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
