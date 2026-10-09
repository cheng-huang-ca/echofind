"""S4: deep rungs R3 (1-D and 2-D small CNNs) and R4 (ResNet-50 frozen, MobileNetV3-Small
fine-tuned) on the W3 splits, scored and compared with W3's R1 and R2 on the same candidates.

Protocol (W3, decision 203; S4 additions in reports/decisions/601-deep-rung-protocol.md):
- Folds: cross-session (4 lake groups held out, pooled), cross-frequency (720 -> 1200 kHz and
  back), sea false alarms (train on all lake).
- Training side of each fold: 20% of its sessions held out for early stopping and for the
  transferred threshold (0.05 FAPF on those sessions' negatives). Fit set: every body
  candidate plus at most `max_neg` negatives (seeded).
- Test side: every candidate of the held-out frames, so false alarms per frame are honest.
- Keep rule: recall at 0.05 FAPF beats R2 with a paired cluster-bootstrap interval above zero
  on cross-session and on both cross-frequency directions.

Run: uv run python scripts/s4_ladder.py [--rungs R3a R3b R4a R4b] [--boot 1000]
Needs scripts/s4_snippets.py (and s4_resnet_features.py for R4a).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import PROC, ROOT, git_commit, manifest_hash  # noqa: E402
from w3_ladder import BODY_GROUPS, FAPF, SEED, load, summarise  # noqa: E402

from echofind.eval import splits as sp  # noqa: E402
from echofind.eval.bootstrap import paired_cluster_bootstrap, recall_metric  # noqa: E402
from echofind.eval.metrics import threshold_at_fapf  # noqa: E402
from echofind.models.cnn import (  # noqa: E402
    MobileNetSmall,
    ProfileCNN,
    SnippetCNN,
    n_params,
    profile_batch,
    snippet_batch,
)
from echofind.models.train import TrainConfig, fit, predict  # noqa: E402

OUT = ROOT / "reports" / "s4"
CKPT = PROC / "s4_ckpt"
RUNGS = {
    "R3a": dict(desc="1-D CNN on range profile", input="profile", max_neg=40000,
                cfg=TrainConfig(epochs=15, batch=512, lr=2e-3)),
    "R3b": dict(desc="2-D CNN on snippet (~110k params)", input="snippet", max_neg=40000,
                cfg=TrainConfig(epochs=12, batch=256, lr=2e-3)),
    "R4a": dict(desc="ResNet-50 ImageNet, frozen; logistic head", input="resnet", max_neg=40000),
    "R4b": dict(desc="MobileNetV3-Small ImageNet, fine-tuned", input="snippet", max_neg=12000,
                cfg=TrainConfig(epochs=8, batch=128, lr=5e-4, patience=3)),
}


def folds(fr):
    lake = fr[fr.site == "lake"]
    sea = fr[fr.site == "sea"]
    out = []
    for name, tr, te in sp.leave_one_group_out(lake, "grp", BODY_GROUPS):
        out.append(("cross-session", name, set(lake.frame_id[tr]), set(lake.frame_id[te])))
    for a, b in ((720, 1200), (1200, 720)):
        n, tr, te = sp.transfer(lake, "freq_khz", a, b)
        out.append(("cross-frequency", f"{a}k->{b}k", set(lake.frame_id[tr]),
                    set(lake.frame_id[te])))
    out.append(("sea-fa", "lake->sea", set(lake.frame_id), set(sea.frame_id)))
    return out


def split_val(ca_tr: pd.DataFrame, seed: int):
    """Hold out ~20% of training sessions (with at least 50 body candidates) for validation."""
    sess = ca_tr.session.unique()
    rng = np.random.default_rng(seed)
    for _ in range(200):
        v = set(rng.choice(sess, max(1, int(0.2 * len(sess))), replace=False))
        m = ca_tr.session.isin(v).to_numpy()
        if ca_tr.y.to_numpy()[m].sum() >= 50 and ca_tr.y.to_numpy()[~m].sum() >= 200:
            return ~m, m
    raise RuntimeError("no validation split with enough positives")


def fit_set(idx: np.ndarray, y: np.ndarray, max_neg: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos, neg = idx[y == 1], idx[y == 0]
    if len(neg) > max_neg:
        neg = rng.choice(neg, max_neg, replace=False)
    return np.sort(np.r_[pos, neg])


def run_rung(rung: str, spec: dict, fr, ca, data, log) -> tuple[np.ndarray, np.ndarray, dict]:
    """Scores for every candidate on the test side of each fold, and the fold thresholds."""
    y_all = ca.y.to_numpy()
    fid = ca.frame_id.to_numpy()
    store = {}
    for split, name, trf, tef in folds(fr):
        t0 = time.time()
        tr = np.flatnonzero(np.isin(fid, list(trf)))
        te = np.flatnonzero(np.isin(fid, list(tef)))
        fit_m, val_m = split_val(ca.iloc[tr], SEED)
        tr_fit, tr_val = tr[fit_m], tr[val_m]
        fset = fit_set(tr_fit, y_all[tr_fit], spec["max_neg"], SEED)
        tag = f"{rung}_{split}_{name}".replace(">", "").replace("-", "_")
        if spec["input"] == "resnet":
            F = data["resnet"]
            head = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
            neg_w = max(1.0, (y_all[tr_fit] == 0).sum() / max((y_all[fset] == 0).sum(), 1))
            w = np.where(y_all[fset] == 1, 1.0, neg_w)
            head.fit(F[fset].astype(np.float32), y_all[fset], logisticregression__sample_weight=w)
            score = lambda idx, h=head, F=F: h.decision_function(F[idx].astype(np.float32))  # noqa: E731
        else:
            arr = data[spec["input"]]
            bf = profile_batch if spec["input"] == "profile" else snippet_batch

            def batcher(idx, aug, rng, arr=arr, bf=bf):
                return bf(arr, idx, aug, rng)

            model = {"R3a": ProfileCNN, "R3b": SnippetCNN,
                     "R4b": MobileNetSmall}[rung]()
            torch.manual_seed(SEED)
            model = fit(model, batcher, fset, y_all[fset], tr_val, y_all[tr_val], spec["cfg"],
                        CKPT / f"{tag}.pt", log=lambda s, t=tag: log(f"  {t} {s}"))
            torch.save(model.state_dict(), CKPT / f"{tag}_best.pt")
            score = lambda idx, m=model, b=batcher: predict(m, b, idx)  # noqa: E731
        s_val = score(tr_val)
        n_val_frames = fr.frame_id.isin(set(fid[tr_val])).sum()
        thr = threshold_at_fapf(s_val[y_all[tr_val] == 0], int(n_val_frames), FAPF)
        s_te = score(te)
        store[(split, name)] = (te, s_te, thr)
        log(f"{rung} {split} {name}: fit {len(fset)} ({int(y_all[fset].sum())} pos), "
            f"test {len(te)}, {(time.time() - t0) / 60:.1f} min")
    return store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="*", default=list(RUNGS))
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "train_log.txt", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    fr, ob, ca = load()
    oof = pd.read_parquet(PROC / "uatd_oof_scores.parquet")
    data = {"snippet": np.load(PROC / "uatd_snips.npy"),
            "profile": np.load(PROC / "uatd_profiles.npy")}
    if "R4a" in args.rungs:
        data["resnet"] = np.load(PROC / "uatd_resnet50.npy", mmap_mode="r")
    deep_path = PROC / "uatd_oof_deep.parquet"
    deep = pd.read_parquet(deep_path) if deep_path.exists() else pd.DataFrame(index=ca.index)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("s4-deep-rungs")

    for rung in args.rungs:
        spec = RUNGS[rung]
        store = run_rung(rung, spec, fr, ca, data, log)
        for split in ("cross-session", "cross-frequency", "sea-fa"):
            s_all = np.full(len(ca), np.nan)
            t_all = np.full(len(ca), np.nan)
            for (sp_, _), (te, s, thr) in store.items():
                if sp_ == split:
                    s_all[te], t_all[te] = s, thr
            deep[f"{split}:{rung}"] = s_all
            deep[f"{split}:{rung}:thr"] = t_all
        deep.to_parquet(deep_path)
        params = {"desc": spec["desc"], "max_neg": spec["max_neg"], "seed": SEED,
                  **({k: v for k, v in vars(spec["cfg"]).items()} if "cfg" in spec else {})}
        if rung in ("R3a", "R3b", "R4b"):
            m = {"R3a": ProfileCNN, "R3b": SnippetCNN, "R4b": lambda: MobileNetSmall(False)}[rung]()
            params["n_params"] = n_params(m)
        with mlflow.start_run(run_name=f"s4/{rung}"):
            mlflow.set_tags({"rung": rung, "git_commit": git_commit(),
                             "data_manifest": manifest_hash()})
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        (OUT / f"{rung}_params.json").write_text(json.dumps(params, indent=2, default=str))
    evaluate(fr, ob, ca, oof, deep, args.boot)


def evaluate(fr, ob, ca, oof, deep, boot) -> None:
    """Results table and paired comparisons with R1 and R2 for every deep rung present."""
    rungs = sorted({c.split(":")[1] for c in deep.columns if c.count(":") == 1})
    lake = fr[fr.site == "lake"]
    rows, paired, sea = [], [], []
    evals = [("cross-session", "pooled", set(lake.frame_id[lake.grp.isin(BODY_GROUPS)]))]
    evals += [("cross-frequency", f"{a}k->{b}k", set(lake.frame_id[lake.freq_khz == b]))
              for a, b in ((720, 1200), (1200, 720))]
    for split, fold, tef in evals:
        fm = fr.loc[sorted(tef)]
        m = ca.frame_id.isin(tef).to_numpy()
        ob_t = ob[ob.frame_id.isin(tef)]
        cl = {}
        for rung in ["R1", "R2"] + rungs:
            src = oof if rung in ("R1", "R2") else deep
            s, t = src[f"{split}:{rung}"].to_numpy()[m], src[f"{split}:{rung}:thr"].to_numpy()[m]
            row, c, _ = summarise(split, fold, rung, ob_t, ca[m], s, None, t, fm, boot)
            cl[rung] = c
            if rung not in ("R1", "R2"):
                rows.append(row)
        for rung in rungs:
            for base in ("R2", "R1"):
                d, lo, hi = paired_cluster_bootstrap(cl[base], cl[rung], recall_metric(FAPF),
                                                     boot, SEED)
                paired.append({"split": split, "fold": fold, "lower": base, "upper": rung,
                               "diff_recall": d, "lo": lo, "hi": hi, "upper_better": lo > 0})
    seam = (ca.site == "sea").to_numpy()
    nsea = int((fr.site == "sea").sum())
    for rung in rungs:
        s, t = deep[f"sea-fa:{rung}"].to_numpy()[seam], deep[f"sea-fa:{rung}:thr"].to_numpy()[seam]
        cls = ca.cls.to_numpy()[seam]
        passed = s >= t
        sea.append({"rung": rung, "fapf_transfer": passed.sum() / nsea,
                    **{f"fa_{c}": int(passed[cls == c].sum()) for c in sorted(set(cls))}})
    pd.DataFrame(rows).to_csv(OUT / "deep_results.csv", index=False, float_format="%.4f")
    pd.DataFrame(paired).to_csv(OUT / "deep_paired.csv", index=False, float_format="%.4f")
    pd.DataFrame(sea).to_csv(OUT / "deep_sea_fa.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows)[["split", "fold", "rung", "recall_at_op", "recall_lo", "recall_hi",
                              "pr_auc", "recall_transfer", "fapf_transfer"]]
          .to_string(index=False, float_format="%.3f"))
    print(pd.DataFrame(paired).to_string(index=False, float_format="%.3f"))
    print(pd.DataFrame(sea).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
