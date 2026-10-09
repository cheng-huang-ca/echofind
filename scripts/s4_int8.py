"""S4 / H8: int8 static quantisation of a CNN rung with ONNX Runtime; recall and latency cost.

For each cross-session fold model of the chosen rung: export to ONNX (fp32), quantise
statically (QDQ, per-channel weights, int8 activations) with 512 snippets from that fold's
training frames as calibration data, then score the held-out frames with both. Reports pooled
cross-session recall at 0.05 FAPF for fp32 and int8 with a paired cluster-bootstrap interval, and
single-thread latency for a batch of 20 candidates (one ping).

H8 (docs/PLAN.md): "int8 costs under 1 point of recall and runs at least 2x faster than fp32".

Run: uv run python scripts/s4_int8.py --rung R3b
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

sys.path.insert(0, str(Path(__file__).parent))
from s4_ladder import CKPT, OUT, folds  # noqa: E402
from w3_common import PROC  # noqa: E402
from w3_ladder import BODY_GROUPS, FAPF, SEED, load, summarise  # noqa: E402

from echofind.eval.bootstrap import paired_cluster_bootstrap, recall_metric  # noqa: E402
from echofind.models.cnn import MobileNetSmall, ProfileCNN, SnippetCNN  # noqa: E402

MODELS = {"R3a": (ProfileCNN, "profile"), "R3b": (SnippetCNN, "snippet"),
          "R4b": (lambda: MobileNetSmall(False), "snippet")}


def inputs(arr, idx, kind):
    x = arr[idx]
    if kind == "profile":
        return ((x.astype(np.float32) + 5.0) / 40.0)
    return x.astype(np.float32) / 255.0


class Reader(CalibrationDataReader):
    def __init__(self, x, name):
        self.it = iter([{name: x[i: i + 32]} for i in range(0, len(x), 32)])

    def get_next(self):
        return next(self.it, None)


def session(path, threads=1):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def run(sess, x, batch=4096):
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: x[i: i + batch]})[0]
                           for i in range(0, len(x), batch)]).ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="R3b")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    ctor, kind = MODELS[args.rung]
    fr, ob, ca = load()
    arr = np.load(PROC / ("uatd_snips.npy" if kind == "snippet" else "uatd_profiles.npy"))
    fid = ca.frame_id.to_numpy()
    s32 = np.full(len(ca), np.nan)
    s8 = np.full(len(ca), np.nan)
    lat = []
    qdir = PROC / "s4_onnx"
    qdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    for split, name, trf, tef in folds(fr):
        if split != "cross-session":
            continue
        tag = f"{args.rung}_{split}_{name}".replace(">", "").replace("-", "_")
        model = ctor()
        model.load_state_dict(torch.load(CKPT / f"{tag}_best.pt", weights_only=True))
        model.eval()
        shape = (1, 128) if kind == "profile" else (1, 64, 32)
        f32 = qdir / f"{tag}.onnx"
        torch.onnx.export(model, torch.zeros(shape), str(f32), input_names=["x"],
                          output_names=["logit"], dynamic_axes={"x": {0: "n"}, "logit": {0: "n"}},
                          opset_version=17, dynamo=False)
        tr = np.flatnonzero(np.isin(fid, list(trf)))
        cal = inputs(arr, np.sort(rng.choice(tr, 512, replace=False)), kind)
        f8 = qdir / f"{tag}.int8.onnx"
        quantize_static(str(f32), str(f8), Reader(cal, "x"), quant_format=QuantFormat.QDQ,
                        per_channel=True, weight_type=QuantType.QInt8,
                        activation_type=QuantType.QInt8)
        te = np.flatnonzero(np.isin(fid, list(tef)))
        x = inputs(arr, te, kind)
        a, b = session(f32, 6), session(f8, 6)
        s32[te], s8[te] = run(a, x), run(b, x)
        ping = x[:20]
        for label, path in (("fp32", f32), ("int8", f8)):
            s1 = session(path, 1)
            run(s1, ping)
            t = []
            for _ in range(300):
                t0 = time.perf_counter()
                run(s1, ping)
                t.append((time.perf_counter() - t0) * 1e6)
            lat.append({"fold": name, "precision": label, "p50_us": np.median(t),
                        "p95_us": np.quantile(t, 0.95), "model_bytes": path.stat().st_size})
        print(name, "done", flush=True)
    lake = fr[fr.site == "lake"]
    tef = set(lake.frame_id[lake.grp.isin(BODY_GROUPS)])
    fm = fr.loc[sorted(tef)]
    m = ca.frame_id.isin(tef).to_numpy()
    ob_t = ob[ob.frame_id.isin(tef)]
    rows, cl = [], {}
    for label, s in (("fp32", s32), ("int8", s8)):
        row, c, _ = summarise("cross-session", "pooled", f"{args.rung} {label}", ob_t, ca[m], s[m],
                              None, np.full(m.sum(), np.inf), fm, args.boot)
        rows.append({k: row[k] for k in ("rung", "recall_at_op", "recall_lo", "recall_hi",
                                         "pr_auc")})
        cl[label] = c
    d, lo, hi = paired_cluster_bootstrap(cl["fp32"], cl["int8"], recall_metric(FAPF), args.boot,
                                         SEED)
    rows.append({"rung": "int8 - fp32", "recall_at_op": d, "recall_lo": lo, "recall_hi": hi})
    L = pd.DataFrame(lat)
    summ = L.groupby("precision")[["p50_us", "p95_us", "model_bytes"]].median()
    rows.append({"rung": "speed-up (fp32 p50 / int8 p50)",
                 "recall_at_op": summ.loc["fp32", "p50_us"] / summ.loc["int8", "p50_us"]})
    pd.DataFrame(rows).to_csv(OUT / f"int8_{args.rung}.csv", index=False, float_format="%.4f")
    L.to_csv(OUT / f"int8_{args.rung}_latency.csv", index=False, float_format="%.1f")
    print(pd.DataFrame(rows).to_string(index=False))
    print(summ)


if __name__ == "__main__":
    main()
