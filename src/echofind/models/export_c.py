"""Compile ladder rungs to dependency-free C for the edge build (W5).

- R2 (LightGBM) becomes nested if/else per tree, reproducing LightGBM's numerical decision:
  with missing type NaN, a NaN goes to the default child; with missing type Zero, |x| <= 1e-35
  does; otherwise NaN is read as 0. Then `x <= threshold` goes left (LightGBM tree.h,
  NumericalDecision). The raw score is the sum of the leaf values, as predict(raw_score=True).
- R1 (standardiser + logistic regression) becomes w . (x - m) / s + b.
- R0 (size-gated SNR) becomes a two-comparison rule.

Each model is emitted twice: a double version (reference) and a float version (device). In the
float version every split threshold is rounded down to the largest float not above it, so for
float inputs the float model takes exactly the branches the double model takes; the only
accuracy cost of the export is rounding the features themselves to float32, which W5 measures.
One C file holds every model plus a dispatch table matching edge/include/echofind/capi.h.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

ZERO = 1e-35


@dataclass
class CModel:
    name: str
    features: list[str]
    body_f64: str          # C statements computing `double s` from `const double* x`
    body_f32: str          # same for `float s` from `const float* x`
    meta: dict = field(default_factory=dict)


def _lit(v: float, f32: bool) -> str:
    if np.isinf(v):
        return ("-" if v < 0 else "") + ("INFINITY")
    if not f32:
        return repr(float(v))
    t = f"{v:.9g}"
    return (t if any(ch in t for ch in ".e") else t + ".0") + "f"


def _node(n: dict, f32: bool, depth: int) -> str:
    pad = "  " * depth
    if "leaf_value" in n:
        return f"{pad}s += {_lit(n['leaf_value'], f32)};\n"
    f, thr = n["split_feature"], n["threshold"]
    if n.get("decision_type", "<=") != "<=":
        raise ValueError("only numerical <= splits are supported")
    mt = n.get("missing_type", "None")
    xv = f"x[{f}]"
    go_default = n.get("default_left", True)
    if mt == "NaN":
        miss = f"isnan({xv})"
    elif mt == "Zero":
        miss = f"(fabs({xv}) <= {ZERO!r})"
    else:
        miss = None
    val = f"(isnan({xv}) ? 0 : {xv})" if mt != "NaN" else xv
    if f32:
        # largest float <= thr, so that for any float x: x <= thr_f exactly when x <= thr
        tf = np.float32(thr)
        if float(tf) > thr:
            tf = np.nextafter(tf, np.float32(-np.inf))
        thr = float(tf)
    cond = f"{val} <= {_lit(thr, f32)}"
    if miss is not None:
        cond = (f"({miss} ? {1 if go_default else 0} : ({cond}))")
    return (f"{pad}if ({cond}) {{\n{_node(n['left_child'], f32, depth + 1)}"
            f"{pad}}} else {{\n{_node(n['right_child'], f32, depth + 1)}{pad}}}\n")


def lgbm_to_c(booster, name: str, features: list[str]) -> CModel:
    """LightGBM booster (lgb.Booster or LGBMClassifier.booster_) to C."""
    dump = booster.dump_model()
    if dump.get("num_class", 1) != 1:
        raise ValueError("binary or regression boosters only")
    trees = dump["tree_info"]
    bodies = []
    for f32 in (False, True):
        bodies.append("".join(_node(t["tree_structure"], f32, 1) for t in trees))
    leaves = sum(t["num_leaves"] for t in trees)
    return CModel(name, features, bodies[0], bodies[1],
                  {"kind": "lightgbm", "trees": len(trees), "leaves": leaves})


def logistic_to_c(pipeline, name: str, used: list[str], features: list[str]) -> CModel:
    """sklearn Pipeline(StandardScaler, LogisticRegression) fitted on columns `used`, to C
    (decision_function). The C function reads the full `features` vector."""
    sc, lr = pipeline.steps[0][1], pipeline.steps[-1][1]
    w = lr.coef_.ravel() / sc.scale_
    b = float(lr.intercept_[0] - np.sum(lr.coef_.ravel() * sc.mean_ / sc.scale_))
    idx = [features.index(f) for f in used]
    bodies = []
    for f32 in (False, True):
        terms = " + ".join(f"{_lit(wi, f32)} * x[{i}]" for i, wi in zip(idx, w, strict=True))
        bodies.append(f"  s = {_lit(b, f32)} + {terms};\n")
    return CModel(name, features, bodies[0], bodies[1], {"kind": "logistic", "weights": len(w)})


def r0_to_c(name: str, gate: tuple[float, float], features: list[str]) -> CModel:
    """R0: the SNR if the longest extent is inside the size gate, else -inf."""
    a, b, c = (features.index(f) for f in ("range_ext_m", "cross_ext_m", "snr_db"))
    bodies = []
    for f32 in (False, True):
        lo, hi = _lit(gate[0], f32), _lit(gate[1], f32)
        t = "float" if f32 else "double"
        bodies.append(f"  {{ {t} e = x[{a}] > x[{b}] ? x[{a}] : x[{b}];\n"
                      f"    s = (e >= {lo} && e <= {hi}) ? x[{c}] : -INFINITY; }}\n")
    return CModel(name, features, bodies[0], bodies[1], {"kind": "rule"})


def write_c(models: list[CModel], c_path, json_path) -> None:
    """One C file with every model and the capi.h dispatch table; a JSON with feature orders."""
    out = ["/* Generated by echofind.models.export_c. Do not edit. */",
           "#include <math.h>", '#include "echofind/capi.h"', ""]
    for m in models:
        for f32, body in ((False, m.body_f64), (True, m.body_f32)):
            t = "float" if f32 else "double"
            out.append(f"static {t} {m.name}_{'f32' if f32 else 'f64'}(const {t}* x) {{\n"
                       f"  {t} s = 0;\n{body}  return s;\n}}\n")
    n = len(models)
    out.append(f"static const char* const NAMES[{n}] = {{"
               + ", ".join(f'"{m.name}"' for m in models) + "};")
    out.append(f"static const int NFEAT[{n}] = {{"
               + ", ".join(str(len(m.features)) for m in models) + "};")
    out.append(f"static double (*const F64[{n}])(const double*) = {{"
               + ", ".join(f"{m.name}_f64" for m in models) + "};")
    out.append(f"static float (*const F32[{n}])(const float*) = {{"
               + ", ".join(f"{m.name}_f32" for m in models) + "};")
    out.append(f"EF_API int ef_model_count(void) {{ return {n}; }}")
    out.append("EF_API const char* ef_model_name(int i) { return NAMES[i]; }")
    out.append("EF_API int ef_model_n_features(int i) { return NFEAT[i]; }")
    out.append("EF_API double ef_model_score_f64(int i, const double* x) { return F64[i](x); }")
    out.append("EF_API float ef_model_score_f32(int i, const float* x) { return F32[i](x); }")
    with open(c_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump([{"name": m.name, "features": m.features, **m.meta} for m in models], f,
                  indent=1)
