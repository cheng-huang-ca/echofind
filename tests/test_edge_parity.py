"""C++ edge library vs Python reference (W5): front end and compiled classifiers.

The library is loaded through its C ABI (ctypes), so the test runs wherever `edge/build` holds a
built ef_edge (cmake -S edge -B edge/build && cmake --build edge/build); otherwise it is skipped.
Error is max |c++ - python| / max |python|. Pass: <= 1e-9 in double, <= 1e-5 in float32
(the success criterion in docs/PLAN.md).
"""

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter

from echofind.dsp.cfar import CFARConfig, cfar_threshold
from echofind.dsp.frontend import demodulate
from echofind.dsp.matched_filter import matched_filter
from echofind.dsp.waveforms import lfm_pulse, to_passband

EDGE = Path(__file__).parents[1] / "edge"
CANDIDATES = ["libef_edge.dll", "ef_edge.dll", "libef_edge.so", "libef_edge.dylib"]


def _lib():
    for name in CANDIDATES:
        p = EDGE / "build" / name
        if p.exists():
            return ctypes.CDLL(str(p))
    pytest.skip("edge library not built (cmake -S edge -B edge/build)")


def _ptr(a, t=ctypes.c_double):
    return a.ctypes.data_as(ctypes.POINTER(t))


def _rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


@pytest.fixture(scope="module")
def ping():
    """A 20 m, 80 kHz LFM ping (B 20 kHz, 1 ms) with two echoes and noise, at passband."""
    fc, bw, dec = 80e3, 20e3, 5
    fs_bb, fs = 4 * bw, 4 * bw * dec
    rng = np.random.default_rng(3)
    rep = lfm_pulse(bw, 1e-3, fs_bb)
    n = int(2 * 20 / 1500 * fs_bb)
    iq = 0.05 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    for k, a in ((400, 1.0), (1500, 0.3)):
        iq[k: k + len(rep)] += a * rep
    up = np.repeat(iq, dec)                         # zero-order hold is enough for parity
    x = to_passband(up, fs, fc)
    return x / np.abs(x).max(), rep, fs, fs_bb, fc, bw, dec


def _sos(fc, bw, fs):
    lo, hi = max(fc - 0.6 * bw, 1.0), min(fc + 0.6 * bw, 0.49 * fs)
    return (butter(6, [lo, hi], btype="bandpass", fs=fs, output="sos"),
            butter(6, min(0.6 * bw, 0.45 * fs, 0.9 * fc), btype="lowpass", fs=fs, output="sos"))


@pytest.mark.parametrize("f32", [False, True])
def test_demodulate_matches_scipy_sosfiltfilt_chain(ping, f32):
    """IQ demodulation: band-pass, mix by 2 exp(-j 2 pi fc t), low-pass, decimate (demod.py)."""
    lib = _lib()
    x, _, fs, _, fc, bw, dec = ping
    bp, lp = (np.ascontiguousarray(s) for s in _sos(fc, bw, fs))
    ref = demodulate(x, fs, fc, bw, decimate=dec)
    out_n = (len(x) + dec - 1) // dec
    if f32:
        xi, out, fn, t = x.astype(np.float32), np.zeros(2 * out_n, np.float32), \
            lib.ef_demodulate_f32, ctypes.c_float
    else:
        xi, out, fn, t = x, np.zeros(2 * out_n), lib.ef_demodulate_f64, ctypes.c_double
    fn.restype = ctypes.c_int
    got = fn(_ptr(xi, t), len(x), ctypes.c_double(fs), ctypes.c_double(fc), _ptr(bp), len(bp),
             _ptr(lp), len(lp), dec, ctypes.c_double(0.0), _ptr(out, t))
    assert got == len(ref)
    y = out[0::2].astype(float) + 1j * out[1::2]
    assert _rel(y, ref) <= (1e-5 if f32 else 1e-9)


@pytest.mark.parametrize("f32", [False, True])
def test_matched_filter_matches_fftconvolve(ping, f32):
    """Correlation with the unit-energy replica, peak at the echo start (matched_filter.py)."""
    lib = _lib()
    _, rep, _, _, _, _, _ = ping
    rng = np.random.default_rng(4)
    x = rng.normal(size=3000) + 1j * rng.normal(size=3000)
    x[700: 700 + len(rep)] += 3 * rep
    ref = matched_filter(x, rep)
    dt, ct = (np.float32, ctypes.c_float) if f32 else (np.float64, ctypes.c_double)
    xi = np.stack([x.real, x.imag], -1).astype(dt).ravel()
    ri = np.stack([rep.real, rep.imag], -1).astype(dt).ravel()
    out = np.zeros(2 * len(x), dt)
    fn = lib.ef_matched_filter_f32 if f32 else lib.ef_matched_filter_f64
    assert fn(_ptr(xi, ct), len(x), _ptr(ri, ct), len(rep), _ptr(out, ct)) == len(x)
    y = out[0::2].astype(float) + 1j * out[1::2]
    assert _rel(y, ref) <= (1e-5 if f32 else 1e-9)
    assert np.argmax(np.abs(y)) == 700


@pytest.mark.parametrize("kind,code", [("ca", 0), ("go", 1), ("os", 3)])
@pytest.mark.parametrize("stride", [1, 4])
def test_cfar_thresholds_and_scale_factor(kind, code, stride):
    """CFAR threshold = scale(Pfa) x statistic of strided reference cells (cfar.py), NaN where
    the window does not fit; the scale factor inverts the closed-form Pfa."""
    lib = _lib()
    rng = np.random.default_rng(5)
    p = rng.exponential(size=2000)
    p[1000:1400] *= 30                       # a clutter edge
    cfg = CFARConfig(kind, n_ref=32, n_guard=2 * stride, pfa=1e-4, stride=stride)
    ref = cfar_threshold(p, cfg)
    thr = np.zeros_like(p)
    lib.ef_cfar_threshold_f64(_ptr(p), len(p), code, 32, 2 * stride, -1, stride,
                              ctypes.c_double(1e-4), _ptr(thr))
    assert np.array_equal(np.isnan(thr), np.isnan(ref))
    ok = ~np.isnan(ref)
    assert _rel(thr[ok], ref[ok]) <= 1e-9


def test_compiled_models_match_python():
    """Generated C classifiers on 300 committed candidates. Double: equal to the Python models
    to 1e-9 of the score range. Float: on float32-rounded features, the same branches as the
    double model on the same rounded features (thresholds are rounded down), so equal to 1e-5 of
    the score range (leaf sums in float). Rounding the features themselves is the export's only
    accuracy cost, measured in scripts/w5_bench.py."""
    lib = _lib()
    meta = json.loads((EDGE / "generated" / "models.json").read_text())
    gold = EDGE / "tests" / "golden"
    nf = len(meta[0]["features"])
    X = np.fromfile(gold / "models_x.bin", "<f8").reshape(-1, nf)
    S = np.fromfile(gold / "models_scores.bin", "<f8").reshape(len(X), -1)
    lib.ef_model_score_f64.restype = ctypes.c_double
    lib.ef_model_score_f32.restype = ctypes.c_float
    lib.ef_model_count.restype = ctypes.c_int
    assert lib.ef_model_count() == len(meta) == S.shape[1]
    for m in range(len(meta)):
        c64 = np.array([lib.ef_model_score_f64(m, _ptr(np.ascontiguousarray(r))) for r in X])
        X32 = X.astype(np.float32)
        c32 = np.array([lib.ef_model_score_f32(m, _ptr(r, ctypes.c_float)) for r in X32],
                       dtype=float)
        c64r = np.array([lib.ef_model_score_f64(m, _ptr(r.astype(float))) for r in X32])
        ref = S[:, m]
        fin = np.isfinite(ref)
        assert np.array_equal(fin, np.isfinite(c64)), meta[m]["name"]
        span = np.ptp(ref[fin])
        assert np.max(np.abs(c64[fin] - ref[fin])) <= 1e-9 * span, meta[m]["name"]
        assert np.array_equal(np.isfinite(c32), np.isfinite(c64r)), meta[m]["name"]
        f = np.isfinite(c64r)
        assert np.max(np.abs(c32[f] - c64r[f])) <= 1e-5 * span, meta[m]["name"]


def test_pybind_module_matches_python(ping):
    """The pybind11 module (built with -DEF_PYBIND=ON, e.g. in CI on Linux) gives the same
    demodulation as echofind.dsp."""
    import sys

    sys.path.insert(0, str(EDGE / "build"))
    mod = pytest.importorskip("_ef_edge")
    x, _, fs, _, fc, bw, dec = ping
    bp, lp = _sos(fc, bw, fs)
    got = mod.demodulate(x, fs, fc, bp, lp, dec)
    assert _rel(got, demodulate(x, fs, fc, bw, decimate=dec)) <= 1e-9
    assert "r2_t300" in mod.model_names()
