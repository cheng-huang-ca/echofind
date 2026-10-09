# 401 · Edge parity: offline filter design, scipy-exact zero-phase filtering, double filter state

**Status:** accepted for W5 · **Owner:** W5

## Context
The plan requires the C++ front end to match Python within 1e-5 (float32). Three choices
decide whether that is possible and cheap.

## Decisions
1. **Filters are designed offline.** `scripts/w5_export.py` calls the same `scipy.signal.butter`
   design as `echofind.dsp.demodulate` and ships the SOS rows to the device. A handheld unit
   needs fixed coefficients, not a filter designer, and porting the design code would only add
   a second source of error.
2. **Zero-phase filtering reproduces `sosfiltfilt` exactly:** odd extension of 3 × ntaps,
   initial state `sosfilt_zi` scaled by the edge sample, a forward pass and a backward pass.
   A causal filter would be cheaper, but then it would not match the Python reference that
   W2 validated against theory. Exact parity first; a causal or polyphase variant is a later,
   separately validated change.
3. **Biquad state runs in double even for float signals.** With float32 state, six cascaded
   band-pass sections at 400 kHz reached 1.01e-5 of peak, just over the bar. Double state
   gives 1.1e-7. A Cortex-A FPU does scalar double at float speed, and only the stored signal
   stays float32.
4. **`-fcx-limited-range`.** GCC otherwise calls the C99 NaN-safe complex multiply for every
   complex product, which made the matched filter slower than SciPy. The inputs are finite
   samples, and the golden tests are unchanged with the flag.

## Evidence
`edge/build/ef_golden` (`reports/w5/golden.txt`) passes 9 of 9 cases (demodulation, matched
filter, CA/GO/OS-CFAR at stride 1 and 4, TVG). Errors are ≤ 2e-11 in double and ≤ 3.1e-7
in float32, against bars of 1e-9 and 1e-5. `tests/test_edge_parity.py` (ctypes, 11 cases)
agrees on a fresh simulated ping.
