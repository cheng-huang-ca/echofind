# 103 · CFAR reference cells are strided at one per resolution cell

**Status:** accepted · **Owner:** W2

## Context
The front end samples the matched-filter output at 4B, so a pulse peak is not missed between
samples. Adjacent samples are then correlated. The CFAR closed forms (CA, GO, SO, OS) assume
independent reference cells. Measured on white noise through the matched filter
(`reports/w2/pfa_empirical.csv`, Pfa design 1e-4, N = 32):

| reference window | CA | GO | OS |
| --- | --- | --- | --- |
| contiguous, 4B sampling | 6.2e-4 | 4.4e-4 | 5.6e-4 |
| every 4th sample (stride 4) | 0.96e-4 | 0.96e-4 | 0.93e-4 |
| 1 sample per cell (B sampling) | 0.90e-4 | 0.94e-4 | 0.93e-4 |

## Options considered
1. Design for N_eff = N/4 independent cells: overcorrects (CA 0.38e-4, OS 0.07e-4), because
   the true equivalent count (about 9.5 here) depends on the pulse's autocorrelation and the
   window edges, and OS has no clean equivalent.
2. Decimate to B before CFAR: holds Pfa but tests only one sample per cell (straddle loss).
3. **Stride the reference cells by the oversampling factor, test every sample.** Chosen.

## Decision
`CFARConfig(stride=m)` takes the N reference cells every m samples (about one per resolution
cell). The closed forms stay valid, and `tests/test_cfar.py` holds the regression.

## Consequences
The reference span is N·m samples (N resolution cells), the same physical extent a
critically sampled CFAR would use. Set `stride = round(fs / B)` and `n_guard ≥ stride`.
