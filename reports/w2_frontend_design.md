# W2 front-end design note: from echoes to candidates

EchoFind W2, Oct 2026. Code: `src/echofind/dsp`, `src/echofind/sim`. Evidence: `reports/w2/*.csv`,
`reports/figures/w2_*.png`, decisions 101–105. Every block is checked against a closed form in `tests/`.

## Chain and defaults

```
passband ─► band-pass fc±0.6B ─► IQ demod ─► matched filter ─► |·|² ─┬─► OS-CFAR (N=32, stride=fs/B) ─► SNR gate ─► candidates
 (or EK80 complex IQ)                       h(t)=s*(−t)            └─► TVG 20/40 log r + 2αr ─► bottom track (persistence)
```

| Block | Choice | Checked against | Measured |
| --- | --- | --- | --- |
| Pulse | LFM, B = 20 kHz at 80 kHz (40 kHz at 200 kHz), T = 1 ms | f(t) = −B/2 + Bt/T | exact |
| Matched filter | FFT correlation, unit-energy replica | peak SNR = E/N0; gain 10 log10 BT | E/N0 within 0.1 dB; gain within 0.7 dB on all 16 B×T grid points (H2) |
| Resolution | 4B sampling | −3 dB width 0.886 c/(2B) | within 7% for BT ≥ 20; 11–31% wide for BT ≤ 10 (H2, partly met) |
| Real pings | EK80 replica from echopype | echopype `compress_pulse` | max relative error 5e-8, three channels, two files |
| TVG | 40 log r + 2αr (targets), 20 log r (volume) | EL = SL − 2TL + TS | within 0.1 dB from 5 to 40 m |
| Absorption | Francois–Garrison; fresh = S 0 | echopype FG; f² law | identical; 80 kHz: 0.0017 dB/m fresh, 0.029 dB/m salt |
| CFAR | CA/GO/SO/OS closed forms | quadrature; Monte Carlo Pfa and Swerling-1 Pd | Pfa within 5σ; Pd within MC error |
| Reverberation | Lambert, μ = −27 dB (sand), K texture | ∬ μ sin²g b² r⁻⁴ dA | within 10% |
| Multipath | image method, surface and bottom, first order | (rᵢ + rⱼ)/2 | MF peaks within res/4 |

## What the numbers say

1. **Near the bottom, a body is reverberation-limited, not noise-limited.** For a body on sand in
   5 m of water, SINR at the matched-filter output is −4 to 16 dB from 10 to 60 m. The
   noise-limited sonar equation (NL 40 dB, on axis) predicts 31–68 dB for the same ranges (`w2_sinr_vs_range.png`). More source
   level buys nothing. What helps is a smaller resolution cell (more B), a narrower beam, and
   features that separate the body from bottom texture (W3).
2. **Beam geometry sets the short-range blind zone.** With a 10° beam tilted 10° down from 0.5 m,
   Pd is ≤ 0.05 at 5 m and about 0.55 at 10 m on the bottom, then ≥ 0.92 from 15 to 50 m
   (`w2_pd_vs_range.png`). A handheld unit needs a steeper tilt or a second, wider beam for the
   near zone. That is a hardware trade, not an algorithm one.
3. **Absorption bites only past about 50 m.** In salt water (α 0.029 dB/m at 80 kHz, 0.066 dB/m
   at 200 kHz), Pd for a body 1 m above the bottom is comparable at 50 m or less (0.81–0.93),
   because reverberation, not absorption, sets the limit. At 60 m it is 0.28 at 200 kHz against
   0.48 at 80 kHz (0.75 in fresh water). Dual frequency (`simulate_pair`) gives W3 a
   frequency-response feature.
4. **Oversampled CFAR needs strided reference cells.** Contiguous windows on 4B-sampled
   matched-filter output false-alarm 4–6× too often. Taking reference cells every 4 samples
   restores Pfa to 0.93–0.96e-4 (decision 103). Swerling-1 CFAR loss is then 1.2–1.7 dB at
   Pd 0.9 (`w2_pd_vs_snr.png`).
5. **Real noise is not Gaussian in every season.** In MBARI 70–90 kHz noise, January, April and
   July follow the exponential tail. September has a heavy tail (P > 9.2 floor ≈ 1e-2), so a
   fixed threshold set for Pfa 1e-4 false-alarms 90× too often, while every CFAR holds Pfa
   (`w2_mbari_noise_tails.png`, `pfa_empirical.csv`).
6. **CFAR choice is a trade (H3).** At a +15 dB clutter edge, GO keeps the false-alarm excess to
   about 3×, against 12× for CA and 15× for OS. With a second target 4–8 cells away, OS keeps
   Pd 0.75 while CA drops to 0.59 and GO to 0.52. Default: OS, which handles adjacent highlights
   and debris; GO is the alternative at bottom transitions (decision 105). In K clutter whose
   texture is patchy on the scale of the window, CFAR re-normalises and holds Pfa. Spiky
   clutter at the cell scale breaks every exponential-model CFAR (28–47× at ν = 0.5), so a
   K-aware threshold built on W1's fits is the next step.
7. **CFAR finds echoes, not targets.** On HB2305 ES70 pings, OS-CFAR at Pfa 1e-4 returns about
   21 echoes per ping in the water column (plankton, fish). A 20 dB SNR gate leaves 4.5 and a
   25 dB gate leaves 0.8 (`w2_noaa_echogram.png`). The gate and the per-scan false-alarm budget
   are set in W4 from mission cost, not here.
8. **The bottom tracker must ignore the target.** A strongest-echo tracker locked onto a body
   lying on the bottom. The persistence tracker (decision 104) does not, and it held on all 184
   real pings.

## Limits and open items

- Body TS (−20 dB, five highlights) is assumed (decision 102). Absolute Pd moves with it; the
  findings above do not.
- Compression gain on the real bottom echo is 3–5 dB, against 10–20 dB BT. The bottom is spread
  over many cells, and BT applies to point targets (verified in the simulator), not to the bottom.
- MBARI noise is uncalibrated (dB re full scale), scaled to a stated floor; the bands stop at
  128 kHz.
- **Pending:** the UATD-beam and Bath ping-row range-profile bridge, and the beamwidth sweep for
  H9 (`sum_adjacent_beams` is ready). Both datasets are blocked by this environment's network
  policy; run them in W3 once fetched.
