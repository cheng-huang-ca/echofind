# 105 · OS-CFAR as the default detector, followed by an SNR gate

**Status:** accepted for W3 input · **Owner:** W2

## Evidence (reports/w2/*.csv)
- Homogeneous noise: all four detectors hold Pfa. Swerling-1 SNR at Pd 0.9 is ideal 19.0,
  CA 20.2, GO 20.5, OS 20.7 dB (white, 4B, strided). The CFAR loss is 1.2–1.7 dB.
- Interfering target 4–8 cells away (two highlights, or a body beside a tyre): Pd goes from
  0.77 to 0.59 for CA and 0.52 for GO, while OS stays at 0.75 and SO at 0.73.
- +15 dB clutter edge, loud side, first 14 cells: mean false-alarm excess is CA 12×, OS 15×,
  GO 3×, SO 217×.
- Real MBARI noise: a fixed threshold false-alarms at 90× the design rate in the September
  file (transients); every CFAR stays at or below the design rate in all four seasons.
- Real water column (HB2305 ES70): OS-CFAR at Pfa 1e-4 returns about 21 echoes per ping from
  plankton and fish; a 20 dB SNR gate leaves 4.5 and a 25 dB gate leaves 0.8.

## Decision
Default `CFARConfig("os", n_ref=32, n_guard=2·stride, pfa=1e-4, stride=round(fs/B))`,
then `min_snr_db` on candidates. Of the four detectors, OS loses least when two highlights or
objects sit close together, and that matters most for a body near debris. GO is the alternative
where clutter edges dominate (bottom transitions), so W3 can compare OS and GO as a rung-0
choice. SO is never used.

## Consequences
H3 as pre-registered ("GO or OS better than CA at clutter edges") is supported for GO and
refuted for OS. Both outcomes are reported. A K-aware threshold, with its scale from W1's
fitted ν (0.4–0.7 near the bottom on HB2305 FM channels), is required next: no exponential-model CFAR holds Pfa in spiky clutter at the
resolution scale (28–47× at ν = 0.5).
