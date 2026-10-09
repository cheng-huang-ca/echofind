# 602 · R3b fails the keep rule on one test of three; it becomes the reference rung, flagged

**Status:** accepted for S4 · **Owner:** S4

## The rule (decision 203, written before any result)
A rung is kept only if it beats the rung below on cross-session and on both cross-frequency
directions, with paired cluster-bootstrap 95% intervals that exclude zero.

## Result (reports/s4/deep_paired.csv)
R3b − R2: cross-session +0.20 [+0.06, +0.33]; 720 → 1,200 kHz +0.48 [+0.36, +0.69];
1,200 → 720 kHz +0.10 [−0.09, +0.25]. Two passes, one interval includes zero.

## Decision
The rule is not changed after the fact: **R3b is not "kept" under decision 203**, and every
report states this.

R3b still becomes the *reference* rung for further work (the W6 trial analysis and the next
edge port), with a flag. Its point estimate beats R2 on all three tests, and the failing test
is the weakest one (501 bodies in 3 sessions). It loses nothing measurable against R2 anywhere.
Deploying it on a product would wait for the paired-frequency data the W6 trial collects,
where the same rule is applied again.

## Consequences
- R2 stays as the shipped edge model in `edge/` (W5) until R3b passes on new data. The
  R3b ONNX models exist (`data/processed/s4_onnx`), but they are not compiled into the edge
  build.
- The model card lists R3b's per-condition results next to R2's.
