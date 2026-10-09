# 203 · Object-level scoring at 0.05 false alarms per frame, on shared CFAR proposals

**Status:** accepted before any ladder result was computed · **Owner:** W3

## Proposals (shared by every rung)
OS-CFAR along range (N = 32, G = 4, Pfa = 1e-3) on block-averaged, range-normalised power
(4 cm range cells; 1 beam at 720 kHz, 2 at 1,200 kHz), connected components, SNR ≥ 8 dB, at least
3 cells, at most 60 per frame. Block averaging makes cells non-exponential, so the Pfa is a
knob, not a guarantee. The settings were picked on a 120-frame sample (60 body frames, 60
random), drawn from all groups, from a 3 × 4 sweep of Pfa (1e-2 to 1e-4) and SNR gate
(6 to 12 dB). Body proposal recall was 1.00 at every setting; the choice keeps about 20
candidates per frame, so later rungs face a realistic clutter load. The sample covers the test
groups, but the setting gave the same recall everywhere, and all rungs share the proposals,
so it cannot favour one rung over another.

## Scoring unit
- **Object-level recall:** a body object counts as found if any candidate whose peak falls inside
  its box (padded by 10%) scores above threshold. Bodies the proposal stage missed count as
  misses for every rung.
- **False alarms:** every other candidate above threshold, including those on confuser
  objects. Each is counted separately, because each would send a diver.
- **Operating point:** 0.05 false alarms per frame (one in 20 frames), fixed now. Secondary
  points (0.01 to 1.0) appear on the FROC curves. A handheld sweep is tens of frames, so
  0.05 per frame is about one false alarm per sweep, the plan's stretch target.

## Two readings of recall at the operating point
- **recall_at_op** uses the threshold that gives 0.05 FAPF on the test data itself (a FROC
  point). It measures ranking quality, and the ladder compares rungs on it.
- **recall_transfer / fapf_transfer** use a threshold set from out-of-fold scores on the
  training side (3-fold, grouped by session), then applied unchanged to the test data. This is
  what a fielded device would do, and it shows whether the false-alarm rate holds under shift.

## Keep rule
A rung is kept only if its recall_at_op beats the rung below on the cross-session and
cross-frequency splits, with a paired cluster-bootstrap 95% interval (by session) that excludes
zero. The random-frame split is reported only to show leakage (H4).
