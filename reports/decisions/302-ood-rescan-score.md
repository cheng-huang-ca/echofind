# 302 · OOD rescan score: Mahalanobis distance on detector features

**Status:** accepted for W4, with known weak spots · **Owner:** W4

## Context
The plan's safety section asks for an out-of-distribution score that asks for a rescan
instead of staying silent. It has to run on the device beside the detector, needs no labels,
and must be explainable to an operator.

## Decision
`echofind.eval.ood.MahalanobisOOD`: standardise the 38 R2 features of every candidate, fit a
Ledoit-Wolf Gaussian on training candidates, and score a candidate by its squared Mahalanobis
distance D². A frame's score is the 90th percentile of its candidates' D². The flag threshold
lets 5% of held-out training frames through. Cost on the device: one 38 × 38 matrix-vector
product per candidate.

Alternatives not taken now: an isolation forest (harder to explain, similar cost); detector
confidence (a shifted detector is confidently wrong, as the cross-frequency results show); a
deep feature density (no deep rung yet; revisit after S4).

## Evidence (reports/w4/ood_flags.csv, ood_vs_errors.csv, fault_injection.csv)
- It catches new *objects*. Of R2's false alarms at sea, the score flags 78% of those on the
  aircraft model, which is far from anything in the lake. It flags 33% of those on
  background, and 52% of all of them.
- It is weak on new *conditions*. AUROC against new lake sessions is 0.53–0.73 for a new
  frequency or site. Frames it flags under frequency shift fail no worse than unflagged ones,
  so it cannot predict the cross-frequency collapse.
- On new lake sessions at the training frequency it flags 3% (1,200 kHz) to 13–16% (720 kHz)
  of frames instead of the nominal 5%, because a new session is already a small shift.
- Under fault injection it flags 53% of frames with a boat-wake cloud, which the input QA
  checks miss entirely.

## Consequences
The score ships as a candidate-level "unfamiliar object here" cue and a frame-level rescan
prompt. It is not evidence that the detector is right. The frequency-shift hazard is handled by
threshold recalibration (decision 303) and per-frequency data (W6), not by this score.
