# 303 · Re-set the false-alarm threshold at a new site from body-free frames

**Status:** proposed for the product; tested in W4 · **Owner:** W4

## Context
W3 found that a threshold set on one frequency misses its false-alarm target by up to 5× at
the other (R1, 720 → 1,200 kHz: 0.26 false alarms per frame against a 0.05 target), or
silences the detector (R2, 1,200 → 720 kHz: recall 0.002). The false-alarm half of the
operating point needs only frames *without* a body. A crew can record those anywhere,
at any time, with no target and no labelling.

## Decision
On arrival at a new site, or after a frequency switch, the device records a short body-free
sweep. It sets the score threshold so that those frames give the target false-alarm rate
(`echofind.eval.metrics.threshold_at_fapf` on their candidates). The model is not retrained.

## Evidence (reports/w4/recalibration.csv; 200 random draws per size)
With 100 body-free frames, about three to four 30-frame sweeps:

| Shift | Rung | FAPF, source threshold | FAPF, 100 frames (10–90%) | Recall, source → recalibrated |
| --- | --- | --- | --- | --- |
| 720 → 1,200 kHz | R1 | 0.26 | 0.06 (0.04–0.10) | 0.52 → 0.21 |
| 720 → 1,200 kHz | R2 | 0.008 | 0.05 (0.03–0.08) | 0.02 → 0.08 |
| 1,200 → 720 kHz | R2 | 0.000 | 0.05 (0.03–0.09) | 0.002 → 0.23 |
| lake → sea | R2 | 0.06 | 0.06 (0.03–0.10) | n/a (no bodies) |

Ten frames are too few (10–90% band up to 0.4). Fifty to a hundred hold the target within
about ×2.

## Consequences
- Recalibration fixes the false-alarm rate, not the ranking. Recall is then whatever the
  shifted model can give at that rate. For R1 at 1,200 kHz that is lower than the inflated
  figure seen before recalibration, which was bought with 5× the false alarms.
- The product needs a "calibrate here" step in the workflow, and the model card states the
  recalibration requirement.
- Body-free frames drawn at random from the same sessions as the test frames are slightly
  optimistic. A field test should take them from a separate sweep.
