# 501 · Field trial: randomised complete blocks by site and day, frequency within placement

**Status:** proposed for AquaEye · **Owner:** W6

## Context
W3 and W4 found three questions the public data cannot answer: frequency shift, ranges past
12 m, and targets other than one mannequin. A trial must separate these from site and session
effects, which UATD confounds.

## Decision
- **Blocks:** site × day (4 sites × 2 days). Sites are chosen to differ by bottom (mud, sand,
  rock, weed), so the bottom effect is confounded with site. With one site per bottom type,
  that is unavoidable, and the report says so.
- **Each block:** a complete replicate of range (10, 20, 50 m) × pose (prone, supine, curled)
  × clothing (light, heavy): 18 body placements in random order. Add 4 confusers (tyre, log,
  rock pile, debris bag) at random ranges, and a body-free sweep at the start and end of the
  day. File: `reports/w6/field_design.csv` (192 rows, seed 0).
- **Frequency** is recorded on every ping (both channels), so the frequency contrast is paired
  within each placement.
- **Body heading** is randomised and logged, not controlled. Aspect is a covariate.
- **Labelling** is blinded: labellers work from placement IDs, never the placement log.
- **Ground truth:** a GPS-tagged buoy plus a drop camera or diver, confirmed before the first
  pass.

## Why not a full factorial with bottom crossed
Crossing bottom with everything needs several sites per bottom type: 4 × more days for one
more contrast. The first product questions are frequency and range, and both are well powered
within blocks.

## Consequences
- 8 field days, 144 body placements, 32 confusers, 16 sweeps; about 6 hours of placements per
  day at 15 minutes each.
- The analysis unit is the placement (decision 502).
