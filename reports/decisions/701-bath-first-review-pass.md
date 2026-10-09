# 701 · Bath labels, first review pass: Underfall Yard usable, Bathampton canal not

**Status:** reviewer 1 done (Claude); reviewer 2 and adjudication needed · **Owner:** Bath labelling

## What was done (protocol steps 2 and 3, one reviewer)
- `scripts/bath_render.py` renders all 222 990 kHz legs (73 target, 149 unlisted) at square
  2 cm pixels with a 1 m grid and one fixed contrast rule, with no model output.
- All 73 target legs were boxed in metres (across track, along track) and graded A/B/C under
  the protocol's definitions: `reports/bath/target_boxes_reviewer1.csv`.
- A seeded random sample of 12 unlisted legs (6 per site) was reviewed for target-free status:
  `reports/bath/unlisted_sample_reviewer1.csv`.

| Site | A | B | C | Not found | Target legs |
| --- | --- | --- | --- | --- | --- |
| Underfall Yard (harbour, 2 Oct 2019) | 16 | 16 | 4 | 1 | 37 |
| Bathampton canal (5 Oct 2017) | 0 | 18 | 16 | 2 | 36 |

## What it showed
- **Underfall Yard:** the mannequin is a distinct highlight with a sharp shadow on a smooth
  harbour floor, mostly at 4–9 m across track. 32 of 37 legs are A or B. In 5 of the 6
  sampled unlisted legs nothing is target-like, but one (L203) has a small highlight with a
  shadow at the start of the leg.
- **Bathampton canal:** the floor is covered with debris that casts the same
  highlight-and-oval-shadow signature. No canal leg reaches grade A, and 16 of 36 are C. In 4
  of the 6 sampled unlisted canal legs I see a target-like mark indistinguishable from the
  ones I boxed in target legs. Either the mannequin also appears in legs the archive did not
  list (one run passed the spot about 65 times), or a single reviewer cannot tell it from
  debris. Either way, canal boxes cannot serve as ground truth, and canal "unlisted" legs
  cannot be negatives.

## Decision
- No Bath detection number is reported yet. In particular, no leave-one-site-out result:
  one of the two sites has unreliable labels.
- Protocol step 4 is required before any use: a second reviewer double-labels at least the
  A/B Underfall legs and the 12 sampled unlisted legs, with kappa and IoU reported.
- For the canal, the next useful input is the USV track (`.mat`, not present for this
  session) or the deployment notes. A box that recurs at one ground position across legs is
  the mannequin; one that moves is debris. Without that, the canal stays out.
- If reviewer 2 agrees on Underfall Yard, the A/B boxes there give a within-site test set
  (32 legs). That is enough for an external check of the side-scan view, not for a
  cross-site claim.

## Consequences
The plan's Bath leave-one-site-out comparison with Nga et al. stays open. The main obstacle
is now named: canal ground truth, not effort.
