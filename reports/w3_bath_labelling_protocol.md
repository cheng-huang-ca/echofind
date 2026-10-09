# Bath side-scan: labelling protocol (W3)

Status: step 1 done (leg-level table); steps 2 to 4 are not done yet and block any Bath number.
Owner: W3. Related: [decision 202](decisions/202-uatd-primary-bath-protocol.md), `scripts/w3_bath_legs.py`,
[`w3/bath_legs.csv`](w3/bath_legs.csv), [`w3/bath_legs_summary.csv`](w3/bath_legs_summary.csv).

## What the archive gives us

Each PNG is one survey leg on one side (port or starboard) at one frequency (450 or 990 kHz).
Its name holds the leg number, the slant range (15 m on almost every leg) and the along-track
length, for example `990_port_cond_12-15_00x18_80m.png` (leg 12, 15.00 m by 18.80 m). Columns
run across track (2,047 samples over 15 m, about 7.3 mm each). Rows run along track, and their
spacing changes with boat speed, so every leg needs rescaling to square pixels before any
geometric feature means anything (paper Figure 3).

The README says the target and background labels were lost. Only two sessions kept a `target/`
folder, which holds copies of the 990 kHz legs that show the mannequin:

| Session | Legs listed as target | Legs in the session not listed |
| --- | --- | --- |
| bathampton_canal / 5 Oct 2017 / run1 | 22, plus 14 from an alternate deployment | 130 |
| underfall_yard / 2 Oct 2019 / run1 | 13 | 19 |
| underfall_yard / 2 Oct 2019 / run2 | 24 (this run has no `990/` folder) | 0 |
| 14 other runs, 4 sites | none | 1,190 legs with no label at all |

The paper used 166 target snippets and 13,054 non-target snippets from four sites. Only 73
target legs at two sites can be traced from the archive, and none has a position for the
mannequin inside the leg.

## Label definitions

- **Leg labels** (step 1, automatic): `target` if the leg is in a `target/` folder;
  `unlisted` if its run has a `target/` folder but the leg is not in it; `unlabelled` if its
  run has no `target/` folder.
- **Box labels** (step 3, by hand): an axis-aligned box in rescaled leg coordinates around the
  mannequin's highlight and its acoustic shadow, with a confidence grade:
  - A: highlight and shadow both clear, extent 1.2 to 2.0 m along the long axis.
  - B: one of the two clear, or the extent is outside that range because of aspect.
  - C: something compatible is visible, but a reviewer could argue otherwise.
- **Snippets** (for the ladder): fixed 3 m by 3 m windows around each A or B box (positives),
  and windows tiled over verified target-free legs (negatives). C boxes are kept for analysis and
  left out of training and headline metrics.

## Steps

1. **Leg table (done).** `scripts/w3_bath_legs.py` writes one row per leg with site, session,
   side, leg number, range, along-track length and leg label.
2. **Verify `unlisted` legs.** Two reviewers look at every `unlisted` 990 kHz leg (149 legs)
   and mark it target-free, target-visible or unusable (smeared by a turn or a stationary
   boat, as the README warns). Any leg marked target-visible becomes a target leg. Only
   legs that both reviewers mark target-free become negatives.
3. **Localise the mannequin in target legs.** Rescale the leg to square pixels, then draw one box
   per visible mannequin and grade it. Use the same session's GPS waypoints (`gps.csv`) and
   the USV track (`.mat`) where they exist to check that boxes in neighbouring legs sit at a
   consistent ground position. Use one viewer and one fixed display stretch for every leg, and
   show no model scores, so a detector cannot steer the labels.
4. **Agreement and adjudication.** Double-label a random 10% of target legs (at least 10).
   Report Cohen's kappa on the leg-level target-visible call and the IoU of matched boxes.
   A third reviewer settles disagreements. Kappa below 0.6 or median IoU below 0.5 stops
   labelling until the box definition is revised.
5. **QA checks** before release: every box sits inside its image; no box sits in the nadir
   gap; along-track scaling has been applied; no target leg is also listed as a negative; and
   the counts per session match the table above.

## How the labels will be used

- Splits are leave-one-site-out (Bathampton canal versus Underfall Yard harbour), grouped by
  session. No session appears on both sides of a split.
- Only 990 kHz legs from sessions before 8 Apr 2022 are compared with Nga et al., to match
  the paper. The 450 kHz legs are a cross-frequency test.
- The 1,190 `unlabelled` legs are not negatives. They may be screened later for hard
  negatives, but only after review.

## Why the W3 checkpoint does not wait for this

Box labels do not exist, and the ladder's ground truth would be whatever box positions this
session guessed. Every Bath number would inherit that unverified guess. The checkpoint therefore
uses UATD, which ships with boxes, as the primary set (decision 202), and keeps Bath as a
labelled-later external test with leave-one-site-out evaluation.
