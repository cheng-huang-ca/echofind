# 202 · UATD is the primary set for rungs R0 to R2; Bath waits for box labels

**Status:** accepted for the W3 checkpoint · **Owner:** W3

## Context
The plan makes Bath the primary set, with leave-one-site-out as in Nga et al. (2024). The
archive's README says the target labels were lost. Only 73 legs at two sites are marked as
showing the mannequin (59 in `target/` folders plus 14 from an alternate deployment), and none
gives the mannequin's position in the leg (`reports/w3/bath_legs_summary.csv`). In a 990 kHz
leg the mannequin is a highlight with a shadow among many similar marks: debris, tyres and
bottom texture. A first look at sample legs found that it cannot be located reliably from one
viewing. The plan's risk table names this case: "make UATD primary if needed".

## Decision
- The W3 ladder (R0 to R2) runs on UATD candidates, which have ground-truth boxes for the
  human body model and nine confuser classes.
- Bath gets a written labelling protocol (`reports/w3_bath_labelling_protocol.md`) and step 1
  of it, a leg-level label table. No Bath detection numbers are reported until steps 2 to 4
  (verified negatives, graded boxes, double-labelling with kappa) are done.
- The Nga et al. comparison (F1 on unseen sites, 0.47) stays a Must item for W4, run on Bath
  once labelled.

## Consequences
- The headline "new site" result for bodies cannot come from UATD (no bodies at sea; decision
  201). Until Bath is labelled, cross-site evidence is limited to cross-session and
  cross-frequency shifts at one lake, plus false alarms at the sea site.
- Labelling Bath is about 4 to 6 hours of reviewer time (149 legs to verify, 73 legs to box,
  10% double-labelled). It needs a human reviewer, not this session, because a model-made
  box would be scored against itself.
