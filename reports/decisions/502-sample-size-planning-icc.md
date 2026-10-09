# 502 · Sample size by placements, with ICC 0.12 (measured) and 0.5 (planning) and a pilot

**Status:** accepted for the W6 memo · **Owner:** W6

## Context
A placement is seen in many pings. Counting pings as independent would overstate power by the
design effect DE = 1 + (m − 1)ρ (Kish 1965). `echofind.eval.power` implements the
two-proportion sample size, the design effect, the ANOVA ICC estimator and a Monte Carlo check
(`tests/test_power.py`: the formula's placements reach 0.80 ± 0.06 power in simulation).

## Evidence
- From UATD W3 R2 cross-session scores at 0.05 FAPF: ICC 0.12 [0.02, 0.24] for "found" across
  frames of one session (16 sessions, 1,434 objects; `reports/w6/icc.json`). A UATD session is
  an inferred recording at one range setting, not a single placement, so the true
  within-placement ICC is likely higher.
- Trial power, 48 placements per arm and 10 looks, simulated: a 0.15 difference has power 0.89
  at ICC 0.12 and 0.52 at ICC 0.5. A 0.20 difference has 0.99 and 0.74 (`reports/w6/trial_power.csv`).

## Decision
Plan with both values and run one pilot day first, then measure ICC on its 18 placements. If
ICC ≤ 0.3, run the 8-day design as proposed (a 0.15 difference detectable). If ICC > 0.3, add
a third day per site (72 placements per arm) rather than more passes, since extra looks at a
correlated placement add little.

## Consequences
Contrasts are analysed at the placement level (cluster bootstrap or a mixed model with a
random placement effect), never with pings as independent trials.
