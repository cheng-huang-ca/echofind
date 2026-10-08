# 102 · A body is five aspect-dependent highlights, TS −20 dB at broadside

**Status:** accepted for W2 (to revisit with field data) · **Owner:** W2

## Context
No public data gives the acoustic target strength of a submerged human body at 50–1200 kHz.
Divers with gas in the lungs report roughly −10 to −20 dB at 70–120 kHz; a drowned body has
partly flooded lungs and should be weaker. The Bath and UATD sets use mannequins, whose
acoustics differ again (PLAN, "Mannequins are proxies").

## Decision
`echofind.sim.target.Body`: five point highlights along 1.7 m (head 10%, chest 50%, pelvis
20%, knees 10%, feet 10% of the power), broadside TS −20 dB, a Gaussian broadside lobe of 30°
falling to an end-on floor of −10 dB relative, random highlight phases per ping (so the sum
fluctuates like a Swerling target when highlights share a resolution cell). Every number is a
parameter.

## Consequences
- Labels and range extent come free (extent = 1.6 m · |cos aspect|).
- Absolute Pd vs range depends on the −20 dB assumption, so W2 reports SINR alongside Pd and
  quotes conclusions that hold across TS (reverberation-limited near the bottom; beam
  geometry dominates at short range).
- This is the first thing a field trial should measure (W6): TS vs aspect and frequency for
  a body proxy that matches tissue and lungs.
