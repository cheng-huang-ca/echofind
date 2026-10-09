# 301 · Mission-cost parameters for the operating point

**Status:** accepted for W4 (illustrative values; replace with AquaEye field numbers) · **Owner:** W4

## Context
The plan asks for the threshold that minimises the expected time to clear a search area,
E[T] = T_scan + N_fa × T_check + P_miss × T_research (`echofind.eval.cost`). No public source
gives these times for a handheld sonar search. A fixed 0.05 false-alarm-per-frame point
(decision 203) compares rungs fairly, but it is not what a crew should run.

## Decision
Default values, each a stated assumption to vary rather than a measurement:

| Parameter | Default | Range swept | Reasoning |
| --- | --- | --- | --- |
| Frames per scan F | 30 | fixed | One 360° handheld sweep at a few frames per second |
| T_scan | 60 s | fixed | Same in every case, so it does not move the optimum |
| T_check per false alarm | 180 s | 60, 180, 600 s | Re-aim and look again (1 min) up to sending a diver down (10 min) |
| T_research per miss | 1,800 s | 600, 1,800, 7,200 s | Second search by other means; in a rescue, minutes matter more than in a recovery |
| Independent looks k | 1 | 1, 3 | k = 1 is conservative; k = 3 assumes independent looks, which multi-look fusion (R5) would have to earn |

## Consequences
- The optimum, and whether a detector helps at all, depends on T_check / T_research. W4
  reports the whole grid (`reports/w4/mission_cost.csv`), not one number.
- With k = 1 and default times, R0 and R1 raise E[T] at every threshold. The best setting
  for them is to raise no alarm, so a crew would do better to ignore them. Only R2 lowers
  E[T] below the "no detector" baseline in the cross-session test.
- A field trial should measure T_check (how long one false alarm actually costs) before any
  threshold ships.
