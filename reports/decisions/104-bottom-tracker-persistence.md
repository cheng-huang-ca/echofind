# 104 · Bottom tracking by persistence, not by the strongest echo

**Status:** accepted · **Owner:** W2

## Context
The first tracker took the strongest smoothed return as the bottom. In the simulator, a body
lying on the bottom (the case that matters) was stronger than the local seabed return and
captured the track; the candidate stage then removed the body as "bottom".

## Decision
The bottom is the range where a forward running median over `persist_m` peaks, with the
leading edge at peak − 6 dB. A compact target shorter than persist_m/2 cannot raise that
median. A cross-ping median removes single-ping outliers.

`persist_m` is a geometry parameter:
- forward-looking or tilted beam (the handheld case): grazing reverberation lasts metres, so
  use 3 m (longer than a body, 1.7 m);
- down-looking echosounder: the bottom tail is only h(1/cos θ − 1) (a few cm at 10 m with a 7°
  beam), so use 0.05–0.3 m.

Removing everything beyond the bottom (`exclude_beyond_bottom`) is right only for
down-looking beams. For a tilted beam the seabed beyond the bottom-entry range is where a
body lies, so the default keeps it.

## Consequences
The tests show that a −5 dB body on the bottom does not capture the track, that a down-looking
track sits within 5 cm of H − z_s, and that the track holds on all 184 HB2305 ES70 pings.
