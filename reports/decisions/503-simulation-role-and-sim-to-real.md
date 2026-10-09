# 503 · Simulation fills geometry and noise, not body acoustics; a pre-registered sim-to-real test

**Status:** accepted for W6 · **Owner:** W6

## Context
The W2 simulator models geometry, spreading and absorption, Lambert and K-distributed
reverberation, image-method multipath and real MBARI noise. Its body is five highlights at an
assumed −20 dB broadside target strength (decision 102). The question is how far to rely on
it once field data exists.

## Decision
Simulation is used for:
- range and tilt planning;
- DSP verification against closed forms;
- ablations, such as beamwidth and SNR sweeps.

It is not a source of training or headline evaluation data until it passes this test, run on
the W6 trial data and fixed now:

1. **Level:** after fitting one target-strength offset per frequency on day 1, simulated SNR
   against range matches measured SNR on days 2–8 within 3 dB RMS, per frequency.
2. **Clutter:** the measured K-distribution shape ν per bottom type falls inside the range
   the simulator uses for that bottom. Cross-check against the W1 fits (ν 0.4–0.7 near the
   bottom on HB2305).
3. **Usefulness:** a detector trained on simulated pings only, tested on field placements,
   loses no more than 0.15 recall at 0.05 FAPF against the same detector trained on field
   data (leave one site out).

## Consequences
If test 1 or 2 fails, the simulator stays a DSP test bench, and the next step is to fit it
to the field data. If test 3 fails while 1 and 2 pass, the missing piece is target shape and
texture, which a simulator does not supply; collect more real placements.
