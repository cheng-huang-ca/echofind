# 101 · Simulate echoes at complex baseband, not passband

**Status:** accepted (W2) · **Owner:** W2

## Context
The simulator must produce thousands of pings for Pd sweeps on a 4-vCPU VM. A passband
simulation at 80–200 kHz needs fs ≥ 0.5–1 MHz; the information lives in a 20–40 kHz band.

## Decision
`echofind.sim.scene` builds each ping as complex baseband at fs = 4·max(B, 1/T):
every scatterer is an impulse with the exact carrier phase exp(−j2π fc τ), placed at the nearest
sample, then convolved once with the baseband pulse. Passband (`dsp.waveforms.to_passband`) is
generated only where the band-pass and IQ demodulation blocks are under test.

## Consequences
- 25–50× fewer samples than passband; a 60 m ping takes ~50 ms including about 50k bottom
  facets.
- Envelope delay is quantised to ±1/(2fs) = ±B⁻¹/8 (under 5 mm at 20 kHz); carrier phase is
  exact, so speckle and multipath interference are right.
- The image-method arrivals and Lambert level are checked against closed forms
  (`tests/test_sim.py`), so the shortcut is verified rather than assumed.
- A C++ port (W5) consumes the same baseband IQ, which is also what EK80 complex files hold.
