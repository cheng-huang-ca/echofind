# 001 · MBARI noise levels are reported in dB re full scale

**Context.** W1 needs ambient-noise spectra from the MBARI MARS 256 kHz archive. MBARI publishes
calibration values only for its 2 kHz decimated product: −168.8 dB re 1 V/µPa before 14 June 2017
and −177.9 dB re 1 V/µPa after it, measured at 26 Hz and 250 Hz respectively. We found no
sensitivity curve, preamp gain or full-scale voltage for the 256 kHz WAV files, and the
hydrophone response near 100 kHz is unlikely to be flat.

**Decision.** Report every MBARI level in dB re digital full scale (dBFS, or dBFS²/Hz for PSDs).
Comparisons between seasons, bands and events stay valid because the recorder chain is the same.
No absolute sound-pressure levels are claimed.

**Consequences.** The simulator (W2) can use MBARI noise for its shape and fluctuation
statistics (spectral slope, tones, impulsiveness), with the level set by a target SNR rather than
by µPa. Converting to µPa needs MBARI's calibration JSON for the MARS deployment
(`MBARI_MARS_Hydrophone_Deployment02.json` in their pacific-sound repository) or a reply from the
maintainers, which stays open as follow-up F1 in the atlas.
