# 002 · NOAA file selection, impulse filter and noise window

**File selection.** `configs/data.yaml` originally took the first 8 `D*-T*.raw` keys of HB2305.
Those are consecutive 20-second files recorded alongside in Newport harbour (41.53 N, 71.32 W),
so they show one place and one minute. The fetcher now supports `spread: true`, which spaces the
picks evenly over all 11,700 files. The 8 chosen files run from 21 to 26 July and sit on the
southern New England shelf, with the seabed 27 to 39 m below the transducer.

**What the files contain.** Contrary to the scaffold's note, the `D*` files are not CW-only.
Every file holds complex FM samples for ES38 (34–45 kHz), ES70 (45–90 kHz) and ES200 (160–260 kHz),
plus an 18 kHz CW channel (ES18), so pulse compression applies to the whole set.

**Impulse filter.** About 5% of ES70 pings carry a burst roughly 2 m long at a random range, up
to −18 dB Sv, louder than fish. That pattern is interference from another sounder in band (the
ship's ME70 multibeam runs at 70–120 kHz). We mask samples that exceed both neighbouring pings by
10 dB after a 0.5 m smooth (Ryan et al. 2015). Statistics use the masked data; echograms show
the raw data so the interference stays visible.

**Noise window.** The noise floor is the median level before TVG over the farthest 15% of the
record. That window is used only for the 500 m records, because in the 100 m records it still
contains the second and third seabed multiples.

**Seabed detection.** The detector picks the strongest sample beyond 5 m in each ping, smooths
the picks into a track with a 9-ping running median, and re-picks any ping more than 1.5 m off
the track. A first-threshold-crossing detector was tried first, but interference bursts
captured it.
