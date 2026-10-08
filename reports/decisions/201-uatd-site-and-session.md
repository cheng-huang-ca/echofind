# 201 · UATD site, session and split groups come from sonar settings in the XML

**Status:** accepted for W3 · **Owner:** W3

## Context
UATD frames carry no site, date or sequence field, and the released files are shuffled across
Training, Test_1 and Test_2. Grouped splits need a site and a session, or near-duplicate frames
leak across folds (CLAUDE.md; H4). Each XML does log frequency, range setting and the sound speed
the sonar used.

## Evidence (all 9,200 frames)
- Sound speed is bimodal: 1,457–1,488 m/s in 6,949 frames, 1,492–1,585 m/s in 2,251 frames.
  Fresh water at 10–20 °C is 1,447–1,482 m/s and sea water at 32 PSU is 1,490–1,522 m/s, so
  the split matches the paper's lake and sea sites.
- Planes and the ROV are present only above 1,490 m/s (sea); the human body, square cages and
  most metal buckets only below it (lake). There are **no human-body frames at sea**.
- Lake body frames fall in four tight blocks: 720 kHz at 1,474–1,475 m/s and 1,200 kHz at
  1,466–1,468 m/s, each at range settings of about 10 m or 15 m. A range setting change means the
  operator stopped and re-configured, so these blocks are the closest thing to sessions.

## Decision
- `site` = sea if sound speed ≥ 1,490 m/s, else lake (`echofind.io.uatd`).
- `session` (bootstrap cluster) = frequency × site × range setting (0.5 m) × sound speed (1 m/s).
- `grp` (split unit) = frequency × range band (range rounded to 5 m). The four lake groups with
  bodies are `720k_r10`, `720k_r15`, `1200k_r10` and `1200k_r15`; cross-session holds out one at a
  time. Cross-frequency trains on one lake frequency and tests on the other.
- Sea frames are a false-alarm-only test of a new site: the lake threshold is applied there,
  and false alarms per frame are reported. Recall at sea cannot be measured.
- The official Training/Test split is not used: it shares sessions across its parts.

## Consequences
- Lake-to-sea transfer of *recall*, planned in W4, is impossible with UATD. The new-site
  claim for bodies has to come from Bath once it is labelled (decision 202).
- 720 kHz and 1,200 kHz frames may show the same scenes, because the Gemini switches
  frequency within a deployment. Cross-frequency therefore isolates the sensor shift, not a
  scene shift.
- Sessions are inferred, not logged. Neighbouring sound-speed bins may still be one recording,
  so the bootstrap clusters are, if anything, too fine, which makes intervals too narrow.
  The split unit (`grp`) is coarser for that reason.
