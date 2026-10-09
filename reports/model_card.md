# Model card: EchoFind R2 candidate classifier (UATD, checkpoint)

**Status:** research checkpoint, not for operational use. Updated Oct 8, 2026 (W4).

## What it is
- **Task:** score each CFAR candidate in a forward-looking sonar frame as body or not body.
  An alarm is a candidate above the threshold.
- **Pipeline:** 8-bit beam × range frame → block-averaged power → range normalisation
  (row median) → OS-CFAR (N 32, G 4, Pfa 1e-3) → connected components with an SNR gate of
  8 dB → 38 physics features → LightGBM (300 trees, 15 leaves).
  Code: `echofind.features.fls`, `echofind.models.ladder.R2Trees`.
- **Alternatives kept for comparison:** R0, a CFAR plus size-gate rule, and R1, a logistic
  model on 10 features. Both are worse than R2 at every condition except 720 → 1,200 kHz,
  where R1 ranks better.

## Training data
UATD (Xie et al., 2022, CC BY 4.0): Tritech Gemini 1200ik at 720 and 1,200 kHz. Lake: 6,949
frames, 1,434 instances of one human-body mannequin, nine confuser classes. Sea: 2,251 frames,
no bodies. Sites and sessions are inferred from the logged sound speed and range setting
(decision 201).

## Intended use and limits
- **Intended:** an aid that points a trained operator at candidates, with the raw sonar view
  always shown. It is never the only basis for ending a search.
- **Not validated for:** real human bodies (mannequin only); range beyond 12 m (3 test bodies);
  any sonar other than the Gemini 1200ik; side-scan data (Bath labels pending); sea or river
  bodies; turbid, weedy or ice conditions.

## Performance (object-level recall at 0.05 false alarms per frame, 95% cluster-bootstrap CI)

| Condition | Recall | False alarms per 30-frame sweep | Notes |
| --- | --- | --- | --- |
| New lake session, same frequency | 0.41 [0.31, 0.56] | 1.5 | 0–6 m: 0.49; 9–12 m: 0.34 |
| Random frames (leaky, not a valid estimate) | 0.86 | 1.5 | Shown only to warn against random splits |
| Trained at 720, used at 1,200 kHz | 0.08 [0.05, 0.12] | 1.5 | With the source threshold: 0.02 recall |
| Trained at 1,200, used at 720 kHz | 0.21 [0.09, 0.33] | 1.5 | With the source threshold: 0.002 recall |
| Lake model at the sea site | no bodies | 1.8 | 45% of false alarms on the aircraft model |
| Input saturated (+12 dB) | 0.34 | 1.7 | Fault injection, 720 kHz |
| 10% of range rows dropped | 0.32 | 1.4 | Fault injection |

Calibration: ECE 0.010 over all candidates, but 0.09 (new session) to 0.43 (new frequency)
among candidates with p ≥ 0.05. Probabilities are not reliable under shift; treat the score as
a ranking.

## Operating requirements
1. **Calibrate at every new site and frequency.** Record 50–100 body-free frames, then set the
   threshold to the target false-alarm rate on them (decision 303). Without this step the
   false-alarm rate is off by up to ×5, or the detector goes silent.
2. **Set the target false-alarm rate from mission cost** (decision 301). The best rate is
   0.008–0.54 per frame for plausible check and re-search times.
3. **Run the input QA checks** (`echofind.eval.qa`) and the OOD score (`echofind.eval.ood`)
   on every frame, and show their state to the operator.

## Known failure modes
- Confuser objects at close range (balls, tyres, cages) are the main source of false alarms,
  not background.
- Missed bodies are mostly thin, line-like returns at 1,200 kHz, 7–11 m, with little shadow.
- A new object type (aircraft model at sea) causes false alarms; the OOD score flags 78% of them.
- A frequency change collapses ranking. Recalibration restores the false-alarm rate but not
  recall.

## Hazards (FMEA-lite)

Severity: 3 = a body is missed with no warning; 2 = a missed body is likely but the operator
is warned, or false alarms flood; 1 = degraded but visible. Likelihood is for a handheld search
in inland water. Evidence links point to measured results where they exist.

| Hazard | Effect on the detector | Sev | Lik | Detection on the device | Mitigation | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Frequency or sonar changed from training | Ranking collapses; FAPF off ×5 or silent | 3 | High (dual-frequency product) | Device knows its own frequency | Per-frequency model; calibrate on body-free frames | W4 §5, decision 303 |
| New site, bottom type or object type | False alarms on unseen objects | 2 | High | OOD score (52% of sea false alarms flagged) | Calibrate on site; OOD cue on candidate | W4 §6 |
| Receiver saturation (close target, high gain) | Recall −9 points | 3 | Medium | QA saturation check (56%) + OOD (66% combined) | Automatic gain limit; QA warning | Fault injection |
| Dropped pings or rows | Recall −11 points | 2 | Low | QA (100%) | Warn and rescan | Fault injection |
| Dead or blocked beams | Recall −7 points; blind sector | 3 | Low | QA (100%) | Warn; show the blind sector on the display | Fault injection |
| Boat wake or bubbles | Clutter in near range; recall unchanged in test | 2 | Medium | OOD only (53%); QA misses it | Rescan after the wake clears | Fault injection |
| Device tilt (beam into surface or bottom) | Bottom or surface floods CFAR; body outside beam | 3 | Medium | Not tested (no tilt data); needs an IMU | IMU tilt gate; prompt the operator | Gap: W6 |
| Body beyond 12 m | Untested | 3 | High (sold to 50 m) | None | Collect 12–50 m data | W4 §2 |
| Real body differs from mannequin | Unknown echo and shadow | 3 | Certain | None | Field target-strength study | W6 |
| Operator over-trusts "no alarm" | Search ended early | 3 | Medium | — | Show raw view; show "coverage incomplete" when QA or OOD fails | Product requirement |

## Ethics and data
UATD is CC BY 4.0. No real victim images are used or shown. SCTD and the FLS victim set have no
stated licence and are not used in this checkpoint.
