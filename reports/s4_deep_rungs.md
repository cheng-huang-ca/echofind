# S4: deep rungs R3 and R4 (and why there is no R5)

Four CNN rungs ran on the W3 protocol: the same 211k CFAR candidates, the same grouped folds,
recall at 0.05 false alarms per frame, and paired cluster-bootstrap comparisons with R2 on
the same candidates. Decisions: [601](decisions/601-deep-rung-protocol.md) (protocol) and
[602](decisions/602-r3b-keep-rule-verdict.md) (verdict). Code: `echofind.models.cnn`,
`echofind.models.train`, `scripts/s4_*.py`.

**Bottom line.** A 113k-parameter 2-D CNN (R3b) is the best rung on every split:
- Cross-session: 0.61 recall against R2's 0.41, a paired difference of +0.20 [+0.06, +0.33].
- 720 → 1,200 kHz: 0.56 against 0.08, +0.48 [+0.36, +0.69]. This was the failure that broke
  every W3 rung.
- 1,200 → 720 kHz: 0.31 against 0.21, but +0.10 [−0.09, +0.25] includes zero.

So R3b misses the pre-registered keep rule on one of three tests. The big pretrained networks
do worse than the small one. A frozen ImageNet ResNet-50, the backbone of Nga et al., is below
R2 everywhere, and MobileNetV3 is in between. What helps is learning the spatial pattern of
highlight and shadow, not model size or ImageNet features. A single-beam 1-D CNN (R3a) only
matches R2, so the gain needs more than one beam.

## Results (recall at 0.05 FAPF, 95% cluster-bootstrap CI)

| Rung | Input, size | Cross-session | 720 → 1,200 kHz | 1,200 → 720 kHz | Sea FAPF (lake threshold) |
| --- | --- | --- | --- | --- | --- |
| R0 rule (W3) | CFAR + size gate | 0.07 [0.01, 0.15] | 0.07 | 0.02 | 0.020 |
| R1 logistic (W3) | 10 features | 0.13 [0.04, 0.30] | 0.19 | 0.05 | 0.016 |
| R2 trees (W3) | 38 features | 0.41 [0.31, 0.56] | 0.08 [0.05, 0.12] | 0.21 [0.09, 0.33] | 0.060 |
| R3a 1-D CNN | range profile, 26k | 0.37 [0.29, 0.47] | 0.19 [0.11, 0.38] | 0.37 [0.11, 0.74] | 0.050 |
| **R3b 2-D CNN** | snippet, 113k | **0.61 [0.45, 0.83]** | **0.56 [0.45, 0.77]** | 0.31 [0.00, 0.46] | 0.183 |
| R4a ResNet-50 frozen | snippet, 23.5M (frozen) | 0.34 [0.18, 0.50] | 0.06 [0.04, 0.07] | 0.00 [0.00, 0.07] | 0.343 |
| R4b MobileNetV3-Small | snippet, 0.93M, fine-tuned | 0.47 [0.26, 0.70] | 0.22 [0.15, 0.33] | 0.15 [0.00, 0.23] | 0.154 |

![Ladder](figures/s4/ladder.png)

Paired differences against R2 ([`s4/deep_paired.csv`](s4/deep_paired.csv)):

| Rung − R2 | Cross-session | 720 → 1,200 | 1,200 → 720 | Keep rule |
| --- | --- | --- | --- | --- |
| R3a | −0.04 [−0.17, +0.09] | +0.11 [+0.03, +0.29] | +0.16 [−0.12, +0.54] | no |
| R3b | **+0.20 [+0.06, +0.33]** | **+0.48 [+0.36, +0.69]** | +0.10 [−0.09, +0.25] | 2 of 3 |
| R4a | −0.07 [−0.19, +0.02] | −0.02 [−0.06, +0.01] | −0.21 [−0.31, −0.09] | no |
| R4b | +0.06 [−0.09, +0.20] | +0.14 [+0.06, +0.26] | −0.07 [−0.26, +0.08] | no |

Training cost on 6 CPU cores, all 7 folds:
- R3a: 13 minutes.
- R3b: 2 hours (about 18 minutes per fold).
- R4a: 45 minutes of one-off feature caching, then 4 minutes.
- R4b: 80 minutes, on 12,000 negatives per fold.

Every training run stayed under the 30-minute limit, with a checkpoint each epoch.

## Is the CNN using a shortcut?

Many UATD frames show a fixed-range seam, a row where every beam drops in level. If R3b's gain
came from the seam rather than the target, it would sit in candidates whose snippet holds one.
It does not ([`s4/seam_check.csv`](s4/seam_check.csv)):
- Only 1.5% of body candidates have a 6 dB or larger full-width step inside the snippet window.
- R3b's recall on those 13 bodies is 0.15; on the other 1,421 it is 0.61.
- The 720 → 1,200 kHz result is the stronger argument. A seam learned on 720 kHz frames would
  not carry to 1,200 kHz bodies, yet that is where R3b gains most.

The step test also counts natural reverberation decay, so it over-counts seams. The conclusion
holds in that direction.

## Thresholds still do not transfer

R3b ranks much better, but its threshold moves with the data, as R2's did:
- From the training sessions' 0.05-FAPF point, it gives 0.095 false alarms per frame at new
  lake sessions, 0.31 at 1,200 kHz, and 0.18 at sea (mostly background and ROV).
- The W4 fix applies unchanged (decision 303): set the threshold from 50–100 body-free frames
  at the new site or frequency.

## Hypotheses

- **H4 (random splits overstate):** shown in W3 for R2; not re-run for CNNs.
- **H5 (normalisation narrows the shift gap more than a bigger network): not supported.**
  Normalising R2's features changes cross-frequency recall by −0.01 and +0.04. R3b gains +0.48
  and +0.10 over R2 ([`s4/h5.csv`](s4/h5.csv)). The big networks gain less or lose (R4a −0.02
  and −0.21; R4b +0.14 and −0.07). Learned spatial structure on normalised input beats both
  hand-built normalisation and model size.
- **H7 (multi-look fusion): not testable on UATD.** Frames are shuffled, with no timestamps or
  sonar pose, so no real scan sequence can be rebuilt (decision 601). R5 waits for Bath legs
  or the W6 trial.
- **H8 (int8 costs under 1 point and runs at least 2× faster): not met on this CPU.**

  | R3b, 4 cross-session folds | Recall @ 0.05 FAPF | Latency, 20 candidates, 1 thread (p50 / p95) | Model size |
  | --- | --- | --- | --- |
  | ONNX fp32 | 0.610 [0.447, 0.829] | 8.4 / 12.1 ms | 456 kB |
  | ONNX int8 (static, QDQ, per channel) | 0.593 [0.441, 0.799] | 8.4 / 11.6 ms | 137 kB |
  | Difference | −0.017 [−0.036, +0.002] | 1.0× | 3.3× smaller |

  The i7-8850H has no int8 dot-product instructions, so ONNX Runtime gains nothing.
  ARM cores with SDOT instructions might; the Raspberry Pi run decides
  ([`s4/int8_R3b.csv`](s4/int8_R3b.csv)).

## What it means for the edge budget and for AquaEye

- **Latency:** R3b adds 8.4 ms per ping for 20 candidates (fp32, one core) to the 16.4 ms front
  end at 200 kHz. That is about 25 ms of 68, against 0.2 ms for R2. The budget still holds on
  x86, but the headroom for a slower ARM core falls from 4.1× to about 2.4×.
- **Geometry:** the gain needs a 2-D view. A single-beam handheld sees a range profile, where
  the CNN (R3a) is no better than R2. AquaEye's 360° scan builds a bearing × range picture over
  a sweep. The W6 trial should record it so a 2-D model can be trained on it.
- **Sea:** R3b's false alarms at sea are mostly background (285), then ROV (74) and aircraft
  (47). Recalibration on site (decision 303) and hard-negative collection (W6 item 4) still
  apply.

## Limits

- Deep rungs train on 80% of each fold's training sessions (20% held out for early stopping
  and the threshold). R2 trained on 100%, so the comparison slightly favours R2.
- The 1,200 → 720 kHz test has 501 bodies in 3 sessions, so its intervals are wide.
- One seed per fold; seed-to-seed spread was not measured (two hours per R3b run).
