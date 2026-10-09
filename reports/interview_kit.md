# Interview kit: EchoFind for the AquaEye Applied Scientist role

Material to rehearse. Every number comes from the repo and is linked to its source. The
technical report is [technical_report.pdf](technical_report.pdf); the deck draft is linked
from the README.

## Pitches

**30 seconds.** "I built EchoFind, an open pipeline that finds a submerged mannequin in
public sonar data, from raw echoes to C++ on an edge CPU. Published CNNs on side-scan body
data fall from 97% to 47% F1 at unseen sites, so I built everything around testing at new
sessions, frequencies and sites. The headline: a small CNN finds 61% of bodies at a new
session at one false alarm per 20 frames, against 41% for my best feature model. But a
frequency change breaks every model, and a short body-free calibration sweep is what keeps
the false-alarm rate honest in the field."

**2 minutes.** Cover these in order:
1. **The metric.** Recall at a fixed false-alarm rate per frame, turned into expected minutes
   to clear an area.
2. **The data and its traps.** The sets have no site or session labels, so I inferred
   sessions from the logged sound speed. Random splits doubled recall (0.86 against 0.41).
   Bath had lost its target positions.
3. **The signal.** Clutter is K-distributed, not Rayleigh, so fixed thresholds fail. Near the
   bottom a body is reverberation-limited.
4. **The ladder and its keep rule.** Trees beat simpler rungs. A 113k-parameter CNN beat the
   trees on 2 of 3 tests, missed the pre-registered rule on the third, and I did not move the
   goalposts. Pretrained ResNet-50 did worse.
5. **The field view.** Threshold drift at a new frequency, fixed by recalibrating from
   50–100 body-free frames. Mission cost showed two of the simpler models make searches
   longer.
6. **The edge.** The C++ front end matches Python to 3e-7 and takes 16 ms per 50 m ping
   against a 68 ms budget.
7. **What I would collect.** An 8-day blocked trial with paired frequencies, sized from a
   measured ICC.

## STAR stories from the project

### 1. Why it failed: the leakage that doubled recall (curiosity)
- **Situation:** UATD has no site or session field, and its official train/test split shares
  sessions. My first gradient-boosted model scored 0.86 recall at the operating point.
- **Task:** decide whether that number would survive a new session before building on it.
- **Action:** I inferred sites from sound speed (fresh water under 1,490 m/s, salt water
  above) and sessions from range setting and sound-speed bins (decision 201). I then reran the
  same model on held-out session groups, and measured nearest-neighbour distances between test
  and training bodies under both splits.
- **Result:** recall fell to 0.41 [0.31, 0.56]. Test bodies sat 20% closer to a training body
  under the random split (median distance 2.37 against 2.94). From then on every split in the
  project is grouped, and the random split is kept only as a warning (H4).

### 2. When simple won, and when it didn't (judgment)
- **Situation:** a ladder from a CFAR rule to pretrained CNNs. The keep rule, a paired win
  with an interval above zero on cross-session and both frequency directions, was committed
  before any result.
- **Action:** I ran the rungs on identical candidates and splits.
  - The 2-D CNN won by +0.20 and +0.48, but only +0.10 [−0.09, +0.25] on 1,200 → 720 kHz.
  - Pretrained ResNet-50 and MobileNetV3 did worse than the small CNN.
  - Mission cost showed the rule and logistic rungs make searches *longer* at any threshold.
- **Result:** I reported the CNN as "best, but not kept" (decision 602) and kept trees as the
  shipped edge model until paired-frequency field data retests it. The lesson I would bring:
  pre-register the bar, and let a cheap model ship while a better one earns its place.

### 3. A hardware trade (collaboration with the hardware lead)
- **Situation:** the handheld must finish each ping before the next one at the 50 m setting,
  which gives 68 ms.
- **Action:**
  - I ported the front end to C++17 and benchmarked per stage on one pinned core.
  - Float32 filter state missed the parity bar (1.01e-5), so I moved the biquad state to double
    and kept the signal in float, which is cheap on a Cortex-A FPU.
  - Tree thresholds are rounded down so float inputs take the same branches.
  - The int8 CNN showed no speed-up on x86, because the CPU lacks int8 dot-product
    instructions.
- **Result:** 16.4 ms p95 at 200 kHz, which leaves 4.1× headroom for an ARM core, or 2.4× with
  the CNN. Three questions for hardware:
  - Does the target core have SDOT (for int8)?
  - Can the front end decimate before filtering?
  - Does the device form a bearing × range image during a 360° sweep? The CNN gain needs a
    2-D view; a single-beam profile model only ties the trees.

### 4. From your career (prepare your own)
Mentoring someone, a practice others adopted, field work in hard conditions, a disagreement
settled with data.

## Hard questions about this project, and short answers

| Question | Answer |
| --- | --- |
| Why no Bath F1 against Nga et al.? | Its target positions were lost; 73 legs are marked "target" with no box. Scoring my own unverified boxes would grade the model against itself. There is a written protocol (about 4–6 reviewer hours). |
| UATD sessions are inferred. How sure are you? | The sound-speed split matches fresh and salt water physically, and planes and the ROV appear only above 1,490 m/s. Inferred sessions may merge or split real ones, so splits use a coarser group than the bootstrap clusters. |
| Would you ship the CNN? | Not yet. It misses the pre-registered rule on one direction, its threshold drifts (0.18 false alarms per frame at sea), and it adds 8 ms per ping. It would be the first thing retested on paired-frequency field data. |
| Your 0.05 false alarms per frame: is that good? | It is about 1.5 per 30-frame sweep. Mission cost says the right point depends on check time: 0.008 if a check means a dive, up to 0.5 if it means a re-aim. |
| Mannequin vs a real body? | A proxy-only claim. The trial adds a tissue phantom with lung-volume air, and target strength against aspect at both frequencies is the first measurement. |
| Simulation instead of field data? | Simulation covers geometry, SNR and DSP tests. It becomes training data only after a pre-registered sim-to-real test: SNR within 3 dB RMS, clutter shape in range, and a recall loss of at most 0.15. |

## First 90 days at AquaEye

| Area | Point to in EchoFind | First 90 days |
| --- | --- | --- |
| Signal processing | Signal atlas, clutter fits, front-end design note | The same atlas on AquaEye field data, by site and frequency |
| ML and algorithms | Ladder, keep rule, cross-session and cross-frequency results | Re-baseline the current detector on session-grouped tests with a simple-feature rung as a check |
| Data strategy | Coverage audit, blocked trial, ICC-based sample size, metadata schema | Audit existing data; run a pilot day; size the first campaign |
| Product and direction | Mission-cost operating point, recalibration step, edge budget | Agree mission metrics with product; propose the "calibrate here" workflow |
| Leadership | Decision records, model card, reproducible `make` targets | Start an experiment register and a monthly technical review |

Questions to ask AquaEye are in [docs/PLAN.md](../docs/PLAN.md#questions-to-ask-aquaeye).
