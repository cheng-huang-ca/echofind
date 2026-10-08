# EchoFind: Project Plan for AquaEye Interview Prep

Oct 6, 2026 · @Huangc55

## Summary

EchoFind is a 4-week portfolio project that finds a submerged human body in public sonar data, from raw echoes to a detector running on ARM. It gives evidence for the 10 required qualifications a project can show (degree and years come from your CV) and depth in four Highly Valued areas, against a minimum of two.

- **The question:** can a handheld-class pipeline separate a body from rocks, tyres, debris and the bottom at known miss and false-alarm rates, and does that hold at a new site, frequency or sonar type?
- **The data:** public forward-looking and side-scan sonar sets that include human-body targets, raw echosounder pings from the NOAA archive on AWS, and real hydrophone noise. A small physics simulator covers controlled what-ifs.
- **The method:** a ladder from CFAR thresholding to engineered features to a small CNN. Each rung must beat the last on cross-site recall at a fixed false-alarm rate, or the simpler one ships.
- **The edge step:** the DSP front end rewritten in C++ with parity tests against Python, plus an int8 model benchmarked for latency and memory on a Raspberry Pi class board.
- **The interview kit:** a public repo, a 2-page technical report, a 10-slide deck, a field-data proposal for AquaEye, and three rehearsed stories.

Interview-ready checkpoint: Oct 20, 2026 (end of week 2). Full scope lands Nov 3, 2026. Open question: your interview date, which decides how much of weeks 3 and 4 to keep.

## What AquaEye is hiring for

The posting values judgment about sonar data more than any one model. It wants someone who learns what the signal contains, then picks the simplest method that survives field conditions. Five themes recur, and each one sets a requirement for the project.

| Theme in the posting | What it asks for | What EchoFind must show |
| --- | --- | --- |
| The data underneath the algorithm | Explore raw sonar; find noise sources, variability and the features that matter | Work starts from raw pings and beamformed frames, not pre-extracted features; a written signal atlas |
| Detection that holds up | FP/FN focus, generalization across environments, simple vs complex approaches | Splits by site and sensor, recall at a fixed false-alarm rate, a baseline that is allowed to win |
| Data strategy and field work | Decide what to collect, design experiments, label and QA data, find gaps and bias | A coverage audit of the public data and a hypothesis-driven field collection plan for AquaEye |
| Embedded product | Production-quality code, optimized for embedded hardware | C++ DSP with parity tests, an int8 model, measured latency and memory |
| Direction and leadership | Choose between data, DSP, ML or hardware fixes; communicate; set rigorous practice; mentor | Decision memos, a reproducible pipeline, model and data cards, one talk for mixed audiences |

One consequence for dataset choice: the classic UCI mines-vs-rocks sonar set cannot be the core. It ships as 60 pre-computed band energies, not raw signals.

## Project design

EchoFind treats AquaEye's task as echo classification under field shift: decide person or not person from sonar returns, then prove the call holds at a site the model has never seen.

- **Input:** returns from a shallow scene: raw pings, forward-looking frames or side-scan lines.
- **Output:** body or not body per candidate, with a calibrated score, range and bearing.
- **Safety:** misses cost most, but false alarms cost dive checks and trust, so both are reported at one fixed operating point.
- **Field shift:** sites, bottoms, frequencies and sensors change, and random splits hide it.
- **Compute:** each ping must be processed within a handheld budget.

&#91;embedded content: EchoFind architecture · 4 layers, 1 feedback loop\]

Each layer hands the next a smaller, cleaner object: pings become candidates, candidates become scores, and only the winning rung is ported to C++. Gaps found in evaluation loop back into the field-collection plan.

## Public datasets

Four free sources carry the project. Two contain human-body targets for detection, and two are raw-signal archives for DSP and noise work. Both body-target sets are CC BY 4.0.

| Dataset | Sensor and scene | Size | Role in EchoFind | Access and license |
| --- | --- | --- | --- | --- |
| [Bath side-scan body-like objects](https://researchdata.bath.ac.uk/1467/) (Rymansaib et al., 2024) | StarFish 450 and 990 side-scan on a small uncrewed boat; a canal, a river and a harbour in Bath and Bristol, UK, freshwater and tidal; a sunken mannequin as drowning-victim proxy; Oct 2017 to Apr 2022 | 6 GB archive; the companion paper used 166 target and 13,054 non-target snippets from 4 sites | Primary body-vs-background set; leave-one-site-out evaluation | CC BY 4.0, DOI 10.15125/BATH-01467 |
| [UATD](https://pmc.ncbi.nlm.nih.gov/articles/PMC9715547/) (Xie et al., 2022) | Tritech Gemini 1200ik multibeam forward-looking sonar at 720 kHz (512 beams) and 1200 kHz (1024 beams); a lake about 4 m deep and shallow sea 4 to 10 m; targets 5 to 25 m away | 9,200 frames (2,900 at 720 kHz, 6,300 at 1200 kHz); 10 classes including a human body model, tyre, cylinder and cube | Confuser classes; per-beam range profiles as AquaEye-like echoes; lake-to-sea and cross-frequency tests | CC BY 4.0, figshare DOI 10.6084/m9.figshare.21331143.v3 |
| [NOAA Water Column Sonar Data Archive](https://registry.opendata.aws/ncei-wcsd-archive/) | Simrad EK60 and EK80 scientific echosounders; raw ping-by-ping files from survey cruises at several frequencies | Pull 2 or 3 cruises (tens of GB) | Real raw active-sonar pings: TVG, bottom detection, noise and interference, single-target detection | Free use under a NOAA disclaimer; bucket s3://noaa-wcsd-pds, no AWS account needed |
| [MBARI Pacific Ocean Sound Recordings](https://registry.opendata.aws/pacific-sound/) | Hydrophone in Monterey Bay; 256 kHz originals plus 16 kHz and 2 kHz decimated files; recording since July 2015 | Continuous archive; sample 20 to 40 hours across seasons | Real ambient and interference noise for spectral analysis and for injection into simulated echoes | CC BY 4.0; per-year buckets such as s3://pacific-sound-256khz-2024, no account needed |
| [SCTD 1.0](https://github.com/TTFF322/SCTD-) (optional) | Side-scan images with ship, plane and drowning-victim classes; Pascal VOC boxes | Two archives (1.0 and 2.0) | External test of the side-scan detector on real victim images | License not stated: evaluate only, never redistribute |
| [FLS Detection Dataset](https://github.com/XingYZhu/Forward-looking-Sonar-Detection-Dataset) (optional) | Oculus M1200d forward-looking sonar; victim, boat and plane classes; Pascal VOC boxes | One zip | External test of the forward-looking detector | License not stated: evaluate only |

The benchmark to beat: on the Bath data, ResNet-50 and Xception averaged 97% F1 in-distribution but 47% on unseen data. An ensemble heatmap still located the target ([Nga et al., 2024](https://researchportal.bath.ac.uk/en/publications/automated-recognition-of-submerged-body-like-objects-in-sonar-ima/)). EchoFind's central claim is a smaller gap, measured the same way.

Caveats that shape the design:

- **Mannequins are proxies.** Their materials and trapped air differ from tissue, bone and lungs, so echo strength may not match a real body. That gap becomes a field-data question for AquaEye.
- **Frames from one survey line are near-duplicates.** Split by site and session, never by random frame, or the scores leak.
- **Bath and UATD ship images, not transducer voltages.** Raw-signal claims rest on the NOAA pings and the simulator.
- **SCTD victim images show real people.** Keep them out of slides and public repos.

Left out on purpose: UCI mines-vs-rocks (pre-computed features; a warm-up at most) and passive ship-noise sets such as DeepShip (AquaEye is an active sonar).

## Workstreams and experiments

Six workstreams run in order, and each ends in an artifact you can show. The experiment register at the end fixes every pass criterion before results arrive.

### W1 · Signal atlas: what the data contains (week 1)

Goal: describe the raw signal, its noise and its variability before training anything.

- **Echosounder pings (NOAA):** echograms per frequency, level vs range before and after TVG, the noise floor from passive or deep samples, and ping-to-ping fluctuation.
- **Noise (MBARI):** Welch PSDs and spectrograms by hour and season. Tag ship passages, rain, biological clicks and electrical tones.
- **Clutter statistics:** envelope histograms for open water vs near the bottom, fitted with Rayleigh, Weibull and K-distributions. Heavy tails explain why fixed thresholds fail.
- **Images (UATD, Bath):** speckle, gain artefacts, the side-scan nadir gap, acoustic shadows, and how the mannequin differs from tyres and cylinders at 720 and 1200 kHz.
- **Variability matrix:** background level, target contrast and SNR by site, depth, range, frequency and sensor.

Deliverable: a signal atlas of 10 to 15 annotated figures, plus a noise-and-variability register (source, signature, where seen, mitigation).

### W2 · Physics and DSP front end (weeks 1 and 2)

Goal: turn raw echoes into calibrated range profiles and candidate detections, with each block checked against theory.

- **Echo simulator:** CW or LFM pulse; two-way spreading and absorption for fresh or salt water; a body as a few aspect-dependent highlights over about 1.7 m; Lambert bottom reverberation; surface and bottom multipath by the image method; real MBARI noise added. Labels come free.
- **Front end:** band-pass and IQ demodulation, matched filter, envelope, TVG, then CA-, GO- and OS-CFAR, bottom tracking, and a candidate list (range, bearing, level, extent).
- **Bridge to AquaEye's form factor:** read each UATD beam, and each side-scan ping row, as a range profile. Summing adjacent beams emulates a wider handheld beam, so beamwidth becomes a testable hardware trade.
- **Dual frequency:** AquaEye is dual-frequency, so the simulator also produces paired low/high-frequency echoes for frequency-response features.
- **Tests:** pytest checks of pulse-compression gain, CFAR false-alarm rate and range resolution against closed-form values.

Deliverable: the `echofind.dsp` package with tests, Pd-vs-SNR curves, and a one-page front-end design note.

### W3 · Body detection: the model ladder (weeks 2 and 3)

Goal: find the simplest method that meets the operating point on sites it has never seen.

1. **R0 rule:** CFAR detection plus a size gate; no learning.
2. **R1 logistic regression** on about 10 physics features: peak level, echo extent, highlight count and spacing, shadow depth and length, local contrast, speckle contrast.
3. **R2 gradient-boosted trees** on about 40 features, with SHAP and ablations to show which features matter.
4. **R3 small CNNs:** a 1D CNN on range profiles, and a 2D CNN of about 100k parameters on snippets.
5. **R4 pretrained CNNs:** ResNet-50 as in Nga et al., and MobileNetV3-Small as the edge-sized option.
6. **R5 multi-look fusion (stretch):** combine scores across adjacent pings, beams or passes, as AquaEye's 360-degree scan and the Bath heatmap do.

A rung is kept only if it beats the rung below on leave-one-site-out recall at the fixed false-alarm rate. The paired bootstrap interval on that difference must exclude zero, and the compute must fit the edge budget.

Deliverable: a results table per rung (in-site, cross-site, cross-frequency), a failure gallery, and a feature-importance page.

### W4 · Field-honest evaluation and safety (weeks 2 and 3)

Goal: measure what a rescue crew would experience, not what a leaderboard rewards.

- **Splits:** leave-one-site-out on Bath; lake-to-sea and 720-to-1200 kHz on UATD; SCTD and the FLS victim set as external tests. Group by session, never by random frame.
- **Metrics:** recall at a fixed false-alarm rate per scan, FROC curves, PR-AUC, results by range band and site, calibration error, and cluster-bootstrap 95% intervals by session.
- **Operating point from mission cost:** expected time to find = scan time + false alarms × check time + miss probability × re-search time. Pick the threshold that minimizes it, then show how the choice moves when the assumptions change.
- **Safety:** an FMEA-lite hazard table covering saturation, unseen bottom types, device tilt, bubbles and wake. Mitigations: input QA checks, an out-of-distribution score that asks for a rescan instead of staying silent, and the raw view always shown to the operator.

Deliverable: a generalization report, and a model card listing per-condition performance and known failure modes.

### W5 · Edge deployment (week 3)

Goal: show that the chosen pipeline fits a handheld budget, with numbers.

- **C++17 port of the front end** (IQ demodulation, FFT matched filter, envelope, TVG, CA- and OS-CFAR), with pybind11 bindings and golden-vector parity tests against Python.
- **Model export:** the winning rung goes to ONNX with int8 static quantization, or a tree model is compiled to plain C. Record the accuracy change.
- **Budget from public specs:** at the 50 m setting, two-way travel time is 2 × 50 / 1480 ≈ 68 ms. Back-to-back pings therefore allow about 68 ms of processing each.
- **Benchmark** on a Raspberry Pi 4 or 5: p50 and p95 latency per ping, peak RAM, binary and model size. With no board, use one pinned laptop core and record its clock.
- **Stretch:** a Q15 fixed-point CFAR and matched filter, with measured SQNR.

Deliverable: the `edge/` library and CLI, a benchmark table, and an accuracy-vs-latency Pareto plot.

### W6 · Data strategy and field-experiment design (week 4)

Goal: turn the gaps you found into a collection plan AquaEye could run.

- **Coverage audit:** example counts by site type, depth, range, bottom, pose and frequency. Expect gaps such as one mannequin, no real bodies, no weed beds, no ice, and few targets beyond 25 m.
- **Factorial field design:** factors such as range (10, 20, 50 m), bottom (mud, sand, rock, weed), pose, clothing and frequency. Block by site and day, randomize placement, blind the labelers, and confirm ground truth with a GPS-tagged buoy plus a diver or camera.
- **Sample size:** a power calculation for detecting a stated recall difference between two conditions.
- **Labeling protocol:** schema and attributes, double-labeling 10% for Cohen's kappa, adjudication, and automatic QA for clipping, dropped pings and missing metadata.
- **Simulation's role:** what it can fill (geometry, range, SNR), what it cannot (real body acoustics), and a sim-to-real test.

Deliverable: a 2-page memo, "What AquaEye should collect next", with a metadata schema.

### Experiment register

| ID | Hypothesis | Test | Pass criterion |
| --- | --- | --- | --- |
| H1 | Background near the bottom is heavier-tailed than Rayleigh | Goodness-of-fit on NOAA and UATD background cells | Rayleigh rejected; K or Weibull preferred by AIC |
| H2 | Pulse compression gains about 10·log10(B·T) dB over the raw echo and keeps resolution at c/(2B) | Simulator sweep of bandwidth B and duration T | Gain within 1 dB of theory; resolution within 10% |
| H3 | GO- or OS-CFAR holds the false-alarm rate at clutter edges better than CA-CFAR | Simulated edges and NOAA bottom transitions | Lower false-alarm excess at edges; detection loss in flat noise reported |
| H4 | Random-frame splits overstate performance | Same model, random split vs leave-one-site-out | Gap reproduced and traced to near-duplicate frames |
| H5 | Physics normalization narrows the cross-site gap more than a larger network | R2 with and without TVG and per-site normalization, vs R4 | Larger cross-site gain from normalization |
| H6 | Shadow and extent features carry most of the separability for bottom-lying bodies | SHAP and drop-one-group ablation on R2 | Top feature groups named, with effect sizes |
| H7 | Multi-look fusion cuts false alarms at equal recall | R5 vs single look | False alarms per scan reduced, with interval |
| H8 | int8 costs under 1 point of recall and runs at least 2x faster than fp32 on ARM | Raspberry Pi benchmark | Both met, or the trade reported |
| H9 | Wider emulated beams lose recall on small targets at long range | UATD beam-summing sweep | A beamwidth-vs-recall curve for the hardware discussion |

## Coverage map

Every required qualification a project can show maps to an artifact, and four Highly Valued areas get headline depth. Degree and years of experience come from your CV; the project does not stand in for them.

| Required qualification | Evidence in EchoFind | Artifact to show |
| --- | --- | --- |
| Raw signals, sensor data, complex physical-world datasets | Raw NOAA pings, MBARI hydrophone audio, UATD and Bath sonar frames (W1) | Signal atlas |
| Signal processing and time-series analysis | IQ demodulation, matched filter, TVG, CFAR, PSDs (W2) | `echofind.dsp` with tests; design note |
| Noisy real-world data: features, patterns, variability | Noise register, variability matrix, clutter fits, SHAP and ablations (W1, W3; H1, H6) | Noise register; feature-importance page |
| Python with scientific and ML libraries | NumPy, SciPy, xarray, echopype, scikit-learn, LightGBM, PyTorch | Repo with CI |
| Algorithms from exploration to validation and implementation | One chain from atlas to ladder to field evaluation to edge (W1 to W5) | Technical report |
| Cleaning, preprocessing, feature engineering, models, experimental design | Input QA, normalization, 40 physics features, pre-registered hypotheses | Experiment register; MLflow runs |
| Math and science turned into working software | Simulator and CFAR code tested against closed-form results; C++ port (H2, H3) | Unit tests that cite the equation they check |
| Working with hardware and multidisciplinary teams | Beamwidth trade study (H9), a processing budget derived from device specs, a C API for firmware | Hardware trade note; C header |
| Technical communication and documentation | Atlas, design notes, model and data cards, a deck for mixed audiences | 2-page report; 10-slide deck |
| Curiosity: why it works or fails, not a black box | Leakage investigation (H4), failure gallery, per-site error analysis | "Why site X fails" write-up |

| Highly Valued area | Emphasis | Where | Evidence |
| --- | --- | --- | --- |
| Sonar, underwater acoustics, hydroacoustics, acoustics | Headline | W1, W2, W6 | Sonar-equation budget for the 10, 20 and 50 m settings; fresh vs salt absorption; reverberation and clutter statistics on real pings |
| Signal processing, DSP, spectral analysis and Fourier transforms | Headline | W1, W2 | IQ demodulation, FFT matched filter, Welch PSDs and CFAR, each tested against theory |
| Detection, classification and pattern recognition | Headline | W3, W4 | Six-rung ladder, FROC curves, operating point from mission cost |
| Edge ML, embedded and resource-constrained computing, C or C++ | Headline | W5 | C++17 front end with parity tests; int8 model; latency and RAM on a Raspberry Pi |
| Time-series or multidimensional sensor data | Supporting | W1, W2 | Ping × range × beam × frequency arrays in xarray |
| PyTorch and scikit-learn | Supporting | W3 | Rungs R1 to R4 |
| MLOps and production model lifecycle | Supporting | Repo | DVC data versions, MLflow registry with champion and challenger aliases, CI, model card |
| Data acquisition and experimental design for physical systems | Supporting | W6 | Factorial field design, power calculation, labeling protocol |
| Safety-critical or high-reliability products | Supporting | W4 | FMEA-lite, out-of-distribution rescan, mission-cost operating point |
| AWS or Azure | Light | W1 | Anonymous S3 reads from AWS Open Data; optional spot-instance training |
| Radar, lidar, remote sensing | Light | W2 | The CFAR family comes from radar; say so in the interview |
| IP development or patent strategy | Light | W6 | Optional one-page prior-art scan of human-detection sonar patents |

## Timeline and milestones

Two weeks buy an interview-ready core; two more add the CNN rungs, the edge port and the field memo. The plan assumes about 15 to 20 hours a week, roughly 70 hours in all. Shift every date by the same amount once your interview date is known.

&#91;embedded content: EchoFind schedule · 6 workstreams, 2 milestones\]

If the interview is under 7 days away, run the compressed core instead:

1. Day 1: pull the NOAA, UATD and Bath data; start the signal atlas.
2. Day 2: atlas figures and the noise register.
3. Day 3: matched filter, TVG and CA- and OS-CFAR, with tests against theory.
4. Day 4: rungs R0 to R2 on Bath, leave-one-site-out, with bootstrap intervals.
5. Day 5: the failure gallery and the mission-cost operating point.
6. Day 6: a one-page report and five slides.
7. Day 7: pitches and the question bank, practiced aloud.

## Running it on Claude Code cloud

Run each workstream as its own cloud session against one GitHub repo, ending in a pull request. The VM is CPU-only x86\_64 with about 4 vCPUs, 16 GB RAM and 30 GB disk, so the plan keeps data subsets small, makes training resumable, and leaves ARM timing to local hardware. Cloud sessions draw on your plan's usage limits; there is no separate compute charge ([cloud sessions](https://code.claude.com/docs/en/claude-code-on-the-web), [cloud environments](https://code.claude.com/docs/en/cloud-environments)).

### One-time setup (day 0)

1. Create a public GitHub repo `echofind` and connect it through the Claude GitHub App or `/web-setup`.
2. Export this doc to Markdown and commit it as `docs/PLAN.md`, plus a `CLAUDE.md` with the conventions from the Repository section.
3. Add `data/`, `mlruns/`, `*.raw`, `*.wav` and the job posting to `.gitignore`. Data never enters git.
4. Create a cloud environment named `echofind` with **Custom** network access, "include default list" checked, plus `figshare.com`, `*.figshare.com`, `researchdata.bath.ac.uk` and `download.pytorch.org`. The defaults already cover `*.amazonaws.com` (NOAA and MBARI buckets), PyPI and GitHub.
5. Set environment variables `BASH_DEFAULT_TIMEOUT_MS=600000` and `BASH_MAX_TIMEOUT_MS=1800000` so long steps are not cut at 2 minutes.
6. Use a setup script that only installs packages. It must finish in about 5 minutes to be cached, so downloads stay out of it.

```bash
#!/bin/bash
set -euo pipefail
uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install --system numpy scipy xarray echopype scikit-learn lightgbm shap timm \
  mlflow dvc hydra-core onnx onnxruntime pybind11 boto3 s3fs pytest ruff
```

7. Make `make data` idempotent with checksums, and have each data-heavy session run it first in the background. A reclaimed VM loses its files, so data must be cheap to fetch again.

### Session map

| Session | Workstream | Model | Can run in parallel with | Ends with |
| --- | --- | --- | --- | --- |
| S0 Scaffold | Repo, CI, make targets, data scripts | Sonnet |  | Merged PR; `make data` works |
| S1 Atlas | W1 | Sonnet | S2 | Atlas figures and noise register committed |
| S2 Physics and DSP | W2 simulator and front end (no downloads needed) | Opus for the design, Sonnet for code | S1 | `echofind.dsp` with passing tests |
| S3 Ladder core | W3 rungs R0 to R2, W4 splits and metrics | Opus |  | Checkpoint results table |
| S4 Deep rungs | W3 rungs R3 to R5 on CPU | Sonnet | S5 | Updated results table |
| S5 Edge | W5 C++ port, parity tests, aarch64 cross-build | Sonnet | S4 | `edge/` library and x86 benchmark |
| S6 Write-up | W6 memo, report and deck drafts | Opus |  | Memo, report and slide outline |

Start each one with a one-line brief, for example `Execute W1 in docs/PLAN.md; exit criterion: atlas figures and noise register committed; open a PR`.

### Disk budget (30 GB)

| Item | What to pull | Cap |
| --- | --- | --- |
| Bath archive | The full 6 GB zip; delete it after extraction | 12 GB at peak, 6 GB after |
| UATD | Check the size first; if large, all 720 kHz frames plus a 1200 kHz subset | 6 GB |
| NOAA pings | 2 or 3 short shallow-water cruise segments | 4 GB |
| MBARI noise | 16 kHz daily files plus a few 256 kHz hours | 3 GB |
| Packages, models, MLflow runs |  | 4 GB |

When two large sets do not fit together, run their sessions separately and commit only derived features (Parquet) and figures.

### Rules that save usage

- One workstream per session, with a written exit criterion, ending in a PR.
- Plan locally, execute in the cloud: settle each workstream's approach in a short local or plan-mode session first.
- Keep each training run under the 30-minute background limit: checkpoint every epoch and resume.
- On CPU, cache frozen ResNet-50 features once, fine-tune MobileNetV3-Small fully, and subsample negatives to about 4,000 per fold.
- Commit small results (metrics CSV, PNG figures) as you go, so a reclaimed VM loses nothing.
- Run sessions in parallel only where the map says so; parallel sessions use limits proportionally.
- Use `/compact` at each milestone, and never print raw data dumps into the conversation.
- If usage runs short, apply the cut order from the Risks section.

### What stays local

- **ARM timing:** the cloud VM is x86\_64, so run the edge benchmark on your Raspberry Pi or an ARM machine. The cloud session only cross-compiles and checks parity.
- **AWS credentials:** not needed, since all buckets allow anonymous reads; keep keys out of the cloud environment.
- **Deck polish and rehearsal:** final slides and practice aloud happen on your side.

## Repository and tooling

One public repo, one command per stage, and every figure in the report regenerates from a tagged commit.

```text
echofind/
├── README.md                 # 60-second pitch, results table, how to reproduce
├── Makefile                  # make data | atlas | dsp | train | eval | edge | report
├── pyproject.toml            # uv-managed, pinned
├── dvc.yaml                  # pipeline stages and data versions
├── configs/                  # Hydra configs: data, features, model, eval
├── data/                     # DVC-tracked; raw/ is read-only
│   ├── raw/                  # NOAA .raw, MBARI .wav, UATD, Bath
│   ├── interim/
│   └── processed/
├── src/echofind/
│   ├── io/                   # readers: echopype, UATD BMP + XML, Bath snippets
│   ├── sim/                  # echo simulator
│   ├── dsp/                  # IQ, matched filter, TVG, CFAR, bottom tracking
│   ├── features/             # physics features
│   ├── models/               # ladder rungs R0 to R5
│   └── eval/                 # splits, FROC, bootstrap, cost model, OOD score
├── edge/                     # C++17 library, pybind11 bindings, benchmarks
├── tests/                    # checks against closed-form results; golden vectors
├── notebooks/                # exploration only; nothing imports from here
├── reports/                  # atlas, design notes, model and data cards, memo, decisions/
└── .github/workflows/ci.yml  # lint, tests, 2-minute smoke run
```

| Layer | Choice | Why |
| --- | --- | --- |
| Environment | Python 3.11+, uv | Fast, locked installs |
| Sonar I/O | echopype (EK60 and EK80 to xarray), pyEcholab as a cross-check | The standard readers for NOAA echosounder files |
| DSP | NumPy, SciPy, xarray | Standard, easy to test |
| Classical ML | scikit-learn, LightGBM, SHAP | Rungs R1 and R2, feature importance |
| Deep learning | PyTorch, timm | Rungs R3 and R4 |
| Tracking | MLflow (local), DVC | Runs, metrics, model registry, data versions |
| Config | Hydra | One config per experiment, logged with the run |
| Edge | C++17, CMake, pocketfft or KissFFT, pybind11, ONNX Runtime | Small, portable, testable from Python |
| Quality | pytest, ruff, pre-commit, GitHub Actions | Tests on every push |
| Optional propagation | arlpy with BELLHOP | Ray-traced shallow-water check on the simple simulator |

Conventions:

- Raw data is read-only; every derived file comes from a DVC stage.
- Seeds are fixed, and each MLflow run logs its config, git commit and data hash.
- Notebooks explore; anything reused moves into `src/` with a test.
- Each test that checks math names the equation it checks.
- Each non-obvious choice, such as OS-CFAR over CA-CFAR, gets a one-page decision record.
- Model and data cards are updated whenever a model is promoted.

## Success criteria

The project is done when the nine artifacts below exist and the cross-site numbers are reported honestly, hit or miss. Must targets prove competence; stretch targets are interview talking points, not promises.

| Measure | Must | Stretch |
| --- | --- | --- |
| Cross-site F1 on Bath, same protocol as Nga et al. | Reported with 95% CI; the in-site vs cross-site gap explained | Above 0.47, their unseen-data average |
| Recall at the operating point on held-out sites | Reported per site, with CI | 0.90 or better with multi-look fusion |
| False alarms per scan (one survey line or 360-degree sweep) at that point | Reported | 1 or fewer |
| Calibration error (ECE) on held-out sites | Reported | 0.05 or lower |
| DSP checks against theory | All pass in CI |  |
| C++ vs Python parity | Max absolute error 1e-5 or less in float32 | Q15 version with SQNR of 60 dB or more |
| Latency per ping on a Raspberry Pi | p95 under 68 ms, the 50 m back-to-back budget | Under 10 ms |
| Recall lost to int8 | 1 point or less |  |
| Reproducibility | A clean clone rebuilds every figure with one command | CI smoke run under 2 minutes |

Definition of done:

- [ ] Signal atlas and noise-and-variability register (W1)
- [ ] `echofind.dsp` package, with all checks passing in CI (W2)
- [ ] Ladder results table with cross-site intervals, plus a failure gallery (W3)
- [ ] Generalization report and model card (W4)
- [ ] Edge benchmark table and Pareto plot (W5)
- [ ] Field-collection memo with a metadata schema (W6)
- [ ] 2-page technical report (PDF) and a 10-slide deck
- [ ] README with a 60-second pitch, the results table and reproduce steps
- [ ] Pitch and three STAR stories rehearsed aloud and recorded once

## Risks and fallbacks

The biggest risk is scope, not data. The cut order is fixed in advance: drop R5, the Q15 port, arlpy and the external tests before touching W1, W2, rungs R0 to R2, or W4.

| Risk | Early sign | Mitigation or fallback |
| --- | --- | --- |
| Scope creep | Week-2 checkpoint at risk on day 10 | Apply the cut order above; ship the core and list the rest as next steps |
| Bath labels sparse or archive layout unclear | Cannot rebuild the paper's snippet counts by day 3 | Follow the paper's snippet protocol; hand-label a few hundred boxes under a written protocol; make UATD primary if needed |
| Few positives (166 target snippets) give wide intervals | Bootstrap intervals wider than the gaps you want to claim | Report intervals and effect sizes; nested cross-validation; no tuning on held-out sites |
| Leakage through near-duplicate frames | Held-out scores close to in-site scores | Group splits by session; deduplicate by perceptual hash; check nearest-neighbour distances across splits |
| Simulator too clean | Simulated clutter statistics differ from the W1 fits | Calibrate noise and clutter to W1; use simulation for DSP tests and ablations, never as the headline number |
| Mannequin is not a body | Interviewer asks about real victims | State it as a limitation and turn it into the W6 field question; claim proxy performance only |
| NOAA files large and slow | One cruise takes over an hour to pull | Pick 2 or 3 small shallow-water cruises; stream from S3; subsample pings |
| No Raspberry Pi | No board by week 3 | Benchmark on an AWS Graviton (ARM) instance or one pinned laptop core |
| Optional datasets lack a license | No license file in the repo | Evaluate only; never redistribute or show victim images |
| Negative results, e.g. the CNN does not beat features | R3 or R4 loses to R2 cross-site | Present it as the finding, with the diagnosis; the posting asks for exactly this judgment |

## Interview preparation

Prepare three things: the EchoFind story at three lengths, fluent answers on sonar and detection basics, and informed questions about AquaEye's device and data.

### Know the product

- VodaSafe is based in Vancouver; founder and CEO Carlyn Loncaric. A $1.4M seed round was led by Vanedge Capital with BDC in July 2020 ([Techcouver](https://techcouver.com/2020/07/30/vodasafe-vanedge-aquaeye/)).
- AquaEye is a handheld scanning sonar with AI; a 360-degree scan takes 3 to 5 minutes and covers about two acres ([Techcouver](https://techcouver.com/2020/07/30/vodasafe-vanedge-aquaeye/)).
- It is dual-frequency with selectable 10, 20 and 50 m ranges; 1.4 kg, IP68, buoyant, rated -10 to +50 °C. An "EchoMap" view shows raw sonar beside AI markers with direction and approximate depth ([Dive Right In Scuba](https://www.diverightinscuba.com/vodasafe-aquaeye-pro-sonar-scanner.html), [Echolotzentrum](https://www.echolotzentrum.de/en/?p=119303)).
- The sales director describes the AI as listening to return echoes and deciding person or not person ([KSBY](https://ksby.com/news/local-news/new-ai-sonar-technology-is-coming-to-slo-county-heres-how-it-can-help-first-responders)).
- A check worth saying aloud: a 360-degree sweep at 50 m covers π × 50² ≈ 7,850 m², which matches the published 8,000 m² per scan.

Talk about victims with care. Rescue teams also run recoveries, and some interviewers will have been on them.

### Three pitches

- **30 seconds:** "I built EchoFind, an open pipeline that finds a submerged mannequin in public sonar data, from raw echoes to an int8 model on a Raspberry Pi. A published CNN on the same data fell from 97% to 47% F1 on unseen sites. My focus was why, and what closes that gap." Then add your headline number.
- **2 minutes:** mission metric, the data and its traps, what the signal atlas showed, the ladder result, the edge numbers, and what you would collect next.
- **20-minute deep dive, 10 slides:** mission metric; data and traps; signal atlas; front end; ladder; cross-site results and failure gallery; operating point from mission cost; edge numbers; what AquaEye should collect next; your first 90 days.

### Map to the five responsibility areas

| Posting area | Point to in EchoFind | First 90 days at AquaEye |
| --- | --- | --- |
| Signal processing and applied data science | Signal atlas, front end, clutter fits | Build the same atlas on AquaEye field data, by site and frequency |
| ML and algorithm development | The ladder and its keep rule; cross-site results | Re-baseline the current detector on a session-grouped test set, with a simple-feature rung as a check |
| Data strategy and field experimentation | Coverage audit, factorial design, labeling protocol | Audit existing data coverage; propose the first blocked field campaign |
| Product development and technical direction | Mission-cost operating point; beamwidth trade note | Agree mission metrics with product; write a data vs DSP vs ML vs hardware decision memo |
| Technical leadership and team development | Reproducible repo, decision records, model cards | Start an experiment register and decision records; run a monthly technical review |

### Technical question bank

| Topic | Likely question | Answer outline |
| --- | --- | --- |
| Sonar equation | Walk me through detecting a body at 50 m | Noise-limited: SL - 2TL + TS - (NL - DI) ≥ DT. Reverberation-limited: SL - 2TL + TS - RL ≥ DT. One-way TL ≈ 20 log r + αr |
| Fresh vs salt water | Does a lake change the design? | Seawater adds boric-acid and magnesium-sulfate relaxation absorption; fresh water does not, so absorption is far lower at the same frequency. Sound speed also shifts with temperature |
| Pulse choice | CW or LFM for a handheld search sonar? | Equal energy gives equal matched-filter SNR. LFM keeps resolution at c/(2B) with a long pulse, and wider bandwidth shrinks the reverberation cell |
| Matched filter | Why is it optimal, and when is it not? | It maximizes output SNR in white Gaussian noise. Colored noise needs prewhitening; multipath, target spread and Doppler cause mismatch |
| CFAR | Which CFAR for a body lying on the bottom? | CA fails at clutter edges and beside other targets; GO handles edges; OS handles multiple targets. CA multiplier: α = N(Pfa^(-1/N) - 1) for N reference cells |
| Clutter statistics | Why do false alarms jump near the bottom? | A few dominant scatterers push the envelope from Rayleigh toward K or Weibull; heavy tails need an adapted threshold |
| Spectral analysis | How would you look at a noisy recording? | Welch PSD with a stated window and overlap; spectrograms trade time for frequency resolution; watch leakage; skip mel scales, which model hearing, not physics |
| Features vs deep learning | Why not just train a CNN? | Few positives, a cross-site gap and an edge budget. Climb the ladder; keep a rung only if it wins cross-site |
| Evaluation | How do you know a gain will hold in the field? | Group splits by site and session, pre-registered metrics, cluster bootstrap, external sets, then a field check |
| Metrics | What would you report to a fire chief? | Recall at a fixed false-alarm rate per scan, translated into expected minutes to find |
| Imbalance | 166 positives vs 13,054 negatives: what changes? | PR curves over ROC alone, class weights or focal loss, hard-negative mining, a threshold from the cost curve |
| Domain shift | A new lake with weed beds: what happens? | An OOD score flags it; normalize per site; request a rescan; then collect, label and monitor drift |
| Quantization | What breaks at int8? | Activation ranges and outliers; per-channel scales; a calibration set that covers field conditions; check recall at the operating point |
| Embedded budget | How fast must detection run? | About 68 ms per ping at 50 m if pings run back to back; memory for one ping plus the reference window; quote p95, not the mean |
| Data strategy | One week of field time: what do you collect? | The top gaps from the audit, a blocked factorial design, ground truth, a labeling protocol and a safety plan |

### Stories to rehearse (STAR)

- **Why it failed (curiosity):** the cross-site gap or leakage investigation: what you suspected, the test you ran, what changed.
- **When simple won (judgment):** a feature model matching or beating the CNN cross-site, and how you made the call.
- **A hardware trade (collaboration):** the beamwidth or latency study, told as you would tell AquaEye's hardware lead.
- **From your career:** mentoring someone, a practice others adopted, field work in hard conditions, a disagreement settled with data.

### Questions to ask AquaEye

- What reaches the classifier today: raw IQ, envelope or features? At which frequencies and beamwidths?
- How is ground truth set in trials and customer use, and do confirmed or rejected detections flow back?
- Which failures cost most today: misses in which conditions, false alarms from what?
- How are training and test data split, and how is generalization to new sites checked?
- What compute runs on the device, and how do model updates reach units in the field?
- How do hardware, firmware and data science make trade-offs together?
- What would success in this role look like at 6 and 12 months?

## Appendix: formulas to whiteboard

These are the relations the question bank leans on; each one is also checked by a unit test in `tests/`.

Active sonar equation, noise-limited and reverberation-limited (signal excess SE above zero means detection):

```latex
\mathrm{SE} = \mathrm{SL} - 2\,\mathrm{TL} + \mathrm{TS} - (\mathrm{NL} - \mathrm{DI}) - \mathrm{DT}, \qquad \mathrm{SE} = \mathrm{SL} - 2\,\mathrm{TL} + \mathrm{TS} - \mathrm{RL} - \mathrm{DT}
```

One-way transmission loss with spherical spreading and absorption α in dB/m:

```latex
\mathrm{TL}(r) = 20\log_{10} r + \alpha r
```

Time-varying gain: point targets need 40 log r + 2αr; volume backscatter needs 20 log r + 2αr, and seabed backscatter at grazing angles falls closer to 30 log r.

Range from two-way travel time, with c ≈ 1,480 m/s (it shifts with temperature and salinity):

```latex
r = \frac{c\,t}{2}
```

Range resolution for a CW pulse of length T, and for an LFM pulse of bandwidth B:

```latex
\Delta r_{\mathrm{CW}} = \frac{c\,T}{2}, \qquad \Delta r_{\mathrm{LFM}} = \frac{c}{2B}
```

Matched filter and its peak SNR, which depends only on pulse energy E over noise density N0; pulse compression gains about 10 log10(B·T) dB over the raw echo:

```latex
h(t) = s^{*}(-t), \qquad \mathrm{SNR}_{\max} = \frac{2E}{N_0}
```

CA-CFAR threshold for a square-law detector in exponential noise with N reference cells:

```latex
T = \alpha \cdot \frac{1}{N}\sum_{i=1}^{N} x_i, \qquad \alpha = N\left(P_{fa}^{-1/N} - 1\right)
```

Half-power beamwidth of a uniform line aperture of length D, in radians (a circular piston gives about 1.03 λ/D):

```latex
\theta_{-3\,\mathrm{dB}} \approx 0.886\,\frac{\lambda}{D}, \qquad \lambda = \frac{c}{f}
```

Near field of a circular piston of radius a ends near the Rayleigh distance:

```latex
R_{0} = \frac{\pi a^{2}}{\lambda}
```

## Sources

Pages opened for this plan; the job posting is the local file Job Posting.docx.

- [UATD dataset paper, Scientific Data (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9715547/)
- [Bath side-scan dataset of submerged body-like objects (2024)](https://researchdata.bath.ac.uk/1467/)
- [Nga et al., Remote Sensing 16(21):4036 (2024)](https://researchportal.bath.ac.uk/en/publications/automated-recognition-of-submerged-body-like-objects-in-sonar-ima/)
- [NOAA Water Column Sonar Data Archive on AWS](https://registry.opendata.aws/ncei-wcsd-archive/)
- [MBARI Pacific Ocean Sound Recordings on AWS](https://registry.opendata.aws/pacific-sound/), with [tutorials](https://docs.mbari.org/pacific-sound/)
- [SCTD repository](https://github.com/TTFF322/SCTD-)
- [FLS Detection Dataset repository](https://github.com/XingYZhu/Forward-looking-Sonar-Detection-Dataset)
- [Sonar image dataset survey (arXiv 2510.03353)](https://arxiv.org/html/2510.03353v1), for further datasets
- [Techcouver: VodaSafe seed round, July 30, 2020](https://techcouver.com/2020/07/30/vodasafe-vanedge-aquaeye/)
- [Dive Right In Scuba: AquaEye Pro product page](https://www.diverightinscuba.com/vodasafe-aquaeye-pro-sonar-scanner.html)
- [Echolotzentrum: AquaEye product page](https://www.echolotzentrum.de/en/?p=119303)
- [KSBY: AquaEye with SLO County first responders](https://ksby.com/news/local-news/new-ai-sonar-technology-is-coming-to-slo-county-heres-how-it-can-help-first-responders)
