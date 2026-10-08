# EchoFind: instructions for Claude sessions

Portfolio project for an Applied Scientist interview (handheld sonar for finding drowning victims).
The plan is `docs/PLAN.md`; read its section for your workstream before starting.

## Session protocol

- One workstream per session. The prompt names it (for example "Execute W1") and an exit criterion.
- Work on a branch `ws/<id>-<slug>` (for example `ws/w1-atlas`) and end with a PR to `main`.
- Update the Status table in `README.md` in the same PR.
- Commit small results as you go (metrics CSV, PNG figures under `reports/`), so a reclaimed VM loses nothing.
- Record each non-obvious choice as a one-page decision record in `reports/decisions/NNN-title.md`.

## Commands

- `make setup`, `make test`, `make lint`
- `make data-dry`, then `make data ONLY="<sources>"` (sources: bath, uatd, noaa, mbari).
  Fetch only what the workstream needs; run long fetches in the background.

## Conventions

- Raw data lives in `data/raw/` and is never committed or modified. Derived data goes to `data/interim/` or `data/processed/`.
- Library code goes in `src/echofind/`; notebooks explore only, and nothing imports from them.
- Every test that checks math names the equation it checks in its docstring.
- Seeds are fixed. Each MLflow run logs its config, git commit and data manifest hash.
- Splits are grouped by site and session, never random frames (near-duplicate frames leak).
- Report recall at a fixed false-alarm rate with cluster-bootstrap intervals; never accuracy alone.

## Cloud VM constraints

About 4 vCPUs, 16 GB RAM, 30 GB disk, x86_64, no GPU.
- Keep each training run under 30 minutes; checkpoint every epoch and resume.
- On CPU: cache frozen backbone features once; subsample negatives (about 4,000 per fold).
- Never print raw data or large arrays into the conversation.
- ARM timing for W5 runs on real ARM hardware, not in the cloud.

## Data facts to remember

- Bath archive: most target/background labels were lost (see its README). Only some directories
  have a `target/` subdirectory. Images are uncorrected and need rescaling (paper Figure 3).
  Only the 990 kHz data was used in the paper; 450 kHz is included too. The `8_apr_2022` directory
  was not used in the paper. W3 starts with a written labelling protocol.
- UATD: BMP frames with XML boxes; 720 kHz and 1200 kHz; lake (Maoming) and sea (Dalian) sites.
- NOAA HB2305: EK80 files; `ComplexSamples-*` are broadband complex echoes, `D*-T*.raw` are CW.
- SCTD and the FLS victim set have no stated license: evaluate only, never redistribute, never show victim images.
