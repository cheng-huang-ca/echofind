# Cloud environment setup

Settings for the `echofind` environment used by Claude Code cloud sessions. The reasons are in
[PLAN.md](PLAN.md), section "Running it on Claude Code cloud".

## Network access

Choose **Custom**, check **Also include default list of common package managers**, and add:

```text
figshare.com
*.figshare.com
researchdata.bath.ac.uk
download.pytorch.org
```

The default list already covers `*.amazonaws.com` (NOAA and MBARI buckets), PyPI and GitHub.

## Environment variables

```text
BASH_DEFAULT_TIMEOUT_MS=600000
BASH_MAX_TIMEOUT_MS=1800000
```

## Setup script

Paste [`scripts/cloud-setup.sh`](../scripts/cloud-setup.sh). It installs packages only.

## VM limits to plan around

About 4 vCPUs, 16 GB RAM, 30 GB disk, x86_64, no GPU. Idle VMs pause and can be reclaimed,
which loses downloaded data, so `make data` is idempotent and safe to rerun.

## Data per session

| Session | Fetch | Approx. size |
| --- | --- | --- |
| S1 Atlas | `make data` (all) | about 15 GB after extraction |
| S2 Physics and DSP | `make data ONLY="noaa mbari"` | 3 to 4 GB |
| S3, S4 Ladder | `make data ONLY="bath uatd"` | about 11 GB |
| S5 Edge | `make data ONLY=noaa` (golden vectors) | 2 to 3 GB |
| S0, S6 | none | |
