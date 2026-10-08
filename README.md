# EchoFind

Submerged-person detection in public sonar data, from raw echoes to a detector on ARM.
A portfolio project: the full plan is in [docs/PLAN.md](docs/PLAN.md).

**Question:** can a handheld-class pipeline separate a body from rocks, tyres, debris and the
bottom at known miss and false-alarm rates, and does that hold at a new site, frequency or sonar?

**Benchmark to beat:** on the Bath side-scan data, published CNNs averaged 97% F1 in-distribution
but 47% on unseen data (Nga et al., Remote Sensing 16(21):4036, 2024).

## Status

| Workstream | State |
| --- | --- |
| S0 Scaffold: repo, CI, data fetcher | done |
| W1 Signal atlas | NOAA and MBARI done ([atlas](reports/signal_atlas.md), [noise register](reports/noise_register.csv)); UATD and Bath pending network access |
| W2 Physics and DSP front end | not started |
| W3 Model ladder | not started |
| W4 Field-honest evaluation | not started |
| W5 Edge deployment | not started |
| W6 Data strategy memo | not started |

## Reproduce

```bash
make setup          # uv sync with dev tools
make data-dry       # list files and sizes
make data           # fetch everything (about 15 GB after extraction)
make test
```

## Data

| Source | License | Use |
| --- | --- | --- |
| [Bath side-scan, mannequin](https://researchdata.bath.ac.uk/1467/) | CC BY 4.0 | Body vs background, leave-one-site-out |
| [UATD forward-looking sonar](https://doi.org/10.6084/m9.figshare.21331143.v3) | CC BY 4.0 | Confuser classes, per-beam range profiles |
| [NOAA WCSD, cruise HB2305 EK80](https://registry.opendata.aws/ncei-wcsd-archive/) | NOAA disclaimer | Raw pings for DSP |
| [MBARI Pacific Sound 256 kHz](https://registry.opendata.aws/pacific-sound/) | CC BY 4.0 | Real ambient noise |

Data is downloaded, never committed. See `configs/data.yaml` for exact files and checksums.
