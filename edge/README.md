# edge/

The C++17 port of the EchoFind front end (`echofind.dsp`) and the ladder classifiers compiled
to plain C, with parity tests and a per-ping benchmark (W5). Results:
[reports/w5_edge.md](../reports/w5_edge.md).

```
include/echofind/edge.hpp   header-only DSP: zero-phase SOS, demodulator, FFT matched filter,
                            power, TVG, CA/GO/SO/OS-CFAR, 1-D candidates (float or double)
include/echofind/capi.h     C ABI (used by ctypes parity tests and any C caller)
src/capi.cpp                C ABI implementation
generated/models.c          R0, R1, R2 (25/50/100/300 trees) as C, double and float versions
generated/models.json       feature order (R2_FEATURES) and model sizes
bindings/pybind.cpp         optional pybind11 module _ef_edge (-DEF_PYBIND=ON)
bench/ef_bench.cpp          ef_bench CLI: one pinned core, p50/p95/max per stage, peak memory
tests/golden_test.cpp       ef_golden: C++-only parity against tests/golden/*.bin
tests/golden/               golden vectors written by scripts/w5_export.py
```

## Build and test

```bash
cmake -S edge -B edge/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build edge/build
ctest --test-dir edge/build --output-on-failure      # golden vectors, double and float
uv run pytest tests/test_edge_parity.py              # live parity with echofind.dsp
```

Needs a C++17 compiler and CMake 3.16 or later. On Windows, WinLibs GCC (MinGW-w64, UCRT) was
used, and the binaries are linked statically. The pybind11 module is built in CI on Linux; on
Windows, Python reaches the library through the C ABI, because MinGW extensions do not load in
an MSVC-built CPython.

## Regenerate inputs and models

```bash
uv run python scripts/w5_export.py            # golden vectors, bench pings, models.c, accuracy
uv run python scripts/w5_bench.py             # benchmark, float32 cost, figures
```

## Run on a Raspberry Pi (ARM timing, still to do)

```bash
sudo apt install build-essential cmake
cmake -S edge -B edge/build -DCMAKE_BUILD_TYPE=Release
cmake --build edge/build -j4
ctest --test-dir edge/build
edge/build/ef_bench data/processed/edge/bench_200k.txt --pings 500 --core 2
edge/build/ef_bench data/processed/edge/bench_80k.txt --pings 500 --core 2
```

Copy `data/processed/edge/` from a machine that ran `w5_export.py`, or run it on the board.
Record `lscpu`, the clock governor (`cpufreq-info`) and the board's temperature; a throttling
Pi 4 is slower. The budget is 68 ms per ping at the 50 m setting.
