# 403 · Benchmark on one pinned x86 core; ARM numbers wait for a board

**Status:** accepted for W5; ARM measurement pending · **Owner:** W5

## Context
The plan asks for p50 and p95 latency per ping on a Raspberry Pi 4 or 5, "with no board, use
one pinned laptop core and record its clock". No ARM board or aarch64 cross toolchain was
available in this session.

## Decision
- `ef_bench` pins itself to one core (core 2) at high thread priority. It runs 500 pings of a
  simulated 50 m ping through the whole chain, at 80 kHz (400 kHz passband sampling, 27,300
  samples) and at 200 kHz (960 kHz, 65,520 samples). It reports p50, p95 and max per stage,
  plus peak resident memory.
- Machine: Intel Core i7-8850H (2.6 GHz base, 6 cores), Windows 11, GCC 16.2 at -O2, no
  -march=native (`reports/w5/machine.json`).
- The ARM run is a single command on the board, with the same config files (edge/README.md).

## How to read the x86 numbers for ARM
The 200 kHz chain takes 16.4 ms at p95 in float32, so a core up to 4.1× slower still meets
68 ms. At 80 kHz (6.1 ms) the margin is 11×. Whether a Pi 4 core is within 4.1× of this one
for this code is exactly what the board run must show; it is not assumed here. If it is not,
the first fix is to decimate before band-pass filtering (a polyphase front end), since
zero-phase filtering at the full passband rate takes half the time (decision 401, item 2).

## Also noted
Windows is not a real-time OS. The max latency (53 ms at 200 kHz, against a 16 ms p95)
comes from scheduler preemption, not the code. A device build on a real-time or isolated core
should report p99.9 as well.
