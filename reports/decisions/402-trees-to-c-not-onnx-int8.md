# 402 · The winning rung ships as generated C, not ONNX int8

**Status:** accepted for W5 · **Owner:** W5

## Context
The plan offers two export paths: ONNX with int8 static quantisation, or a tree model compiled
to plain C. W3 made R2 (LightGBM, 300 trees of 15 leaves) the best rung. No CNN rung exists
yet (S4).

## Decision
`echofind.models.export_c` writes every rung as dependency-free C, in a double version and a
float version, into `edge/generated/models.c`:
- Trees become nested `if` statements reproducing LightGBM's numerical decision, including
  its missing-value routing.
- In the float version each threshold is rounded **down** to the largest float not above it.
  For any float input, `x <= t_float` then holds exactly when `x <= t_double`, so the float
  model takes the same branches as the double model would on the same rounded input.
- R1 becomes one dot product, and R0 a two-comparison rule.

int8 is not used. A tree compares features with thresholds, so quantising features to 8 bits
would move samples across splits for no speed gain: a comparison costs the same in any width.

## Evidence (reports/w5/float32_cost.csv, model_accuracy.csv)
- Parity: the double C code equals LightGBM's `predict(raw_score=True)` to 1e-9 of the score
  range on 300 committed candidates. The float code equals the double code on float32 inputs
  to 1e-5 (`tests/test_edge_parity.py`).
- The only accuracy cost is rounding features to float32. Cross-session recall at 0.05 FAPF is
  0.410 against 0.412 in double, a difference of −0.002 [−0.007, +0.005] (paired cluster
  bootstrap). 0.011% of alarm decisions flip. The plan's limit is 1 point.
- Cost: 0.20 ms per ping for 20 candidates at 300 trees (p95). Code size is about 0.94 MB of
  machine code for all six models in both precisions; R2-300 is about 63% of the leaves.

## Consequences
H8 ("int8 costs under 1 point of recall and runs at least 2× faster than fp32") does not apply
to the tree rung; the float32 cost is reported in its place. H8 returns if S4 produces a CNN.
