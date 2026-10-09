# 601 · Deep rungs on CPU: same candidates and splits as W3, early stopping on held-out sessions

**Status:** accepted for S4 · **Owner:** S4

## Context
R3 and R4 must be compared with R2 on exactly the W3 protocol: the same CFAR candidates, the
same grouped folds, the 0.05 false-alarm-per-frame point and the paired keep rule. They also
have to fit a CPU-only budget, with each training run under 30 minutes and checkpoints every
epoch (CLAUDE.md).

## Decisions
- **Inputs** (`scripts/s4_snippets.py`): cut from the same range-normalised grid R2's features
  use, so the comparison is about the model, not the preprocessing.
  - A 64 × 32 snippet: 2.56 m of range, with the peak at row 24 so the shadow behind the
    target is in view.
  - A 128-cell range profile through the peak beam: the 1-D echo a single-beam handheld has.
- **Rungs:**
  - R3a: 1-D CNN on the profile, 26k parameters.
  - R3b: 2-D CNN on the snippet, 113k parameters.
  - R4a: frozen ImageNet ResNet-50 with features cached once and a logistic head.
  - R4b: ImageNet MobileNetV3-Small, fine-tuned.
  - The pretrained models see the snippet resized to 128 × 64 on 3 identical channels.
- **Training set per fold:** every body candidate plus at most 40,000 negatives (12,000 for
  R4b, whose epochs are about 10× slower), seeded. R2 used the same 40,000 cap, so R3 and R4a
  see the same amount of data. A positive weight of n_neg/n_pos balances the loss.
- **Validation:** 20% of the training side's sessions, chosen with a fixed seed so that they
  hold at least 50 body candidates. They give early stopping (best average precision,
  patience 3–4) and the transferred threshold. Test frames are never used for training,
  stopping or thresholds.
- **Augmentation:** beam-axis flip (the body seen from the other side) and shifts of up to
  2 cells. No range flip, because shadow direction is physical.
- **No R5 (multi-look fusion) on UATD.** UATD frames are shuffled and carry no timestamp or
  sonar pose, so a "scan" of consecutive looks cannot be rebuilt. Grouping random frames from
  one session would fake independent looks and overstate fusion gains. H7 needs sequential
  data: Bath survey legs once labelled, or the W6 trial (3 passes per placement).

## Consequences
Every deep result is directly comparable with W3's R1 and R2 numbers, paired candidate by
candidate (`reports/s4/deep_paired.csv`).
