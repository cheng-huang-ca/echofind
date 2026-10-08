"""Train/test splits by group, never by random frame (except the random split kept to show H4).

Each split is a list of folds; a fold is (name, train frame mask, test frame mask) over a frame
table. Candidates inherit their frame's fold. Splits are deterministic for a fixed seed.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

Fold = tuple[str, np.ndarray, np.ndarray]


def random_frames(frames: pd.DataFrame, k: int = 5, seed: int = 0) -> Iterator[Fold]:
    """K-fold over individual frames. Leaks near-duplicate frames; kept only for H4."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, len(frames))
    for i in range(k):
        yield f"random{i}", fold != i, fold == i


def leave_one_group_out(frames: pd.DataFrame, col: str,
                        only: list[str] | None = None) -> Iterator[Fold]:
    """Hold out each value of `col` in turn (restricted to `only` if given); train on the rest."""
    g = frames[col].to_numpy()
    for v in only if only is not None else sorted(pd.unique(g)):
        yield str(v), g != v, g == v


def transfer(frames: pd.DataFrame, col: str, train_value, test_value) -> Fold:
    """Train on rows where col == train_value, test on col == test_value (domain shift)."""
    g = frames[col].to_numpy()
    return f"{train_value}->{test_value}", g == train_value, g == test_value


def check_disjoint(frames: pd.DataFrame, fold: Fold, col: str) -> None:
    """Raise if any group in `col` appears on both sides of the fold."""
    _, tr, te = fold
    both = set(frames.loc[tr, col]) & set(frames.loc[te, col])
    if both:
        raise ValueError(f"groups on both sides of {fold[0]}: {sorted(both)[:5]}")
