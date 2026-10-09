"""Out-of-distribution score that asks for a rescan instead of staying silent (W4 safety).

Candidates are described by the same physics features the detector uses. A Gaussian fitted
to the training candidates (Ledoit-Wolf shrinkage covariance on standardised features) gives
each new candidate a squared Mahalanobis distance,

    D^2(x) = (z - mu)^T S^-1 (z - mu),   z = (x - m) / s,

and a frame's score is a high quantile of its candidates' D^2 (one odd object is enough to
matter, but the maximum alone is noisy). A frame whose score exceeds the threshold, set so that
a fixed share (default 5%) of held-out in-distribution frames are flagged, should prompt
"conditions unlike training: rescan or switch view" rather than a silent "no body".

This is deliberately simple: it is cheap on the device, needs no labels, and its behaviour is
easy to explain to an operator. See reports/decisions/302-ood-rescan-score.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass
class MahalanobisOOD:
    features: list[str]
    frame_quantile: float = 0.9

    def fit(self, X: pd.DataFrame) -> MahalanobisOOD:
        Z = X[self.features].to_numpy(dtype=float)
        self.m_ = np.nanmean(Z, axis=0)
        self.s_ = np.nanstd(Z, axis=0) + 1e-9
        Z = np.nan_to_num((Z - self.m_) / self.s_)
        lw = LedoitWolf().fit(Z)
        self.mu_, self.prec_ = lw.location_, lw.precision_
        return self

    def candidate_score(self, X: pd.DataFrame) -> np.ndarray:
        """Squared Mahalanobis distance D^2 of each candidate."""
        Z = np.nan_to_num((X[self.features].to_numpy(dtype=float) - self.m_) / self.s_)
        d = Z - self.mu_
        return np.einsum("ij,jk,ik->i", d, self.prec_, d)

    def frame_score(self, X: pd.DataFrame, frame_id: np.ndarray) -> pd.Series:
        """Per-frame score: the frame_quantile of its candidates' D^2."""
        s = pd.Series(self.candidate_score(X), index=np.asarray(frame_id))
        return s.groupby(level=0).quantile(self.frame_quantile)


def flag_threshold(in_dist_scores: np.ndarray, flag_rate: float = 0.05) -> float:
    """Threshold that flags `flag_rate` of in-distribution scores."""
    return float(np.quantile(np.asarray(in_dist_scores, dtype=float), 1 - flag_rate))
