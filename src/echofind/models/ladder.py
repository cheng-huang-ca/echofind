"""Model ladder rungs R0 to R2 for body-vs-not-body on CFAR candidates (docs/PLAN.md, W3).

Every rung maps a candidate feature table to a score; higher means more body-like.

- R0 rule: CFAR detection plus a physical size gate, no learning. Score = CFAR SNR (dB) when the
  candidate's longest extent fits a body (0.3 to 3.0 m, set from a 1.7 m adult in any pose,
  before seeing any results), else -inf.
- R1 logistic regression on 10 physics features (features.fls.R1_FEATURES), standardised.
- R2 gradient-boosted trees (LightGBM) on 38 features (features.fls.R2_FEATURES).

Learned rungs train on all positive candidates and at most `max_neg` negatives, with the kept
negatives up-weighted by N_neg / max_neg so predicted probabilities stay on the original prior
(needed for the calibration error). Hyperparameters are fixed in advance, not tuned on test data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from echofind.features.fls import R1_FEATURES, R2_FEATURES

SIZE_GATE_M = (0.3, 3.0)


def raw_view(cols: list[str]) -> list[str]:
    """Same features, with level features read from raw power instead of normalised power."""
    from echofind.features.fls import LEVEL_FEATURES
    return [f"{c}_raw" if c in LEVEL_FEATURES else c for c in cols]


def _subsample(X: pd.DataFrame, y: np.ndarray, max_neg: int, seed: int):
    neg = np.flatnonzero(y == 0)
    w = np.ones(len(y))
    if len(neg) > max_neg:
        rng = np.random.default_rng(seed)
        keep_neg = rng.choice(neg, max_neg, replace=False)
        keep = np.sort(np.r_[np.flatnonzero(y == 1), keep_neg])
        w[keep_neg] = len(neg) / max_neg
        return X.iloc[keep], y[keep], w[keep]
    return X, y, w


@dataclass
class Rung:
    name: str
    features: list[str] = field(default_factory=list)
    max_neg: int = 40000
    seed: int = 0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Rung:
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def prob(self, X: pd.DataFrame) -> np.ndarray | None:
        return None


@dataclass
class R0Rule(Rung):
    name: str = "R0"
    gate: tuple[float, float] = SIZE_GATE_M

    def score(self, X):
        ext = np.maximum(X.range_ext_m.to_numpy(), X.cross_ext_m.to_numpy())
        ok = (ext >= self.gate[0]) & (ext <= self.gate[1])
        return np.where(ok, X.snr_db.to_numpy(), -np.inf)


@dataclass
class R1Logistic(Rung):
    name: str = "R1"
    features: list[str] = field(default_factory=lambda: list(R1_FEATURES))
    C: float = 1.0

    def fit(self, X, y):
        Xs, ys, w = _subsample(X[self.features], y, self.max_neg, self.seed)
        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(C=self.C, max_iter=5000))
        self.model.fit(Xs.to_numpy(), ys, logisticregression__sample_weight=w)
        return self

    def prob(self, X):
        return self.model.predict_proba(X[self.features].to_numpy())[:, 1]

    def score(self, X):
        return self.model.decision_function(X[self.features].to_numpy())


@dataclass
class R2Trees(Rung):
    name: str = "R2"
    features: list[str] = field(default_factory=lambda: list(R2_FEATURES))
    params: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        verbose=-1, n_jobs=4))

    def fit(self, X, y):
        Xs, ys, w = _subsample(X[self.features], y, self.max_neg, self.seed)
        self.model = lgb.LGBMClassifier(random_state=self.seed, **self.params)
        self.model.fit(Xs, ys, sample_weight=w)
        return self

    def prob(self, X):
        return self.model.predict_proba(X[self.features])[:, 1]

    def score(self, X):
        return self.model.predict(X[self.features], raw_score=True)


def make(name: str, **kw) -> Rung:
    return {"R0": R0Rule, "R1": R1Logistic, "R2": R2Trees}[name](**kw)
