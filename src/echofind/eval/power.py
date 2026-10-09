"""Sample size for comparing detection recall between two field conditions (W6).

In a field trial the unit that is randomised is a *placement* (one body proxy put down at one
spot, in one condition), and each placement is seen in m looks (pings or passes). Looks at one
placement are correlated, so they are worth fewer than m independent trials:

- Two-proportion sample size (Fleiss, Levin and Paik 2003, eq. 4.14, no continuity correction),
  looks per arm for recall p1 against p2 at two-sided level alpha and the given power:
      n = [z_{1-a/2} sqrt(2 pbar qbar) + z_{1-b} sqrt(p1 q1 + p2 q2)]^2 / (p1 - p2)^2
- Design effect for clusters of m looks with intraclass correlation rho (Kish 1965):
      DE = 1 + (m - 1) rho
- Placements per arm = ceil(n * DE / m). With rho near 1, extra looks add nothing and the
  answer tends to the independent-placement n.

rho is estimated from data with the one-way ANOVA estimator for binary outcomes
(Donner and Koval 1980):
      rho = (MSB - MSW) / (MSB + (n0 - 1) MSW),  n0 = (N - sum n_i^2 / N) / (k - 1).
simulate_power() checks the formula by Monte Carlo with beta-binomial placements and a
cluster-level test, which is also how the trial should be analysed.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def n_two_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Independent looks per arm to tell p1 from p2 (two-sided)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return num**2 / (p1 - p2) ** 2


def design_effect(m: float, rho: float) -> float:
    """DE = 1 + (m - 1) rho."""
    return 1 + (m - 1) * rho


def placements_per_arm(p1: float, p2: float, looks: int, rho: float, alpha: float = 0.05,
                       power: float = 0.8) -> int:
    """ceil(n * DE / m): placements per condition, each seen in `looks` correlated looks."""
    n = n_two_proportions(p1, p2, alpha, power)
    return math.ceil(n * design_effect(looks, rho) / looks)


def icc_binary(y: np.ndarray, cluster: np.ndarray) -> float:
    """One-way ANOVA estimator of the intraclass correlation of a 0/1 outcome."""
    y = np.asarray(y, dtype=float)
    c = np.asarray(cluster)
    keys, inv = np.unique(c, return_inverse=True)
    k, N = len(keys), len(y)
    n_i = np.bincount(inv)
    mean_i = np.bincount(inv, weights=y) / n_i
    grand = y.mean()
    ssb = np.sum(n_i * (mean_i - grand) ** 2)
    ssw = np.sum((y - mean_i[inv]) ** 2)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - np.sum(n_i**2) / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def simulate_power(p1: float, p2: float, placements: int, looks: int, rho: float,
                   alpha: float = 0.05, n_sims: int = 2000, seed: int = 0) -> float:
    """Monte Carlo power: placement recall ~ Beta(mean p, ICC rho), looks ~ Bernoulli; test the
    difference of mean placement recall with a Welch t-test (one value per placement)."""
    rng = np.random.default_rng(seed)

    def arm(p):
        if rho <= 0:
            q = np.full(placements, p)
        else:
            a, b = p * (1 / rho - 1), (1 - p) * (1 / rho - 1)
            q = rng.beta(a, b, placements)
        return rng.binomial(looks, q) / looks

    hits = 0
    for _ in range(n_sims):
        x, y = arm(p1), arm(p2)
        if np.var(x) == 0 and np.var(y) == 0:
            hits += x.mean() != y.mean()
            continue
        hits += stats.ttest_ind(x, y, equal_var=False).pvalue < alpha
    return hits / n_sims
