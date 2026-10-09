"""Field-trial sample size (W6): formula, design effect, ICC estimator, Monte Carlo check."""

import numpy as np
import pytest

from echofind.eval.power import (
    design_effect,
    icc_binary,
    n_two_proportions,
    placements_per_arm,
    simulate_power,
)


def test_two_proportion_n_matches_textbook_value():
    """Fleiss eq. 4.14 without continuity correction: p1 0.6 vs p2 0.4, alpha 0.05, power 0.8
    gives about 97 per arm."""
    assert n_two_proportions(0.6, 0.4) == pytest.approx(96.9, abs=0.5)


def test_design_effect_limits():
    """DE = 1 + (m - 1) rho: independent looks (rho 0) give 1, identical looks (rho 1) give m."""
    assert design_effect(10, 0.0) == 1 and design_effect(10, 1.0) == 10
    n = n_two_proportions(0.7, 0.5)
    assert placements_per_arm(0.7, 0.5, looks=10, rho=1.0) == pytest.approx(np.ceil(n), abs=1)
    assert placements_per_arm(0.7, 0.5, looks=10, rho=0.0) == np.ceil(n / 10)


def test_icc_estimator_recovers_simulated_value():
    """ANOVA ICC of beta-binomial clusters with rho 0.3 is close to 0.3."""
    rng = np.random.default_rng(0)
    rho, p = 0.3, 0.5
    q = rng.beta(p * (1 / rho - 1), (1 - p) * (1 / rho - 1), 400)
    y = (rng.random((400, 20)) < q[:, None]).astype(float)
    assert icc_binary(y.ravel(), np.repeat(np.arange(400), 20)) == pytest.approx(rho, abs=0.05)


def test_formula_gives_nominal_power_by_simulation():
    """Placements from the design-effect formula reach about 80% power in Monte Carlo."""
    k = placements_per_arm(0.7, 0.5, looks=8, rho=0.4)
    pw = simulate_power(0.7, 0.5, k, looks=8, rho=0.4, n_sims=1500, seed=1)
    assert pw == pytest.approx(0.8, abs=0.06)
