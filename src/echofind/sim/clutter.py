"""Clutter sampling: compound-Gaussian K-distributed returns.

A K-distributed envelope is complex Gaussian speckle whose local power (the texture) is
Gamma-distributed with shape nu and unit mean: z = sqrt(tau) * w, tau ~ Gamma(nu, 1/nu),
w ~ CN(0, 1). The intensity I = |z|^2 then has mean 1 and E[I^2] / E[I]^2 = 2 (1 + 1/nu);
nu -> infinity gives Rayleigh (exponential intensity, ratio 2). Small nu means heavy tails, so
a threshold set for Rayleigh noise false-alarms far more often.

Only sampling lives here; fitting K, Weibull and Rayleigh to data is W1's
(echofind.features.clutter).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def k_texture(shape, nu: float, rng: np.random.Generator, corr_cells: int = 1) -> np.ndarray:
    """Unit-mean Gamma(nu) texture; corr_cells > 1 holds it roughly constant over that many
    cells along the last axis (patchy seabed), by drawing on a coarse grid and repeating."""
    shape = tuple(np.atleast_1d(shape))
    if not np.isfinite(nu):
        return np.ones(shape)
    if corr_cells <= 1:
        return rng.gamma(nu, 1 / nu, size=shape)
    n_coarse = -(-shape[-1] // corr_cells)
    coarse = rng.gamma(nu, 1 / nu, size=shape[:-1] + (n_coarse,))
    fine = np.repeat(coarse, corr_cells, axis=-1)[..., : shape[-1]]
    # soften the steps so the texture has no hard edges; keeps the unit mean
    return uniform_filter1d(fine, max(corr_cells // 2, 1), axis=-1, mode="nearest")


def complex_gaussian(shape, rng: np.random.Generator) -> np.ndarray:
    """CN(0, 1): unit-power circular complex Gaussian."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def k_clutter(shape, nu: float, rng: np.random.Generator, corr_cells: int = 1) -> np.ndarray:
    """Unit-mean-power K-distributed complex samples."""
    return np.sqrt(k_texture(shape, nu, rng, corr_cells)) * complex_gaussian(shape, rng)
