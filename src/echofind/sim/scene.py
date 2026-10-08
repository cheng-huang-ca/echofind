"""Echo simulator: one ping of complex baseband echoes from a shallow scene, with labels.

Geometry (metres, z positive down): the transducer sits at (0, 0, z_s) with its beam axis in the
x-z plane, depressed `tilt_deg` below horizontal. The bottom is flat at depth H. Every echo
component is a sum of delayed, scaled copies of the transmit pulse:

    e(t) = sum_k a_k s(t - tau_k) exp(-j 2 pi fc tau_k)

with pressure amplitude (uPa) for a scatterer of cross-section sigma reached over one-way paths
of lengths r_i, r_j:

    a = sqrt(SL) * sqrt(sigma) * R_i R_j * sqrt(b_i b_j) / (r_i r_j) * 10^(-alpha (r_i + r_j) / 20)

where SL is the linear source level, b the one-way beam intensity pattern and R the reflection
coefficients along the path. For the direct path this is SL - 2 TL + TS in dB.

- Target: Body highlights over three one-way paths, direct (D), surface image (S) and bottom
  image (B), giving two-way arrivals DD, DS+SD, SS, DB+BD, BB, SB+BS (image method, first order).
- Bottom reverberation: Lambert's law, sigma = mu sin^2(grazing) dA per patch, with
  K-distributed texture on patch power (compound-Gaussian clutter).
- Noise: white with spectrum level NL, or any callable n -> complex samples (real MBARI noise).

Each component is returned separately so labels and the true SNR come for free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from ..dsp.waveforms import cw_pulse, lfm_pulse
from .acoustics import FRESH, Water, gaussian_beam
from .clutter import complex_gaussian, k_texture
from .noise import white_noise
from .target import Body


@dataclass(frozen=True)
class Sonar:
    fc: float = 80e3              # carrier (Hz)
    bandwidth: float = 20e3       # LFM sweep; 0 for CW
    duration: float = 1e-3        # pulse length (s)
    sl_db: float = 200.0          # source level, dB re 1 uPa at 1 m
    beamwidth_deg: float = 10.0   # -3 dB, one-way, conical
    depth_m: float = 0.5          # transducer depth
    tilt_deg: float = 10.0        # beam axis depression below horizontal
    max_range_m: float = 60.0
    oversample: int = 4           # fs = oversample * max(B, 1/T)
    taper: str | None = None

    @property
    def fs(self) -> float:
        return self.oversample * max(self.bandwidth, 1 / self.duration)

    def pulse(self) -> np.ndarray:
        if self.bandwidth > 0:
            return lfm_pulse(self.bandwidth, self.duration, self.fs, self.taper)
        return cw_pulse(self.duration, self.fs, self.taper)

    @property
    def axis(self) -> np.ndarray:
        t = np.deg2rad(self.tilt_deg)
        return np.array([np.cos(t), 0.0, np.sin(t)])


@dataclass(frozen=True)
class Seabed:
    mu_db: float = -27.0           # Lambert constant (sand about -27, mud about -35)
    reflection_db: float = -10.0   # specular reflection loss for the bottom image path
    k_nu: float = np.inf           # K-distribution shape of the texture; inf = Rayleigh
    texture_m: float = 0.5         # texture correlation length along range


@dataclass(frozen=True)
class Scene:
    water: Water = FRESH
    water_depth_m: float = 5.0
    seabed: Seabed = field(default_factory=Seabed)
    body: Body | None = field(default_factory=Body)
    target_range_m: float = 20.0       # horizontal range to the body centre
    target_bearing_deg: float = 0.0    # azimuth off the beam axis
    target_height_m: float = 0.15      # body centre above the bottom
    target_aspect_deg: float = 90.0
    multipath: bool = True
    surface_loss_db: float = 1.0       # |R_s| = 10^(-loss/20); R_s is negative
    reverb: bool = True
    nl_db: float = 40.0                # noise spectrum level, dB re 1 uPa^2/Hz
    noise: Callable[[int, np.random.Generator], np.ndarray] | None = None  # overrides white

    @property
    def target_depth_m(self) -> float:
        return self.water_depth_m - self.target_height_m


@dataclass
class Ping:
    iq: np.ndarray             # total complex baseband echo (uPa)
    fs: float
    replica: np.ndarray        # transmit pulse for the matched filter
    target: np.ndarray         # components, same length as iq
    reverb: np.ndarray
    noise: np.ndarray
    truth: dict

    @property
    def r(self) -> np.ndarray:
        return self.truth["c"] * np.arange(len(self.iq)) / self.fs / 2


def _off_axis(vec: np.ndarray, axis: np.ndarray) -> np.ndarray:
    u = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
    return np.arccos(np.clip(u @ axis, -1, 1))


def _accumulate(n: int, fs: float, fc: float, tau: np.ndarray, amp: np.ndarray) -> np.ndarray:
    """Impulse train with carrier phase exp(-j 2 pi fc tau), at the nearest sample."""
    h = np.zeros(n, complex)
    idx = np.round(np.asarray(tau) * fs).astype(int)
    ok = (idx >= 0) & (idx < n)
    np.add.at(h, idx[ok], (amp * np.exp(-2j * np.pi * fc * np.asarray(tau)))[ok])
    return h


def target_arrivals(sonar: Sonar, scene: Scene, rng: np.random.Generator):
    """Delays (s) and amplitudes (uPa) of every target arrival, plus a path label each."""
    if scene.body is None:
        return np.zeros(0), np.zeros(0, complex), []
    c, alpha = scene.water.c, scene.water.alpha(sonar.fc)
    zs, H, zt = sonar.depth_m, scene.water_depth_m, scene.target_depth_m
    x = scene.target_range_m
    y = x * np.tan(np.deg2rad(scene.target_bearing_deg))
    rs = 10 ** (-scene.surface_loss_db / 20)
    rb = 10 ** (scene.seabed.reflection_db / 20)
    one_way = {"D": (zt, 1.0)}
    if scene.multipath:
        one_way["S"] = (-zt, -rs)
        one_way["B"] = (2 * H - zt, rb)
    bw = np.deg2rad(sonar.beamwidth_deg)
    legs = {}
    for k, (z_img, refl) in one_way.items():
        v = np.array([x, y, z_img - zs])
        legs[k] = (np.linalg.norm(v), refl, gaussian_beam(_off_axis(v, sonar.axis), bw))
    offsets, hl_amp = scene.body.scatterers(scene.target_aspect_deg, rng)
    sl = np.sqrt(10 ** (sonar.sl_db / 10))
    taus, amps, labels = [], [], []
    for i, (ri, Ri, bi) in legs.items():
        for j, (rj, Rj, bj) in legs.items():
            geo = sl * Ri * Rj * np.sqrt(bi * bj) / (ri * rj) * 10 ** (-alpha * (ri + rj) / 20)
            taus.append((ri + rj + 2 * offsets) / c)
            amps.append(geo * hl_amp)
            labels += [i + j] * len(offsets)
    return np.concatenate(taus), np.concatenate(amps), labels


@dataclass
class BottomPatches:
    """Bottom facets in the beam footprint, with fixed K texture (shared across frequencies)."""
    tau: np.ndarray
    power: np.ndarray   # SL * sigma * b^2 / r^4 * absorption, times texture (uPa^2)

    @staticmethod
    def build(sonar: Sonar, scene: Scene, rng: np.random.Generator, cells_per_res: int = 4,
              n_az: int = 41) -> BottomPatches:
        c, alpha = scene.water.c, scene.water.alpha(sonar.fc)
        h = scene.water_depth_m - sonar.depth_m
        res = c / (2 * max(sonar.bandwidth, 1 / sonar.duration))
        rho_max = np.sqrt(max(sonar.max_range_m**2 - h**2, 0.0))
        rho = np.arange(res / cells_per_res / 2, rho_max, res / cells_per_res)
        bw = np.deg2rad(sonar.beamwidth_deg)
        phi = np.linspace(-1.5 * bw, 1.5 * bw, n_az)
        d_rho, d_phi = rho[1] - rho[0], phi[1] - phi[0]
        P, F = np.meshgrid(rho, phi, indexing="ij")
        vec = np.stack([P * np.cos(F), P * np.sin(F), np.full_like(P, h)], axis=-1)
        r = np.linalg.norm(vec, axis=-1)
        b = gaussian_beam(_off_axis(vec, sonar.axis), bw)
        sin_g = h / r
        mu = 10 ** (scene.seabed.mu_db / 10)
        sigma = mu * sin_g**2 * P * d_rho * d_phi
        sl = 10 ** (sonar.sl_db / 10)
        power = sl * sigma * b**2 / r**4 * 10 ** (-2 * alpha * r / 10)
        corr = max(int(round(scene.seabed.texture_m / (res / cells_per_res))), 1)
        tex = k_texture((n_az, len(rho)), scene.seabed.k_nu, rng, corr).T
        keep = b**2 > 1e-4
        return BottomPatches(tau=(2 * r / c)[keep], power=(power * tex)[keep])


def simulate_ping(sonar: Sonar, scene: Scene, rng: np.random.Generator,
                  patches: BottomPatches | None = None) -> Ping:
    """Simulate one ping; returns total echo, components and ground truth."""
    fs, c = sonar.fs, scene.water.c
    pulse = sonar.pulse()
    n = int(np.ceil(2 * sonar.max_range_m / c * fs))

    tau_t, amp_t, labels = target_arrivals(sonar, scene, rng)
    target = fftconvolve(_accumulate(n, fs, sonar.fc, tau_t, amp_t), pulse)[:n]

    if scene.reverb:
        patches = patches or BottomPatches.build(sonar, scene, rng)
        amp_b = np.sqrt(patches.power) * complex_gaussian(len(patches.tau), rng)
        reverb = fftconvolve(_accumulate(n, fs, sonar.fc, patches.tau, amp_b), pulse)[:n]
    else:
        reverb = np.zeros(n, complex)

    noise = (scene.noise(n, rng) if scene.noise is not None
             else white_noise(n, scene.nl_db, fs, rng))

    truth = {"c": c, "alpha_db_per_m": scene.water.alpha(sonar.fc), "fc": sonar.fc}
    if scene.body is not None:
        direct = [t for t, lab in zip(tau_t, labels, strict=True) if lab == "DD"]
        truth.update(target_range_m=c * float(np.mean(direct)) / 2,
                     target_extent_m=c * float(np.ptp(direct)) / 2,
                     arrivals={lab: c * float(np.mean([t for t, q in zip(tau_t, labels,
                                                                          strict=True)
                                                       if q == lab])) / 2
                               for lab in dict.fromkeys(labels)})
    h = scene.water_depth_m - sonar.depth_m
    t_ax = np.deg2rad(sonar.tilt_deg)
    truth["bottom_range_m"] = h / np.sin(t_ax + np.deg2rad(sonar.beamwidth_deg) / 2)
    return Ping(target + reverb + noise, fs, pulse, target, reverb, noise, truth)


def simulate_pair(lo: Sonar, hi: Sonar, scene: Scene, rng: np.random.Generator,
                  ts_hi_offset_db: float = 0.0) -> tuple[Ping, Ping]:
    """Paired low/high-frequency pings of the same scene and body pose.

    The seabed texture is shared (same physical patches); speckle and body highlight phases are
    drawn independently per frequency, as they decorrelate across widely spaced bands.
    ts_hi_offset_db shifts the body TS at the high frequency (frequency response of the target).
    """
    seed = int(rng.integers(2**31))
    p_lo = BottomPatches.build(lo, scene, np.random.default_rng(seed)) if scene.reverb else None
    p_hi = BottomPatches.build(hi, scene, np.random.default_rng(seed)) if scene.reverb else None
    a = simulate_ping(lo, scene, rng, p_lo)
    scene_hi = scene
    if scene.body is not None and ts_hi_offset_db:
        from dataclasses import replace

        scene_hi = replace(scene, body=replace(
            scene.body, ts_broadside_db=scene.body.ts_broadside_db + ts_hi_offset_db))
    b = simulate_ping(hi, scene_hi, rng, p_hi)
    return a, b
