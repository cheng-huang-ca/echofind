"""Simulator physics against closed forms: sonar equation, TVG, image-method multipath, Lambert
reverberation, K-distributed clutter, the body model, and the front end on simulated scenes."""

import wave
from dataclasses import replace

import numpy as np
import pytest
from scipy import integrate
from scipy.signal import find_peaks

from echofind.dsp.bottom import BottomConfig, track_bottom
from echofind.dsp.cfar import CFARConfig
from echofind.dsp.frontend import FrontEndConfig, process_iq
from echofind.dsp.matched_filter import matched_filter
from echofind.dsp.tvg import apply_tvg, range_from_time, tvg_gain_db
from echofind.sim import SALT, Body, Scene, Seabed, Sonar, simulate_pair, simulate_ping
from echofind.sim.acoustics import (
    absorption_fg,
    beamwidth_line,
    gaussian_beam,
    mf_snr_db,
    sound_speed_mackenzie,
)
from echofind.sim.clutter import k_clutter
from echofind.sim.noise import mbari_baseband, read_wav_segment
from echofind.sim.scene import BottomPatches

POINT = Body(ts_broadside_db=-20.0, end_on_db=0.0, highlights=((0.0, 1.0),), fluctuating=False)
SILENT = lambda n, rng: np.zeros(n, complex)  # noqa: E731


def _on_axis_scene(r, water=SALT, **kw):
    """Point target at the transducer depth, straight ahead (beam axis horizontal)."""
    return Scene(water=water, water_depth_m=50.0, body=POINT, target_range_m=r,
                 target_height_m=50.0 - 1.0, multipath=False, reverb=False, noise=SILENT, **kw)


# ---------------------------------------------------------------- acoustics
def test_mackenzie_check_value():
    """Mackenzie (1981) check value: c = 1550.744 m/s at T = 25 C, S = 35, D = 1000 m."""
    assert sound_speed_mackenzie(25, 35, 1000) == pytest.approx(1550.744, abs=0.001)


def test_absorption_francois_garrison():
    """Francois-Garrison (1982): matches echopype's implementation; fresh water is the pure-water
    term A3 f^2 (so alpha(2f) = 4 alpha(f)); sea water absorbs more."""
    ep = pytest.importorskip("echopype.utils.uwa")
    for f in (38e3, 80e3, 200e3):
        ref = ep.calc_absorption(frequency=f, temperature=12, salinity=35, pressure=5, pH=8.0,
                                 formula_source="FG")
        assert absorption_fg(f, 12, 35, 5, 8.0) == pytest.approx(float(ref), rel=1e-9)
    a1, a2 = absorption_fg(100e3, 15, 0, 0), absorption_fg(200e3, 15, 0, 0)
    assert a2 / a1 == pytest.approx(4.0, rel=1e-9)
    assert absorption_fg(80e3, 15, 35, 0) > 10 * absorption_fg(80e3, 15, 0, 0)


def test_beam_patterns():
    """Gaussian beam: one-way intensity 1/2 at theta = bw/2; line aperture bw = 0.886 lambda/D."""
    bw = np.deg2rad(10)
    assert gaussian_beam(bw / 2, bw) == pytest.approx(0.5)
    assert beamwidth_line(100e3, 0.15, 1500) == pytest.approx(0.886 * 0.015 / 0.15)


def test_range_from_time():
    """r = c t / 2."""
    assert range_from_time(0.02, 1480.0) == pytest.approx(14.8)


# ---------------------------------------------------------------- sonar equation and TVG
@pytest.mark.parametrize("r", [5.0, 10.0, 20.0, 40.0])
def test_echo_level_and_40logr_tvg(r):
    """Echo level EL = SL - 2 TL + TS with TL = 20 log10 r + alpha r; after 40 log10 r + 2 alpha r
    TVG every point target reads SL + TS, whatever its range."""
    son = Sonar(tilt_deg=0.0, depth_m=1.0, max_range_m=50.0)
    sc = _on_axis_scene(r)
    p = simulate_ping(son, sc, np.random.default_rng(0))
    n = len(p.replica)
    peak_pow = np.abs(matched_filter(p.iq, p.replica)).max() ** 2 / n   # = A^2
    alpha = SALT.alpha(son.fc)
    el = son.sl_db - 2 * (20 * np.log10(r) + alpha * r) + POINT.ts_broadside_db
    assert 10 * np.log10(peak_pow) == pytest.approx(el, abs=0.05)
    k = int(np.argmax(np.abs(matched_filter(p.iq, p.replica))))
    tvg = 10 * np.log10(apply_tvg(np.array([peak_pow]), np.array([p.r[k]]), alpha)[0])
    assert tvg == pytest.approx(son.sl_db + POINT.ts_broadside_db, abs=0.1)


def test_tvg_gain_formula():
    """TVG = spreading log10 r + 2 alpha r."""
    assert tvg_gain_db(np.array([100.0]), 0.01, 40)[0] == pytest.approx(80 + 2.0)
    assert tvg_gain_db(np.array([100.0]), 0.01, 20)[0] == pytest.approx(40 + 2.0)


def test_sonar_equation_snr():
    """Noise-limited MF SNR = SL - 2 TL + TS - NL + 10 log10 T (DI = 0 here: NL is the noise at
    the receiver output), measured within 0.3 dB."""
    rng = np.random.default_rng(1)
    son = Sonar(tilt_deg=0.0, depth_m=1.0, max_range_m=50.0)
    sc = replace(_on_axis_scene(30.0), noise=None, nl_db=60.0)
    p = simulate_ping(son, sc, rng)
    sig = np.abs(matched_filter(p.target, p.replica)).max() ** 2
    noise = np.mean([np.mean(np.abs(matched_filter(simulate_ping(son, sc, rng).noise,
                                                   p.replica)) ** 2) for _ in range(20)])
    want = mf_snr_db(son.sl_db, POINT.ts_broadside_db, 30.0, SALT.alpha(son.fc), 60.0, 0.0,
                     son.duration)
    assert 10 * np.log10(sig / noise) == pytest.approx(want, abs=0.3)


# ---------------------------------------------------------------- multipath
def test_image_method_arrivals():
    """Image method: one-way paths to the target and to its surface (-z_t) and bottom (2H - z_t)
    images; two-way arrivals sit at (r_i + r_j)/2 and appear as MF peaks there."""
    son = Sonar(beamwidth_deg=80.0, tilt_deg=30.0, depth_m=1.0, max_range_m=20.0)
    sc = Scene(water_depth_m=10.0, body=POINT, target_range_m=10.0, target_height_m=3.0,
               multipath=True, reverb=False, noise=SILENT, surface_loss_db=0.0,
               seabed=Seabed(reflection_db=-3.0))
    p = simulate_ping(son, sc, np.random.default_rng(0))
    x, zs, zt, H = 10.0, 1.0, 7.0, 10.0  # arrivals at least 0.5 m apart
    r = {"D": np.hypot(x, zt - zs), "S": np.hypot(x, zt + zs), "B": np.hypot(x, 2 * H - zt - zs)}
    for lab, got in p.truth["arrivals"].items():
        assert got == pytest.approx((r[lab[0]] + r[lab[1]]) / 2, abs=1e-9)
    env = np.abs(matched_filter(p.iq, p.replica))
    pk, _ = find_peaks(env, height=env.max() * 0.05)
    found = p.r[pk]
    res = p.truth["c"] / (2 * son.bandwidth)
    for want in {(r[a] + r[b]) / 2 for a in r for b in r}:
        assert np.min(np.abs(found - want)) <= res / 4


# ---------------------------------------------------------------- reverberation and clutter
def test_lambert_reverberation_level():
    """Mean reverberation intensity from the Lambert seabed equals the integral over the
    insonified annulus: I(t) = SL int int mu sin^2(g) b^2 r^-4 10^(-2 alpha r/10) rho drho dphi,
    over slant ranges c (t - T)/2 .. c t/2 (CW pulse of length T, within 10%)."""
    son = Sonar(bandwidth=0.0, duration=0.5e-3, beamwidth_deg=12.0, tilt_deg=15.0, depth_m=0.5,
                max_range_m=30.0, oversample=16)
    sc = Scene(water_depth_m=5.0, body=None, multipath=False, reverb=True, noise=SILENT)
    rng = np.random.default_rng(2)
    patches = BottomPatches.build(son, sc, rng, cells_per_res=32, n_az=121)
    pw = np.mean([np.abs(simulate_ping(son, sc, rng, patches).reverb) ** 2
                  for _ in range(300)], axis=0)
    c, alpha, h = sc.water.c, sc.water.alpha(son.fc), sc.water_depth_m - son.depth_m
    bw, ax = np.deg2rad(son.beamwidth_deg), son.axis
    sl, mu = 10 ** (son.sl_db / 10), 10 ** (sc.seabed.mu_db / 10)

    def dens(rho, phi):
        v = np.array([rho * np.cos(phi), rho * np.sin(phi), h])
        r = np.linalg.norm(v)
        b = gaussian_beam(np.arccos(np.clip(v @ ax / r, -1, 1)), bw)
        return sl * mu * (h / r) ** 2 * b**2 / r**4 * 10 ** (-2 * alpha * r / 10) * rho

    for rng_m in (16.0, 20.0, 26.0):
        t = 2 * rng_m / c
        r_lo, r_hi = c * (t - son.duration) / 2, rng_m
        rho_lo, rho_hi = np.sqrt(r_lo**2 - h**2), np.sqrt(r_hi**2 - h**2)
        want = integrate.dblquad(lambda rho, phi: dens(rho, phi), -1.5 * bw, 1.5 * bw,
                                 rho_lo, rho_hi)[0]
        k = int(round(t * son.fs))
        assert pw[k] == pytest.approx(want, rel=0.10)


@pytest.mark.parametrize("nu", [0.5, 2.0, 10.0, np.inf])
def test_k_clutter_moments(nu):
    """K-distribution (compound Gaussian, Gamma(nu) texture): E[I] = 1 and
    E[I^2]/E[I]^2 = 2 (1 + 1/nu); Rayleigh (nu = inf) gives 2."""
    z = k_clutter(400_000, nu, np.random.default_rng(3))
    i = np.abs(z) ** 2
    assert i.mean() == pytest.approx(1.0, rel=0.03)
    want = 2 * (1 + 1 / nu) if np.isfinite(nu) else 2.0
    assert (i**2).mean() / i.mean() ** 2 == pytest.approx(want, rel=0.06)


# ---------------------------------------------------------------- body model
def test_body_target_strength_and_extent():
    """Body: sum of highlight cross-sections = 10^(TS/10) x aspect gain; range extent is
    L_highlights cos(aspect): zero broadside, the full highlight span head-on; end-on gain is
    floored at end_on_db."""
    b = Body(ts_broadside_db=-20.0, end_on_db=-10.0)
    off, amp = b.scatterers(90.0, np.random.default_rng(0))
    assert np.sum(np.abs(amp) ** 2) == pytest.approx(0.01)
    assert np.ptp(off) == pytest.approx(0.0, abs=1e-12)
    off0, amp0 = b.scatterers(0.0, np.random.default_rng(0))
    assert np.ptp(off0) == pytest.approx(1.6)
    assert 10 * np.log10(np.sum(np.abs(amp0) ** 2)) == pytest.approx(-30.0)


def test_paired_frequencies_share_geometry():
    """simulate_pair: same body range at both frequencies, different absorption."""
    lo = Sonar(fc=80e3)
    hi = Sonar(fc=200e3, bandwidth=40e3)
    a, b = simulate_pair(lo, hi, Scene(water=SALT), np.random.default_rng(4))
    assert a.truth["target_range_m"] == pytest.approx(b.truth["target_range_m"])
    assert b.truth["alpha_db_per_m"] > 2 * a.truth["alpha_db_per_m"]


# ---------------------------------------------------------------- noise I/O
def _write_wav24(path, x, fs):
    q = np.clip(np.round(x * (1 << 23)), -(1 << 23), (1 << 23) - 1).astype(np.int32)
    b = np.stack([q & 0xFF, (q >> 8) & 0xFF, (q >> 16) & 0xFF], -1).astype(np.uint8)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(fs)
        w.writeframes(b.tobytes())


def test_wav24_reader_and_noise_floor_scaling(tmp_path):
    """24-bit PCM round-trips to +/-1 full scale; mbari_baseband sets the Gaussian floor to
    spectrum level NL: variance = 10^(NL/10) x (1.2 B) for the IQ filter's pass band."""
    rng = np.random.default_rng(5)
    fs = 256_000
    x = 0.01 * rng.standard_normal(fs) + 0.2      # with a DC offset like the MARS files
    path = tmp_path / "n.wav"
    _write_wav24(path, x, fs)
    fs_r, y = read_wav_segment(path, 0.25, 0.5)
    assert fs_r == fs
    np.testing.assert_allclose(y, x[fs // 4 : fs // 4 + fs // 2], atol=2 ** -22)
    bb = mbari_baseband(path, 0.1, 20_000, 80e3, 80e3, 20e3, nl_db=40.0)
    assert np.mean(np.abs(bb) ** 2) == pytest.approx(1e4 * 1.2 * 20e3, rel=0.1)


# ---------------------------------------------------------------- front end on simulated scenes
def test_bottom_tracker_ignores_body_on_bottom():
    """A compact body on the seabed must not capture the bottom track (persistence criterion)."""
    son = Sonar()
    sc = Scene(body=replace(Body(), ts_broadside_db=-5.0), target_range_m=22.0)
    rng = np.random.default_rng(6)
    pings = [simulate_ping(son, sc, rng) for _ in range(5)]
    pw = np.array([np.abs(matched_filter(p.iq, p.replica)) ** 2 for p in pings])
    r = pings[0].r
    bt = track_bottom(apply_tvg(pw, r, 0.0), r, BottomConfig())
    assert np.all(bt < pings[0].truth["target_range_m"] - 1.0)
    assert np.all(bt > pings[0].truth["bottom_range_m"] - 1.5)


def test_bottom_tracker_down_looking():
    """Down-looking beam over a flat bottom: the track sits at the vertical range H - z_s.
    The bottom tail is only h (1/cos(theta) - 1), a few cm here, so persistence must be shorter."""
    son = Sonar(tilt_deg=90.0, beamwidth_deg=7.0, max_range_m=15.0)
    sc = Scene(water_depth_m=10.0, body=None, nl_db=30.0)
    rng = np.random.default_rng(7)
    pw = np.array([np.abs(matched_filter(p.iq, p.replica)) ** 2
                   for p in (simulate_ping(son, sc, rng) for _ in range(5))])
    r = simulate_ping(son, sc, rng).r
    bt = track_bottom(apply_tvg(pw, r, 0.0, 20.0), r,
                      BottomConfig(persist_m=0.06, cell_m=0.02, median_pings=1))
    np.testing.assert_allclose(bt, 10.0 - son.depth_m, atol=0.05)


def test_front_end_finds_body_on_bottom():
    """End to end: a body lying on a sand bottom at 20 m in fresh water is a candidate within
    two resolution cells of its true range."""
    son = Sonar()
    sc = Scene()
    p = simulate_ping(son, sc, np.random.default_rng(8))
    cfg = FrontEndConfig(c=p.truth["c"], alpha_db_per_m=p.truth["alpha_db_per_m"],
                         cfar=CFARConfig("os", n_ref=32, n_guard=6, pfa=1e-4))
    res = process_iq(p.iq, p.fs, p.replica, cfg)
    assert len(res.candidates) >= 1
    err = np.abs(res.candidates.range_m - p.truth["target_range_m"]).min()
    assert err < 2 * p.truth["c"] / (2 * son.bandwidth)
