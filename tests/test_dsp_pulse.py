"""Waveforms, IQ demodulation and matched filtering against closed-form results (H2)."""

import numpy as np
import pytest

from echofind.dsp.demod import iq_demodulate
from echofind.dsp.frontend import demodulate
from echofind.dsp.matched_filter import (
    compression_gain_theory_db,
    cw_resolution_theory,
    lfm_resolution_theory,
    matched_filter,
    width_at_level,
)
from echofind.dsp.waveforms import cw_pulse, lfm_pulse, pulse_energy, to_passband

C = 1500.0


def _cn(n, rng):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


def test_lfm_instantaneous_frequency():
    """LFM sweep: f(t) = -B/2 + (B/T) t, from d(phi)/dt / (2 pi)."""
    B, T, fs = 20e3, 2e-3, 400e3
    s = lfm_pulse(B, T, fs)
    f_inst = np.diff(np.unwrap(np.angle(s))) * fs / (2 * np.pi)
    t_mid = (np.arange(len(f_inst)) + 0.5) / fs
    np.testing.assert_allclose(f_inst, -B / 2 + B / T * t_mid, atol=B * 1e-3)


def test_pulse_energy():
    """E = integral |s|^2 dt = A^2 T for a constant-envelope pulse."""
    assert pulse_energy(lfm_pulse(10e3, 1e-3, 100e3), 100e3) == pytest.approx(1e-3)


def test_matched_filter_peak_at_echo_start():
    """Correlation with h(t) = s*(-t) peaks where the echo starts."""
    fs = 80e3
    s = lfm_pulse(20e3, 1e-3, fs)
    x = np.zeros(2000, complex)
    x[700 : 700 + len(s)] = s
    assert int(np.argmax(np.abs(matched_filter(x, s)))) == 700


def test_matched_filter_peak_snr_is_e_over_n0():
    """Peak output SNR = E / N0 (complex baseband), independent of pulse shape."""
    rng = np.random.default_rng(1)
    fs, sigma2 = 100e3, 2.0                      # N0 = sigma^2 / fs
    for s in (cw_pulse(1e-3, fs), lfm_pulse(30e3, 1e-3, fs, taper="hann")):
        peak = np.abs(matched_filter(np.r_[s, np.zeros(50)], s)).max() ** 2
        noise_out = np.mean(np.abs(matched_filter(np.sqrt(sigma2) * _cn(200_000, rng), s)) ** 2)
        e_over_n0 = pulse_energy(s, fs) / (sigma2 / fs)
        assert 10 * np.log10(peak / noise_out) == pytest.approx(10 * np.log10(e_over_n0),
                                                                abs=0.1)


@pytest.mark.parametrize("B,T", [(5e3, 1e-3), (20e3, 1e-3), (20e3, 5e-3), (50e3, 2e-3)])
def test_pulse_compression_gain_baseband(B, T):
    """H2: gain = SNR_out / SNR_in(band B) = B T, i.e. 10 log10(B T) dB, within 1 dB."""
    rng = np.random.default_rng(2)
    fs = 4 * B
    s = lfm_pulse(B, T, fs)
    noise = _cn(400_000, rng)                                 # sigma^2 = 1 at rate fs
    snr_in = 1.0 / (np.var(noise) * B / fs)                   # A = 1, noise power in band B
    peak = np.abs(matched_filter(np.r_[s, np.zeros(10)], s)).max() ** 2
    snr_out = peak / np.mean(np.abs(matched_filter(noise, s)) ** 2)
    gain = 10 * np.log10(snr_out / snr_in)
    assert gain == pytest.approx(compression_gain_theory_db(B, T), abs=1.0)


def test_pulse_compression_gain_passband_chain():
    """H2 through band-pass + IQ demodulation: gain within 1 dB of 10 log10(B T)."""
    rng = np.random.default_rng(3)
    fc, B, T, fs = 80e3, 20e3, 2e-3, 400e3
    s_pb = to_passband(lfm_pulse(B, T, fs), fs, fc)
    sig = np.zeros(40_000)
    sig[10_000 : 10_000 + len(s_pb)] = s_pb
    noise = rng.standard_normal(400_000)
    # raw in-band SNR: signal power A^2/2 (real) over real noise power in band B
    snr_in = 0.5 / (np.var(noise) * 2 * B / fs)
    dec = 5
    rep = lfm_pulse(B, T, fs / dec)
    peak = np.abs(matched_filter(demodulate(sig, fs, fc, B, dec), rep)).max() ** 2
    n_out = matched_filter(demodulate(noise, fs, fc, B, dec), rep)[2000:-2000]
    gain = 10 * np.log10(peak / np.mean(np.abs(n_out) ** 2) / snr_in)
    assert gain == pytest.approx(compression_gain_theory_db(B, T), abs=1.0)


@pytest.mark.parametrize("B", [5e3, 20e3, 50e3])
def test_lfm_range_resolution(B):
    """H2: LFM -3 dB width = 0.886 / B (sinc main lobe), i.e. 0.886 c/(2B) in range, +/-10%."""
    fs, T = 8 * B, 4e-3
    s = lfm_pulse(B, T, fs)
    y = matched_filter(np.r_[np.zeros(len(s)), s, np.zeros(len(s))], s)
    width_m = width_at_level(y) / fs * C / 2
    assert width_m == pytest.approx(0.886 * lfm_resolution_theory(B, C), rel=0.10)


def test_cw_range_resolution():
    """CW: envelope width T gives c T / 2; matched-filter (triangle) -3 dB width is
    2 (1 - 1/sqrt 2) T."""
    fs, T = 200e3, 1e-3
    s = cw_pulse(T, fs)
    env = np.r_[np.zeros(100), s, np.zeros(100)]
    assert width_at_level(env, -3.0, upsample=1) / fs * C / 2 == pytest.approx(
        cw_resolution_theory(T, C), rel=0.02)
    y = matched_filter(np.r_[np.zeros(len(s)), s, np.zeros(len(s))], s)
    assert width_at_level(y) / fs == pytest.approx(2 * (1 - 1 / np.sqrt(2)) * T, rel=0.02)


def test_iq_demodulation_recovers_baseband():
    """Mixing by 2 exp(-j w t) and low-pass returns s(t) from Re{s exp(j w t)}."""
    fc, B, T, fs = 100e3, 20e3, 2e-3, 500e3
    s = lfm_pulse(B, T, fs, taper="tukey")
    x = to_passband(np.r_[np.zeros(1000), s, np.zeros(1000)], fs, fc)
    bb = iq_demodulate(x, fs, fc, B)[1000 : 1000 + len(s)]
    core = slice(len(s) // 10, -len(s) // 10)
    rho = np.abs(np.vdot(bb[core], s[core])) / np.linalg.norm(bb[core]) / np.linalg.norm(s[core])
    assert rho > 0.995
    assert np.abs(bb[core]).mean() == pytest.approx(np.abs(s[core]).mean(), rel=0.03)
