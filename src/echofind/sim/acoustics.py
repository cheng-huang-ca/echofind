"""Underwater acoustics for the simulator: sound speed, absorption, transmission loss, sonar
equation and beam patterns.

- Sound speed: Mackenzie (1981), J. Acoust. Soc. Am. 70(3):807-812.
- Absorption: Francois and Garrison (1982), J. Acoust. Soc. Am. 72(6):1879-1890. With S = 0
  the boric-acid and MgSO4 relaxation terms vanish, leaving pure-water viscous absorption, which
  is the fresh-water case at the frequencies EchoFind uses (above about 20 kHz).
- One-way transmission loss TL(r) = 20 log10 r + alpha r (spherical spreading, alpha in dB/m).
- Active sonar equation, noise-limited, matched-filter output SNR for a pulse of duration T
  (noise spectrum level NL in dB re 1 uPa^2/Hz): SNR = SL - 2 TL + TS - (NL - DI) + 10 log10 T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sound_speed_mackenzie(temperature: float, salinity: float, depth: float) -> float:
    """Mackenzie (1981) nine-term sound speed (m/s); T in deg C, S in PSU, D in m."""
    T, S, D = temperature, salinity, depth
    return float(1448.96 + 4.591 * T - 5.304e-2 * T**2 + 2.374e-4 * T**3
                 + 1.340 * (S - 35) + 1.630e-2 * D + 1.675e-7 * D**2
                 - 1.025e-2 * T * (S - 35) - 7.139e-13 * T * D**3)


def absorption_fg(frequency_hz, temperature: float, salinity: float, depth: float = 0.0,
                  ph: float = 8.0) -> np.ndarray:
    """Francois-Garrison (1982) absorption coefficient in dB/m."""
    f = np.asarray(frequency_hz, dtype=float) / 1e3  # kHz
    T, S, D = temperature, salinity, depth
    c = 1412 + 3.21 * T + 1.19 * S + 0.0167 * D
    # boric acid
    a1 = 8.86 / c * 10 ** (0.78 * ph - 5)
    f1 = 2.8 * np.sqrt(S / 35) * 10 ** (4 - 1245 / (273 + T))
    # magnesium sulphate
    a2 = 21.44 * S / c * (1 + 0.025 * T)
    p2 = 1 - 1.37e-4 * D + 6.2e-9 * D**2
    f2 = 8.17 * 10 ** (8 - 1990 / (273 + T)) / (1 + 0.0018 * (S - 35))
    # pure water
    p3 = 1 - 3.83e-5 * D + 4.9e-10 * D**2
    if T <= 20:
        a3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T**2 - 1.50e-8 * T**3
    else:
        a3 = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T**2 - 6.5e-10 * T**3
    boric = a1 * f1 * f**2 / (f**2 + f1**2) if S > 0 else 0.0 * f
    mgso4 = a2 * p2 * f2 * f**2 / (f**2 + f2**2)
    alpha_db_per_km = boric + mgso4 + a3 * p3 * f**2
    return alpha_db_per_km / 1e3


def transmission_loss(r, alpha_db_per_m: float) -> np.ndarray:
    """One-way TL = 20 log10 r + alpha r (dB)."""
    r = np.asarray(r, dtype=float)
    return 20 * np.log10(r) + alpha_db_per_m * r


def mf_snr_db(sl: float, ts: float, r, alpha_db_per_m: float, nl: float, di: float,
              duration: float) -> np.ndarray:
    """Noise-limited matched-filter SNR: SL - 2 TL + TS - (NL - DI) + 10 log10 T."""
    return (sl - 2 * transmission_loss(r, alpha_db_per_m) + ts - (nl - di)
            + 10 * np.log10(duration))


def gaussian_beam(off_axis_rad, beamwidth_rad: float) -> np.ndarray:
    """One-way intensity pattern exp(-4 ln2 theta^2 / bw^2): 0.5 at theta = bw/2."""
    th = np.asarray(off_axis_rad, dtype=float)
    return np.exp(-4 * np.log(2) * th**2 / beamwidth_rad**2)


def beamwidth_line(frequency_hz: float, aperture_m: float, c: float = 1500.0) -> float:
    """Half-power beamwidth of a uniform line aperture: 0.886 lambda / D (radians)."""
    return 0.886 * (c / frequency_hz) / aperture_m


def directivity_index_db(beamwidth_rad: float) -> float:
    """DI of an axisymmetric pencil beam, approx 10 log10(4 pi / bw^2) for small bw."""
    return float(10 * np.log10(4 * np.pi / beamwidth_rad**2))


@dataclass(frozen=True)
class Water:
    temperature: float = 15.0
    salinity: float = 0.0       # 0 for fresh water, about 35 for sea water
    depth: float = 5.0          # used for absorption and sound speed
    ph: float = 7.5

    @property
    def c(self) -> float:
        return sound_speed_mackenzie(self.temperature, self.salinity, self.depth)

    def alpha(self, frequency_hz: float) -> float:
        return float(absorption_fg(frequency_hz, self.temperature, self.salinity, self.depth,
                                   self.ph))


FRESH = Water(temperature=15.0, salinity=0.0, ph=7.5)
SALT = Water(temperature=15.0, salinity=35.0, ph=8.0)
