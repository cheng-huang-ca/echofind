"""Physics-based echo simulator (W2)."""

from .acoustics import FRESH, SALT, Water, absorption_fg, mf_snr_db, sound_speed_mackenzie
from .scene import BottomPatches, Ping, Scene, Seabed, Sonar, simulate_pair, simulate_ping
from .target import Body

__all__ = [
    "FRESH", "SALT", "Body", "BottomPatches", "Ping", "Scene", "Seabed", "Sonar", "Water",
    "absorption_fg", "mf_snr_db", "simulate_pair", "simulate_ping", "sound_speed_mackenzie",
]
