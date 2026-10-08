"""DSP front end: IQ demodulation, matched filter, TVG, CFAR, bottom tracking (W2)."""

from .cfar import CFARConfig, cfar_detect, cfar_threshold, scale_factor
from .demod import bandpass, iq_demodulate
from .frontend import FrontEndConfig, FrontEndResult, demodulate, process_iq, sum_adjacent_beams
from .matched_filter import matched_filter
from .tvg import apply_tvg, range_axis, tvg_gain_db
from .waveforms import cw_pulse, lfm_pulse, to_passband

__all__ = [
    "CFARConfig", "FrontEndConfig", "FrontEndResult", "apply_tvg", "bandpass", "cfar_detect",
    "cfar_threshold", "cw_pulse", "demodulate", "iq_demodulate", "lfm_pulse", "matched_filter",
    "process_iq", "range_axis", "scale_factor", "sum_adjacent_beams", "to_passband",
    "tvg_gain_db",
]
