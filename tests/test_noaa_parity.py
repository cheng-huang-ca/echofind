"""Matched filter on real EK80 broadband pings matches echopype's pulse compression.

Skipped unless the NOAA HB2305 data (`make data ONLY=noaa`) and echopype are present.
"""

from pathlib import Path

import numpy as np
import pytest

from echofind.dsp.matched_filter import matched_filter

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "noaa" / "D20230722-T131829.raw"


@pytest.mark.skipif(not RAW.exists(), reason="NOAA data not fetched")
def test_matched_filter_matches_echopype_on_ek80():
    """y[k] = sum_n x[k+n] s*[n] (correlation with h(t) = s*(-t), aligned to the echo start)
    equals echopype's compress_pulse within float32 precision on every channel."""
    ep = pytest.importorskip("echopype")
    from echopype.calibrate.ek80_complex import (
        compress_pulse,
        get_filter_coeff,
        get_transmit_signal,
    )

    ed = ep.open_raw(str(RAW), sonar_model="EK80")
    beam = ed["Sonar/Beam_group1"].isel(ping_time=slice(0, 2))
    vend = ed["Vendor_specific"]
    chirp, _ = get_transmit_signal(beam, get_filter_coeff(vend), "BB",
                                   vend["receiver_sampling_frequency"])
    for ch in beam["channel"].values:
        bs = (beam["backscatter_r"] + 1j * beam["backscatter_i"]).sel(channel=[ch])
        ref = compress_pulse(bs, chirp).sel(channel=ch).transpose(
            "ping_time", "range_sample", "beam").values
        x = np.nan_to_num(bs.sel(channel=ch).transpose("ping_time", "range_sample",
                                                       "beam").values)
        ours = matched_filter(x, np.asarray(chirp[str(ch)]), axis=1, normalize=False)
        ok = np.isfinite(ref)
        assert np.max(np.abs(ours[ok] - ref[ok])) / np.max(np.abs(ref[ok])) < 1e-5
