"""FLS proposals and features on a synthetic frame, the UATD reader, and the R0 size gate."""

import numpy as np
import pandas as pd
import pytest

from echofind.features.fls import ProposalConfig, frame_candidates
from echofind.io.uatd import assign_groups, parse_xml
from echofind.models.ladder import R0Rule

RANGE_M, AZ = 10.0, 120.0


def synthetic_frame(seed=0):
    """Rayleigh speckle with a 30 log R reverberation ramp, plus a body at 6 m drawn as five
    highlights along 1.5 m of range, 0.5 m across (decision 102: head, chest, pelvis, knees,
    feet), and a -20 dB acoustic shadow 1.2 m long behind it (rows = range, cols = beams)."""
    rng = np.random.default_rng(seed)
    H, W = 1000, 512
    r = (np.arange(H) + 0.5) / H * RANGE_M
    amp = 0.04 * (r / 2.0) ** -1.5
    a = amp[:, None] * rng.rayleigh(1 / np.sqrt(2), (H, W))
    dr = RANGE_M / H
    dx = 6.0 * np.deg2rad(AZ / W)                            # cross-range m per beam at 6 m
    b0, b1 = W // 2 - int(0.25 / dx), W // 2 + int(0.25 / dx)
    r0, r1 = int(5.25 / dr), int(6.75 / dr)
    for frac, rel in zip((0.0, 0.3, 0.55, 0.8, 0.95), (0.1, 0.5, 0.2, 0.1, 0.1), strict=True):
        c = r0 + int(frac * (r1 - r0))
        a[c - 3 : c + 4, b0:b1] = np.sqrt(rel) * 0.9 * rng.rayleigh(1 / np.sqrt(2), (7, b1 - b0))
    s1 = r1 + int(1.2 / dr)
    a[r1 + 4 : s1, b0:b1] *= 0.1                             # shadow: -20 dB
    box = {"cls": "human body", "xmin": b0, "xmax": b1, "ymin": r0 - 4, "ymax": r1 + 4}
    return np.clip(a, 0, 1).astype(np.float32), box


def test_proposal_finds_target_and_features_measure_it():
    """The matched candidates span the body's 1.5 m within a few cells, are about 0.5 m wide,
    and the farthest one sees the injected shadow, 1.2 m long. The shadow is injected 20 dB
    deep but the 8-bit floor (half a grey level) limits what can be measured to about 9 dB."""
    a, box = synthetic_frame()
    frame = pd.Series({"range_m": RANGE_M, "azimuth_deg": AZ, "freq_khz": 720})
    objs = pd.DataFrame([box], index=[7])
    c, g = frame_candidates(a, frame, objs, ProposalConfig())
    hit = c[c.cls == "human body"]
    assert len(hit) >= 1 and set(hit.obj_idx) == {7}
    span = (hit.r1.max() - hit.r0.min()) * g.dr
    assert span == pytest.approx(1.5, abs=4 * g.dr)
    assert hit.cross_ext_m.max() == pytest.approx(0.5, abs=0.15)
    last = hit.loc[hit.r1.idxmax()]
    assert last.shadow_depth_db > 6
    assert last.shadow_len_m == pytest.approx(1.2, abs=0.25)
    assert hit.snr_db.max() > 15


def test_r0_size_gate():
    X = pd.DataFrame({"range_ext_m": [1.5, 0.1, 5.0], "cross_ext_m": [0.4, 0.1, 0.5],
                      "snr_db": [20.0, 30.0, 25.0]})
    s = R0Rule().score(X)
    assert s[0] == 20.0 and np.isneginf(s[1]) and np.isneginf(s[2])


def test_uatd_xml_and_site_inference(tmp_path):
    """Sound speed splits fresh (< 1490 m/s) from salt water; elevation text like 12(+/-6)."""
    x = tmp_path / "00001.xml"
    x.write_text("""<annotation><sonar><range>10.0</range><azimuth>120</azimuth>
      <elevation>12(+/-6) </elevation><soundspeed>1466.1</soundspeed><frequency>1200k</frequency>
      </sonar><file><filename>00001</filename></file><size><width>1024</width>
      <height>1028</height></size><object><name>human body</name><bndbox><xmin>5</xmin>
      <ymin>6</ymin><xmax>50</xmax><ymax>60</ymax></bndbox></object></annotation>""")
    f, objs = parse_xml(x)
    assert f["elevation_deg"] == 12 and f["freq_khz"] == 1200 and objs[0]["cls"] == "human body"
    fr = pd.DataFrame([{**f, "sound_speed": c} for c in (1466.1, 1467.0, 1527.2)])
    g = assign_groups(fr)
    assert list(g.site) == ["lake", "lake", "sea"]
    assert g.deployment[0] == g.deployment[1] != g.deployment[2]
