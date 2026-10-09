"""Reader for UATD (Xie et al., 2022): Tritech Gemini 1200ik forward-looking sonar frames.

Each frame is a BMP in beam x range layout (columns are beams across the azimuth sector, rows are
range samples, row 0 nearest the sonar) with a Pascal VOC style XML holding the sonar settings and
boxes for 10 object classes. The XML has no site or session field, so both are inferred here:

- Site: the logged sound speed separates fresh from salt water. The lake frames log about
  1458-1484 m/s and the sea frames about 1490-1585 m/s (Chen-Millero: fresh water at 10-20 C is
  1447-1482 m/s; sea water at 32 PSU and the same temperatures is 1490-1522 m/s).
- Session: frames of one deployment share frequency, range setting and a sound speed that drifts
  slowly with temperature. Frames are grouped by (frequency, site, range setting rounded to 0.5 m,
  sound speed rounded to 1 m/s). This over-splits rather than merges, which is the safe direction
  for leakage only when the split later groups by a coarser key too, so `deployment` merges
  sessions whose sound speeds are within 3 m/s at the same frequency and site.

See reports/decisions/201-uatd-site-and-session.md.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("UATD_Training", "UATD_Test_1", "UATD_Test_2")
CLASSES = ("human body", "tyre", "cylinder", "cube", "ball", "circle cage", "square cage",
           "metal bucket", "plane", "rov")
SEA_MIN_C = 1490.0  # m/s; see module docstring


def _root(raw: Path, split: str) -> Path:
    """Extraction nests one folder inside another of the same name; accept either layout."""
    base = raw / split
    return base / split if (base / split).is_dir() else base


def _num(text: str) -> float:
    return float(re.match(r"\s*([-+]?[\d.]+)", text).group(1))


def parse_xml(path: Path) -> tuple[dict, list[dict]]:
    r = ET.parse(path).getroot()
    s = r.find("sonar")
    frame = {
        "file": r.findtext("file/filename"),
        "range_m": float(s.findtext("range")),
        "azimuth_deg": _num(s.findtext("azimuth")),
        "elevation_deg": _num(s.findtext("elevation")),  # some files say "12(+/-6)"
        "sound_speed": float(s.findtext("soundspeed")),
        "freq_khz": int(s.findtext("frequency").lower().rstrip("k")),
        "width": int(r.findtext("size/width")),
        "height": int(r.findtext("size/height")),
    }
    objs = []
    for o in r.findall("object"):
        b = o.find("bndbox")
        objs.append({"cls": o.findtext("name").strip().lower(),
                     **{k: int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")}})
    return frame, objs


def load_index(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frame table and object table for every split present under `raw` (data/raw/uatd)."""
    frames, objects = [], []
    for split in SPLITS:
        root = _root(raw, split)
        if not (root / "annotations").is_dir():
            continue
        for x in sorted((root / "annotations").glob("*.xml")):
            f, objs = parse_xml(x)
            fid = f"{split}/{f['file']}"
            f.update(frame_id=fid, split=split, image=str(root / "images" / f"{f['file']}.bmp"))
            frames.append(f)
            objects += [{"frame_id": fid, **o} for o in objs]
    fr = pd.DataFrame(frames)
    ob = pd.DataFrame(objects)
    return assign_groups(fr), ob


def assign_groups(fr: pd.DataFrame) -> pd.DataFrame:
    """Add site, session and deployment columns (see module docstring)."""
    fr = fr.copy()
    fr["site"] = np.where(fr.sound_speed >= SEA_MIN_C, "sea", "lake")
    fr["session"] = (fr.freq_khz.astype(str) + "k_" + fr.site + "_r"
                     + (fr.range_m * 2).round().div(2).astype(str) + "_c"
                     + fr.sound_speed.round().astype(int).astype(str))
    dep = np.empty(len(fr), dtype=object)
    for (_, _), g in fr.groupby(["freq_khz", "site"]):
        c = np.sort(g.sound_speed.unique())
        cut = np.r_[0, np.cumsum(np.diff(c) > 3.0)]
        lut = dict(zip(c, cut, strict=True))
        for i, row in g.iterrows():
            dep[fr.index.get_loc(i)] = f"{row.freq_khz}k_{row.site}_d{lut[row.sound_speed]}"
    fr["deployment"] = dep
    return fr


def range_axis(frame: pd.Series) -> np.ndarray:
    """Range (m) of each image row; row 0 is at the sonar."""
    return np.linspace(0, frame.range_m, frame.height)


def bearing_axis(frame: pd.Series) -> np.ndarray:
    """Bearing (deg) of each image column across the azimuth sector."""
    half = frame.azimuth_deg / 2
    return np.linspace(-half, half, frame.width)


def load_image(frame: pd.Series) -> np.ndarray:
    """Grayscale frame as float32 in [0, 1], shape (range rows, beams)."""
    from PIL import Image

    with Image.open(frame.image) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0
