"""W3: leg-level label table for the Bath side-scan archive (labelling protocol, step 1).

Each PNG in the archive is one survey leg on one side (port or starboard) at one frequency.
The archive's README says the full labels were lost; some runs keep a target/ folder holding
copies of the legs that show the mannequin. This script applies the first step of the protocol
in reports/w3_bath_labelling_protocol.md and writes reports/w3/bath_legs.csv, one row per leg:

- label = "target"     the 990 kHz leg appears in that run's target/ folder (or its
                       extra_target_images_alt_deployment/ subfolder, deployment "alt")
- label = "unlisted"   the run has a target/ folder but this leg is not in it; presumed
                       target-free, NOT yet verified (protocol step 2)
- label = "unlabelled" the run has no target/ folder; unusable until reviewed

Run: uv run python scripts/w3_bath_legs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from w3_common import OUT, RAW  # noqa: E402

ROOT = RAW / "bath" / "sidescan_sonar_images_PRIME_CNN_training"
# e.g. 990_port_cond_12-15_00x18_80m.png: leg 12, 15.00 m slant range, 18.80 m along track
NAME = re.compile(r"(?P<freq>450|990)_(?P<side>port|stbd)_cond_(?P<leg>\d+)-"
                  r"(?P<rng>\d+)_(?P<rngf>\d+)x(?P<al>\d+)_(?P<alf>\d+)m\.png$")


def parse(path: Path) -> dict | None:
    m = NAME.search(path.name)
    if not m:
        return None
    rel = path.relative_to(ROOT).parts
    return {"site": rel[0], "date": rel[1], "run": rel[2], "freq_khz": int(m["freq"]),
            "side": m["side"], "leg": int(m["leg"]),
            "range_m": float(f"{m['rng']}.{m['rngf']}"),
            "along_m": float(f"{m['al']}.{m['alf']}"), "file": path.name,
            "path": "/".join(rel)}


def build() -> pd.DataFrame:
    rows = []
    for run in sorted(p for p in ROOT.glob("*/*/run*") if p.is_dir()):
        tdir = run / "target"
        primary = {p.name for p in tdir.glob("*.png")} if tdir.is_dir() else set()
        alt = {p.name for p in (tdir / "extra_target_images_alt_deployment").glob("*.png")} \
            if tdir.is_dir() else set()
        for freq in ("990", "450"):
            for p in sorted((run / freq).glob("*.png")):
                r = parse(p)
                if r is None:
                    continue
                listed = freq == "990" and p.name in primary
                r["label"] = "target" if listed else ("unlisted" if tdir.is_dir()
                                                      else "unlabelled")
                r["deployment"] = "main"
                rows.append(r)
        # target images whose leg is missing from the run's 990/ folder, and the alt deployment
        have = {r["file"] for r in rows if r["path"].startswith("/".join(run.relative_to(ROOT)
                                                                          .parts))}
        for names, sub, dep in ((primary, tdir, "main"),
                                (alt, tdir / "extra_target_images_alt_deployment", "alt")):
            for n in sorted(names):
                if dep == "main" and n in have:
                    continue
                r = parse(sub / n)
                if r:
                    rows.append({**r, "label": "target", "deployment": dep})
    df = pd.DataFrame(rows)
    df["session"] = df.site + "/" + df.date + "/" + df.run
    df["used_in_paper"] = (df.freq_khz == 990) & (df.date != "8_apr_2022")
    return df


def main() -> None:
    df = build()
    OUT.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["path"]).to_csv(OUT / "bath_legs.csv", index=False)
    t = (df[df.freq_khz == 990].groupby(["site", "date", "run", "deployment", "label"]).size()
         .unstack(fill_value=0))
    t.to_csv(OUT / "bath_legs_summary.csv")
    print(t.to_string())


if __name__ == "__main__":
    main()
