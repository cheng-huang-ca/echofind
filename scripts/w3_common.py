"""Shared helpers for W3 scripts: paths and run metadata (config, git commit, data hash)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "w3"
FIG = ROOT / "reports" / "figures" / "w3"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def manifest_hash() -> str:
    """Hash of the raw-data manifest if present, else of the UATD annotation file list."""
    m = RAW / "MANIFEST.json"
    if m.exists():
        return file_hash(m)
    names = sorted(p.relative_to(RAW).as_posix() for p in (RAW / "uatd").rglob("*.xml"))
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


def write_meta(name: str, config: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"script": name, "git_commit": git_commit(), "data_manifest": manifest_hash(),
            "config": config}
    (OUT / f"{name}_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta
