"""Idempotent downloader for the EchoFind datasets in configs/data.yaml.

Run `python -m echofind.data.fetch --only uatd noaa` (or `make data`). A file that is already
present with the expected size (and md5, when known) is skipped, partial downloads resume with
HTTP Range requests, and archives are extracted once and then deleted to save disk.
Every run rewrites data/raw/MANIFEST.json with what is on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree

import yaml

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs" / "data.yaml"
RAW = REPO / "data" / "raw"
CHUNK = 8 * 1024 * 1024
USER_AGENT = "echofind-fetch/0.1 (+https://github.com/cheng-huang-ca/echofind)"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


@dataclass
class Item:
    source: str
    url: str
    name: str
    size: int | None = None
    md5: str | None = None
    extract: bool = False


def _open(url: str, start: int = 0):
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def list_s3(endpoint: str, prefix: str):
    """Yield (key, size) for an anonymous S3 bucket, in key order, following pagination."""
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        with _open(f"{endpoint}/?{urllib.parse.urlencode(query)}") as resp:
            root = ElementTree.fromstring(resp.read())
        for c in root.iter(f"{S3_NS}Contents"):
            yield c.findtext(f"{S3_NS}Key"), int(c.findtext(f"{S3_NS}Size"))
        if root.findtext(f"{S3_NS}IsTruncated") != "true":
            return
        token = root.findtext(f"{S3_NS}NextContinuationToken")


def select_s3(source: str, spec: dict, lister=list_s3) -> list[Item]:
    """Resolve S3 selections to items: first `count` keys per prefix whose basename matches."""
    endpoint = spec["endpoint"]
    items: list[Item] = []
    for sel in spec["selections"]:
        pattern = re.compile(sel["pattern"])
        found = 0
        for key, size in lister(endpoint, sel["prefix"]):
            base = key.rsplit("/", 1)[-1]
            if pattern.search(base):
                items.append(Item(source, f"{endpoint}/{urllib.parse.quote(key)}", base, size))
                found += 1
                if found >= sel["count"]:
                    break
        if found < sel["count"]:
            print(f"  ! {source}: only {found}/{sel['count']} matches under {sel['prefix']}")
    return items


def plan(config: dict, only: list[str] | None = None, lister=list_s3) -> list[Item]:
    items: list[Item] = []
    for source, spec in config.items():
        if only and source not in only:
            continue
        if "files" in spec:
            items += [Item(source, f["url"], f["name"], f.get("size"), f.get("md5"),
                           f.get("extract", False)) for f in spec["files"]]
        if "s3" in spec:
            chosen = select_s3(source, spec["s3"], lister)
            cap = spec.get("max_bytes")
            total = sum(i.size or 0 for i in chosen)
            if cap and total > cap:
                raise SystemExit(f"{source}: selection is {total / 1e9:.2f} GB, over its "
                                 f"{cap / 1e9:.2f} GB cap; lower the counts in {CONFIG.name}")
            items += chosen
    return items


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _done_marker(item: Item, dest: Path) -> Path:
    return dest / f".{item.name}.extracted"


def fetch(item: Item, dest: Path) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / item.name
    if item.extract and _done_marker(item, dest).exists():
        return "extracted earlier"
    have = target.stat().st_size if target.exists() else 0
    if item.size is not None and have > item.size:
        target.unlink()
        have = 0
    if item.size is None or have < item.size:
        with _open(item.url, have) as resp, target.open("ab" if have else "wb") as out:
            if have and resp.status != 206:  # server ignored Range: start over
                out.truncate(0)
            shutil.copyfileobj(resp, out, CHUNK)
    if item.size is not None and target.stat().st_size != item.size:
        raise SystemExit(f"{item.name}: size {target.stat().st_size} != expected {item.size}")
    if item.md5 and md5sum(target) != item.md5:
        target.unlink()
        raise SystemExit(f"{item.name}: md5 mismatch; deleted, run again")
    if item.extract:
        with zipfile.ZipFile(target) as z:
            z.extractall(dest / Path(item.name).stem)
        _done_marker(item, dest).touch()
        target.unlink()  # keep the disk budget: the archive is no longer needed
        return "downloaded and extracted"
    return "downloaded"


def write_manifest(items: list[Item]) -> None:
    rows = []
    for i in items:
        dest = RAW / i.source
        present = (dest / i.name).exists() or _done_marker(i, dest).exists()
        rows.append({**asdict(i), "present": present})
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "MANIFEST.json").write_text(json.dumps(rows, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", help="sources to fetch, e.g. uatd noaa")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    args = ap.parse_args(argv)

    config = yaml.safe_load(CONFIG.read_text())
    unknown = set(args.only or []) - set(config)
    if unknown:
        ap.error(f"unknown sources: {sorted(unknown)}; choose from {sorted(config)}")
    items = plan(config, args.only)
    total = sum(i.size or 0 for i in items)
    print(f"{len(items)} files, {total / 1e9:.2f} GB")
    if args.dry_run:
        for i in items:
            print(f"  {i.source:6} {(i.size or 0) / 1e6:9.1f} MB  {i.name}")
        return 0
    free = shutil.disk_usage(RAW.parent if RAW.parent.exists() else REPO).free
    if total > free:
        print(f"  ! only {free / 1e9:.1f} GB free; fetch fewer sources with --only")
    for i in items:
        print(f"  {i.source:6} {i.name}: {fetch(i, RAW / i.source)}", flush=True)
    write_manifest(items)
    return 0


if __name__ == "__main__":
    sys.exit(main())
