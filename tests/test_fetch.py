"""Offline tests for the data fetcher: selection, caps and resume logic, no network."""

import hashlib

import pytest
import yaml

from echofind.data import fetch


def fake_lister(keys):
    def lister(endpoint, prefix):
        yield from ((k, s) for k, s in sorted(keys) if k.startswith(prefix))
    return lister


KEYS = [
    ("cruise/EK80/ComplexSamples-D1-T1.raw", 10),
    ("cruise/EK80/ComplexSamples-D1-T2.raw", 10),
    ("cruise/EK80/D20230721-T173135.raw", 5),
    ("cruise/EK80/D20230721-T173200.raw", 5),
    ("cruise/README_x.md", 1),
]


def test_select_first_n_matching_basenames():
    spec = {"endpoint": "https://bucket", "selections": [
        {"prefix": "cruise/EK80/", "pattern": r"^ComplexSamples-.*\.raw$", "count": 1},
        {"prefix": "cruise/EK80/", "pattern": r"^D\d{8}-T\d{6}\.raw$", "count": 2},
    ]}
    items = fetch.select_s3("noaa", spec, fake_lister(KEYS))
    assert [i.name for i in items] == [
        "ComplexSamples-D1-T1.raw", "D20230721-T173135.raw", "D20230721-T173200.raw"]
    assert items[0].url == "https://bucket/cruise/EK80/ComplexSamples-D1-T1.raw"


def test_plan_enforces_size_cap():
    config = {"noaa": {"s3": {"endpoint": "e", "selections": [
        {"prefix": "cruise/EK80/", "pattern": r"\.raw$", "count": 4}]}, "max_bytes": 20}}
    with pytest.raises(SystemExit, match="cap"):
        fetch.plan(config, lister=fake_lister(KEYS))


def test_plan_only_filters_sources():
    config = {"a": {"files": [{"url": "u", "name": "a.txt"}]},
              "b": {"files": [{"url": "u", "name": "b.txt"}]}}
    assert [i.name for i in fetch.plan(config, only=["b"])] == ["b.txt"]


def test_config_is_well_formed():
    config = yaml.safe_load(fetch.CONFIG.read_text())
    assert set(config) == {"bath", "uatd", "noaa", "mbari"}
    for spec in config.values():
        assert "license" in spec
        for f in spec.get("files", []):
            assert f["url"].startswith("https://") and f["name"]


def test_fetch_skips_complete_file_and_checks_md5(tmp_path, monkeypatch):
    data = b"echo" * 100
    (tmp_path / "x.bin").write_bytes(data)
    item = fetch.Item("t", "https://unused", "x.bin", len(data), hashlib.md5(data).hexdigest())
    monkeypatch.setattr(fetch, "_open", lambda *a, **k: pytest.fail("should not download"))
    assert fetch.fetch(item, tmp_path) == "downloaded"
    bad = fetch.Item("t", "https://unused", "x.bin", len(data), "0" * 32)
    with pytest.raises(SystemExit, match="md5"):
        fetch.fetch(bad, tmp_path)
