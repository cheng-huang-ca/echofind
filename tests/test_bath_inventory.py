"""The Bath inventory groups images by folder and flags those under target/."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "w1_bath_inventory", Path(__file__).parents[1] / "scripts" / "w1_bath_inventory.py")
inv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inv)


def test_inventory_counts_target_folders():
    paths = [
        "root/site_a/990/12_may_2021/img1.png",
        "root/site_a/990/12_may_2021/img2.png",
        "root/site_a/990/12_may_2021/target/t1.png",
        "root/site_b/450/8_apr_2022/img1.jpg",
        "root/README.txt",
    ]
    df = inv.inventory(paths)
    t = df[df.subset == "target"].iloc[0]
    assert t.images == 1 and t.freq_khz == 990 and t.survey_date == "12_may_2021"
    assert df[df.subset == "other"].images.sum() == 3
    assert set(df.freq_khz) == {450, 990}
