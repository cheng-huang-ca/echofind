"""The field-collection metadata schema (W6) accepts the example and rejects the gaps W1-W5
found: no raw data, no IMU, a body without ground truth, an unblinded label."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

CFG = Path(__file__).parents[1] / "configs"
SCHEMA = json.loads((CFG / "field_metadata.schema.json").read_text(encoding="utf-8"))
EXAMPLE = json.loads((CFG / "field_metadata.example.json").read_text(encoding="utf-8"))
V = Draft202012Validator(SCHEMA)


def test_schema_is_valid_and_example_passes():
    Draft202012Validator.check_schema(SCHEMA)
    assert list(V.iter_errors(EXAMPLE)) == []


def _broken(path, value=None, delete=False):
    rec = copy.deepcopy(EXAMPLE)
    obj = rec
    for k in path[:-1]:
        obj = obj[k]
    if delete:
        del obj[path[-1]]
    else:
        obj[path[-1]] = value
    return rec


@pytest.mark.parametrize("path,value,delete", [
    (["device", "raw_recording"], False, False),          # display images only
    (["device", "imu"], None, True),                       # no tilt logging
    (["placements", 1, "ground_truth"], None, True),       # body without confirmed position
    (["labels", 0, "blind"], False, False),                # labeller saw the placement log
    (["site", "bottom_type"], None, True),                 # bottom not logged
    (["placements", 1, "target", "type"], "dummy", False), # unknown target type
])
def test_schema_rejects_known_gaps(path, value, delete):
    assert list(V.iter_errors(_broken(path, value, delete)))
