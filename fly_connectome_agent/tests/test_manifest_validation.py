"""P0 test: manifest schema validation.

Validates that:
- examples/manifest.valid.json passes the ExperimentManifest schema.
- examples/manifest.invalid.json fails the ExperimentManifest schema.
- Cross-field validation: quantiles.low < quantiles.high, w_exc_max > 0, w_inh_max > 0.

No SNN runtime is involved.
"""
import json
from pathlib import Path

import pytest

try:
    from jsonschema import Draft202012Validator, ValidationError
except ImportError as e:
    pytest.skip(
        "jsonschema not installed; run `pip install jsonschema`",
        allow_module_level=True,
    )

from fly_connectome_agent.src.science.manifest.cross_field import (
    cross_field_errors,
    validate_cross_field,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "experiment_manifest.schema.json"
VALID_PATH = ROOT / "examples" / "manifest.valid.json"
INVALID_PATH = ROOT / "examples" / "manifest.invalid.json"


@pytest.fixture(scope="module")
def schema():
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def validator(schema):
    return Draft202012Validator(schema)


def test_valid_manifest_passes(validator):
    with VALID_PATH.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    validator.validate(doc)  # raises ValidationError on failure


def test_invalid_manifest_fails(validator):
    with INVALID_PATH.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    with pytest.raises(ValidationError):
        validator.validate(doc)


def test_invalid_manifest_rejects_biological_claim(validator):
    with INVALID_PATH.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    errors = list(validator.iter_errors(doc))
    assert any(
        "biological_plausibility" in str(e.message)
        or "biological_target" in str(e.message)
        for e in errors
    ), "schema must reject biological_plausibility claim and biological_target status"


def test_valid_manifest_quantiles_ordered():
    """Valid manifest must have quantiles.low < quantiles.high."""
    with VALID_PATH.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    q = doc["weight_mapping"]["quantiles"]
    assert q["low"] < q["high"], "quantiles.low must be < quantiles.high"


def test_valid_manifest_weight_ranges_positive():
    """Valid manifest must have positive w_exc_max and w_inh_max."""
    with VALID_PATH.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    wm = doc["weight_mapping"]
    assert wm["w_exc_max"] > 0, "w_exc_max must be > 0"
    assert wm["w_inh_max"] > 0, "w_inh_max must be > 0"


def test_cross_field_validation_quantiles():
    """Cross-field validation: quantiles.low < quantiles.high.

    JSON Schema не умеет сравнивать два числа между собой, поэтому правило
    живёт в src/science/manifest/cross_field.py, а не в схеме.
    """
    bad = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    bad["weight_mapping"]["quantiles"] = {"low": 0.95, "high": 0.05}  # reversed
    problems = cross_field_errors(bad)
    assert any("quantiles" in p.lower() for p in problems), \
        "reversed quantiles must be rejected"
    with pytest.raises(ValueError):
        validate_cross_field(bad)


def test_cross_field_validation_weight_ranges():
    """Schema-level validation: w_exc_max > 0 and w_inh_max > 0."""
    bad = json.loads(VALID_PATH.read_text(encoding="utf-8"))
    bad["weight_mapping"]["w_exc_max"] = -0.1
    bad["weight_mapping"]["w_inh_max"] = -0.2
    validator = Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    errors = list(validator.iter_errors(bad))
    assert any("exclusiveMinimum" in str(e) or "w_exc_max" in str(e) or "w_inh_max" in str(e)
               for e in errors), "schema should reject non-positive w_exc_max/w_inh_max"


def test_schema_itself_is_valid():
    """The schema must itself be a valid Draft 2020-12 schema."""
    Draft202012Validator.check_schema(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))