"""P4.2b shard schema and identity tests.

Each test proves that the shard manifest schema refuses a specific
malformation. The point: a shard is infrastructure, not science; but
its identity must be locked tightly enough that no one can pass a
partial result as a complete experiment.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

try:
    from jsonschema import Draft202012Validator
except ImportError:
    pytest.skip("jsonschema not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "p4_shard_manifest.schema.json"


VALID_SHARD = {
    "shard_schema_version": "1.0.0",
    "protocol_id": "p4.2.oracle-baseline.v3",
    "manifest_digest": "sha256:" + "a" * 64,
    "git_sha": "a" * 40,
    "shard_id": "shard-arm-group-a",
    "shard_axis": "arms",
    "seed_subset": [180, 181, 182],
    "arm_subset": ["rstdp_mixed", "no_plasticity", "m_zero"],
    "task_streams": ["left_target", "right_target", "mixed"],
    "episodes_per_seed": 80,
    "partial": True,
    "non_confirmatory": False,
    "csv_sha256": "sha256:" + "b" * 64,
    "provenance_jsonl_sha256": "sha256:" + "c" * 64,
    "provenance_head_sha256": "sha256:" + "d" * 64,
    "environment_fingerprint": {"python": "3.13.0", "numpy": "2.0.0"},
    "started_at_utc": "2026-10-04T12:00:00Z",
    "finished_at_utc": "2026-10-04T12:15:00Z",
    "cells_in_shard": 27,
    "wall_clock_seconds": 378.2,
}


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _errors(validator, doc) -> list[str]:
    return [e.message for e in validator.iter_errors(doc)]


def _failed_props(validator, doc) -> set:
    """Имена полей, на которых схема отвергла документ (через error path).

    Сообщение jsonschema для `const` не содержит имени поля, поэтому ловить
    «именно это поле сломалось» надо по absolute_path, а не по тексту.
    """
    return {e.absolute_path[-1] for e in validator.iter_errors(doc)
            if len(e.absolute_path) > 0}


def _mutate(fn):
    d = copy.deepcopy(VALID_SHARD)
    fn(d)
    return d


# --- 1. Valid shard passes ---

def test_valid_shard_manifest_passes(validator):
    validator.validate(VALID_SHARD)


def test_valid_seed_axis_shard_passes(validator):
    """P4.2c: shard_axis="seeds" теперь валиден — seed-axis shard держит ВСЕ
    arms и ВСЕ streams и режет лишь seeds. Схема это разрешает; точное разбиение
    проверяет код coverage, не схема."""
    doc = copy.deepcopy(VALID_SHARD)
    doc["shard_axis"] = "seeds"
    doc["shard_id"] = "shard-seed-group-a"
    doc["seed_subset"] = [180, 181, 182, 183, 184, 185, 186, 187, 188, 189]
    doc["arm_subset"] = ["rstdp_mixed", "no_plasticity", "m_zero",
                        "weight_shuffled_frozen", "direction_shuffled_frozen", "oracle_reflex"]
    doc["cells_in_shard"] = len(doc["arm_subset"]) * len(doc["seed_subset"]) * len(doc["task_streams"])
    validator.validate(doc)


def test_schema_itself_is_valid():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


# --- 2. Missing git_sha fails ---

def test_missing_git_sha_fails(validator):
    doc = _mutate(lambda d: d.pop("git_sha"))
    assert _errors(validator, doc)


# --- 3. Missing manifest_digest fails ---

def test_missing_manifest_digest_fails(validator):
    doc = _mutate(lambda d: d.pop("manifest_digest"))
    assert _errors(validator, doc)


# --- 4. partial=false on a shard fails ---

def test_partial_false_on_shard_is_rejected(validator):
    """Shard по определению partial=true. False означает «это полный результат»,
    а полный результат — это merged manifest, не shard."""
    doc = _mutate(lambda d: d.__setitem__("partial", False))
    assert _errors(validator, doc)
    assert "partial" in _failed_props(validator, doc)


# --- 5. Empty arm subset fails ---

def test_empty_arm_subset_fails(validator):
    doc = _mutate(lambda d: d.__setitem__("arm_subset", []))
    assert _errors(validator, doc)


# --- 6. Empty seed subset fails ---

def test_empty_seed_subset_fails(validator):
    doc = _mutate(lambda d: d.__setitem__("seed_subset", []))
    assert _errors(validator, doc)


# --- 7. Invalid shard axis fails ---

def test_invalid_shard_axis_fails(validator):
    """Разрешены только arms и seeds. Значение 'episodes' (резать episodes)
    схемой отвергается: делить прогон по episodes нельзя — это меняет научный план."""
    doc = _mutate(lambda d: d.__setitem__("shard_axis", "episodes"))
    assert _errors(validator, doc)
    assert "shard_axis" in _failed_props(validator, doc)


# --- 8. non_confirmatory=true on a shard fails ---

def test_non_confirmatory_shard_is_rejected(validator):
    """non_confirmatory=true означает «этот shard нельзя merge-ить в научный
    результат». Схема закрепляет const false."""
    doc = _mutate(lambda d: d.__setitem__("non_confirmatory", True))
    assert _errors(validator, doc)
    assert "non_confirmatory" in _failed_props(validator, doc)


# --- Additional identity checks ---

def test_unknown_field_is_rejected(validator):
    doc = _mutate(lambda d: d.__setitem__("__injection", "payload"))
    assert _errors(validator, doc)


def test_duplicate_arms_in_subset_fails(validator):
    doc = _mutate(lambda d: d.__setitem__("arm_subset", ["rstdp_mixed", "rstdp_mixed"]))
    errors = _errors(validator, doc)
    assert errors


def test_git_sha_wrong_format_fails(validator):
    doc = _mutate(lambda d: d.__setitem__("git_sha", "short"))
    assert _errors(validator, doc)


def test_manifest_digest_wrong_format_fails(validator):
    doc = _mutate(lambda d: d.__setitem__("manifest_digest", "md5:abc"))
    assert _errors(validator, doc)
