"""P6 test: TeacherAnalysisProposal schema boundary enforcement.

Все 13 требований review проверены как refusals: если boundary перестаёт
держаться, тест краснеет. valid-пример обязан проходить, а любая попытка
наделить LLM правом на действие — закончится ошибкой валидации.

Схема использует additionalProperties:false на каждом уровне, поэтому
запрещённые поля (action, weight_update, manifest_override) отвергаются не
списком «не разрешено», а структурой: их нет в allowed properties.

Этот файл НЕ трогает SNN/LIF/STDP/GateKeeper/provenance. Чисто формальный
контракт формата.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

try:
    from jsonschema import Draft202012Validator, ValidationError
except ImportError as exc:
    pytest.skip("jsonschema not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "teacher_analysis_proposal.schema.json"
VALID_EXAMPLE = ROOT / "examples" / "teacher_analysis_proposal.valid.json"


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validator(schema) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.fixture()
def valid_doc() -> dict:
    return json.loads(VALID_EXAMPLE.read_text(encoding="utf-8"))


def _mutate(doc: dict, fn) -> dict:
    """Deep copy, apply mutation, return for validation."""
    d = copy.deepcopy(doc)
    fn(d)
    return d


def _refused(validator, doc, mutate):
    """Assert that the mutation causes at least one schema error."""
    candidate = _mutate(doc, mutate)
    errors = list(validator.iter_errors(candidate))
    assert errors, "schema accepted a document that must be rejected"
    return errors


# --- 1. Valid advisory proposal passes --------------------------------------

def test_valid_example_passes(validator, valid_doc):
    """Базовая проверка: эталонный заполненный proposal валиден."""
    validator.validate(valid_doc)


def test_schema_is_valid_draft_2020_12(schema):
    Draft202012Validator.check_schema(schema)


# --- 2–6. Safety const flags ------------------------------------------------

def test_may_modify_snn_true_is_rejected(validator, valid_doc):
    """may_modify_snn зафиксирован как const false: LLM не трогает SNN."""
    _refused(validator, valid_doc,
             lambda d: d["safety"].__setitem__("may_modify_snn", True))


def test_may_execute_action_true_is_rejected(validator, valid_doc):
    """may_execute_action: const false."""
    _refused(validator, valid_doc,
             lambda d: d["safety"].__setitem__("may_execute_action", True))


def test_may_write_provenance_true_is_rejected(validator, valid_doc):
    """may_write_provenance: const false."""
    _refused(validator, valid_doc,
             lambda d: d["safety"].__setitem__("may_write_provenance", True))


def test_policy_eligible_true_is_rejected(validator, valid_doc):
    """policy_eligible: const false; proposal не меняет политику."""
    _refused(validator, valid_doc,
             lambda d: d["safety"].__setitem__("policy_eligible", True))


def test_scientific_claim_eligible_true_is_rejected(validator, valid_doc):
    """scientific_claim_eligible: const false; LLM не порождает научных выводов."""
    _refused(validator, valid_doc,
             lambda d: d["safety"].__setitem__("scientific_claim_eligible", True))


# --- 7–9. Forbidden top-level fields ----------------------------------------

def test_field_action_is_rejected(validator, valid_doc):
    """Поле 'action' отсутствует в properties, additionalProperties:false ловит."""
    errors = _refused(validator, valid_doc,
                      lambda d: d.__setitem__("action", {"type": "turn_left"}))
    assert any("action" in str(e.message) or "Additional" in str(e.message)
               for e in errors)


def test_field_weight_update_is_rejected(validator, valid_doc):
    """Поле 'weight_update' не входит в схему."""
    _refused(validator, valid_doc,
             lambda d: d.__setitem__("weight_update", {"delta": 0.1}))


def test_field_manifest_override_is_rejected(validator, valid_doc):
    """Поле 'manifest_override' не входит в схему."""
    _refused(validator, valid_doc,
             lambda d: d.__setitem__("manifest_override", {"episodes": 200}))


# --- 10–11. Missing required source_bundle fields ---------------------------

def test_missing_manifest_digest_is_rejected(validator, valid_doc):
    """manifest_digest обязателен: без него proposal не привязан к протоколу."""
    _refused(validator, valid_doc,
             lambda d: d["source_bundle"].pop("manifest_digest"))


def test_missing_result_bundle_hash_is_rejected(validator, valid_doc):
    """result_bundle_hash обязателен: без него не доказано, какой bundle прочитан."""
    _refused(validator, valid_doc,
             lambda d: d["source_bundle"].pop("result_bundle_hash"))


# --- 12. must_use_new_seed_set must be const true ----------------------------

def test_must_use_new_seed_set_false_is_rejected(validator, valid_doc):
    """Постоянно true: LLM не может разрешить переиспользование seed'ов."""
    _refused(validator, valid_doc,
             lambda d: d["analysis"]["next_experiment_suggestions"][0]
             .__setitem__("must_use_new_seed_set", False))


# --- 13. Unknown extra fields rejected ---------------------------------------

def test_unknown_top_level_field_is_rejected(validator, valid_doc):
    """additionalProperties:false: любое незнакомое поле — ошибка."""
    _refused(validator, valid_doc,
             lambda d: d.__setitem__("__unknown_injection", "payload"))


def test_unknown_field_in_safety_is_rejected(validator, valid_doc):
    """Safety-объект тоже закрыт от дополнительных полей."""
    _refused(validator, valid_doc,
             lambda d: d["safety"].__setitem__("may_bypass_review", True))


def test_unknown_field_in_analysis_is_rejected(validator, valid_doc):
    """Analysis не принимает лишнего."""
    _refused(validator, valid_doc,
             lambda d: d["analysis"].__setitem__("shell_command", "rm -rf /"))
