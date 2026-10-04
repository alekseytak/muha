"""Требования review к P4.2.v2, проверенные кодом, а не абзацем в README.

Review выдал шесть проверяемых пунктов (protocol_id, seed-множество, сохранённый
научный план, состав bundle, свежий provenance-путь, поля преемственности).
Каждый из них здесь написан как отказ: если пункт перестаёт выполняться, тест
краснеет. Тест, который не может упасть, — это пересказ требования.

Три вещи здесь специально проверены против литералов, а не против манифеста:
несечение с 0..119 и 900..903, имя протокола, девять ролей bundle. Если сверять
манифест с самим собой, правка манифеста вместе с правкой теста проходила бы
незамеченной — ровно та тавтология, из-за которой v1 получил отдельный якорь в
коде.

Ни один тест не запускает confirmatory-прогон: v2 авторизован, но сам прогон
не стартует из тестов; проверяется целостность авторизации и неизменность
научного протокола.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import p4_2_protocol as proto  # noqa: E402
import check_p4_2_oracle_gate as gate  # noqa: E402
import collect_p4_2_bundle as bundle  # noqa: E402
import p4_2_burned_seeds as burned  # noqa: E402
import run_p4_2_oracle as runner  # noqa: E402

MANIFEST_PATH = REPO / "fly_connectome_agent/manifests/p4_2_oracle_confirmatory_v2.json"
EVIDENCE_PATH = REPO / "fly_connectome_agent/manifests/p4_2_v1_burned_seeds.json"

# Литералы review, не выводимые из манифеста.
V2_ID = "p4.2.oracle-baseline.v2"
V1_ID = "p4.2.oracle-baseline.v1"
V1_STATUS = "aborted_infrastructure_performance_defect"
CONFIRMATORY = list(range(120, 180))
BURNED_V1 = list(range(60, 120))
P4_1_USED = list(range(0, 60))
FIXTURE_PROBE = list(range(900, 904))
REVIEW_ROLES = {
    "run_csv", "csv_sidecar", "provenance_jsonl", "provenance_head_witness",
    "gate_stdout", "gate_verdict_json", "per_seed_outcomes",
    "environment_fingerprint", "file_hashes_manifest",
}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return proto.load(MANIFEST_PATH)


def _tampered(manifest: dict, mutate) -> dict:
    """Копия без служебного _digest: validate() обязана поймать правку сама."""
    m = copy.deepcopy(manifest)
    m.pop("_digest", None)
    mutate(m)
    return m


def _refuses(manifest: dict, mutate, needle: str) -> None:
    with pytest.raises(proto.ProtocolError) as exc:
        proto.validate(_tampered(manifest, mutate))
    assert needle in str(exc.value), str(exc.value)


def _entry(manifest: dict, role_or_id: str) -> dict:
    for art in manifest["result_bundle"]["artifacts"]:
        if art["role"] == role_or_id:
            return art
    for other in manifest["seed_sets"]["disjoint_from"]:
        if other["id"] == role_or_id:
            return other
    raise KeyError(role_or_id)


# --- 1. идентичность и преемственность (пункт 6 review) ---------------------

def test_protocol_id_is_the_name_the_review_asked_for(manifest):
    assert manifest["protocol_id"] == V2_ID
    assert manifest["pre_registered"] is True


def test_supersedes_fields_are_exactly_the_values_from_the_review(manifest):
    assert manifest["supersedes_protocol_id"] == V1_ID
    assert manifest["supersedes_status"] == V1_STATUS


def test_the_inherited_status_is_taken_from_the_code_not_from_the_file(manifest):
    """Итог закрытого прогона задаёт реестр в коде. Манифест-преемник обязан его
    повторить, а не назначить: иначе «aborted» превращается в «completed» одной
    правкой в файле, который ничего не решает."""
    assert manifest["supersedes_status"] == proto.FROZEN_PROTOCOLS[V1_ID]["status"]
    assert proto.frozen_entry(V1_ID)["manifest"] == \
        "fly_connectome_agent/manifests/p4_2_oracle_confirmatory.json"


def test_dropping_the_lineage_pair_is_refused_by_code_and_schema(manifest):
    _refuses(manifest, lambda m: (m.pop("supersedes_protocol_id", None),
                                  m.pop("supersedes_status", None)),
             "обязан нести supersedes_protocol_id")
    errors = _schema_errors(_tampered(manifest, lambda m: (
        m.pop("supersedes_protocol_id"), m.pop("supersedes_status"))))
    assert len(errors) == 2, errors
    assert any("supersedes_protocol_id" in e for e in errors), errors
    assert any("supersedes_status" in e for e in errors), errors


def test_the_lineage_fields_are_a_pair_not_two_switches(manifest):
    """Одно поле без другого не читается: «мы кого-то замещаем» без ответа «с каким
    итогом» — это снова разрешение на повторный запуск закрытого протокола."""
    _refuses(manifest, lambda m: m.pop("supersedes_status"), "это пара")
    _refuses(manifest, lambda m: m.pop("supersedes_protocol_id"), "это пара")


def test_supersedes_status_cannot_be_softened_by_the_successor(manifest):
    _refuses(manifest, lambda m: m.__setitem__("supersedes_status", "superseded"),
             "против 'aborted_infrastructure_performance_defect' в коде")
    _refuses(manifest, lambda m: m.__setitem__("supersedes_status", "completed"),
             "итог закрытого")


def test_supersedes_cannot_name_an_unknown_protocol_or_itself(manifest):
    _refuses(manifest, lambda m: m.__setitem__("supersedes_protocol_id", "p4.2.oracle-baseline.v0"),
             "нет в реестре замороженных протоколов")
    _refuses(manifest, lambda m: m.__setitem__("supersedes_protocol_id", V2_ID),
             "не может supersede сам себя")


def test_v1_remains_valid_without_lineage(manifest):
    """Требование появилось в v2 и не имеет права задним числом сделать v1
    невалидным: его digest заперт в коде, а файл обязан читаться и через год."""
    v1 = proto.load(proto.V1_MANIFEST)
    assert v1["protocol_id"] == V1_ID
    assert "supersedes_protocol_id" not in v1 and "result_bundle" not in v1
    assert proto.digest_of_file(proto.V1_MANIFEST) == proto.FROZEN_PROTOCOLS[V1_ID]["digest"]
    assert _schema_errors(v1) == []
    # иерархия проверок: v2 без lineage — две ошибки (схема), v1 — ноль.
    assert len(_schema_errors(_tampered(manifest, lambda m: (
        m.pop("supersedes_protocol_id"), m.pop("supersedes_status"))))) == 2


def test_closed_v1_is_recognised_as_closed_not_as_wrong(manifest):
    """Разница между «это не тот протокол» и «этот протокол закрыт» важна для
    отказа: вторая формулировка объясняет, почему v1 нельзя допровадить."""
    v1_digest = proto.FROZEN_PROTOCOLS[V1_ID]["digest"]
    assert proto.require_frozen(v1_digest, where="тест", protocol_id=V1_ID) == v1_digest
    with pytest.raises(proto.ProtocolError, match="закрытому протоколу не считают"):
        proto.require_frozen(v1_digest, where="тест")
    assert manifest["_digest"] == proto.FROZEN_PROTOCOLS[V2_ID]["digest"]


def _schema_errors(manifest: dict) -> list[str]:
    """Только форма, без кросс-полей. Служебное _digest, которое добавляет load(),
    схема отвергает как незнакомое поле, поэтому перед проверкой оно снимается."""
    candidate = copy.deepcopy(manifest)
    candidate.pop("_digest", None)
    schema = json.loads(proto.SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return sorted(f"{list(e.path)}: {e.message}" for e in validator.iter_errors(candidate))


# --- 2. seed-множество (пункт 2 review) -------------------------------------

def test_confirmatory_set_records_both_the_range_and_the_count(manifest):
    spec = manifest["seed_sets"]["confirmatory"]
    assert spec["range"] == [120, 179]
    assert spec["count"] == 60
    assert proto.confirmatory_seeds(manifest) == CONFIRMATORY
    assert len(proto.confirmatory_seeds(manifest)) == spec["count"]


def test_confirmatory_set_avoids_every_seed_the_review_listed(manifest):
    """Сверка с литералами, а не с disjoint_from из того же файла: манифест мог бы
    объявить неудобное множество и остаться правым сам с собой."""
    chosen = set(proto.confirmatory_seeds(manifest))
    assert not chosen & set(P4_1_USED), f"пересечение с P4.1: {sorted(chosen & set(P4_1_USED))[:5]}"
    assert not chosen & set(BURNED_V1), "пересечение с сожжённым v1"
    assert not chosen & set(FIXTURE_PROBE), "пересечение с fixture-пробами"
    assert not chosen & set(proto.probe_seeds(manifest))


def test_the_burned_record_still_matches_the_aborted_journal_on_disk():
    """Запись в git сверяется с журналом, из которого она получена. Это единственная
    проверка в файле, которая читает 17 МБ прерванного прогона, — и именно она
    отделяет «мы так написали» от «так выглядит журнал»."""
    record = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert record["protocol_id"] == V1_ID and record["status"] == V1_STATUS
    scans = [burned.scan_provenance(REPO / j["source"]["path"]) for j in record["journals"]]
    rebuilt = burned.build_record(scans, protocol_id=record["protocol_id"],
                                  status=record["status"], note=record["note"])
    assert rebuilt["burned_seed_set"] == record["burned_seed_set"]
    assert rebuilt["burned_seed_set"]["values"] == BURNED_V1
    assert rebuilt["coverage"] == record["coverage"]
    assert record["coverage"]["evidence_complete"] is True


def test_the_declared_disjoint_set_agrees_with_the_record(manifest):
    entry = _entry(manifest, "p4_2_v1_aborted_run")
    record = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert entry["range"] == record["burned_seed_set"]["range"] == [60, 119]
    assert entry["count"] == record["burned_seed_set"]["count"] == 60
    assert entry["evidence"] == "fly_connectome_agent/manifests/p4_2_v1_burned_seeds.json"


def test_narrowing_the_disjoint_range_below_the_record_is_refused(manifest):
    """Единственный способ «освежить» сожжённые seed'ы — переписать несечение.
    59 seed'ов вместо 60 внутренне согласованы (диапазон == count), поэтому ловить
    их обязан именно пересчёт по записи-доказательству."""
    def drop(m):
        e = _entry(m, "p4_2_v1_aborted_run")
        e["range"], e["count"] = [60, 118], 59
    _refuses(manifest, drop, "разошёлся с фактом журнала")


def test_dropping_the_evidence_pointer_is_refused(manifest):
    """Без evidence проверка вырождается в сравнение двух строк манифеста, и снятие
    указателя было бы дешевле правки диапазона."""
    def drop(m):
        _entry(m, "p4_2_v1_aborted_run").pop("evidence")
    _refuses(manifest, drop, "поле evidence снято")


def test_evidence_must_come_from_another_protocol(manifest, tmp_path):
    """Запись, «подтверждающая» несечение собственного протокола, — не свидетельство:
    это пересказ того же множества. Проверяется на подставной записи с protocol_id
    самого v2."""
    forged = tmp_path / "burned_self.json"
    record = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    record["protocol_id"] = V2_ID
    forged.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    def repoint(m):
        _entry(m, "p4_2_v1_aborted_run")["evidence"] = str(forged)
    _refuses(manifest, repoint, "доказательство получено из прогона того же протокола")


def test_missing_evidence_file_is_refused(manifest):
    def rename(m):
        _entry(m, "p4_2_v1_aborted_run")["evidence"] = \
            "fly_connectome_agent/manifests/p4_2_v1_burned_seeds_absent.json"
    _refuses(manifest, rename, "не найдена")


# --- 3. научный план не двигался (пункт 3 review) ---------------------------

def test_frozen_plan_carries_the_reviewed_science_literals(manifest):
    assert len(manifest["arms"]) == 6
    assert manifest["task_streams"] == ["left_target", "right_target", "mixed"]
    assert manifest["episodes_per_seed"] == 80
    assert manifest["primary_metric"] == "mixed_min_half_success"
    st = manifest["statistical_test"]
    assert st["method"] == "paired_exact_sign_test"
    assert st["unit_of_analysis"] == "seed"
    assert st["multiple_comparison_policy"] == "Holm"
    assert st["family"] == ["H1", "H2", "H3", "H4"]
    h5 = next(h for h in manifest["hypotheses"] if h["id"] == "H5")
    assert h5["kind"] == "characterization"
    assert h5["direction"] == "difference"
    assert next(a for a in manifest["arms"] if a["arm_id"] == "oracle_reflex")["gate"] == "none"


def test_v2_changed_nothing_in_the_scientific_plan_of_v1(manifest):
    """Смысл замены — инфраструктура, поэтому научная часть обязана совпасть с v1
    побайтово по полям. Расхождение здесь означает, что под видом нового run
    протащили другой эксперимент."""
    v1 = proto.load(proto.V1_MANIFEST)
    for key in ("arms", "task_streams", "episodes_per_seed", "primary_metric",
                "primary_metric_definition", "secondary_metrics", "hypotheses",
                "statistical_test", "provenance", "decision_rule_after_p42", "claims_allowed"):
        assert manifest[key] == v1[key], f"v2 двигает научное поле {key!r}"
    op_new, op_old = manifest["operating_point"], v1["operating_point"]
    assert op_new["values"] == op_old["values"], "рабочая точка сдвинута"
    assert op_new["overrides"] == op_old["overrides"], "override'ы разъехались"
    assert set(manifest) - set(v1) == {"supersedes_protocol_id", "supersedes_status",
                                       "result_bundle"}, "v2 принёс лишние поля"


def test_the_only_differences_are_the_replacement_itself(manifest):
    v1 = proto.load(proto.V1_MANIFEST)
    v1.pop("_digest")
    new = copy.deepcopy(manifest)
    new.pop("_digest")
    changed = sorted(k for k in set(new) | set(v1) if new.get(k) != v1.get(k))
    assert changed == ["created_at", "not_in_scope", "operating_point", "protocol_id",
                       "result_bundle", "run_budget", "seed_sets", "supersedes_protocol_id",
                       "supersedes_status"], changed
    # operating_point: разница ровно в добавленном note, значения не сдвинуты;
    # not_in_scope: v2 обязан только добавлять запреты, но не снимать их.
    assert set(new["operating_point"]) - set(v1["operating_point"]) == {"note"}
    assert new["operating_point"]["values"] == v1["operating_point"]["values"]
    assert set(v1["not_in_scope"]) <= set(new["not_in_scope"])
    assert manifest["run_budget"]["cells"] == 6 * 60 * 3
    assert manifest["run_budget"]["episodes"] == manifest["run_budget"]["cells"] * 80


def test_operating_point_override_is_declared_not_silent(manifest):
    """Рабочая точка перенесена из v1; единственные отличия от дефолтов кода —
    eta и episodes, и оба объявлены с обоснованием. Тихий сдвиг любой другой
    величины ловится validate()."""
    assert set(manifest["operating_point"]["overrides"]) == {"eta", "episodes"}
    for field in manifest["operating_point"]["overrides"]:
        assert manifest["operating_point"]["override_rationale"][field].strip()
    _refuses(manifest, lambda m: m["operating_point"]["values"].__setitem__("sigma", 4.0),
             "sigma")
    # удаление поля из values — не «упрощение записи», а тихое наследование дефолта
    # из кода: протокол обязан объявлять рабочую точку целиком.
    _refuses(manifest, lambda m: m["operating_point"]["values"].pop("eta"),
             "объявлен не полностью")
    _refuses(manifest, lambda m: m["operating_point"]["overrides"].pop("eta"),
             "не объявлен в overrides")


# --- 4. состав result_bundle (пункт 4 review) -------------------------------

def test_bundle_declares_exactly_the_nine_reviewed_artifacts(manifest):
    bundle_meta = manifest["result_bundle"]
    assert {a["role"] for a in bundle_meta["artifacts"]} == REVIEW_ROLES
    assert set(proto.REQUIRED_BUNDLE_ROLES) == REVIEW_ROLES
    assert bundle_meta["directory"] == "var/p4_2_v2"
    assert bundle_meta["hashing"] == "sha256"
    assert bundle_meta["hashes_manifest"] == "bundle_hashes.json"


def test_every_artifact_is_sha256_required_and_produced_by_someone(manifest):
    for art in manifest["result_bundle"]["artifacts"]:
        assert art["hash"] == "sha256", art["role"]
        assert art["required"] is True, art["role"]
        assert art["producer"].strip(), art["role"]


def test_witness_is_named_by_the_writer_contract_and_lives_in_the_bundle(manifest):
    log = _entry(manifest, "provenance_jsonl")
    witness = _entry(manifest, "provenance_head_witness")
    assert witness["filename"] == log["filename"] + proto.HEAD_SUFFIX
    assert manifest["result_bundle"]["hashes_manifest"] in \
        [a["filename"] for a in manifest["result_bundle"]["artifacts"]]
    assert sorted(a["filename"] for a in manifest["result_bundle"]["artifacts"]) == \
        sorted(set(a["filename"] for a in manifest["result_bundle"]["artifacts"]))


def test_the_bundle_contract_is_enforced_against_the_code_registry(manifest):
    """Реестр ролей живёт в коде: вычеркнуть witness из манифеста — значит вычеркнуть
    единственное доказательство, что журнал не обрезан."""
    def drop(m):
        arts = m["result_bundle"]["artifacts"]
        arts.remove(_entry(m, "provenance_head_witness"))
    _refuses(manifest, drop, "не объявляет обязательный артефакт 'provenance_head_witness'")

    def rename_witness(m):
        _entry(m, "provenance_head_witness")["filename"] = "head.json"
    _refuses(manifest, rename_witness, "обязан называться")

    def relax(m):
        _entry(m, "gate_verdict_json")["required"] = False
    _refuses(manifest, relax, "required!=true")

    def extra_role(m):
        m["result_bundle"]["artifacts"].append(
            {"role": "my_notes", "filename": "notes.md", "hash": "sha256",
             "required": True, "producer": "я"})
    _refuses(manifest, extra_role, "которой нет в реестре кода")

    def into_the_common_dir(m):
        m["result_bundle"]["directory"] = "var"
    _refuses(manifest, into_the_common_dir, "обязана быть отдельной")

    def hashes_outside(m):
        m["result_bundle"]["hashes_manifest"] = "elsewhere.json"
    _refuses(manifest, hashes_outside, "отсутствует в списке")


def test_gate_refuses_to_write_its_verdict_outside_the_bundle(manifest, tmp_path):
    inside = REPO / "var/p4_2_v2"
    good = gate.bundle_gate(manifest, csv_paths=[str(inside / "run.csv")],
                            per_seed_out=str(inside / "per_seed_outcomes.csv"),
                            json_out=str(inside / "gate.verdict.json"))
    assert good and "var/p4_2_v2" in good[0]
    with pytest.raises(gate.Refused, match="не передан"):
        gate.bundle_gate(manifest, csv_paths=[str(inside / "run.csv")],
                         per_seed_out=str(inside / "per_seed_outcomes.csv"), json_out=None)
    with pytest.raises(gate.Refused, match="вне объявленного result_bundle"):
        gate.bundle_gate(manifest, csv_paths=[str(tmp_path / "run.csv")],
                         per_seed_out=str(inside / "per_seed_outcomes.csv"),
                         json_out=str(inside / "gate.verdict.json"))


def test_bundle_gate_is_a_no_op_for_protocols_that_never_declared_it(manifest):
    """v1 bundle не объявлял — его прогон уже посчитан по старому контракту, и
    заднее число не должно ломать разбор существующих артефактов."""
    v1 = proto.load(proto.V1_MANIFEST)
    assert gate.bundle_gate(v1, csv_paths=["var/p4_2_oracle.csv"],
                            per_seed_out=None, json_out=None) == []


# --- 5. сборка и проверка хешей ---------------------------------------------

@pytest.fixture()
def fake_bundle(tmp_path, monkeypatch):
    """Настоящий манифест, но bundle считается в tmp: девять файлов писать дорого,
    а проверять надо именно поведение сборщика, а не наличие результата."""
    tmp = tmp_path.resolve()
    monkeypatch.setattr(bundle, "REPO", tmp)
    m = copy.deepcopy(proto.load(MANIFEST_PATH))
    m.pop("_digest", None)
    base = tmp / m["result_bundle"]["directory"]
    base.mkdir(parents=True, exist_ok=True)
    for entry in m["result_bundle"]["artifacts"]:
        if entry["role"] != "file_hashes_manifest":
            (base / entry["filename"]).write_text(f"содержимое {entry['role']}\n",
                                                  encoding="utf-8")
    return m, tmp


def test_hashes_manifest_covers_every_declared_file_except_itself(fake_bundle):
    m, _ = fake_bundle
    record = bundle.hashes_manifest(m, "sha256:" + "a" * 64)
    listed = {e["role"] for e in record["files"]}
    assert listed == REVIEW_ROLES - {"file_hashes_manifest"}
    assert record["self_excluded"] == "bundle_hashes.json"
    assert record["bundle_digest"].startswith("sha256:")
    assert all(e["sha256"].startswith("sha256:") and e["bytes"] > 0 for e in record["files"])


def test_hashes_manifest_refuses_an_incomplete_bundle(fake_bundle):
    """required=true в манифесте нельзя снять молча: отсутствие файла — незавершённый
    прогон, а не часть результата."""
    m, tmp = fake_bundle
    (tmp / m["result_bundle"]["directory"] / "run.csv").unlink()
    with pytest.raises(bundle.Refused, match="run_csv"):
        bundle.hashes_manifest(m, "sha256:" + "a" * 64)


def test_a_journal_without_its_witness_is_not_a_bundle(fake_bundle):
    m, _ = fake_bundle
    arts = m["result_bundle"]["artifacts"]
    arts.remove(next(a for a in arts if a["role"] == "provenance_head_witness"))
    with pytest.raises(bundle.Refused, match="witness"):
        bundle.hashes_manifest(m, "sha256:" + "a" * 64)


def test_verify_catches_an_artifact_changed_after_the_verdict(fake_bundle):
    m, tmp = fake_bundle
    digest = "sha256:" + "a" * 64
    base = tmp / m["result_bundle"]["directory"]
    record = bundle.hashes_manifest(m, digest)
    (base / m["result_bundle"]["hashes_manifest"]).write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8")
    ok, problems = bundle.verify(m, digest)
    assert ok and problems == []

    (base / "run.csv").write_text("CSV переписан после вердикта\n", encoding="utf-8")
    ok, problems = bundle.verify(m, digest)
    assert not ok
    assert any("изменился после сборки" in p and "run.csv" in p for p in problems), problems


def test_verify_catches_a_file_deleted_or_swapped_for_another_protocol(fake_bundle):
    m, tmp = fake_bundle
    digest = "sha256:" + "a" * 64
    base = tmp / m["result_bundle"]["directory"]
    record = bundle.hashes_manifest(m, digest)
    (base / m["result_bundle"]["hashes_manifest"]).write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8")
    (base / "per_seed_outcomes.csv").unlink()
    ok, problems = bundle.verify(m, digest)
    assert not ok and any("удалён с диска" in p for p in problems), problems

    other = copy.deepcopy(m)
    other["protocol_id"] = "p4.2.oracle-baseline.v9"
    ok, problems = bundle.verify(other, digest)
    assert not ok and any("манифест p4.2.oracle-baseline.v9" in p for p in problems), problems


def test_check_empty_reports_the_directory_it_promised_to_guard(fake_bundle):
    m, tmp = fake_bundle
    base = tmp / m["result_bundle"]["directory"]
    for f in base.iterdir():
        f.unlink()
    report = bundle.check_empty(m)
    assert report["clean"] is True and report["occupied"] == []
    (base / "run.csv").write_text("начатый прогон\n", encoding="utf-8")
    (base / "shard_leftover.csv").write_text("не из списка\n", encoding="utf-8")
    report = bundle.check_empty(m)
    assert [o["role"] for o in report["occupied"]] == ["run_csv"]
    assert report["stray_files"] == ["var/p4_2_v2/shard_leftover.csv"]
    assert report["clean"] is False


# --- 6. runner: свежий путь и запрет запуска (пункты 5 и 7 review) ----------

def test_the_v2_paths_are_the_declared_bundle_paths_and_not_the_v1_ones(manifest):
    paths = {a["role"]: a["filename"] for a in manifest["result_bundle"]["artifacts"]}
    assert str(runner.DEFAULT_OUT) == str(REPO / "var/p4_2_v2" / paths["run_csv"])
    assert str(runner.DEFAULT_PROV) == str(REPO / "var/p4_2_v2" / paths["provenance_jsonl"])
    assert runner.V1_PROV != runner.DEFAULT_PROV
    assert runner.ABORTED_ROOT == REPO / "var/aborted"


def test_confirmatory_run_is_authorized_after_review_commit(manifest):
    """Авторизующий коммит переключил run на authorized: раннер больше не
    встаёт на authorize_run. Это административный permit-state, а не научная
    правка: протокол, манифест и seed-множество не тронуты."""
    allowed, flag = proto.run_authorization(V2_ID)
    assert allowed is True and flag == "authorized"
    runner.authorize_run(manifest)  # не бросает


def test_revoking_authorization_closes_the_confirmatory_run(manifest, monkeypatch):
    """Откат флага в «not_authorized» обязан вернуть отказ: право на запуск
    держится в коде, а не в записанном где-то вердикте."""
    monkeypatch.setitem(proto.FROZEN_PROTOCOLS[V2_ID], "run", "not_authorized")
    assert proto.run_authorization(V2_ID) == (False, "not_authorized")
    with pytest.raises(runner.Refused, match="не авторизован"):
        runner.authorize_run(manifest)


def test_provenance_guard_refuses_the_v1_log_for_every_kind_of_run(manifest):
    """Ни continue, ни recover_head на прерванном журнале: цепь, у которой начало из
    закрытого протокола, а продолжение из v2, не доказывает ничего."""
    for experimental in (False, True):
        with pytest.raises(runner.Refused, match="прерванному P4.2.v1"):
            runner.provenance_guard(manifest, experimental, str(runner.V1_PROV))
        archived = runner.ABORTED_ROOT / "p4_2_v1_infrastructure_abort" / "prov.jsonl"
        with pytest.raises(runner.Refused, match="прерванному P4.2.v1"):
            runner.provenance_guard(manifest, experimental, str(archived))


def test_provenance_guard_demands_a_fresh_file_for_confirmatory(manifest, tmp_path):
    stale = tmp_path / "prov.jsonl"
    stale.write_text('{"уже начатый прогон": 1}\n', encoding="utf-8")
    with pytest.raises(runner.Refused, match="уже непустой"):
        runner.provenance_guard(manifest, False, str(stale))
    fresh = tmp_path / "fresh.jsonl"
    runner.provenance_guard(manifest, False, str(fresh))       # путь свободен — пропускает
    fresh.write_text("", encoding="utf-8")
    runner.provenance_guard(manifest, False, str(fresh))       # пустой файл — тоже свежий


def test_experimental_run_may_not_dirty_the_bundle_directory(manifest):
    inside = str(REPO / "var/p4_2_v2" / "prov.jsonl")
    with pytest.raises(runner.Refused, match="bundle-каталог"):
        runner.provenance_guard(manifest, True, inside)
    with pytest.raises(runner.Refused, match="bundle-каталог"):
        runner.bundle_guard(manifest, True, str(REPO / "var/p4_2_v2" / "expl.csv"))
    runner.provenance_guard(manifest, True, "var/expl.prov.jsonl")


def test_bundle_guard_keeps_confirmatory_output_inside_the_declared_bundle(manifest):
    runner.bundle_guard(manifest, False, str(runner.DEFAULT_OUT))
    with pytest.raises(runner.Refused, match="вне объявленного result_bundle"):
        runner.bundle_guard(manifest, False, "var/p4_2_oracle.csv")


def test_runner_cli_refuses_confirmatory_to_foreign_path_after_authorization(tmp_path):
    """После авторизации CLI пропускает authorize_run, но bundle_guard всё ещё
    запрещает писать confirmatory-артефакт вне var/p4_2_v2/. Выход 2 — «проверки
    не было»."""
    out, prov = tmp_path / "run.csv", tmp_path / "provenance.jsonl"
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts/run_p4_2_oracle.py"),
         "--seeds", "120", "--arms", "oracle_reflex", "--out", str(out),
         "--provenance", str(prov)],
        capture_output=True, text=True, timeout=180)
    combined = res.stdout + res.stderr
    assert res.returncode == runner.REFUSE, combined[-800:]
    # Причина отказа — bundle placement, а не authorization
    assert "result_bundle" in combined or "bundle-каталог" in combined, combined[-800:]
    assert not out.exists() and not prov.exists()


def test_bundle_tool_refuses_an_unfrozen_manifest_and_reports_a_clean_v2_tree(tmp_path):
    """Доказательство «ничего не записано» — машинное, а не глазами: rc=0 по
    check-empty, и подменённый манифест не собирается (rc=2)."""
    def run(*args):
        return subprocess.run([sys.executable, str(REPO / "scripts/collect_p4_2_bundle.py"), *args],
                              capture_output=True, text=True, timeout=180)

    declared = [REPO / "var/p4_2_v2" / a["filename"]
                for a in proto.load(MANIFEST_PATH)["result_bundle"]["artifacts"]]
    res = run("--mode", "check-empty")
    assert res.returncode == 0, (res.stdout + res.stderr)[-800:]
    assert "все объявленные пути свободны" in res.stdout, res.stdout[-800:]
    assert not [p for p in declared if p.exists()], "на путях прогона уже что-то лежит"

    # правка, невидимая схеме: пробел в свободном тексте меняет canonical digest,
    # и сборщик обязан встать на якоре, а не собрать bundle по другому плану
    edited = tmp_path / "edited.json"
    m = copy.deepcopy(proto.load(MANIFEST_PATH))
    m.pop("_digest")
    m["question"] += " "
    edited.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    res = run("--manifest", edited, "--mode", "check-empty")
    assert res.returncode == 2, (res.stdout + res.stderr)[-800:]
    assert "BUNDLE НЕ СОБИРАЕТСЯ" in res.stderr, res.stderr[-800:]


# --- 7. authorization record integrity (P4.2.v2 run authorization commit) ----

AUTH_RECORD_PATH = REPO / "fly_connectome_agent/manifests/p4_2_v2_authorization.json"


@pytest.fixture(scope="module")
def auth_record() -> dict:
    return json.loads(AUTH_RECORD_PATH.read_text(encoding="utf-8"))


def test_auth_record_digest_matches_frozen_manifest(auth_record):
    """Авторизация выдаётся на конкретный digest. Если запись ссылается на
    другой digest, она не имеет силы."""
    expected = ("sha256:5f5cae42a2f0373933ead1c61307eac"
                "97f6a864f25b0927dbda4643e94d75123")
    assert auth_record["manifest_digest"] == proto.FROZEN_PROTOCOLS[V2_ID]["digest"]
    assert auth_record["manifest_digest"] == expected


def test_auth_record_seed_range_matches_confirmatory_set(auth_record):
    """Авторизованы ровно 120–179; ни шире, ни уже."""
    assert auth_record["seed_range"] == [120, 179]
    assert auth_record["seed_count"] == 60
    assert list(range(*[auth_record["seed_range"][0], auth_record["seed_range"][1] + 1])) == CONFIRMATORY


def test_auth_record_protocol_id_is_v2(auth_record):
    assert auth_record["protocol_id"] == V2_ID


def test_auth_record_declares_authorized_run_in_code(auth_record):
    """Запись и реестр кода обязаны совпадать: либо run=authorized, либо записи
    не должно быть."""
    allowed, flag = proto.run_authorization(V2_ID)
    assert allowed and flag == "authorized", (
        f"авторизация снята ({flag!r}), но p4_2_v2_authorization.json осталась — "
        "это противоречие")


def test_auth_record_does_not_claim_scientific_change(auth_record):
    """Ни одно constraint-поле не разрешает менять научный протокол."""
    constraints = auth_record["constraints"]
    assert constraints["only_run_flag_changed"] is True
    for key in ("manifest_content_modified", "seeds_modified", "arms_modified",
                "operating_point_modified", "hypothesis_family_modified",
                "statistical_plan_modified", "provenance_writer_modified",
                "gate_modified"):
        assert constraints[key] is False, f"{key}=true недопустимо в authorization record"


def test_v2_manifest_digest_unchanged_by_authorization_commit():
    """Главная проверка: authorization commit не тронул научный протокол.
    Digest файла манифеста на диске обязан совпасть с frozen-якорем в коде."""
    actual = proto.digest_of_file(proto.DEFAULT_MANIFEST)
    assert actual == proto.FROZEN_PROTOCOLS[V2_ID]["digest"], (
        f"digest манифеста изменился: {actual} vs {proto.FROZEN_PROTOCOLS[V2_ID]['digest']}")


def test_v2_manifest_content_equals_pre_registration(
        manifest, auth_record):
    """Научные поля манифеста те же, что были на reviewed SHA. Правка любого
    числа (arms, episodes, seeds, operating_point, hypotheses) после
    авторизации — это другой протокол."""
    assert len(manifest["arms"]) == 6
    assert manifest["episodes_per_seed"] == 80
    assert manifest["task_streams"] == ["left_target", "right_target", "mixed"]
    assert manifest["primary_metric"] == "mixed_min_half_success"
    assert manifest["statistical_test"]["family"] == ["H1", "H2", "H3", "H4"]
    assert manifest["seed_sets"]["confirmatory"]["range"] == [120, 179]
    assert manifest["protocol_id"] == V2_ID
