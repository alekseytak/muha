#!/usr/bin/env python
"""Загрузчик и сторож pre-registered протокола P4.2.

Зачем это отдельным модулем: манифест читают два разных скрипта (прогон и
гейт). Если каждый будет сам разбирать seed-диапазоны и сверять operating
point, они разъедутся молча — а ровно на этом различии и держится весь смысл
pre-registration. Здесь одна реализация, и она обязана падать на любой правке
протокола после запуска.

Что проверяется (и почему это не косметика):
  schema shape      — поля, типы, enum'ы;
  seed count        — `count` должен совпадать с длиной диапазона: несопадение
                      значит, что протокол правили, не сверяясь;
  seed disjointness — confirmatory не пересекается с множествами P4.1, иначе
                      «fresh seeds» означают просто повторный анализ старых;
  operating point   — каждое значение обязано быть либо равным дефолту кода,
                      либо перечисленным в overrides. Молча сдвинуть eta после
                      факта нельзя;
  arms против кода  — learner/init/reward_coupled из манифеста сверяются с
                      ARM_SPECS. Манифест не может объявить oracle «non-plastic»,
                      если код для него течёт plasticity;
  budget            — cells/episodes из заявленного run_budget пересчитываются.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fly_connectome_agent.src.engineering.harness.symmetric_toy import (  # noqa: E402
    MIXED,
    ARM_SPECS,
    ToySettings,
)
from fly_connectome_agent.src.engineering.logging.provenance_log import (  # noqa: E402
    HEAD_SUFFIX,
)

DEFAULT_MANIFEST = REPO / "fly_connectome_agent" / "manifests" / "p4_2_oracle_confirmatory_v2.json"
V1_MANIFEST = REPO / "fly_connectome_agent" / "manifests" / "p4_2_oracle_confirmatory.json"
SCHEMA_PATH = REPO / "fly_connectome_agent" / "schemas" / "p4_protocol_manifest.schema.json"

# Digest'ы замороженных pre-registration как константы в коде (v1 — commit 58de288).
# Зачем именно константа: сравнение «digest файла» с «digest'ом загруженного
# манифеста» — тавтология, обе величины считаются из одних байт, и подмена
# манифеста через --manifest её бы не заметила. Якорь обязан жить вне
# проверяемого файла; тогда правка протокола после заморозки ловится всегда,
# включая правки, которые остаются валидными по схеме (например текст
# гипотезы).
#
# Реестр, а не одно значение, потому что протоколы версионируются: v1 остаётся на
# якоре как закрывшийся прогон (его digest должен проверяться и через год), а
# активным является v2. Статусы и флаг run живут здесь же, а не в манифесте: файл
# не может выдать разрешение на собственный запуск. Разрешение — отдельный
# review-акт, и выглядеть он обязан как diff одной строки в коде, а не как правка
# замороженного плана.
ACTIVE_PROTOCOL_ID = "p4.2.oracle-baseline.v2"
RUN_AUTHORIZED_VALUE = "authorized"

FROZEN_PROTOCOLS: dict[str, dict[str, str]] = {
    "p4.2.oracle-baseline.v1": {
        "digest": "sha256:6e343c298c5367ad1cc713db8a8d4e1f981467432a6536120286305476de6f62",
        "status": "aborted_infrastructure_performance_defect",
        "run": "not_authorized",
        "manifest": "fly_connectome_agent/manifests/p4_2_oracle_confirmatory.json",
        "note": "Остановлен на 24 422-й строке журнала: append перечитывал весь лог, O(n) на "
                "запись. Научная часть не начиналась: CSV, sidecar и вердикт гейта не создавались.",
    },
    "p4.2.oracle-baseline.v2": {
        "digest": "sha256:5f5cae42a2f0373933ead1c61307eac97f6a864f25b0927dbda4643e94d75123",
        "status": "frozen_pending_independent_review",
        "run": "authorized",
        "manifest": "fly_connectome_agent/manifests/p4_2_oracle_confirmatory_v2.json",
        "note": "Замена v1 на том же научном плане: disjoint seeds 120–179, контракт bundle из "
                "девяти артефактов, свежий provenance-путь. Запуск запрещён до review этого "
                "коммита: run меняется на authorized только отдельным актом.",
    },
}

# Совместимость с кодом, который сверяет «тот ли это протокол»: активный digest —
# единственный, по которому сегодня разрешено считать confirmatory-вердикт.
FROZEN_PROTOCOL_DIGEST = FROZEN_PROTOCOLS[ACTIVE_PROTOCOL_ID]["digest"]

# Роли артефактов, обязанные быть в bundle. Список сверяется с манифестом кодом:
# файл укоротить можно, этот — нельзя, иначе контракт «что лежит в bundle» снова
# станет самодекларацией.
REQUIRED_BUNDLE_ROLES = (
    "run_csv",
    "csv_sidecar",
    "provenance_jsonl",
    "provenance_head_witness",
    "gate_stdout",
    "gate_verdict_json",
    "per_seed_outcomes",
    "environment_fingerprint",
    "file_hashes_manifest",
)

# Начиная с v2 lineage и bundle обязательны. Проверка живёт и в схеме, и здесь:
# схема ловит правку файла до запуска, а код — случай, когда validate() вызвали
# напрямую (тесты, аудит), минуя load() со схемой.
LINEAGE_FROM_VERSION = re.compile(r"\.v([2-9]|[1-9][0-9]+)$")

TASK_STREAM_TO_CODE = {"left_target": "left", "right_target": "right", "mixed": MIXED}
PRIMARY_METRIC_TO_COLUMN = {
    "mixed_min_half_success": "success_mirror_min",
    "success_rate": "success_rate",
}


class ProtocolError(ValueError):
    """Протокол полезен только тогда, когда его можно отклонить."""


def frozen_entry(protocol_id: str) -> dict[str, str]:
    """Якорь из кода по имени протокола. Неизвестный protocol_id — отказ, а не None."""
    entry = FROZEN_PROTOCOLS.get(protocol_id)
    if entry is None:
        raise ProtocolError(
            f"протокол {protocol_id!r} не заморожен в коде: в реестре "
            f"{sorted(FROZEN_PROTOCOLS)}. Прогон по протоколу, якоря которого нет, не "
            "confirmatory: его план можно менять на ходу, и это другой эксперимент.")
    return entry


def run_authorization(protocol_id: str) -> tuple[bool, str]:
    """Разрешён ли запуск этого протокола. Разрешение даёт только review-акт в коде."""
    entry = frozen_entry(protocol_id)
    return entry["run"] == RUN_AUTHORIZED_VALUE, entry["run"]


def require_frozen(digest: str, *, where: str, protocol_id: str | None = None) -> str:
    """Отклонить любой манифест, которого нет в замороженной pre-registration.

    Без protocol_id сверка идёт с активным протоколом (обратная совместимость).
    С protocol_id становится понятно, что именно разошлось: правка активного
    протокола и попытка посчитать прогон по закрытому v1 — разные находки, и
    вторая обязана звучать как «этот протокол закрыт», а не как «digest не тот».
    """
    if protocol_id is not None:
        entry = frozen_entry(protocol_id)
        if digest != entry["digest"]:
            raise ProtocolError(
                f"{where}: digest {digest} не совпадает с замороженным {protocol_id} "
                f"({entry['digest']}). Протокол после заморозки не правится: правка — это "
                "другой протокол с другим id и новым review.")
        return digest
    if digest != FROZEN_PROTOCOL_DIGEST:
        known = {e["digest"]: pid for pid, e in FROZEN_PROTOCOLS.items()}
        if digest in known:
            closed = FROZEN_PROTOCOLS[known[digest]]
            raise ProtocolError(
                f"{where}: это замороженный {known[digest]} со статусом "
                f"«{closed['status']}». По закрытому протоколу не считают: его прогон не "
                f"был завершён, а повторить его на тех же seed'ах нельзя.")
        raise ProtocolError(
            f"{where}: digest {digest} не совпадает с замороженным протоколом "
            f"{FROZEN_PROTOCOL_DIGEST}. Протокол после заморозки не правится: либо это "
            "тот же манифест, либо прогон не confirmatory и не может быть принят гейтом.")
    return digest


def _eq(a: Any, b: Any) -> bool:
    """Сравнение значения из манифеста с дефолтом кода.

    JSON не различает tuple и list, а ToySettings живёт кортежами: без этой
    функции plastic_slots=[0,1,2,3] вечно «не совпадал бы» с (0,1,2,3).
    """
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return bool(a == b)


def canonical_digest(manifest: dict[str, Any]) -> str:
    """SHA-256 канонной формы: sorted keys, без пробелов, ensure_ascii=False.

    Порядок ключей в файле на digest не влияет — правкой порядка нельзя
    «изменить протокол, не изменив хеш», потому что хеш считается по содержимому.
    """
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expand_seed_range(spec: dict[str, Any], where: str) -> list[int]:
    lo, hi = spec["range"]
    seeds = list(range(lo, hi + 1))
    if spec["count"] != len(seeds):
        raise ProtocolError(
            f"{where}: declared count {spec['count']} != range [{lo}, {hi}] -> {len(seeds)} seeds"
        )
    return seeds


def _overlap(seeds: list[int], other: set[int]) -> list[int]:
    return sorted(set(seeds) & other)


def _repo_relative(path: Any) -> str:
    """Путь от корня репозитория, если он внутри; иначе как задан.

    Нужен, потому что служебное `_manifest_path` приходит из load() абсолютным, а
    в реестре протоколы записаны относительными путями.
    """
    p = pathlib.Path(str(path))
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def _check_lineage(manifest: dict[str, Any], problems: list[str],
                   manifest_path: Any = None) -> None:
    """Преемственность протоколов обязана упираться в реестр кода.

    Заявление «мы закрываем v1 и открываем v2» — это факт про историю прогонов,
    а не про текст манифеста. Если бы статус читался только из файла, правка
    supersedes_status превратила бы «aborted» в «completed» без единого review, и
    по aborted-протоколу снова начали бы считать.
    """
    sup = manifest.get("supersedes_protocol_id")
    status = manifest.get("supersedes_status")
    needs_lineage = bool(LINEAGE_FROM_VERSION.search(manifest.get("protocol_id", "")))
    if needs_lineage and not (sup and status):
        problems.append(f"{manifest['protocol_id']}: начиная с версии v2 манифест обязан нести "
                        "supersedes_protocol_id и supersedes_status — без них неотличимо, какой "
                        "прогон этому предшествовал и с каким итогом")
    if sup is None and status is None:
        return
    if (sup is None) != (status is None):
        problems.append("lineage неполный: supersedes_protocol_id и supersedes_status — это пара, "
                        "по отдельности они не читаются")
        return
    if sup == manifest["protocol_id"]:
        problems.append("протокол не может supersede сам себя")
        return
    entry = FROZEN_PROTOCOLS.get(sup)
    if entry is None:
        problems.append(f"supersedes_protocol_id={sup!r} нет в реестре замороженных протоколов "
                        f"({sorted(FROZEN_PROTOCOLS)}): преемственность нельзя заявить по "
                        "протоколу, которого в коде не существовало")
        return
    if entry["status"] != status:
        problems.append(f"supersedes_status={status!r} против {entry['status']!r} в коде: итог "
                        f"закрытого {sup} задаёт review-акт, а не манифест-преемник")
    if manifest_path is not None and entry["manifest"] == _repo_relative(manifest_path):
        problems.append("supersedes_protocol_id указывает на манифест того же файла")


def _check_result_bundle(manifest: dict[str, Any], problems: list[str]) -> None:
    """Bundle-контракт: роли задаёт код, а не манифест.

    Список REQUIRED_BUNDLE_ROLES живёт в p4_2_protocol.py, потому что договорённость
    «в результат входит witness и вердикт гейта» — это то, что прогон делает
    проверяемым постфактум. Из замороженного файла список можно вычеркнуть одним
    delete, и тогда bundle снова станет самодекларацией: что посчитали удобным,
    то и приложили.
    """
    bundle = manifest.get("result_bundle")
    if bundle is None:
        return
    directory = bundle["directory"]
    if directory.rstrip("/") == "var":
        problems.append("result_bundle.directory = var/: в общий каталог уже лежит journal "
                        "прерванного v1 и CSV прошлых прогонов — bundle-папка обязана быть "
                        "отдельной, иначе 'ничего не записано' недоказуемо")
    artifacts = bundle["artifacts"]
    roles = [a["role"] for a in artifacts]
    names = [a["filename"] for a in artifacts]
    if len(roles) != len(set(roles)):
        problems.append(f"result_bundle: повторяющиеся роли — {[r for r in set(roles) if roles.count(r) > 1]}")
    if len(names) != len(set(names)):
        problems.append(f"result_bundle: в одной папке два артефакта с одинаковым именем — "
                        f"{sorted(n for n in set(names) if names.count(n) > 1)}")
    for role in sorted(set(REQUIRED_BUNDLE_ROLES) - set(roles)):
        problems.append(f"result_bundle не объявляет обязательный артефакт {role!r} (реестр кода) — "
                        "без него вердикт нельзя пересобрать заново")
    for role in sorted(set(roles) - set(REQUIRED_BUNDLE_ROLES)):
        problems.append(f"result_bundle объявляет роль {role!r}, которой нет в реестре кода "
                        f"({sorted(REQUIRED_BUNDLE_ROLES)})")
    by_role = {a["role"]: a for a in artifacts}
    for art in artifacts:
        if art.get("hash") != "sha256":
            problems.append(f"артефакт {art['role']}: hash={art.get('hash')!r} — bundle "
                            "хешится только sha256, иначе список хешей несопоставим")
        if art.get("required") is not True:
            problems.append(f"артефакт {art['role']} помечен required!=true: обязательность "
                            "вычеркивается из манифеста одним delete — реестр кода этого не разрешает")
        if not str(art.get("producer", "")).strip():
            problems.append(f"артефакт {art['role']}: пустой producer — непонятно, кто его создаёт")
    # witness и журнал — один артефакт из двух файлов (контракт писателя):
    # в bundle разрешено положить JSONL без .head.json, и тогда обрезанный журнал
    # снова неотличим от полного.
    log, witness = by_role.get("provenance_jsonl"), by_role.get("provenance_head_witness")
    if log and witness and witness["filename"] != log["filename"] + HEAD_SUFFIX:
        problems.append(f"provenance_head_witness обязан называться {log['filename'] + HEAD_SUFFIX!r} "
                        f"(в манифестве {witness['filename']!r}): писатель пишет witness по суффиксу, "
                        "и другое имя в bundle означает файл, которого прогон не создавал")
    if bundle.get("hashes_manifest") not in names:
        problems.append(f"hashes_manifest={bundle.get('hashes_manifest')!r} отсутствует в списке "
                        "артефактов: список хешей обязан быть частью bundle, а не висеть снаружи")


def _check_seed_evidence(manifest: dict[str, Any], problems: list[str]) -> None:
    """Заявление «эти seed'ы уже сожжены» обязано сверяться с записью-доказательством.

    Диапазон в disjoint_from — это строка в манифесте. Запись, выпущенная
    p4_2_burned_seeds.py из самого журнала, — внешнее свидетельство. Пока они не
    сверяются кодом, в frozen-протокол можно вписать любой диапазон и назвать его
    «несечением с aborted v1»; проверка пройдёт, а сожжённые seed'ы останутся свежими.

    Отдельно ловится снятие самого поля evidence: множество, названное по версии
    закрытого прогона, обязано иметь свидетельство, иначе «вычеркнуть доказательство»
    обходилось бы дешевле, чем «вычеркнуть диапазон».
    """
    sup = str(manifest.get("supersedes_protocol_id") or "")
    tail = re.search(r"v(\d+)$", sup)
    token = f"v{tail.group(1)}" if tail else None
    for other in manifest["seed_sets"]["disjoint_from"]:
        ev = other.get("evidence")
        if token and token in str(other.get("id", "")) and not ev:
            problems.append(
                f"{other['id']}: множество названо по закрытому {sup}, но поле evidence снято — "
                "без записи-доказательства «эти seed'ы сожжены» остаётся строкой в том же "
                "манифесте, который обязан её подтвердить")
        if not ev:
            continue
        path = pathlib.Path(ev)
        if not path.is_absolute():
            path = REPO / ev
        if not path.exists():
            problems.append(f"{other['id']}: запись-доказательство {ev} не найдена — "
                            "несечение нечем подтвердить")
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{other['id']}: запись {ev} не читается как JSON ({exc})")
            continue
        burned = rec.get("burned_seed_set", {})
        if burned.get("range") != other["range"] or burned.get("count") != other["count"]:
            problems.append(f"{other['id']}: заявлено {other['range']} count={other['count']}, "
                            f"а в {path.name} лежит {burned.get('range')} "
                            f"count={burned.get('count')} — манифест разошёлся с фактом журнала")
        if rec.get("protocol_id") == manifest["protocol_id"]:
            problems.append(f"{other['id']}: доказательство получено из прогона того же "
                            "протокола — несечение само себе не свидетельство")
        coverage = rec.get("coverage", {})
        if coverage.get("evidence_complete") is not True:
            problems.append(f"{other['id']}: {path.name} помечен evidence_complete!=true "
                            "(часть журнала не разобрана) — полное несечение по нему "
                            "подтвердить нельзя")


def validate(manifest: dict[str, Any], settings_defaults: dict[str, Any] | None = None, *,
             manifest_path: Any = None) -> None:
    """Кросс-полевые проверки. Первая же находка — исключение, не предупреждение.

    manifest_path передаётся снаружи, а не дописывается в сам манифест: load()
    возвращает словарь, который тесты и скрипты перечитывают и пересохраняют, а
    служебное поле в файле схема отвергает как подделку протокола.
    """
    problems: list[str] = []

    conf = expand_seed_range(manifest["seed_sets"]["confirmatory"], "confirmatory")
    probe = manifest["seed_sets"].get("fixture_probe")
    probe_seeds = expand_seed_range(probe, "fixture_probe") if probe else []

    for other in manifest["seed_sets"]["disjoint_from"]:
        overlap = _overlap(conf, set(expand_seed_range(other, other["id"])))
        if overlap:
            problems.append(f"confirmatory seeds пересекаются с {other['id']}: {overlap[:5]}")
    if probe_seeds:
        for other in manifest["seed_sets"]["disjoint_from"]:
            other_seeds = set(expand_seed_range(other, other["id"]))
            if _overlap(probe_seeds, other_seeds):
                problems.append(f"fixture_probe пересекается с {other['id']}")
        if set(probe_seeds) & set(conf):
            problems.append("fixture_probe использует confirmatory seeds — проба попала в статистику")

    _check_seed_evidence(manifest, problems)
    _check_lineage(manifest, problems, manifest_path)
    _check_result_bundle(manifest, problems)

    if manifest["episodes_per_seed"] != manifest["operating_point"]["values"]["episodes"]:
        problems.append(
            f"episodes_per_seed={manifest['episodes_per_seed']} != "
            f"operating_point.values.episodes={manifest['operating_point']['values']['episodes']}"
        )

    op = manifest["operating_point"]
    defaults = settings_defaults or dataclasses.asdict(ToySettings())
    for key, value in op["values"].items():
        declared_override = key in op.get("overrides", {})
        same_as_code = _eq(defaults.get(key), value)
        if not same_as_code and not declared_override:
            problems.append(f"operating_point.values.{key}={value} не равен дефолту кода "
                            f"({defaults.get(key)}) и не объявлен в overrides")
        if declared_override and same_as_code:
            problems.append(f"overrides.{key} объявлен, но значение равно дефолту — "
                            f"лишняя запись в замороженном протоколе")
        if declared_override and not _eq(op["overrides"][key], value):
            problems.append(f"overrides.{key}={op['overrides'][key]} против values.{key}={value}")
    missing_rationale = sorted(set(op.get("overrides", {})) - set(op.get("override_rationale", {})))
    if missing_rationale:
        problems.append(f"override без обоснования: {missing_rationale}")
    # Рабочая точка обязана быть объявлена целиком. Необъявленное поле означает
    # «возьмём дефолт кода», а дефолт живёт не в замороженном плане: правка
    # ToySettings позже молча меняет прогон, не трогая digest манифеста.
    undeclared = sorted(set(defaults) - set(op["values"]))
    if undeclared:
        problems.append(f"operating_point объявлен не полностью, нет полей {undeclared}: "
                        "молча наследовать дефолт кода протокол не вправе")

    seen_arms: set[str] = set()
    for arm in manifest["arms"]:
        if arm["arm_id"] in seen_arms:
            problems.append(f"повторяющийся arm_id: {arm['arm_id']}")
        seen_arms.add(arm["arm_id"])
        spec = ARM_SPECS.get(arm["code_arm"])
        if spec is None:
            problems.append(f"arm {arm['arm_id']}: code_arm {arm['code_arm']!r} нет в ARM_SPECS")
            continue
        for field, code_value in (("learner", spec["learner"]), ("init_regime", spec["init"]),
                                  ("reward_coupled", spec["reward_coupled"])):
            if arm[field] != code_value:
                problems.append(
                    f"arm {arm['arm_id']}: манифест говорит {field}={arm[field]!r}, "
                    f"код — {code_value!r}"
                )

    roles = {a["role"] for a in manifest["arms"]}
    for arm in manifest["arms"]:
        if arm["role"] == "upper_bound_characterization" and arm.get("gate") != "none":
            problems.append(f"arm {arm['arm_id']}: characterization обязан иметь gate=none")
    if "upper_bound_characterization" not in roles:
        problems.append("в протоколе нет oracle-arms: без upper bound P4.2 не отвечает на свой вопрос")

    hyp_ids = {h["id"] for h in manifest["hypotheses"]}
    family = set(manifest["statistical_test"]["family"])
    if not family <= hyp_ids:
        problems.append(f"family ссылается на неизвестные гипотезы: {sorted(family - hyp_ids)}")
    for h in manifest["hypotheses"]:
        if h["comparator"] not in seen_arms or h["treatment"] not in seen_arms:
            problems.append(f"{h['id']} ссылается на arm вне списка arms")
        in_family = h["id"] in family
        if h["kind"] == "characterization" and in_family:
            problems.append(f"{h['id']}: characterization не входит в поправку")
        if h["kind"] != "characterization" and not in_family:
            problems.append(f"{h['id']}: required/desirable обязаны входить в family")

    rb = manifest.get("run_budget")
    if rb:
        cells = len(manifest["arms"]) * len(conf) * len(manifest["task_streams"])
        if rb["cells"] != cells:
            problems.append(f"run_budget.cells={rb['cells']}, пересчёт даёт {cells}")
        if rb["episodes"] != cells * manifest["episodes_per_seed"]:
            problems.append(f"run_budget.episodes={rb['episodes']}, пересчёт даёт "
                            f"{cells * manifest['episodes_per_seed']}")

    if PRIMARY_METRIC_TO_COLUMN.get(manifest["primary_metric"]) is None:
        problems.append(f"неизвестный primary_metric: {manifest['primary_metric']}")

    if problems:
        raise ProtocolError("\n".join(f"  - {p}" for p in problems))


def load(path: pathlib.Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Схема + кросс-поля + digest. Падает на любом несоответствии протокола коду."""
    manifest = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError("jsonschema обязателен: замороженный протокол без shape-проверки не заморожен") from exc

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda e: list(e.path))
    if errors:
        detail = "\n".join(f"  - {'/'.join(map(str, e.path))}: {e.message}" for e in errors[:10])
        raise ProtocolError(f"манифест не проходит схему {SCHEMA_PATH.name}:\n{detail}")

    validate(manifest, manifest_path=path)
    # digest считается ДО вставки служебного поля и только из содержимого файла;
    # гейт пересчитывает его сам через digest_of_file, сверяясь с этим значением.
    manifest["_digest"] = canonical_digest(manifest)
    return manifest


def digest_of_file(path: pathlib.Path) -> str:
    """Digest так, как его видит гейт: по файлу на диске, без служебных полей."""
    return canonical_digest(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


def confirmatory_seeds(manifest: dict[str, Any]) -> list[int]:
    return expand_seed_range(manifest["seed_sets"]["confirmatory"], "confirmatory")


def probe_seeds(manifest: dict[str, Any]) -> list[int]:
    spec = manifest["seed_sets"].get("fixture_probe")
    return expand_seed_range(spec, "fixture_probe") if spec else []


def build_settings(manifest: dict[str, Any]) -> ToySettings:
    """ToySettings из манифеста. plastic_slots приходит списком — dataclass ждёт tuple."""
    values = dict(manifest["operating_point"]["values"])
    if isinstance(values.get("plastic_slots"), list):
        values["plastic_slots"] = tuple(values["plastic_slots"])
    return ToySettings(**values)


def code_arms(manifest: dict[str, Any]) -> dict[str, str]:
    """arm_id -> code_arm. Единственная точка, где имена протокола и кода встречаются."""
    return {a["arm_id"]: a["code_arm"] for a in manifest["arms"]}


def arm_roles(manifest: dict[str, Any]) -> dict[str, str]:
    return {a["arm_id"]: a["role"] for a in manifest["arms"]}


def task_streams(manifest: dict[str, Any]) -> list[str]:
    return [TASK_STREAM_TO_CODE[t] for t in manifest["task_streams"]]


def describe(manifest: dict[str, Any]) -> str:
    arms = ", ".join(a["arm_id"] for a in manifest["arms"])
    conf = manifest["seed_sets"]["confirmatory"]["range"]
    try:
        allowed, run_flag = run_authorization(manifest["protocol_id"])
    except ProtocolError:
        # describe() вызывается и на experimental-манифестах с чужим protocol_id:
        # показывать план — не право считать, падать здесь не за чем.
        allowed, run_flag = False, "нет в реестре замороженных протоколов"
    lineage = (f"supersedes {manifest['supersedes_protocol_id']} "
               f"({manifest['supersedes_status']})"
               if manifest.get("supersedes_protocol_id") else "первый протокол серии")
    bundle = (f"{manifest['result_bundle']['directory']} / "
              f"{len(manifest['result_bundle']['artifacts'])} артефактов"
              if manifest.get("result_bundle") else "bundle не объявлен (до v2)")
    return (
        f"protocol {manifest['protocol_id']}  digest {manifest['_digest'][:19]}…\n"
        f"  вопрос: {manifest['question']}\n"
        f"  lineage: {lineage}\n"
        f"  bundle : {bundle}\n"
        f"  arms  : {arms}\n"
        f"  seeds : {conf[0]}..{conf[1]} (n={conf[1] - conf[0] + 1}), "
        f"episodes/seed={manifest['episodes_per_seed']}, streams={manifest['task_streams']}\n"
        f"  метрика: {manifest['primary_metric']} -> колонка "
        f"'{PRIMARY_METRIC_TO_COLUMN[manifest['primary_metric']]}', "
        f"тест {manifest['statistical_test']['method']} "
        f"(alpha={manifest['statistical_test']['alpha']}, "
        f"{manifest['statistical_test']['multiple_comparison_policy']}, "
        f"family={manifest['statistical_test']['family']})\n"
        f"  запуск: {run_flag}"
        f"{' — confirmatory прогон разрешён' if allowed else ' — confirmatory прогон запрещён, пока review не поменяет run в коде'}"
    )


if __name__ == "__main__":
    m = load()
    print(describe(m))
    print("  governance violation budget:", m["provenance"]["violation_budget"])
    print("  claims:", ", ".join(m["claims_allowed"]))
