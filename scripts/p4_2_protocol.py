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
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fly_connectome_agent.src.engineering.harness.symmetric_toy import (  # noqa: E402
    MIXED,
    ARM_SPECS,
    ToySettings,
)

DEFAULT_MANIFEST = REPO / "fly_connectome_agent" / "manifests" / "p4_2_oracle_confirmatory.json"
SCHEMA_PATH = REPO / "fly_connectome_agent" / "schemas" / "p4_protocol_manifest.schema.json"

# Digest замороженной pre-registration (commit 58de288) как константа в коде.
# Зачем именно константа: сравнение «digest файла» с «digest'ом загруженного
# манифеста» — тавтология, обе величины считаются из одних байт, и подмена
# манифеста через --manifest её бы не заметила. Якорь обязан жить вне
# проверяемого файла; тогда правка протокола после заморозки ловится всегда,
# включая edits, которые остаются валидными по схеме (например правка текста
# гипотезы).
FROZEN_PROTOCOL_DIGEST = (
    "sha256:6e343c298c5367ad1cc713db8a8d4e1f981467432a6536120286305476de6f62"
)

TASK_STREAM_TO_CODE = {"left_target": "left", "right_target": "right", "mixed": MIXED}
PRIMARY_METRIC_TO_COLUMN = {
    "mixed_min_half_success": "success_mirror_min",
    "success_rate": "success_rate",
}


class ProtocolError(ValueError):
    """Протокол полезен только тогда, когда его можно отклонить."""


def require_frozen(digest: str, *, where: str) -> str:
    """Отклонить любой манифест, которого нет в замороженной pre-registration."""
    if digest != FROZEN_PROTOCOL_DIGEST:
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


def validate(manifest: dict[str, Any], settings_defaults: dict[str, Any] | None = None) -> None:
    """Кросс-полевые проверки. Первая же находка — исключение, не предупреждение."""
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

    validate(manifest)
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
    """ToySettings из манифеста. plastic_slots приходит списком — данныеclass ждёт tuple."""
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
    return (
        f"protocol {manifest['protocol_id']}  digest {manifest['_digest'][:19]}…\n"
        f"  вопрос: {manifest['question']}\n"
        f"  arms  : {arms}\n"
        f"  seeds : {conf[0]}..{conf[1]} (n={conf[1] - conf[0] + 1}), "
        f"episodes/seed={manifest['episodes_per_seed']}, streams={manifest['task_streams']}\n"
        f"  метрика: {manifest['primary_metric']} -> колонка "
        f"'{PRIMARY_METRIC_TO_COLUMN[manifest['primary_metric']]}', "
        f"тест {manifest['statistical_test']['method']} "
        f"(alpha={manifest['statistical_test']['alpha']}, "
        f"{manifest['statistical_test']['multiple_comparison_policy']}, "
        f"family={manifest['statistical_test']['family']})"
    )


if __name__ == "__main__":
    m = load()
    print(describe(m))
    print("  governance violation budget:", m["provenance"]["violation_budget"])
    print("  claims:", ", ".join(m["claims_allowed"]))
