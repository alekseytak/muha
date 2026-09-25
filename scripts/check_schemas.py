#!/usr/bin/env python3
"""Проверка схем манифестов: правильный манифест проходит, неправильный падает.

Правило репозитория: проверка, которая не может упасть, бесполезна. Поэтому
здесь три пробы, и третья ломает манифест нарочно — она обязана покраснеть,
иначе схема ничего не значит.

Запуск: python scripts/check_schemas.py
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    print("нужен jsonschema: pip install jsonschema", file=sys.stderr)
    raise SystemExit(2)

ROOT = pathlib.Path(__file__).resolve().parent.parent
AGENT = ROOT / "fly_connectome_agent"
EXPERIMENT_SCHEMA = AGENT / "schemas" / "experiment_manifest.schema.json"
CONNECTOME_SCHEMA = AGENT / "schemas" / "connectome_manifest.schema.json"
EXAMPLES = AGENT / "examples"

RANDOMISING = (
    "directed_in_out_degree_preserving_rewire",
    "weight_shuffle",
    "density_matched_random_sparse",
)

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ок  ' if ok else 'ПАДЕНИЕ'} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def errors(schema_path: pathlib.Path, document: dict) -> list:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    return sorted(validator.iter_errors(document), key=lambda e: list(e.path))


def main() -> int:
    print("схемы манифестов")

    # Проба 1: правильный манифест обязан пройти.
    valid_path = EXAMPLES / "manifest.valid.json"
    if not valid_path.exists():
        check("правильный манифест проходит", False, f"нет файла {valid_path.name}")
        return report()
    valid = json.loads(valid_path.read_text(encoding="utf-8"))
    found = errors(EXPERIMENT_SCHEMA, valid)
    check(
        "правильный манифест проходит",
        not found,
        "" if not found else f"первая ошибка: {found[0].message}",
    )

    # Проба 2: неправильный обязан упасть — иначе проверка бессмысленна.
    invalid_path = EXAMPLES / "manifest.invalid.json"
    if not invalid_path.exists():
        check("неправильный манифест падает", False, f"нет файла {invalid_path.name}")
    else:
        invalid = json.loads(invalid_path.read_text(encoding="utf-8"))
        found = errors(EXPERIMENT_SCHEMA, invalid)
        check("неправильный манифест падает", bool(found), f"ошибок: {len(found)}")

    # Проба 3: у рандомизирующего базлайна seed обязателен. Убираем его у копии
    # правильного манифеста — схема обязана это заметить.
    broken = copy.deepcopy(valid)
    stripped = None
    for baseline in broken.get("baselines", []):
        if baseline.get("baseline_id") in RANDOMISING and "seed" in baseline:
            stripped = baseline.pop("seed")
            stripped = baseline["baseline_id"]
            break
    if stripped is None:
        check(
            "рандомизирующий базлайн без seed отвергается",
            False,
            "в правильном манифесте нет базлайна со случайностью и seed — проба вхолостую",
        )
    else:
        found = errors(EXPERIMENT_SCHEMA, broken)
        mentions_seed = any("seed" in err.message for err in found)
        check(
            "рандомизирующий базлайн без seed отвергается",
            bool(found) and mentions_seed,
            f"снят seed у {stripped}; ошибок {len(found)}",
        )

    # Проба 4: контрольный базлайн без seed, наоборот, обязан проходить —
    # иначе схема требовала бы сеять то, где случайности нет.
    control = copy.deepcopy(valid)
    planted = False
    for baseline in control.get("baselines", []):
        if baseline.get("baseline_id") in ("no_plasticity", "no_modulator"):
            baseline.pop("seed", None)
            planted = True
    found = errors(EXPERIMENT_SCHEMA, control) if planted else [1]
    check(
        "контрольный базлайн без seed проходит",
        planted and not found,
        "" if not found else f"ошибок: {len(found)}",
    )

    # Проба 5: схема коннектома читается и не пуста.
    if not CONNECTOME_SCHEMA.exists():
        check("схема коннектома на месте", False, "файла нет")
    else:
        schema = json.loads(CONNECTOME_SCHEMA.read_text(encoding="utf-8"))
        check(
            "схема коннектома на месте",
            schema.get("type") == "object" and bool(schema.get("properties")),
            f"свойств: {len(schema.get('properties', {}))}",
        )

    return report()


def report() -> int:
    if failures:
        print(f"\nпровалено проб: {len(failures)} — {', '.join(failures)}")
        return 1
    print("\nвсе пробы схем прошли")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
