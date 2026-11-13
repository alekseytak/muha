#!/usr/bin/env python3
"""Обучение каскада на разметке омеги и замер на невиданных семействах.

Разделение — ПО СЕМЕЙСТВАМ шаблонов (eval/gen_cases.py), а не по фразам:
отложенная часть состоит из форм запроса, которых в обучении не было вовсе.
Это отвечает на вопрос «переносит ли каскад навык», а не «запомнил ли строки».

Сравниваются три конфигурации:
  NB        — только наивный Байес (что было до правил);
  NB+μ1a    — плюс правила высокой точности (текущий каскад);
  омега     — верхняя граница: разметка большой модели.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.micromodels import MicroCascade, selective_report   # noqa: E402

DATA = ROOT / "eval" / "cases_distill.json"
FLOORS = (0.55, 0.70, 0.85)


def load_cases() -> list:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    return [case for case in payload["cases"] if case.get("verbs")]


def split_by_family(cases: list) -> tuple:
    """Чётные семейства — обучение, нечётные — отложенная часть."""
    families = sorted({case["family"] for case in cases})
    train_families = {name for index, name in enumerate(families) if index % 2 == 0}
    train = [case for case in cases if case["family"] in train_families]
    held = [case for case in cases if case["family"] not in train_families]
    return train, held, sorted(train_families), sorted(set(families) - train_families)


def without_rules(pairs: list, floor: float) -> MicroCascade:
    """Каскад без μ1a — чтобы честно увидеть вклад правил."""
    return MicroCascade(pairs, floor=floor, use_rules=False)


def main() -> int:
    cases = load_cases()
    train, held, train_families, held_families = split_by_family(cases)
    print(f"всего размечено омегой: {len(cases)}")
    print(f"обучение: {len(train)} фраз, семейства: {len(train_families)}")
    print(f"отложено: {len(held)} фраз, семейства: {len(held_families)}")
    print(f"  отложенные: {', '.join(held_families)}")
    print()

    pairs = [(case["prompt"], case["verbs"][0]) for case in train]
    for name, factory in (
        ("только правила μ1a", lambda p, f: MicroCascade(p, floor=f, use_stats=False)),
        ("только NB (без правил)", without_rules),
        ("NB+μ1a (каскад)", lambda p, f: MicroCascade(p, floor=f)),
    ):
        print(f"--- {name} ---")
        for floor in FLOORS:
            cascade = factory(pairs, floor)
            report = selective_report(cascade, held)
            print(f"  порог {floor:.2f}: закрыто без модели {report['covered']:>3}/{report['cases']}"
                  f" ({report['coverage']:.0%}), точность {report['selective_accuracy']}"
                  f", глагол {report['verb_accuracy_on_covered']}"
                  f", домен {report['domain_accuracy_on_covered']}")
        print()

    print("--- что важно для безопасности: только глагол ---")
    for floor in FLOORS:
        cascade = MicroCascade(pairs, floor=floor)
        report = selective_report(cascade, held)
        print(f"  порог {floor:.2f}: глагол верен в {report['verb_accuracy_on_covered']} "
              f"случаев при покрытии {report['coverage']:.0%} "
              f"(домен {report['domain_accuracy_on_covered']} — метки омеги шумные)")
    print()

    cascade = MicroCascade(pairs, floor=0.55)
    report = selective_report(cascade, held)
    print(f"выбрано: порог 0.55 — покрытие {report['coverage']:.0%}, "
          f"точность {report['selective_accuracy']}")
    if report["errors"]:
        print("\nошибки на отложенных семействах:")
        for error in report["errors"][:12]:
            print(f"  {error['phrase'][:46]:48s} ждали {error['want']:24s} "
                  f"получили {error['got']} (conf {error['conf']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())