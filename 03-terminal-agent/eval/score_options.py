#!/usr/bin/env python3
"""Замер оценки вариантов через logprobs на отложенных семействах.

Сравниваются два дешёвых способа выбрать глагол:
  * локальный каскад микромоделей (бесплатно, 0 обращений к сети);
  * распределение по вариантам у mistral-nemo через OpenRouter
    (≈ $0.0000012 за решение, но настоящие вероятности).

Метки — разметка омеги (дистилляция). Отличие по домену не считаем: домен
определяется грамматикой по префиксу объекта, а метки омеги по домену шумные.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.micromodels import MicroCascade            # noqa: E402
from core.option_scorer import score_options         # noqa: E402

DATA = ROOT / "eval" / "cases_distill.json"
CREDS = str(Path.home() / ".dsh" / ".credentials.yaml")
MARGIN_FLOOR = 0.15


def split_by_family(cases: list) -> tuple:
    families = sorted({case["family"] for case in cases})
    train_families = {name for index, name in enumerate(families) if index % 2 == 0}
    train = [case for case in cases if case["family"] in train_families]
    held = [case for case in cases if case["family"] not in train_families]
    return train, held


def main() -> int:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    cases = [case for case in payload["cases"] if case.get("verbs")]
    train, held = split_by_family(cases)
    print(f"обучение: {len(train)}, отложено: {len(held)}")

    cascade = MicroCascade([(case["prompt"], case["verbs"][0]) for case in train])
    rows, cost, tokens = [], 0.0, 0
    answered = correct = abstained = failed = 0
    for index, case in enumerate(held, 1):
        want = case["verbs"][0]
        try:
            scores = score_options(case["prompt"], credentials_file=CREDS)
        except Exception as exc:                       # сеть/лимит/провайдер
            failed += 1
            print(f"  [{index}] провал: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        cost += scores.cost_usd
        tokens += scores.tokens
        confident = scores.margin >= MARGIN_FLOOR and scores.mass_on_options >= 0.5
        if confident:
            answered += 1
            correct += int(scores.best == want)
        else:
            abstained += 1
        decision = cascade.decide(case["prompt"])
        rows.append({
            "phrase": case["prompt"], "want": want,
            "scorer": scores.best, "margin": round(scores.margin, 3),
            "mass": round(scores.mass_on_options, 3),
            "confident": confident,
            "cascade": decision.verb if decision else None,
        })

    total = len(rows) + failed
    print()
    print(f"вызовов: {total}, провалов: {failed}")
    print(f"токенов: {tokens}, расход: ${cost:.6f}")
    if answered:
        print(f"оценка вариантов: отвечено {answered}, верно {correct}, "
              f"точность {correct / answered:.3f}, воздержаний {abstained}")
        print(f"покрытие: {answered / total:.0%}")
    cascade_answered = sum(1 for row in rows if row["cascade"])
    cascade_correct = sum(1 for row in rows if row["cascade"] == row["want"])
    print(f"каскад: отвечено {cascade_answered}, верно {cascade_correct}, "
          f"точность {cascade_correct / max(cascade_answered, 1):.3f}")
    both = [row for row in rows if row["confident"] and row["cascade"]]
    agree = sum(1 for row in both if row["scorer"] == row["cascade"])
    print(f"оба ответили на {len(both)} фразах, совпали в {agree} "
          f"({agree / max(len(both), 1):.0%})")
    disagree = [row for row in both if row["scorer"] != row["cascade"]]
    if disagree:
        print("\nрасхождения (кто прав по метке омеги):")
        for row in disagree[:10]:
            who = "оценка" if row["scorer"] == row["want"] else (
                "каскад" if row["cascade"] == row["want"] else "оба мимо")
            print(f"  {row['phrase'][:42]:44s} метка {row['want']:10s} "
                  f"оценка {row['scorer']:10s} каскад {row['cascade']:10s} -> {who}")
    out = ROOT / "eval" / "score_options_report.json"
    out.write_text(json.dumps({"answered": answered, "correct": correct,
                               "abstained": abstained, "failed": failed,
                               "cost_usd": round(cost, 6), "tokens": tokens,
                               "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\nотчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
