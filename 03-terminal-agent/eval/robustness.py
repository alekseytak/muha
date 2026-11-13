#!/usr/bin/env python3
"""Три проверки по следам материалов из ГЛАВНОЙ ПАПКИ.

1. Кривая обучения: растёт ли точность каскада от числа фраз? Если нет —
   «прогнать больше фраз» не поможет, и это видно без Colab.
2. Бутстрап-доверительный интервал для точности: у меня до сих пор были
   только точечные оценки, что для утверждения «0.986» неаккуратно.
   Метод взят из kem-bootloops-pilot/docs/E01_REPORT.md — там парные
   перевыборки и CI обязательны по протоколу.
3. Метод значимости из Математический_движок_октав.md: фон = 30-й процентиль,
   разброс считается только по точкам НИЖЕ фона, значимость
   σ = (пик − фон)/разброс. Применяем это к спектру вероятностей глаголов и
   сравниваем с моим самодельным порогом запаса.

Ничего не подгоняем: все пороги проверяются на отложенных семействах.
"""

import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.micromodels import MicroCascade, selective_report   # noqa: E402
from core.option_scorer import VERBS, score_options           # noqa: E402

DATA = ROOT / "eval" / "cases_distill.json"
CREDS = str(Path.home() / ".dsh" / ".credentials.yaml")
SEED = 20260101


def load():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    return [case for case in payload["cases"] if case.get("verbs")]


def by_family(cases):
    families = sorted({case["family"] for case in cases})
    train_f = {name for index, name in enumerate(families) if index % 2 == 0}
    return ([case for case in cases if case["family"] in train_f],
            [case for case in cases if case["family"] not in train_f],
            sorted(train_f))


def bootstrap_ci(hits, rounds=2000, seed=SEED):
    """Точечная оценка и 95% CI для доли по перевыборкам."""
    if not hits:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(hits)
    point = sum(hits) / n
    means = []
    for _ in range(rounds):
        sample = [hits[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return point, means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def learning_curve(cases):
    print("=== 1. Кривая обучения (отложенные семейства не участвуют) ===")
    train, held, families = by_family(cases)
    pairs = [(case["prompt"], case["verbs"][0]) for case in train]
    print(f"обучение: {len(train)} фраз в {len(families)} семействах, "
          f"отложено: {len(held)}")
    print(f"{'семейств':>9} {'фраз':>6} {'покрытие':>9} {'точность':>9} {'95% CI':>18}")
    for share in (0.25, 0.5, 0.75, 1.0):
        keep = max(1, int(len(families) * share))
        chosen = set(families[:keep])
        subset = [(case["prompt"], case["verbs"][0])
                  for case in train if case["family"] in chosen]
        cascade = MicroCascade(subset)
        report = selective_report(cascade, held)
        hits = [1 if error is None else 0 for error in [None]]  # placeholder
        # Пересчитываем попадания по фразам: selective_report отдаёт только
        # агрегаты и ошибки, поэтому считаем сами.
        decided = []
        for case in held:
            decision = cascade.decide(case["prompt"])
            if decision is not None:
                decided.append(int(decision.verb in case["verbs"]))
        point, low, high = bootstrap_ci(decided)
        print(f"{keep:>9} {len(subset):>6} {report['coverage']:>8.0%} "
              f"{point:>9.3f}  [{low:.3f}, {high:.3f}]")


def cascade_ci(cases):
    print()
    print("=== 2. Доверительный интервал точности каскада ===")
    train, held, _ = by_family(cases)
    cascade = MicroCascade([(case["prompt"], case["verbs"][0]) for case in train])
    decided, hits = 0, []
    for case in held:
        decision = cascade.decide(case["prompt"])
        if decision is None:
            continue
        decided += 1
        hits.append(int(decision.verb in case["verbs"]))
    point, low, high = bootstrap_ci(hits)
    print(f"решено {decided} из {len(held)} ({decided / len(held):.0%}), "
          f"точность {point:.3f}, 95% CI [{low:.3f}, {high:.3f}]")
    print(f"порог 0.95 внутри интервала: {'да' if low <= 0.95 <= high else 'нет'} "
          f"(нижняя граница {low:.3f})")


def octave_thresholds(cases):
    """Метод значимости из движка октав, применённый к вероятностям глаголов."""
    print()
    print("=== 3. Порог значимости σ против моего запаса ===")
    train, held, _ = by_family(cases)
    rows, cost = [], 0.0
    for case in held:
        try:
            scores = score_options(case["prompt"], credentials_file=CREDS)
        except Exception as exc:
            print(f"  пропуск: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        cost += scores.cost_usd
        probabilities = [scores.probabilities.get(verb, 0.0) for verb in VERBS]
        ordered = sorted(probabilities)
        # Фон и разброс — ровно как в движке октав: 30-й процентиль и std по
        # точкам ниже фона.
        background = ordered[min(int(0.3 * len(ordered)), len(ordered) - 1)]
        below = [value for value in ordered if value < background]
        spread = statistics.pstdev(below) if len(below) > 1 else 0.0
        peak = max(probabilities)
        sigma = (peak - background) / spread if spread > 0 else float("inf")
        rows.append({"want": case["verbs"][0], "best": scores.best,
                     "margin": round(scores.margin, 3), "sigma": sigma})
    if not rows:
        print("нет данных")
        return
    print(f"фраз: {len(rows)}, расход ${cost:.6f}")
    for name, rule in (
        ("запас ≥ 0.15 (мой)", lambda row: row["margin"] >= 0.15),
        ("запас ≥ 0.30", lambda row: row["margin"] >= 0.30),
        ("σ ≥ 2.0 (как в движке)", lambda row: row["sigma"] >= 2.0),
        ("σ ≥ 3.0", lambda row: row["sigma"] >= 3.0),
        ("σ ≥ 2.0 и запас ≥ 0.15", lambda row: row["sigma"] >= 2.0 and row["margin"] >= 0.15),
    ):
        chosen = [row for row in rows if rule(row)]
        if not chosen:
            print(f"  {name:28s}: не выбрано ничего")
            continue
        hits = [int(row["best"] == row["want"]) for row in chosen]
        point, low, high = bootstrap_ci(hits)
        print(f"  {name:28s}: покрытие {len(chosen) / len(rows):>4.0%} "
              f"точность {point:.3f}  [{low:.3f}, {high:.3f}]")


def main() -> int:
    cases = load()
    learning_curve(cases)
    cascade_ci(cases)
    octave_thresholds(cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())