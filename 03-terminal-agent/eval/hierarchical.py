#!/usr/bin/env python3
"""Сравнение плоского каскада и иерархии L0–L5 на отложенных семействах.

Уровни взяты из `OS-Glagolov-spec.md:41-49` (см.
`ГЛАВНАЯ ПАПКА/ОС_глаголов_система.md`). Боевой класс — `core/hierarchy.py`,
здесь только замер.

История замера (важно, чтобы не повторять ошибку):
  1. первый вариант спрашивал уровень ПЕРВЫМ и дал 96%/0.993 — это был
     артефакт: модель уровней вообще не обучалась (метки L0–L5 отбрасывались
     фильтром по VERBS), а покрытие росло из лазейки «в уровне один глагол —
     отвечаем без улики»;
  2. после исправления обучения и закрытия лазейки иерархия дала 68%/0.495 —
     потому что ошибка уровня выбрасывала верную улику правила;
  3. итоговый порядок «правила первыми, иерархия только при молчании правил»
     даёт то, что напечатано ниже.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.hierarchy import HierarchicalCascade        # noqa: E402
from core.micromodels import MicroCascade, selective_report  # noqa: E402

DATA = ROOT / "eval" / "cases_distill.json"
SEED = 20260101


def bootstrap(hits, rounds=2000, seed=SEED):
    """Обычный bootstrap по отдельным фразам.

    Оставлен для сравнения, но основной вывод теперь на cluster bootstrap:
    фразы внутри шаблонного семейства зависимы, поэтому перевыборка отдельных
    фраз занижает интервал и выдаёт больше уверенности, чем есть.
    """
    if not hits:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(hits)
    means = []
    for _ in range(rounds):
        sample = [hits[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return sum(hits) / n, means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def cluster_bootstrap(rows, rounds=2000, seed=SEED):
    """Bootstrap по семействам: перевыбираем семьи, а не отдельные фразы.

    Только он отвечает на вопрос «как покажет себя на НОВЫХ формулировках»:
    фразы одного семейства — это одна и та же форма запроса.
    """
    if not rows:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    by_family = {}
    for family, hit in rows:
        by_family.setdefault(family, []).append(hit)
    families = list(by_family)
    total = len(rows)
    point = sum(hit for _, hit in rows) / total
    means = []
    for _ in range(rounds):
        chosen = [families[rng.randrange(len(families))]
                  for _ in range(len(families))]
        hits = sum(sum(by_family[name]) for name in chosen)
        size = sum(len(by_family[name]) for name in chosen)
        means.append(hits / size if size else 0.0)
    means.sort()
    return point, means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def main() -> int:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    cases = [case for case in payload["cases"] if case.get("verbs")]
    families = sorted({case["family"] for case in cases})
    train_f = {name for index, name in enumerate(families) if index % 2 == 0}
    train = [case for case in cases if case["family"] in train_f]
    held = [case for case in cases if case["family"] not in train_f]
    pairs = [(case["prompt"], case["verbs"][0]) for case in train]
    print(f"обучение {len(train)}, отложено {len(held)}, "
          f"семейств в обучении {len(train_f)}")

    flat, hier = MicroCascade(pairs), HierarchicalCascade(pairs)
    for name, cascade in (("плоский каскад", flat), ("иерархия L0–L5", hier)):
        report = selective_report(cascade, held)
        rows, sources = [], {}
        for case in held:
            decision = cascade.decide(case["prompt"])
            if decision is None:
                continue
            hit = 1 if decision.verb in case["verbs"] else 0
            rows.append((case["family"], hit))
            key = decision.source.split(" (")[0]
            sources[key] = sources.get(key, 0) + 1
        point, low, high = cluster_bootstrap(rows)
        plain = bootstrap([hit for _, hit in rows])
        print(f"{name:16s}: покрытие {report['coverage']:>4.0%} "
              f"({len(rows)}/{len(held)}) точность {point:.3f} "
              f"95% CI по семьям [{low:.3f}, {high:.3f}] "
              f"(по фразам [{plain[1]:.3f}, {plain[2]:.3f}])")
        print(f"{'':16s}  источники решений: {sources}")

    # Регрессия: там, где плоский каскад решал, иерархия обязана решать так же.
    changed = []
    for case in held:
        left = flat.decide(case["prompt"])
        right = hier.decide(case["prompt"])
        if left is not None and (right is None or right.verb != left.verb):
            changed.append((case["prompt"], left.verb, right.verb if right else None))
    print(f"\nизменённых ответов на решённых плоским каскадом: {len(changed)}")
    for phrase, was, now in changed[:5]:
        print(f"  {phrase[:44]:46s} было {was:10s} стало {now}")

    extra = [(case, hier.decide(case["prompt"])) for case in held
             if flat.decide(case["prompt"]) is None]
    print(f"\nиерархия добавила решений: {len(extra)}")
    for case, decision in extra:
        if decision is None:
            continue
        mark = "верно" if decision.verb in case["verbs"] else "МИМО"
        rule = decision.notes[-1] if decision.notes else ""
        print(f"  {case['prompt'][:40]:42s} -> {decision.verb:10s} "
              f"ждали {case['verbs'][0]:10s} {mark:5s} | {rule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())