#!/usr/bin/env python3
"""Проверка: рой против одиночного агента.

Критерии зафиксированы ДО прогона (см. AUDIT §30), чтобы потом не подгонять:

  K1. В рое не меньше трёх мух, и профили реально расходятся — то есть
      находятся фразы, где мухи голосуют по-разному.
  K2. Рой меняет решение одиночного пути не реже, чем в 10% фраз.
  K3. На семейном контроле рой НЕ ХУЖЕ одиночного агента по покрытию и по
      точности среди отвеченных.
  K4. Цена по умолчанию нулевая: модель зовётся только при расколе, и доля
      расколов известна.

Если K3 не выполняется — это тоже результат, и он будет записан как есть.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.agent import Agent                                  # noqa: E402
from core.micromodels import MicroCascade                     # noqa: E402
from core.swarm_agents import FLIES, FlySwarm                 # noqa: E402


def family_holdout() -> list:
    """Нечётные семейства — те же 152 фразы, что в eval/hierarchical.py."""
    payload = json.loads((ROOT / "eval" / "cases_distill.json")
                         .read_text(encoding="utf-8"))
    cases = payload.get("cases") or payload
    families = sorted({c["family"] for c in cases if c.get("family")})
    train = {name for index, name in enumerate(families) if index % 2 == 0}
    return [c for c in cases if c.get("family") not in train]


def measure(runner, held) -> dict:
    answered = correct = 0
    wrong = []
    for case in held:
        act = runner(case["prompt"])
        if act is None:
            continue
        answered += 1
        verb = act.verb.value if hasattr(act.verb, "value") else str(act.verb)
        if verb in case["verbs"]:
            correct += 1
        else:
            wrong.append({"phrase": case["prompt"], "expected": case["verbs"],
                          "got": verb})
    return {
        "answered": answered,
        "total": len(held),
        "coverage": round(answered / len(held), 4),
        "correct": correct,
        "accuracy_answered": round(correct / answered, 4) if answered else 0.0,
        "accuracy_all": round(correct / len(held), 4),
        "errors": wrong[:20],
    }


def main() -> int:
    held = family_holdout()
    agent = Agent(config_path=str(ROOT / "config.yaml"),
                  confirm_policy="allow")
    agent.llm = None
    agent.fallback_llm = None

    single = measure(agent.propose_action_fast, held)
    swarm = measure(agent.propose_action_swarm, held)

    # K1: расходятся ли мухи между собой.
    # Проба берёт НАСТОЯЩИЕ движки агента: если собрать её из одного
    # каскада, все мухи смотрят одинаково и расходов не будет по построению.
    probe = FlySwarm(agent.fly_swarm.cascades)
    diverge_inside = 0
    splits = 0
    for case in held:
        decision = probe.vote(case["prompt"])
        if decision.needs_omega:
            splits += 1
        spoken = {vote.verb for vote in decision.votes if vote.spoke}
        if len(spoken) > 1:
            diverge_inside += 1

    # K2: меняет ли рой ответ одиночного пути.
    changed = 0
    examples = []
    for case in held:
        one = agent.propose_action_fast(case["prompt"])
        many = agent.propose_action_swarm(case["prompt"])
        one_verb = (one.verb.value if one is not None else None)
        many_verb = (many.verb.value if many is not None else None)
        if one_verb != many_verb:
            changed += 1
            if len(examples) < 8:
                examples.append({"phrase": case["prompt"], "single": one_verb,
                                 "swarm": many_verb})

    # Отдельно проверить, добавляет ли ЧТО-ТО представление фразы:
    # считаем, где структурная муха сказала другое или промолчала.
    other = silent = 0
    probe_repr = FlySwarm(agent.fly_swarm.cascades)
    for case in held:
        decision = probe_repr.vote(case["prompt"])
        views = {vote.fly: vote for vote in decision.votes}
        raw = views.get("полный")
        masked = views.get("структурная")
        if raw is None or masked is None:
            continue
        if masked.verb is None and raw.verb is not None:
            silent += 1
        elif masked.verb is not None and masked.verb != raw.verb:
            other += 1

    report = {
        "dataset": "eval/cases_distill.json",
        "holdout": "нечётные семейства",
        "phrases": len(held),
        "files_count": len(FLIES),
        "single": single,
        "swarm": swarm,
        "K1_flies_diverge_phrases": diverge_inside,
        "K2_changed_phrases": changed,
        "K2_changed_share": round(changed / len(held), 4),
        "K4_splits": splits,
        "K4_split_share": round(splits / len(held), 4),
        "change_examples": examples,
        "representation_other_verb": other,
        "representation_silent": silent,
    }
    report["K3_swarm_not_worse"] = (
        swarm["accuracy_answered"] >= single["accuracy_answered"]
        and swarm["coverage"] >= single["coverage"]
    )

    print(f"фраз: {len(held)}, мух: {len(FLIES)}")
    print(f"одиночный: покрытие {single['coverage']:.1%} "
          f"({single['answered']}/{len(held)}), точность среди отвеченных "
          f"{single['accuracy_answered']:.3f}, от всех {single['accuracy_all']:.3f}")
    print(f"рой:       покрытие {swarm['coverage']:.1%} "
          f"({swarm['answered']}/{len(held)}), точность среди отвеченных "
          f"{swarm['accuracy_answered']:.3f}, от всех {swarm['accuracy_all']:.3f}")
    print(f"K1 мухи разошлись между собой: {diverge_inside} фраз")
    print(f"K2 рой изменил решение: {changed} фраз "
          f"({changed / len(held):.1%})")
    print(f"K3 рой не хуже одиночного: {report['K3_swarm_not_worse']}")
    print(f"K4 расколов (сюда пошла бы модель): {splits} ({splits / len(held):.1%})")
    print(f"представление «маска»: другой глагол {other}, промолчала {silent}")
    if examples:
        print("примеры расхождений:")
        for item in examples:
            print(f"  {item['phrase'][:44]:46s} одиночный={item['single']} "
                  f"рой={item['swarm']}")

    out = ROOT / "eval" / "swarm_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nотчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())