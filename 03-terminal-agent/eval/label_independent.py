#!/usr/bin/env python3
"""Разметка 80 фраз НЕЗАВИСИМЫМ учителем — моделью другого семейства.

Зачем. Все текущие метки поставлены `deepseek-v4.1-flash`, то есть той же
моделью, что и разбирает фразы. Согласие с такой меткой не является
проверкой качества. Здесь метки ставит `qwen3.8-flash` — другое семейство,
другая обученность. Это всё ещё НЕ человеческий gold, но это уже
независимая от нашего разбора точка зрения.

Правило интерпретации зафиксировано заранее, до прогона:
  расхождение < 10%  -> текущие метки, вероятно, не переобучены под одного
                        учителя; можно писать «согласие двух моделей»;
  10-25%             -> метки шумные, точность выше 0.95 подозрительна;
  > 25%              -> корпус надо переразмечать, и это будет сказано прямо.

Стоимость: 80 фраз ≈ $0.002. Модель дешёвая, но не бесплатная, поэтому
запросов ровно столько, сколько нужно.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.agent import Agent                            # noqa: E402

TEACHER = "qwen3.8-flash"
IN = ROOT / "eval" / "gold_candidates.tsv"
OUT = ROOT / "eval" / "gold_independent.tsv"
REPORT = ROOT / "eval" / "gold_independent_report.json"

VERBS = ("read", "write", "execute", "think", "log", "version", "anchor",
         "veto", "escalate", "recommend", "select")
DOMAINS = ("ui", "filesystem", "system", "registry", "swarm", "connectome",
           "web")

SYSTEM = (
    "Ты разметчик команд терминального агента. Для каждой фразы верни глагол "
    "действия и домен.\n"
    f"Допустимые глаголы: {', '.join(VERBS)}.\n"
    f"Допустимые домены: {', '.join(DOMAINS)}.\n"
    "Формат ответа — строго по одной строке на фразу, без пояснений:\n"
    "N. глагол | домен\n"
    "Пример:\n1. read | filesystem\n2. execute | system\n"
    "Если фраза просит прочитать веб-страницу — домен web. Если просит "
    "запустить команду оболочки — system. Если выбор варианта — select и ui."
)


def load_rows() -> list:
    with IN.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse(answer: str, count: int) -> dict:
    """Разобрать нумерованный ответ. Кривые строки не выдумываются."""
    found = {}
    for line in (answer or "").splitlines():
        line = line.strip()
        if not line or "." not in line:
            continue
        head, _, tail = line.partition(".")
        try:
            index = int(head.strip())
        except ValueError:
            continue
        parts = [part.strip().strip("`*") for part in tail.split("|")]
        if not parts or parts[0] not in VERBS:
            continue
        domain = parts[1] if len(parts) > 1 and parts[1] in DOMAINS else "ui"
        if 1 <= index <= count:
            found[index] = {"verb": parts[0], "domain": domain}
    return found


def main() -> int:
    rows = load_rows()
    agent = Agent(config_path=str(ROOT / "config.yaml"),
                  confirm_policy="allow")
    agent.llm.model = TEACHER

    numbered = "\n".join(f"{index}. {row['prompt']}"
                         for index, row in enumerate(rows, start=1))
    answer = agent.llm.chat(numbered, system_prompt=SYSTEM)
    labels = parse(answer, len(rows))

    missing = [index for index in range(1, len(rows) + 1) if index not in labels]
    for index, row in enumerate(rows, start=1):
        label = labels.get(index) or {}
        row["ind_verb"] = label.get("verb", "")
        row["ind_domain"] = label.get("domain", "")

    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t",
                                fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Сверить независимую разметку с тем, что делает наш каскад.
    agree_verb = agree_domain = compared = 0
    disagreements = []
    for row in rows:
        if not row.get("ind_verb"):
            continue
        act = agent.propose_action_fast(row["prompt"])
        if act is None:
            continue
        compared += 1
        same_verb = act.verb.value == row["ind_verb"]
        if same_verb:
            agree_verb += 1
        if act.domain == row["ind_domain"]:
            agree_domain += 1
        if not same_verb:
            disagreements.append({
                "id": row["id"], "prompt": row["prompt"],
                "наш": act.verb.value, "учитель": row["ind_verb"],
                "домен_учителя": row["ind_domain"],
            })

    share = (len(disagreements) / compared) if compared else 0.0
    if share < 0.10:
        verdict = ("расхождение ниже 10%: метки, вероятно, не переобучены "
                   "под одного учителя; публиковать как согласие двух моделей")
    elif share <= 0.25:
        verdict = ("расхождение 10-25%: метки шумные, точность выше 0.95 "
                   "подозрительна")
    else:
        verdict = ("расхождение выше 25%: корпус надо переразмечать")

    report = {
        "teacher": TEACHER,
        "phrases": len(rows),
        "labelled": len(labels),
        "unparsed": missing,
        "compared": compared,
        "agree_verb": agree_verb,
        "agree_domain": agree_domain,
        "disagreement_share": round(share, 4),
        "verdict": verdict,
        "disagreements": disagreements[:25],
        "prompt_tokens": getattr(agent.llm, "total_prompt_tokens", None),
        "completion_tokens": getattr(agent.llm, "total_completion_tokens", None),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")

    print(f"учитель: {TEACHER}")
    print(f"размечено {len(labels)} из {len(rows)}"
          + (f", не разобрано: {missing}" if missing else ""))
    print(f"сверено с каскадом: {compared}")
    print(f"совпал глагол: {agree_verb}/{compared} "
          f"({agree_verb / compared:.1%})" if compared else "сверять нечего")
    print(f"совпал домен: {agree_domain}/{compared}")
    print(f"расхождение: {share:.1%} -> {verdict}")
    if disagreements:
        print("где разошлись:")
        for item in disagreements[:10]:
            print(f"  {item['id']} {item['prompt'][:40]:42s} "
                  f"наш={item['наш']:9s} учитель={item['учитель']}")
    print(f"\nметки: {OUT}\nотчёт: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())