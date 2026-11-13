#!/usr/bin/env python3
"""Разметка корпуса омегой (дистилляция) — метки ставит большая модель.

Почему не правила: правила μ1a писал я, и размечать ими корпус значит
проверять правила на самих себе. Здесь метка приходит от той модели, которую
каскад замещает, поэтому замер отвечает на вопрос «насколько хорошо дешёвая
модель воспроизводит дорогую», а не «согласна ли она сама с собой».

Экономия: батчи по 20 фраз, temperature 0, докачка с места обрыва,
учёт реального usage из ответа. Ключи не печатаются.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.llm_client import LLMClient          # noqa: E402
from core.micromodels import DOMAIN_CUES, VERBS  # noqa: E402

RAW = ROOT / "eval" / "cases_distill_raw.json"
OUT = ROOT / "eval" / "cases_distill.json"

DOMAINS = ("ui", "filesystem", "system", "registry", "swarm", "connectome", "web")
BATCH = 20
PRICE_PER_TOKEN = 0.000000038        # oneprovider/deepseek-v4.1-flash, $/токен

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


def load() -> dict:
    return json.loads(RAW.read_text(encoding="utf-8"))


def already_labeled() -> dict:
    if not OUT.exists():
        return {}
    payload = json.loads(OUT.read_text(encoding="utf-8"))
    return {case["id"]: case for case in payload.get("cases", [])}


def parse_batch(answer: str, size: int) -> dict:
    """Разбирает ответ построчно. Всё, что не разобралось, честно теряется."""
    found = {}
    for line in answer.splitlines():
        line = line.strip().lstrip("-•* ").strip()
        if not line or "." not in line:
            continue
        head, _, tail = line.partition(".")
        head = head.strip()
        if not head.isdigit():
            continue
        number = int(head)
        if not 1 <= number <= size:
            continue
        parts = [piece.strip().lower() for piece in tail.split("|")]
        if len(parts) < 2:
            continue
        verb, domain = parts[0], parts[1]
        if verb not in VERBS or domain not in DOMAINS:
            continue
        found[number] = {"verbs": [verb], "domains": [domain]}
    return found


def main() -> int:
    payload = load()
    phrases = payload["phrases"]
    done = already_labeled()
    todo = [item for item in phrases if item["id"] not in done]
    print(f"всего фраз: {len(phrases)}, уже размечено: {len(done)}, осталось: {len(todo)}")
    if not todo:
        print("разметка полная, делать нечего")
        return 0

    client = LLMClient(
        base_url="https://api.oneprovider.dev/v1",
        model="deepseek-v4.1-flash",
        provider="oneprovider",
        temperature=0.0,
        max_tokens=2500,          # модель рассуждающая: 700 уходило на мысли
        timeout=90,
        api_key_env="ONEPROVIDER_API_KEY",
        credentials_file=Path.home() / ".dsh" / ".credentials.yaml",
    )

    failures = []

    def label_chunk(chunk: list, depth: int = 0) -> None:
        """Размечает пачку; если разобралась половина — делит её пополам.

        Дробление дешевле слепого повтора: причиной провала обычно бывает
        обрезанный ответ, а не плохие фразы.
        """
        numbered = "\n".join(f"{i}. {item['phrase']}" for i, item in enumerate(chunk, 1))
        answer = ""
        for attempt in (1, 2, 3):
            try:
                answer = client.chat(numbered, system_prompt=SYSTEM)
                break
            except Exception as exc:                        # сеть/лимит/ключ
                print(f"  попытка {attempt} не удалась: {type(exc).__name__}: {exc}")
                time.sleep(2 * attempt)
        parsed = parse_batch(answer, len(chunk))
        for index, item in enumerate(chunk, 1):
            if index in parsed:
                done[item["id"]] = {
                    "id": item["id"], "family": item["family"],
                    "prompt": item["phrase"], **parsed[index],
                }
        missing = [item for index, item in enumerate(chunk, 1) if index not in parsed]
        if not missing:
            return
        if depth < 3 and len(missing) < len(chunk) or depth < 2:
            if len(missing) > 1:
                middle = len(missing) // 2
                label_chunk(missing[:middle], depth + 1)
                label_chunk(missing[middle:], depth + 1)
                return
        failures.extend(item["id"] for item in missing)

    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        label_chunk(chunk)
        OUT.write_text(json.dumps({
            "meta": {
                "name": "distill-label",
                "labeled_by": "oneprovider/deepseek-v4.1-flash",
                "temperature": 0.0,
                "calls": client.total_calls,
                "prompt_tokens": client.total_prompt_tokens,
                "completion_tokens": client.total_completion_tokens,
                "est_cost_usd": round(
                    (client.total_prompt_tokens + client.total_completion_tokens)
                    * PRICE_PER_TOKEN, 6),
                "unparsed": failures,
            },
            "cases": sorted(done.values(), key=lambda case: case["id"]),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        spent = client.total_prompt_tokens + client.total_completion_tokens
        print(f"  батч {start // BATCH + 1}: всего размечено {len(done)}/{len(phrases)}, "
              f"расход {spent} ток.")

    tokens = client.total_prompt_tokens + client.total_completion_tokens
    print(f"\nвызовов: {client.total_calls}, токенов: {tokens}, "
          f"оценка расходов: ${tokens * PRICE_PER_TOKEN:.5f}")
    print(f"не размечено: {len(failures)}")
    print(f"файл: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())