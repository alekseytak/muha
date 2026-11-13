"""Оценка вариантов через вероятности токенов (logprobs), а не через разбор текста.

Идея, ради которой это сделано: не просить модель «написать глагол», а прочитать
вероятности, с которыми она готова выдать каждый из одиннадцати глаголов. Тогда
решение — это argmax по распределению с честным запасом, а не парсинг строки,
который может не удаться.

Это перенос идеи SemIf-OpenJev на доступный интерфейс: там вероятности читаются
из локальной модели, здесь — из OpenAI-совместимого API.

Проверено на живом API (см. AUDIT.md):
  * oneprovider игнорирует logprobs и возвращает null;
  * OpenRouter отдаёт logprobs, но только у провайдера, который их поддерживает —
    поэтому нужен provider.require_parameters = true;
  * mistralai/mistral-nemo (DekaLLM) отдал распределение:
    read 0.819 / write 0.001 на фразе «прочитай файл config.yaml».
"""

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

try:
    from .llm_client import build_ssl_context, resolve_api_key
except ImportError:                               # запуск из eval/ как скрипта
    from core.llm_client import build_ssl_context, resolve_api_key

VERBS: Sequence[str] = ("read", "write", "execute", "think", "log", "version",
                        "anchor", "veto", "escalate", "recommend", "select")
DEFAULT_MODEL = "mistralai/mistral-nemo"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM = ("Classify the user's request into exactly one action word. "
          "Available action words: " + ", ".join(VERBS) + ". "
          "Reply with one word from that list and nothing else.")
# Сколько верхних токенов просить. Мало — потеряем варианты, много — не
# поддерживается провайдерами.
TOP_LOGPROBS = 20


class ScoringUnavailable(RuntimeError):
    """Провайдер не дал распределение — решать по нему нельзя."""


@dataclass
class OptionScores:
    """Результат оценки вариантов по одному запросу."""

    probabilities: Dict[str, float] = field(default_factory=dict)
    best: Optional[str] = None
    best_prob: float = 0.0
    margin: float = 0.0
    mass_on_options: float = 0.0
    model: str = ""
    provider: str = ""
    tokens: int = 0
    cost_usd: float = 0.0
    tokens_seen: List[str] = field(default_factory=list)

    def as_line(self) -> str:
        top = ", ".join(f"{verb} {prob:.2f}" for verb, prob
                        in sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:3])
        return f"{self.best} (запас {self.margin:.2f}) | {top}"


def _canonical(token: str) -> str:
    """Токен к нормальному виду: пробелы, регистр, знаки по краям.

    Одна и та же буква приходит как «read», « read», «read\\n» — это один
    вариант, и его вероятности нужно складывать, а не выбирать максимум.
    """
    return token.strip().strip("\"'«».,:;\n\t ").lower()


def aggregate(top_logprobs: List[dict], options: Sequence[str]) -> tuple:
    """Складывает вероятности по каноническим токенам, оставляя только варианты.

    Возвращает (вероятности по вариантам, доля показанной массы на вариантах,
    увиденные токены). Отсутствующий в top-logprobs вариант получает 0 — это
    нижняя оценка, а не «вероятность ноль» в строгом смысле.
    """
    allowed = {option.lower() for option in options}
    probabilities: Dict[str, float] = {option: 0.0 for option in options}
    total_shown = 0.0
    seen: List[str] = []
    for item in top_logprobs:
        token = _canonical(str(item.get("token") or ""))
        if not token:
            continue
        seen.append(token)
        probability = math.exp(float(item.get("logprob") or -100.0))
        total_shown += probability
        if token in allowed:
            probabilities[token] += probability
    mass = sum(probabilities.values())
    # Доля показанной массы, которая пришлась на допустимые варианты. Низкая
    # доля значит, что модель отвечает «не из списка» — доверять выбору нельзя.
    share = (mass / total_shown) if total_shown > 0 else 0.0
    # Нормируем на массу вариантов: нас интересует выбор среди них, а не
    # абсолютная калибровка всего словаря.
    if mass > 0:
        probabilities = {key: value / mass for key, value in probabilities.items()}
    return probabilities, share, seen


def score_options(phrase: str, options: Sequence[str] = VERBS,
                  model: str = DEFAULT_MODEL, api_key_env: str = "OPENROUTER_API_KEY",
                  credentials_file: Optional[str] = None, url: str = OPENROUTER_URL,
                  timeout: int = 90, retries: int = 3,
                  require_parameters: bool = True) -> OptionScores:
    """Просит у модели распределение по вариантам и возвращает его разбор."""
    key = resolve_api_key(api_key_env, credentials_file)
    if not key:
        raise ScoringUnavailable(f"нет ключа {api_key_env}")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": phrase}],
        "temperature": 0,
        "max_tokens": 3,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
    }
    if require_parameters:
        # Без этого OpenRouter может выбрать провайдера без поддержки logprobs
        # и вернуть logprobs: null — что и случилось с gemma-3-4b-it (Parasail).
        payload["provider"] = {"require_parameters": True}
    body = json.dumps(payload).encode("utf-8")
    last_error = ""
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout,
                                        context=build_ssl_context()) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.read()[:200]!r}"
            if exc.code in (429, 502, 503, 504) and attempt < retries:
                import time
                time.sleep(2 ** attempt)          # 429 у бесплатных моделей — норма
                continue
            raise ScoringUnavailable(last_error) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                import time
                time.sleep(2 ** attempt)
                continue
            raise ScoringUnavailable(last_error) from exc

        choice = (data.get("choices") or [{}])[0]
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        items = ((choice.get("logprobs") or {}).get("content") or [])
        if not items:
            raise ScoringUnavailable(
                "провайдер не вернул logprobs "
                f"(content={content!r}); попробуйте другую модель или "
                "provider.require_parameters"
            )
        # Первый содержательный токен — это и есть ответ. Пустые пропускаем.
        first = None
        for item in items:
            if _canonical(str(item.get("token") or "")):
                first = item
                break
        if first is None:
            raise ScoringUnavailable("logprobs есть, но все токены пустые")
        probabilities, mass, seen = aggregate(first.get("top_logprobs") or [], options)
        ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        usage = data.get("usage") or {}
        return OptionScores(
            probabilities=probabilities,
            best=ranked[0][0],
            best_prob=ranked[0][1],
            margin=ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0),
            mass_on_options=mass,
            model=str(data.get("model") or model),
            provider=str(data.get("provider") or ""),
            tokens=int(usage.get("total_tokens") or 0),
            cost_usd=float(usage.get("cost") or 0.0),
            tokens_seen=seen[:TOP_LOGPROBS],
        )
    raise ScoringUnavailable(last_error or "не удалось получить распределение")