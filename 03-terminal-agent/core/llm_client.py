"""Клиент LLM: [OI]-совместимые провайдеры и Ollama.

Поддерживаются два протокола:
  * `openai` — POST {base_url}/chat/completions (oneprovider, literouter,
    openrouter и любой другой [OI]-совместимый шлюз);
  * `ollama` — POST {base_url}/api/chat (локальная модель).

Ключи НЕ хранятся в конфиге: значение берётся из переменной окружения
(`api_key_env`), затем из YAML-файла с учётными данными
(`credentials_file`, формат `VAR_NAME: value`).
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _as_data_url(image: str) -> str:
    """Путь к файлу или готовый data-URL — в вид, понятный провайдеру.

    Если строка уже начинается с `data:`, она возвращается как есть: так
    вызывающий может передать готовый URL и не заставлять нас читать файл.
    """
    if image.startswith("data:"):
        return image
    if image.startswith("http://") or image.startswith("https://"):
        return image
    path = Path(image)
    if not path.exists():
        raise LLMUnavailable(f"картинка не найдена: {image}")
    suffix = path.suffix.lower().lstrip(".") or "png"
    if suffix == "jpg":
        suffix = "jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{payload}"


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str
    # Картинки, которые нужно показать модели. Пути к файлам или готовые
    # data-URL. Без этого поля сообщение остаётся строкой и уходит как
    # строка: обычный текстовый путь не меняется.
    images: Optional[List[str]] = None


@dataclass
class LLMResponse:
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None
    model: str = ""
    provider: str = ""
    # Расход по данным провайдера. None означает «провайдер не сообщил»,
    # а не «ноль»: это разные вещи, и путать их в отчётах о стоимости нельзя.
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMUnavailable(RuntimeError):
    """Провайдер недоступен: сеть, ключ, HTTP-ошибка."""


def _ca_candidates(explicit: Optional[str]) -> List[str]:
    """Где искать корневые сертификаты: конфиг, certifi, системный бандл."""
    found: List[str] = []
    if explicit:
        found.append(os.path.expanduser(str(explicit)))
    try:
        import certifi  # необязательная зависимость
        found.append(certifi.where())
    except Exception:
        pass
    found.extend([
        "/etc/ssl/cert.pem",              # macOS
        "/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu
        "/etc/pki/tls/certs/ca-bundle.crt",    # RHEL/Fedora
        "/usr/local/etc/openssl/cert.pem",
    ])
    return [path for path in found if path and os.path.exists(path)]


def build_ssl_context(verify: bool = True, ca_bundle: Optional[str] = None):
    """SSL-контекст с рабочим бандлом CA.

    Штатный `ssl.create_default_context()` во framework-сборках Python на macOS
    не находит корневые сертификаты (`CERTIFICATE_VERIFY_FAILED`), поэтому пути
    перебираются явно.
    """
    import ssl

    if not verify:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    for candidate in _ca_candidates(ca_bundle):
        try:
            return ssl.create_default_context(cafile=candidate)
        except Exception:
            continue
    return ssl.create_default_context()


def load_credentials(path: str | Path | None) -> Dict[str, str]:
    """Читает плоский YAML `KEY: value` с учётными данными."""
    if not path:
        return {}
    target = Path(os.path.expanduser(str(path)))
    if not target.exists():
        return {}
    try:
        from .miniyaml import loads
        data = loads(target.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v is not None}


def resolve_api_key(env_name: Optional[str], credentials_file: Optional[str]) -> Optional[str]:
    """Ключ из окружения, иначе из файла учётных данных."""
    if env_name:
        value = os.environ.get(env_name)
        if value:
            return value.strip()
    creds = load_credentials(credentials_file)
    if env_name and env_name in creds:
        return creds[env_name].strip()
    return None


class LLMClient:
    """Единый клиент поверх разных провайдеров."""

    def __init__(
        self,
        base_url: str = "https://api.oneprovider.dev/v1",
        model: str = "deepseek-v4.1-flash",
        provider: str = "oneprovider",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: int = 60,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        credentials_file: Optional[str] = None,
        tls_verify: bool = True,
        ca_bundle: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key_env = api_key_env
        self.credentials_file = credentials_file
        self.tls_verify = tls_verify
        self.ca_bundle = ca_bundle
        self.seed = seed
        # Накопительная статистика расхода: нужна, чтобы автономный цикл мог
        # отчитаться, во что обошёлся прогон, а не «примерно».
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.unreported_calls = 0
        self._api_key = api_key
        self._ssl_context = None

    # ---------- конфигурация ----------

    @property
    def api_style(self) -> str:
        """`ollama` для локального сервера, иначе [OI]-совместимый."""
        return "ollama" if self.provider.lower() == "ollama" else "openai"

    @property
    def ssl_context(self):
        if self._ssl_context is None:
            self._ssl_context = build_ssl_context(self.tls_verify, self.ca_bundle)
        return self._ssl_context

    @property
    def api_key(self) -> Optional[str]:
        if self._api_key:
            return self._api_key
        self._api_key = resolve_api_key(self.api_key_env, self.credentials_file)
        return self._api_key

    def reset_counters(self) -> None:
        """Обнулить счётчики расхода (перед замером прогона)."""
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.unreported_calls = 0

    def usage_report(self) -> Dict[str, Any]:
        """Сколько запросов и токенов потрачено этим клиентом."""
        return {
            "provider": self.provider,
            "model": self.model,
            "calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "calls_without_usage": self.unreported_calls,
        }

    def _account(self, response: "LLMResponse") -> "LLMResponse":
        """Учесть расход по ответу. Отсутствие usage считаем отдельно."""
        self.total_calls += 1
        if response.total_tokens is None and response.prompt_tokens is None:
            self.unreported_calls += 1
        else:
            self.total_prompt_tokens += response.prompt_tokens or 0
            self.total_completion_tokens += response.completion_tokens or 0
        return response

    def describe(self) -> str:
        key_state = "ключ есть" if self.api_key else "ключа нет"
        return f"{self.provider}/{self.model} ({self.api_style}, {key_state})"

    # ---------- проверка доступности ----------

    def is_available(self) -> bool:
        """Сервер отвечает и отвечает ли он на запросы (без генерации)."""
        try:
            if self.api_style == "ollama":
                req = urllib.request.Request(f"{self.base_url}/api/tags")
                with urllib.request.urlopen(req, timeout=5, context=self.ssl_context) as resp:
                    return resp.status == 200
            if not self.api_key:
                return False
            req = urllib.request.Request(
                f"{self.base_url}/models",
                headers=self._headers(),
            )
            with urllib.request.urlopen(req, timeout=8, context=self.ssl_context) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Список моделей провайдера (может быть недоступен)."""
        try:
            if self.api_style == "ollama":
                req = urllib.request.Request(f"{self.base_url}/api/tags")
                with urllib.request.urlopen(req, timeout=5, context=self.ssl_context) as resp:
                    data = json.loads(resp.read().decode())
                    return [m.get("name", "") for m in data.get("models", [])]
            if not self.api_key:
                return []
            req = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=10, context=self.ssl_context) as resp:
                data = json.loads(resp.read().decode())
                return [m.get("id", "") for m in data.get("data", [])]
        except Exception:
            return []

    def ensure_model_available(self) -> bool:
        models = self.list_models()
        if not models:  # провайдер не отдаёт список — не блокируем
            return bool(self.api_key) or self.api_style == "ollama"
        return any(self.model in m for m in models)

    # ---------- генерация ----------

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def generate(
        self,
        messages: List[LLMMessage],
        system_prompt: Optional[str] = None,
        format_json: bool = False,
    ) -> LLMResponse:
        """Генерирует ответ. Бросает LLMUnavailable при сбое."""
        if self.api_style == "ollama":
            return self._generate_ollama(messages, system_prompt, format_json)
        return self._generate_openai(messages, system_prompt, format_json)

    def _generate_ollama(
        self,
        messages: List[LLMMessage],
        system_prompt: Optional[str],
        format_json: bool,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if format_json:
            payload["format"] = "json"

        data = self._post(f"{self.base_url}/api/chat", payload)
        message = data.get("message", {}) or {}
        return self._account(LLMResponse(
            content=message.get("content", ""),
            tool_calls=message.get("tool_calls") or [],
            raw=data,
            model=self.model,
            provider=self.provider,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_tokens=(
                (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
                if data.get("prompt_eval_count") is not None
                or data.get("eval_count") is not None
                else None
            ),
        ))

    def _generate_openai(
        self,
        messages: List[LLMMessage],
        system_prompt: Optional[str],
        format_json: bool,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMUnavailable(
                f"нет ключа для {self.provider}: задайте переменную "
                f"{self.api_key_env or '<api_key_env>'} или credentials_file"
            )
        chat: List[Dict[str, Any]] = []
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        for message in messages:
            if message.images:
                # Мультимодальное сообщение: текст и картинки одним списком
                # частей, как того требует OpenAI-совместимый API.
                parts: List[Dict[str, Any]] = [
                    {"type": "text", "text": message.content}]
                for image in message.images:
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": _as_data_url(image)},
                    })
                chat.append({"role": message.role, "content": parts})
            else:
                chat.append({"role": message.role, "content": message.content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": chat,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.seed is not None:
            # Не все провайдеры поддерживают seed; молча добавляем и смотрим
            # на ответ — при отказе провайдер вернёт 4xx, а не тихо проигнорирует.
            payload["seed"] = self.seed
        if format_json:
            payload["response_format"] = {"type": "json_object"}

        data = self._post(f"{self.base_url}/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise LLMUnavailable(f"пустой ответ провайдера {self.provider}: {str(data)[:200]}")
        message = choices[0].get("message", {}) or {}
        usage = data.get("usage") or {}
        content = message.get("content") or ""
        finish = choices[0].get("finish_reason")
        reasoning_tokens = ((usage.get("completion_tokens_details") or {})
                            .get("reasoning_tokens") or 0)
        if not content.strip() and reasoning_tokens:
            # Рассуждающая модель потратила весь max_tokens на размышление и
            # не успела выдать ответ. Молча вернуть "" значит соврать вызывающему.
            raise LLMUnavailable(
                f"{self.model}: весь бюджет max_tokens={self.max_tokens} ушёл на "
                f"рассуждение ({reasoning_tokens} ток.), ответа нет "
                f"(finish_reason={finish}). Увеличьте max_tokens."
            )
        return self._account(LLMResponse(
            content=content,
            tool_calls=message.get("tool_calls") or [],
            raw=data,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            model=data.get("model", self.model),
            provider=self.provider,
        ))

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl_context) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            raise LLMUnavailable(
                f"HTTP {exc.code} от {self.provider}: {detail or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMUnavailable(f"нет связи с {self.provider} ({url}): {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMUnavailable(f"таймаут {self.timeout}s от {self.provider}") from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"некорректный JSON от {self.provider}: {raw[:200]}") from exc

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        format_json: bool = False,
    ) -> str:
        """Упрощённый чат: только текст ответа."""
        response = self.generate(
            [LLMMessage(role="user", content=user_message)], system_prompt, format_json
        )
        return response.content
