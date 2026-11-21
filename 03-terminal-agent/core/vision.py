"""Зрение: муха смотрит на картинку и превращает её в наблюдение.

Зачем отдельный слой. Зрение не добавляет двенадцатый глагол — оно даёт то,
чего у агента не было: возможность получить задачу не текстом, а картинкой.
Скриншот ошибки, фотография экрана, снимок списка файлов — всё это сначала
становится текстом, а дальше работает та же грамматика из одиннадцати
глаголов и та же граница готовности.

Честно про «мозг заточен под зрение». У настоящей дрозофилы оптические доли
занимают большую часть мозга, и это правда. Но в нашем коннектоме
(`flywire.fafb.v783`, 1515 узлов) только три класса: descending 1215,
ascending 256, sensory 44 — оптических долей там нет. Поэтому зрение здесь
делает модель, а не коннектом, и называть это иначе было бы обманом.

Цена. Один вызов на картинку. Модель просят ТОЛЬКО распознать, без советов:
иначе она начинает лечить вместо того, чтобы читать.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Модель, которая умеет смотреть. Проверена живым вызовом: распознаёт текст
# на скриншоте терминала построчно.
VISION_MODEL = "deepseek-v4-flash-vision-exp"

SYSTEM = (
    "Ты — зрение терминального агента. Тебе дают изображение экрана или "
    "документа. Твоя задача — РАСПОЗНАТЬ, а не советовать.\n"
    "Ответ строго в таком виде, без вступлений:\n"
    "ТИП: ошибка | команда | список | текст | неизвестно\n"
    "ТЕКСТ:\n"
    "<всё, что видно, построчно, дословно>\n"
    "ПОДСКАЗКА: <одна короткая строка, что это значит; если непонятно — "
    "напиши «нет»>"
)

KINDS = ("ошибка", "команда", "список", "текст", "неизвестно")


@dataclass
class Observation:
    """Что муха увидела."""

    kind: str
    text: str
    hint: str = ""
    raw: str = ""
    model: str = ""
    image: str = ""
    tokens: Tuple[Optional[int], Optional[int]] = (None, None)

    @property
    def is_error(self) -> bool:
        return self.kind == "ошибка"

    def as_phrase(self) -> str:
        """Фраза, которую можно отдать обычному разбору.

        Наблюдение превращается в текст задачи, а не в готовое действие:
        решение всё равно принимает грамматика глаголов и граница готовности.
        """
        if self.is_error:
            return f"Объясни ошибку: {self.text.strip().splitlines()[-1][:120]}"
        if self.kind == "команда":
            return f"Запусти команду {self.text.strip().splitlines()[0][:80]}"
        if self.kind == "список":
            return f"Прочитай список: {self.text.strip()[:120]}"
        return f"Объясни, что на картинке: {self.text.strip()[:120]}"


def parse(answer: str) -> Tuple[str, str, str]:
    """Разобрать ответ модели. Непонятное не выдумывается."""
    kind = "неизвестно"
    text_lines: List[str] = []
    hint = ""
    section = None
    for line in (answer or "").splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("ТИП:"):
            value = stripped.split(":", 1)[1].strip().lower()
            for known in KINDS:
                if known in value:
                    kind = known
                    break
            section = None
        elif upper.startswith("ТЕКСТ:"):
            section = "text"
        elif upper.startswith("ПОДСКАЗКА:"):
            hint = stripped.split(":", 1)[1].strip()
            if hint.lower() in ("нет", "-", ""):
                hint = ""
            section = None
        elif section == "text" and stripped:
            text_lines.append(stripped)
    return kind, "\n".join(text_lines), hint


def perceive(client: Any, image: str, model: str = VISION_MODEL) -> Observation:
    """Посмотреть на картинку одним вызовом модели."""
    from .llm_client import LLMMessage

    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"картинка не найдена: {image}")

    previous = getattr(client, "model", None)
    try:
        client.model = model
        response = client.generate(
            [LLMMessage("user", "Распознай изображение.", images=[str(path)])],
            system_prompt=SYSTEM,
        )
    finally:
        if previous is not None:
            client.model = previous

    kind, text, hint = parse(response.content)
    return Observation(
        kind=kind, text=text, hint=hint, raw=response.content,
        model=model, image=str(path),
        tokens=(response.prompt_tokens, response.completion_tokens),
    )