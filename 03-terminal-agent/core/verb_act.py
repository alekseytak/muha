"""Система глаголов и действий (VerbAct) — грамматика действий агента."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import uuid
import time


class VerbActType(Enum):
    """Глаголы системы — базовые типы действий."""
    READ = "read"           # чтение данных
    WRITE = "write"         # запись данных
    EXECUTE = "execute"     # выполнение команды
    THINK = "think"         # размышление (LLM.generate)
    LOG = "log"             # запись в реестр
    VERSION = "version"     # версиярирование
    ANCHOR = "anchor"       # якорить (канонизировать)
    VETO = "veto"           # вето (блокировка)
    ESCALATE = "escalate"   # эскалация (передача на уровень выше)
    RECOMMEND = "recommend" # рекомендация
    SELECT = "select"       # выбор варианта


@dataclass
class VerbAct:
    """Одно действие в грамматике глаголов."""
    id: str
    verb: VerbActType
    object: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    actor: str = "muha-agent"
    gate_decision: Optional[str] = None  # ALLOW/DENY/ESCALATE
    gate_reason: Optional[str] = None
    result: Optional[Any] = None
    success: bool = False
    tool: Optional[str] = None           # какой инструмент реально сработал
    domain: Optional[str] = None         # домен, определённый по объекту
    parse_reason: Optional[str] = None   # как получен глагол (для диагностики)
    # Действие разрешено заранее оператором (автономный режим + pre_approved):
    # подтверждение не спрашивается, но факт попадает в реестр.
    auto_approved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "verb": self.verb.value,
            "object": self.object,
            "parameters": self.parameters,
            "context": self.context,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "gate_decision": self.gate_decision,
            "gate_reason": self.gate_reason,
            "result": self.result,
            "success": self.success,
            "tool": self.tool,
            "domain": self.domain,
            "parse_reason": self.parse_reason,
            "auto_approved": self.auto_approved,
        }

    @classmethod
    def create(
        cls,
        verb: VerbActType,
        object: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        actor: str = "muha-agent",
    ) -> "VerbAct":
        return cls(
            id=str(uuid.uuid4())[:8],
            verb=verb,
            object=object,
            parameters=parameters or {},
            context=context or {},
            actor=actor,
        )

    def __str__(self) -> str:
        return f"[{self.id}] {self.verb.value} {self.object}"


class VerbActGrammar:
    """Грамматика: какой глагол допустим в каком домене.

    Домены описывают область действия, а не права роли — права проверяет
    GateKeeper. Здесь отсекаются заведомо бессмысленные пары
    (например, запись в интерфейс или исполнение в домене UI).
    """
    DOMAIN_RULES: Dict[str, Dict[VerbActType, bool]] = {
        "ui": {
            VerbActType.READ: True,
            VerbActType.WRITE: False,
            VerbActType.EXECUTE: False,
            VerbActType.THINK: True,
            VerbActType.LOG: True,
            VerbActType.VERSION: True,
            VerbActType.RECOMMEND: True,
            VerbActType.SELECT: True,
            VerbActType.ESCALATE: True,
        },
        "filesystem": {
            VerbActType.READ: True,
            VerbActType.WRITE: True,
            VerbActType.EXECUTE: False,
            VerbActType.THINK: True,
            VerbActType.LOG: True,
            VerbActType.SELECT: True,
            VerbActType.ESCALATE: True,
        },
        "web": {
            # Чтение веба: загружать можно, менять внешний мир — нет.
            VerbActType.READ: True,
            VerbActType.SELECT: True,   # метаданные страницы
            VerbActType.THINK: True,
            VerbActType.LOG: True,
            VerbActType.ESCALATE: True,
        },
        "system": {
            VerbActType.READ: True,
            VerbActType.WRITE: False,
            VerbActType.EXECUTE: True,  # требует подтверждения
            VerbActType.THINK: True,
            VerbActType.LOG: True,
            VerbActType.ESCALATE: True,
        },
        "registry": {
            VerbActType.READ: True,
            VerbActType.WRITE: True,
            VerbActType.EXECUTE: False,
            VerbActType.THINK: True,
            VerbActType.LOG: True,
            VerbActType.ANCHOR: True,
            VerbActType.VERSION: True,
            VerbActType.VETO: True,
            VerbActType.ESCALATE: True,
            VerbActType.SELECT: True,
        },
        "swarm": {
            VerbActType.READ: True,
            VerbActType.WRITE: True,
            VerbActType.EXECUTE: True,
            VerbActType.THINK: True,
            VerbActType.LOG: True,
            VerbActType.RECOMMEND: True,
            VerbActType.VETO: True,
            VerbActType.ESCALATE: True,
            VerbActType.SELECT: True,
        },
        # Коннектом — только чтение и рассуждение; исполнения нет.
        "connectome": {
            VerbActType.READ: True,
            VerbActType.WRITE: False,
            VerbActType.EXECUTE: False,
            VerbActType.THINK: True,
            VerbActType.LOG: True,
            VerbActType.SELECT: True,
            VerbActType.ESCALATE: True,
        },
    }

    # Префиксы объектов -> домен. Порядок важен: первый совпавший выигрывает.
    OBJECT_PREFIXES: tuple[tuple[str, str], ...] = (
        ("file:", "filesystem"),
        ("path:", "filesystem"),
        ("dir:", "filesystem"),
        ("folder:", "filesystem"),
        ("fs:", "filesystem"),
        ("sys:", "system"),
        ("cmd:", "system"),
        ("sh:", "system"),
        ("exec:", "system"),
        ("web:", "web"),
        ("url:", "web"),
        ("http://", "web"),
        ("https://", "web"),
        ("reg:", "registry"),
        ("registry:", "registry"),
        ("swarm:", "swarm"),
        ("pheromone:", "swarm"),
        ("connectome:", "connectome"),
        ("connectom:", "connectome"),
        ("neuron:", "connectome"),
        ("fly:", "connectome"),
        ("fw:", "connectome"),
    )

    @classmethod
    def is_allowed(cls, domain: str, verb: VerbActType) -> bool:
        return cls.DOMAIN_RULES.get(domain, {}).get(verb, False)

    @classmethod
    def domain_for_object(cls, obj: str) -> str:
        """Определяет домен по префиксу объекта."""
        text = (obj or "").strip()
        if text.startswith("/") or text.startswith("./") or text.startswith("../"):
            # «голый» путь без префикса тоже файловая система
            return "filesystem"
        for prefix, domain in cls.OBJECT_PREFIXES:
            if text.lower().startswith(prefix):
                return domain
        return "ui"

    @classmethod
    def describe(cls) -> str:
        """Текстовая матрица домен x глагол — для справки."""
        verbs = [v for v in VerbActType]
        head = "глагол     " + " ".join(f"{d[:6]:6s}" for d in cls.DOMAIN_RULES)
        lines = [head]
        for verb in verbs:
            cells = [
                f"{'ok' if cls.is_allowed(domain, verb) else '-':6s}"
                for domain in cls.DOMAIN_RULES
            ]
            lines.append(f"{verb.value:10s} " + " ".join(cells))
        return "\n".join(lines)
