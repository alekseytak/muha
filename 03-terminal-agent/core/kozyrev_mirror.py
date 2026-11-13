"""Зеркало Козырева — парадокс и рефлексия."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class ParadoxLevel(Enum):
    """Уровни парадокса."""
    STABLE = "stable"
    TENSION = "tension"
    PARADOX = "paradox"
    ESCALATION = "escalation"


@dataclass
class ParadoxEvent:
    """Событие парадокса."""
    id: str
    description: str
    level: float  # 0.0 - 1.0
    timestamp: float = field(default_factory=time.time)
    escalated: bool = False


class KozyrevMirror:
    """Зеркало отражает состояние агента и выявляет парадоксы."""

    def __init__(self, paradox_threshold: float = 0.66) -> None:
        self.paradox_threshold = paradox_threshold
        self.events: List[ParadoxEvent] = []
        self.reflections: List[Dict[str, Any]] = []

    def reflect(
        self,
        state_description: str,
        hormone_dopamine: float,
        hormone_cortisol: float,
        risk_appetite: float,
    ) -> ParadoxEvent:
        """Отразить состояние и вычислить уровень парадокса."""
        paradox = (
            hormone_cortisol * 0.4
            + (1.0 - hormone_dopamine) * 0.3
            + risk_appetite * 0.3
        )
        paradox = min(1.0, max(0.0, paradox))

        event = ParadoxEvent(
            id=f"paradox:{int(time.time() * 1000)}",
            description=state_description,
            level=paradox,
        )

        if paradox > self.paradox_threshold:
            event.escalated = True

        self.events.append(event)
        if len(self.events) > 500:
            self.events = self.events[-500:]

        self.reflections.append({
            "description": state_description,
            "paradox_level": paradox,
            "escalated": event.escalated,
            "timestamp": time.time(),
        })

        return event

    def get_recent_paradoxes(self, limit: int = 10) -> List[ParadoxEvent]:
        return sorted(self.events, key=lambda e: -e.timestamp)[:limit]

    def stats(self) -> Dict[str, Any]:
        total = len(self.events)
        escalated = sum(1 for e in self.events if e.escalated)
        avg_level = sum(e.level for e in self.events) / max(1, total)
        return {
            "total_events": total,
            "escalated": escalated,
            "avg_paradox_level": round(avg_level, 3),
        }
