"""Гормоны — дофамин/серотонин/кортизол, влияют на риск-аппетит."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import time


@dataclass
class HormoneState:
    """Текущее состояние гормонов."""
    dopamine: float = 0.5
    serotonin: float = 0.5
    cortisol: float = 0.1
    risk_appetite: float = 0.5
    updated_at: float = field(default_factory=time.time)


class Hormones:
    """Гормональная система — модулирует поведение агента."""

    def __init__(
        self,
        dopamine_base: float = 0.5,
        serotonin_base: float = 0.5,
        cortisol_base: float = 0.1,
        risk_appetite_base: float = 0.5,
    ) -> None:
        self.state = HormoneState(
            dopamine=dopamine_base,
            serotonin=serotonin_base,
            cortisol=cortisol_base,
            risk_appetite=risk_appetite_base,
        )
        self.history: List[Dict[str, Any]] = []

    def reward(self, amount: float = 0.1) -> None:
        """Положительное вознаграждение."""
        self.state.dopamine = min(1.0, self.state.dopamine + amount)
        self.state.cortisol = max(0.0, self.state.cortisol - amount * 0.5)
        self.state.risk_appetite = min(1.0, self.state.risk_appetite + amount * 0.3)
        self.state.serotonin = min(1.0, self.state.serotonin + amount * 0.2)
        self._record("reward", amount)

    def stress(self, amount: float = 0.1) -> None:
        """Отрицательное воздействие."""
        self.state.cortisol = min(1.0, self.state.cortisol + amount)
        self.state.dopamine = max(0.0, self.state.dopamine - amount * 0.5)
        self.state.risk_appetite = max(0.0, self.state.risk_appetite - amount * 0.4)
        self.state.serotonin = max(0.0, self.state.serotonin - amount * 0.2)
        self._record("stress", amount)

    def calm(self, amount: float = 0.05) -> None:
        """Успокоение."""
        self.state.serotonin = min(1.0, self.state.serotonin + amount)
        self.state.cortisol = max(0.0, self.state.cortisol - amount * 0.3)
        self._record("calm", amount)

    def _record(self, event: str, amount: float) -> None:
        self.history.append({
            "event": event,
            "amount": amount,
            "timestamp": time.time(),
            "state": {
                "dopamine": self.state.dopamine,
                "serotonin": self.state.serotonin,
                "cortisol": self.state.cortisol,
                "risk_appetite": self.state.risk_appetite,
            },
        })
        if len(self.history) > 500:
            self.history = self.history[-500:]
        self.state.updated_at = time.time()

    def risk_modifier(self) -> float:
        """Модификатор риска на основе гормонов."""
        return (self.state.dopamine - self.state.cortisol) * 0.5 + 0.5

    def stats(self) -> Dict[str, Any]:
        return {
            "dopamine": round(self.state.dopamine, 3),
            "serotonin": round(self.state.serotonin, 3),
            "cortisol": round(self.state.cortisol, 3),
            "risk_appetite": round(self.state.risk_appetite, 3),
            "risk_modifier": round(self.risk_modifier(), 3),
        }
