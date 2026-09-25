"""Координация роя: феромоны + консенсус + вето Стража.

Феромон — ATQEC-совместимый сигнал: он маркирует и оценивает,
но НЕ выполняет финальный commit (OS-Glagolov-Charter.md п.7).
Решение роя — это L3 «рекомендует»: сначала policy gate,
потом export, и только Registrar якорит запись.

Локальные правила эмерджентности (Бергсон, elan vital):
  separation / alignment / cohesion + феромонный градиент.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Pheromone:
    x: int
    y: int
    kind: str           # reward | danger | resource
    strength: float
    source_agent: str
    ttl: float = 60.0
    born: float = field(default_factory=time.time)

    def expired(self) -> bool:
        return time.time() - self.born > self.ttl


class PheromoneField:
    """Виртуальное феромонное поле (2D-сетка с затуханием)."""

    KINDS = ("reward", "danger", "resource")

    def __init__(self, width: int, height: int, decay: float = 0.995):
        self.w, self.h, self.decay = width, height, decay
        self.grid = {k: np.zeros((height, width)) for k in self.KINDS}

    def deposit(self, kind: str, x: int, y: int, amount: float) -> None:
        self.grid[kind][y % self.h, x % self.w] += amount

    def sense(self, kind: str, x: int, y: int, radius: int = 1) -> float:
        g = self.grid[kind]
        y0, y1 = max(0, y - radius), min(self.h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(self.w, x + radius + 1)
        return float(g[y0:y1, x0:x1].sum())

    def gradient(self, kind: str, x: int, y: int) -> tuple:
        """Куда расти: (dx, dy) по феромонному градиенту."""
        best, bdx, bdy = self.sense(kind, x, y), 0, 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s = self.sense(kind, x + dx, y + dy)
            if s > best:
                best, bdx, bdy = s, dx, dy
        return bdx, bdy

    def tick(self) -> None:
        for g in self.grid.values():
            g *= self.decay


@dataclass
class Proposal:
    action: str
    confidence: float
    risk: float
    evidence_quality: float
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def score(self) -> float:
        return self.confidence * (1.0 - self.risk) * self.evidence_quality


class SwarmCoordinator:
    """Агрегирование предложений + вето Guardian.

    Коллективное решение НЕ является commit — это verb-act
    с ролью Decider и глаголом «рекомендует» (L3).
    """

    def __init__(self, veto_threshold: float = 0.9):
        self.veto_threshold = veto_threshold

    def decide(self, proposals: list, guardian_risk: float) -> dict:
        if not proposals:
            return {"action": "hold", "reason": "нет предложений",
                    "escalate": False}
        if guardian_risk >= self.veto_threshold:
            return {"action": "blocked", "reason": "вето Стража",
                    "escalate": True,
                    "verb_act_hint": {"role": "Guardian",
                                      "verb": "блокирует"}}
        best = max(proposals, key=lambda pr: pr.score)
        return {
            "action": best.action,
            "score": round(best.score, 4),
            "escalate": False,
            "verb_act_hint": {"role": "Decider", "verb": "рекомендует"},
            "note": "финальный commit — только через policy gate и Registrar",
        }


def emergent_move(pos, neighbors, field: PheromoneField, goal_kind="reward"):
    """Один шаг агента из 3 локальных правил + феромонного градиента."""
    x, y = pos
    sep = np.zeros(2)
    ali = np.zeros(2)
    coh = np.zeros(2)
    for nx, ny in neighbors:
        d = np.array([x - nx, y - ny], dtype=float)
        dist = max(np.linalg.norm(d), 1e-6)
        if dist < 2.0:
            sep += d / dist          # separation
        coh += np.array([nx, ny])    # cohesion
    if neighbors:
        coh = coh / len(neighbors) - np.array([x, y])
    gx, gy = field.gradient(goal_kind, int(x), int(y))
    step = 0.3 * sep + 0.2 * coh + 0.5 * np.array([gx, gy]) + 0.2 * ali
    n = np.linalg.norm(step)
    if n > 0:
        step = step / n
    return (x + step[0], y + step[1])
