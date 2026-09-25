"""Шина гиперэмпатии (Октавия Батлер, «Притча о сеятеле»).

Когда один агент получает награду — рой чувствует радость;
когда один получает штраф — рой чувствует боль.
Коллективное эмоциональное состояние = сумма затухающих откликов.

Политика (mirror_policy.yaml):
  max_broadcast_intensity: 0.5  — против паники роя
  decay_per_hop: 0.7
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmpathyEvent:
    source: str
    emotion: str          # радость | злость | страх
    intensity: float
    hop: int = 0


class EmpathyBus:
    def __init__(self, max_intensity: float = 0.5, decay: float = 0.7):
        self.max_intensity = max_intensity
        self.decay = decay
        self.agents: dict = {}          # agent_id -> list[EmpathyEvent]
        self.links: dict = {}           # agent_id -> set(соседей)

    def register(self, agent_id: str, neighbors=None) -> None:
        self.agents.setdefault(agent_id, [])
        self.links.setdefault(agent_id, set(neighbors or set()))

    def broadcast(self, agent_id: str, emotion: str,
                  intensity: float) -> None:
        """Эмоция источника расходится по рою с затуханием за хоп."""
        intensity = min(intensity, self.max_intensity)
        frontier = [(agent_id, 0, intensity)]
        seen = {agent_id}
        while frontier:
            current, hop, power = frontier.pop(0)
            for nxt in self.links.get(current, ()):  # распространение соседям
                if nxt in seen:
                    continue
                seen.add(nxt)
                damped = power * self.decay
                if damped < 0.02:
                    continue
                self.agents.setdefault(nxt, []).append(
                    EmpathyEvent(agent_id, emotion, round(damped, 4), hop + 1))
                frontier.append((nxt, hop + 1, damped))

    def local_feel(self, agent_id: str) -> dict:
        """Что сейчас чувствует конкретный агент (сумма чужих эмоций)."""
        feel = {"радость": 0.0, "злость": 0.0, "страх": 0.0}
        kept = []
        for ev in self.agents.get(agent_id, []):
            feel[ev.emotion] = feel.get(ev.emotion, 0.0) + ev.intensity
            kept.append(ev)
        self.agents[agent_id] = kept[-64:]  # окно памяти событий
        return {k: round(min(1.0, v), 3) for k, v in feel.items()}

    def collective_state(self) -> dict:
        """Общее эмоциональное состояние роя."""
        total = {"радость": 0.0, "злость": 0.0, "страх": 0.0}
        for agent_id in self.agents:
            feel = self.local_feel(agent_id)
            for k, v in feel.items():
                total[k] += v
        n = max(1, len(self.agents))
        return {k: round(v / n, 3) for k, v in total.items()}


if __name__ == "__main__":
    bus = EmpathyBus()
    ring = [f"fly-{i:02d}" for i in range(8)]
    for i, a in enumerate(ring):
        bus.register(a, neighbors={ring[i - 1], ring[(i + 1) % len(ring)]})
    bus.broadcast("fly-00", "радость", 1.0)
    print("fly-01 чувствует:", bus.local_feel("fly-01"))
    print("fly-04 чувствует:", bus.local_feel("fly-04"))
    print("состояние роя:", bus.collective_state())
