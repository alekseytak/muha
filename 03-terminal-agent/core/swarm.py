"""Рой — феромонное поле и координация агентов."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import math


@dataclass
class Pheromone:
    """Феромонная метка."""
    id: str
    position: tuple[float, float]
    strength: float
    decay_rate: float
    created_at: float = field(default_factory=time.time)
    agent_id: str = "swarm"
    type: str = "general"

    def age(self) -> float:
        return time.time() - self.created_at

    def current_strength(self) -> float:
        return self.strength * (self.decay_rate ** self.age())


class PheromoneField:
    """Поле феромонов — агентский координационный слой."""

    def __init__(self, decay_rate: float = 0.95) -> None:
        self.decay_rate = decay_rate
        self.pheromones: List[Pheromone] = []

    def deposit(
        self,
        position: tuple[float, float],
        strength: float,
        agent_id: str = "swarm",
        pheromone_type: str = "general",
    ) -> str:
        """Оставить феромонную метку."""
        pheromone = Pheromone(
            id=f"pher:{len(self.pheromones)}",
            position=position,
            strength=strength,
            decay_rate=self.decay_rate,
            agent_id=agent_id,
            type=pheromone_type,
        )
        self.pheromones.append(pheromone)
        return pheromone.id

    def sense(
        self,
        position: tuple[float, float],
        radius: float = 50.0,
        pheromone_type: Optional[str] = None,
    ) -> List[Pheromone]:
        """Почувствовать феромоны в радиусе."""
        results = []
        for p in self.pheromones:
            if p.current_strength() < 0.01:
                continue
            if pheromone_type and p.type != pheromone_type:
                continue
            dist = math.dist(position, p.position)
            if dist <= radius:
                results.append(p)
        return sorted(results, key=lambda p: -p.current_strength())

    def evaporate(self) -> None:
        """Устаревшие феромоны исчезают."""
        self.pheromones = [
            p for p in self.pheromones
            if p.current_strength() >= 0.01
        ]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_pheromones": len(self.pheromones),
            "avg_strength": sum(p.current_strength() for p in self.pheromones) / max(1, len(self.pheromones)),
        }


class Swarm:
    """Рой агентов с феромонной координацией."""

    def __init__(self, min_agents: int = 3) -> None:
        self.min_agents = min_agents
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.field = PheromoneField()
        self.decisions: List[Dict[str, Any]] = []

    def register_agent(self, agent_id: str, position: tuple[float, float] = (0.0, 0.0)) -> None:
        self.agents[agent_id] = {
            "position": position,
            "role": "scout",
            "active": True,
            "joined_at": time.time(),
        }

    def propose(self, agent_id: str, proposal: str, position: tuple[float, float]) -> str:
        """Агент предлагает действие с феромонной меткой."""
        pheromone_id = self.field.deposit(
            position=position,
            strength=1.0,
            agent_id=agent_id,
            pheromone_type="proposal",
        )
        self.decisions.append({
            "agent_id": agent_id,
            "proposal": proposal,
            "position": position,
            "pheromone_id": pheromone_id,
            "timestamp": time.time(),
        })
        return pheromone_id

    def sense_swarm(self, position: tuple[float, float], radius: float = 50.0) -> List[Dict[str, Any]]:
        """Почувствовать предложения роя рядом."""
        pheromones = self.field.sense(position, radius, "proposal")
        return [
            {
                "id": p.id,
                "position": p.position,
                "strength": p.current_strength(),
                "agent_id": p.agent_id,
                "age": p.age(),
            }
            for p in pheromones
        ]

    def stats(self) -> Dict[str, Any]:
        return {
            "agents": len(self.agents),
            "pheromones": len(self.field.pheromones),
            "decisions": len(self.decisions),
        }
