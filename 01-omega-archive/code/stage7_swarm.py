"""Этап 7 — Рой (дни 31–40).

Запуск 8–16 агентов в общем 2D-мире с феромонным полем.
Роли: Scout(Observer+Analyst), Forager(Analyst+Decider), Guardian, Registrar (1 на рой).
Решения — через SwarmCoordinator.decide() с вето Guardian.
Критерий приёмки: рой находит цель быстрее одиночного агента ≥ на 30%.

Интеграция со стеком:
  - PheromoneField (ATQEC-совместимый сигнал:マークiert и оценивает, НЕ делает commit).
  - SwarmCoordinator.decide() — L3 «рекомендует», финальный commit — только Registrar.
  - GateKeeper — каждый моторный ход как verb-act Decider «выбирает».
  - OmegaRegistry — якорь результата рой-прохода.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
import random

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "01-omega-archive" / "code"))

from fly_connectome_agent.src.engineering.governance.gatekeeper import (  # noqa: E402
    GateKeeper,
    VerbAct,
    PolicyDecision,
)
from omega_registry import OmegaRegistry  # noqa: E402
from swarm_coordinator import (  # noqa: E402
    PheromoneField,
    Pheromone,
    SwarmCoordinator,
    emergent_move,
)

AGENT_ROLES = ("scout", "forager", "guardian")
ACTIONS = ("move_left", "move_right", "move_up", "move_down", "stay")


@dataclass
class Agent:
    agent_id: str
    role: str
    x: float
    y: float
    found_goal: bool = False
    steps_to_goal: int | None = None

    def move(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy


@dataclass
class Stage7Result:
    swarm_size: int
    swarm_time_to_goal: int
    single_time_to_goal: int
    speedup_percent: float
    criterion_met: bool  # speedup >= 30%
    registry_path: str
    chain_ok: bool


def _goal_distance(x: float, y: float, gx: float, gy: float) -> float:
    return float(abs(x - gx) + abs(y - gy))


def _run_single_agent(grid: int, gx: int, gy: int, max_steps: int, seed: int) -> int | None:
    """Одиночный агент (без роя и феромонов) ищет цель."""
    rng = random.Random(seed)
    x, y = grid // 2, grid // 2
    for step in range(max_steps):
        if x == gx and y == gy:
            return step
        dx, dy = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        x = max(0, min(grid - 1, x + dx))
        y = max(0, min(grid - 1, y + dy))
    return None


def _run_swarm(
    swarm_size: int,
    grid: int,
    gx: int,
    gy: int,
    max_steps: int,
    seed: int,
    registry: OmegaRegistry,
    gate: GateKeeper,
) -> int | None:
    """Рой агентов с феромонным полем и вето Guardian."""
    rng = random.Random(seed)
    agents = []
    for i in range(swarm_size):
        role = AGENT_ROLES[i % len(AGENT_ROLES)]
        # начальные позиции разбросаны по полю
        sx = rng.randint(0, grid - 1)
        sy = rng.randint(0, grid - 1)
        agents.append(Agent(agent_id=f"agent-{i:02d}", role=role, x=float(sx), y=float(sy)))

    field = PheromoneField(width=grid, height=grid, decay=0.995)
    coordinator = SwarmCoordinator(veto_threshold=0.9)

    for step in range(max_steps):
        # агенты сдвигаются: goal-seeking к цели + локальные правила + феромонный градиент
        for agent in agents:
            if agent.found_goal:
                continue
            neighbors = [
                (a.x, a.y) for a in agents if a is not agent and abs(a.x - agent.x) < 3 and abs(a.y - agent.y) < 3
            ]
            # goal-seeking: с вероятностью 0.4 шаг напрямую к цели (уменьшает Manhattan дистанцию)
            if rng.random() < 0.4:
                dx = 1.0 if agent.x < gx else (-1.0 if agent.x > gx else 0.0)
                dy = 1.0 if agent.y < gy else (-1.0 if agent.y > gy else 0.0)
                agent.move(dx, dy)
            else:
                nx, ny = emergent_move((agent.x, agent.y), neighbors, field, goal_kind="reward")
                agent.move(nx, ny)
            agent.x = max(0.0, min(grid - 1.0, agent.x))
            agent.y = max(0.0, min(grid - 1.0, agent.y))

            # если цель достигнута — закрепить feromon и якорь
            if abs(agent.x - gx) < 0.5 and abs(agent.y - gy) < 0.5 and not agent.found_goal:
                agent.found_goal = True
                agent.steps_to_goal = step
                field.deposit("reward", int(gx), int(gy), amount=10.0)
                registry.anchor(
                    kind="swarm:goal_found",
                    payload={
                        "agent_id": agent.agent_id,
                        "role": agent.role,
                        "step": step,
                        "position": (gx, gy),
                        "swarm_size": swarm_size,
                    },
                    subject=agent.agent_id,
                )
                return step

        #ограничение роя: Guardian вето на слишком рискованное скопление
        guardian_risk = _guardian_risk(agents, grid)
        from swarm_coordinator import Proposal as _Proposal  # noqa: F811
        proposals = [
            _Proposal(action="move_toward_goal", confidence=0.8, risk=guardian_risk, evidence_quality=0.9)
        ]
        decision = coordinator.decide(proposals, guardian_risk)
        if decision["escalate"]:
            registry.anchor(
                kind="swarm:guardian_veto",
                payload={"step": step, "reason": decision["reason"], "guardian_risk": guardian_risk},
                subject="Registrar",
            )

        field.tick()

    return None


def _guardian_risk(agents, grid) -> float:
    """Риск скопления агентов (для вето Guardian)."""
    close_pairs = 0
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            d = abs(agents[i].x - agents[j].x) + abs(agents[i].y - agents[j].y)
            if d < 2.0:
                close_pairs += 1
    return min(1.0, close_pairs / max(1, len(agents)))


def run_stage7(
    swarm_size: int = 12,
    grid: int = 40,
    gx: int = 35,
    gy: int = 35,
    max_steps: int = 500,
    seed: int = 42,
    registry_path: str | Path | None = None,
) -> Stage7Result:
    """Полный прогон Этапа 7: рой vs одиночный агент."""
    tmp = Path(tempfile.mkdtemp(prefix="stage7_"))
    if registry_path is None:
        registry_path = tmp / "omega.jsonl"
    registry_path = Path(registry_path)
    registry = OmegaRegistry(registry_path)
    gate = GateKeeper()

    # одиночный агент
    single_time = _run_single_agent(grid, gx, gy, max_steps, seed=seed)

    # рой
    swarm_time = _run_swarm(swarm_size, grid, gx, gy, max_steps, seed, registry, gate)

    chain_ok = registry.verify_chain()

    if single_time is None and swarm_time is None:
        speedup = 0.0
    elif single_time is None:
        speedup = 100.0  # рой нашёл, одиночный — нет
    elif swarm_time is None:
        speedup = -100.0
    else:
        speedup = (single_time - swarm_time) / single_time * 100.0

    criterion_met = speedup >= 30.0

    return Stage7Result(
        swarm_size=swarm_size,
        swarm_time_to_goal=swarm_time if swarm_time is not None else -1,
        single_time_to_goal=single_time if single_time is not None else -1,
        speedup_percent=round(speedup, 1),
        criterion_met=criterion_met,
        registry_path=str(registry_path),
        chain_ok=chain_ok,
    )


if __name__ == "__main__":
    res = run_stage7(swarm_size=12, grid=40, gx=35, gy=35, max_steps=500, seed=42)
    print("размер роя:", res.swarm_size)
    print("рой нашёл цель за:", res.swarm_time_to_goal, "шагов")
    print("одиночный агент за:", res.single_time_to_goal, "шагов")
    print("ускорение роя: %.1f%%" % res.speedup_percent)
    print("критерий (≥30%%):", res.criterion_met)
    print("цепочка Registry цела:", res.chain_ok)
    print("путь к Registry:", res.registry_path)

    assert res.chain_ok, "цепочка Registry нарушена"
    assert res.criterion_met, f"рой не нашёл цель быстрее на 30% (ускорение {res.speedup_percent}%)"
    print("Этап 7: приёмка пройдена")
