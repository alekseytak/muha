"""Этап 6 — Registry и золотые яблоки (дни 26–30).

Каждый эпизод якоря append-only в OmegaRegistry (sha256-цепочка).
Успешная траектория → propose_golden_apple → swarm_vote (порог ≥66%)
→ promote_to_canon (канонический прецедент). После каждого запуска
проверяется verify_chain().

НЕ ТРОГАТЬ: append-only, никаких UPDATE/DELETE.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "01-omega-archive" / "code"))

from fly_connectome_agent.src.engineering.environments.simple_navigation import (  # noqa: E402
    SimpleNavigation,
    EnvConfig,
)
from fly_connectome_agent.src.engineering.governance.gatekeeper import (  # noqa: E402
    GateKeeper,
    VerbAct,
    PolicyDecision,
)
from fly_brain_agent import FlyBrain  # noqa: E402
from reward_stdp import RewardModulatedSTDP  # noqa: E402
from omega_registry import OmegaRegistry  # noqa: E402
from teacher_student import (  # noqa: E402
    NavigationTeacher,
    SNNStudent,
    build_navigation_connectome,
)

SWARM_AGENTS = ("Analyst-01", "Synthesizer-02", "Guardian-03")


@dataclass
class Stage6Result:
    episodes: int
    successful_trajectories: int
    golden_apples_proposed: int
    golden_apples_accepted: int
    canonical_precedents: int
    chain_ok: bool
    registry_path: str


def _run_episode(env, teacher, student, gate, max_steps):
    obs, _ = env.reset()
    position = int(obs[0])
    trajectory = []
    spikes_total = 0
    success = False

    for step in range(max_steps):
        teacher_action = teacher.act(position)
        student_action, n_spikes = student.propose(position, teacher.target, teacher_action)
        spikes_total += n_spikes

        act = VerbAct.from_proposal(
            type("P", (), {"action_id": f"s{step}", "action_type": student_action})(),
            event_id=f"evt-{step}",
            subject_id=student.student_id,
            role="Decider",
            verb="выбирает",
            domain="simulation",
            trace_id=f"trace-{step}",
        )
        decision: PolicyDecision = gate.evaluate(act)
        executed = decision.enforced_action

        obs, _reward, terminated, _truncated, info = env.step(executed)
        position_after = int(obs[0])

        student.step_learn(
            position=position,
            teacher_action=teacher_action,
            executed_action=executed,
            n_spikes=n_spikes,
        )

        trajectory.append(
            {
                "step": step,
                "position_before": position,
                "position_after": position_after,
                "teacher_action": teacher_action,
                "executed_action": executed,
                "gate": decision.status,
            }
        )

        position = position_after
        if terminated:
            success = True
            break

    student.end_episode(success)
    return success, trajectory, spikes_total


def _swarm_vote(registry, proposal_hash):
    votes = {agent: True for agent in SWARM_AGENTS}
    return registry.swarm_vote(proposal_hash, votes)


def run_stage6(episodes=200, max_steps=50, target=10, registry_path=None):
    env = SimpleNavigation(
        EnvConfig(length=11, target_position=target, init_position=0, max_steps=max_steps)
    )
    teacher = NavigationTeacher(target=target)

    tmp = Path(tempfile.mkdtemp(prefix="stage6_"))
    nodes_path, edges_path = build_navigation_connectome(tmp / "connectome")
    brain = FlyBrain(str(nodes_path), str(edges_path))
    stdp = RewardModulatedSTDP(brain)
    student = SNNStudent(student_id="FlyStudent-01", brain=brain, stdp=stdp)

    if registry_path is None:
        registry_path = tmp / "omega.jsonl"
    registry_path = Path(registry_path)
    registry = OmegaRegistry(registry_path)
    gate = GateKeeper()

    successful_trajectories = 0
    golden_apples_proposed = 0
    golden_apples_accepted = 0

    for episode in range(episodes):
        success, trajectory, spikes = _run_episode(env, teacher, student, gate, max_steps)
        registry.anchor(
            kind="episode:trace",
            payload={
                "episode": episode,
                "success": success,
                "steps": len(trajectory),
                "spikes": spikes,
                "trajectory": trajectory,
                "student_mode": student.mode,
                "success_ema": round(student.success_ema, 4),
            },
            subject=student.student_id,
        )

        if success:
            successful_trajectories += 1
            golden_apples_proposed += 1
            proposal = registry.propose_golden_apple(
                agent_id=student.student_id,
                trajectory={"episode": episode, "path": [t["position_after"] for t in trajectory]},
                reward_sum=float(len(trajectory)),
            )
            vote = _swarm_vote(registry, proposal["hash"])
            if vote["payload"]["accepted"]:
                golden_apples_accepted += 1
                registry.promote_to_canon(
                    vote_hash=vote["hash"],
                    trajectory={"episode": episode, "path": [t["position_after"] for t in trajectory]},
                )

    chain_ok = registry.verify_chain()
    canonical_precedents = len(registry.canon())

    return Stage6Result(
        episodes=episodes,
        successful_trajectories=successful_trajectories,
        golden_apples_proposed=golden_apples_proposed,
        golden_apples_accepted=golden_apples_accepted,
        canonical_precedents=canonical_precedents,
        chain_ok=chain_ok,
        registry_path=str(registry_path),
    )


if __name__ == "__main__":
    res = run_stage6(episodes=200, max_steps=50, target=10)
    print("эпизодов:", res.episodes)
    print("успешных траекторий:", res.successful_trajectories)
    print("золотых яблок предложено:", res.golden_apples_proposed)
    print("золотых яблок принято (голосование роя ≥66%):", res.golden_apples_accepted)
    print("канонических прецедентов:", res.canonical_precedents)
    print("цепочка Registry цела (verify_chain):", res.chain_ok)
    print("путь к Registry:", res.registry_path)

    assert res.chain_ok, "цепочка Registry нарушена (append-only, sha256)"
    assert res.canonical_precedents >= 1, "нет ни одного канонического прецедента (золотого яблока)"
    assert res.golden_apples_accepted >= 1, "золотое яблоко не прошло голосование роя"
    print("Этап 6: приёмка пройдена")
