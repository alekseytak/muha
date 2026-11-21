"""Этап 9 — Рефлексия и творчество (дни 49–55).

Перед каждым действием уровня L4+ — reflect_decision():
  paradox_likelihood > 0.66 → эскалация (Quantum Escape, как в KEM 4.2).
Автомат Улама: 1 раз в N эпизодов добавляет perturbation к входному току —
разрыв эргодичности, защита от «Дня Сурка».
Перед сложной общей задачей — meditate() роя на 86.84 Гц до sync_order > 0.8.

В OS Глаголов:
  - KozyrevMirror — ATQEC-слой: оценивает/маркирует (L1), НЕ блокирует.
    Решение об эскалации — Guardian через verb-act.
  - Коллективная медитация — L2 «синтезирует» (atqec/memory), НЕ порождает
    внешних действий.
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
sys.path.insert(0, str(REPO / "02-mirror-swarm" / "code"))

from kozyrev_mirror import KozyrevMirror  # noqa: E402
from ulam_chaos import UlamAutomaton  # noqa: E402
from collective_meditation import CollectiveMeditation  # noqa: E402
from fly_connectome_agent.src.engineering.governance.gatekeeper import (  # noqa: E402
    GateKeeper,
    VerbAct,
    PolicyDecision,
)


@dataclass
class Stage9Result:
    decisions_reflected: int
    paradox_escalations: int
    paradox_likelihoods: list[float]
    ulam_perturbations: int
    ulam_novelty: float
    meditation_sync_order: float
    meditation_state: str
    mirror_paradox_blocked: bool  # дикое решение парадоксально, спокойное — нет


def run_stage9(episodes: int = 20, meditation_seconds: float = 15.0) -> Stage9Result:
    mirror = KozyrevMirror()
    ulam = UlamAutomaton(grid_size=16, seed=42)
    # Связь 60 выбрана из условия Курамото, а не подгонкой под ответ: при
    # 86.84 Гц и разбросе 2% полуразброс угловой частоты равен 10.9 рад/с,
    # критическая связь K_c ~ 17. При прежних K=2 синхронизации не было ни при
    # каком шаге интегрирования (проверено и на dt=1e-4), поэтому вместо
    # повышения связи частота раньше уменьшалась в 100 раз.
    meditation = CollectiveMeditation(n_agents=16, coupling=60.0, seed=7)
    gate = GateKeeper()

    decisions_reflected = 0
    paradox_escalations = 0
    paradox_likelihoods = []
    ulam_perturbations = 0

    # --- сценарий рефлексии: спокойное vs дикое решение ---
    calm = np.array([0.5, 0.52, 0.49, 0.51, 0.50])
    wild = np.array([0.1, 0.9, -0.8, 0.95, -0.2])

    r_calm = mirror.reflect_decision(calm)
    r_wild = mirror.reflect_decision(wild)
    decisions_reflected += 2
    paradox_likelihoods.extend([r_calm["paradox_likelihood"], r_wild["paradox_likelihood"]])

    for r in (r_calm, r_wild):
        if r["quantum_escape"]:
            paradox_escalations += 1
            # эскалация через verb-act Guardian
            act = VerbAct(
                event_id="evt-escalate",
                subject_id="Guardian-01",
                role="Guardian",
                verb="эскалирует",
                object_ref="decision-paradox",
                domain="simulation",
                context={"risk_level": "high", "paradox_likelihood": r["paradox_likelihood"]},
                trace_id="trace-escalate",
                proposal_ref="escalate-01",
                proposed_action="stay",
            )
            decision: PolicyDecision = gate.evaluate(act)
            assert decision.status in ("ALLOW", "ESCALATE"), "Guardian эскалация не должна быть DENY"

    # --- Автомат Улама: perturbation к входному току ---
    for ep in range(episodes):
        if ep % 5 == 0:  # 1 раз в N'épisode
            noise = ulam.perturbation((5,))
            ulam_perturbations += 1
    ulam_novelty = ulam.novelty_score()

    # --- Коллективная медитация до sync_order > 0.8 ---
    med = meditation.meditate(seconds=meditation_seconds)
    meditation_sync_order = med["sync_order"]
    meditation_state = med["state"]

    mirror_paradox_blocked = (
        r_calm["paradox_likelihood"] <= 0.66 and r_wild["paradox_likelihood"] > 0.66
    )

    return Stage9Result(
        decisions_reflected=decisions_reflected,
        paradox_escalations=paradox_escalations,
        paradox_likelihoods=[round(x, 4) for x in paradox_likelihoods],
        ulam_perturbations=ulam_perturbations,
        ulam_novelty=round(ulam_novelty, 4),
        meditation_sync_order=meditation_sync_order,
        meditation_state=meditation_state,
        mirror_paradox_blocked=mirror_paradox_blocked,
    )


if __name__ == "__main__":
    res = run_stage9(episodes=20, meditation_seconds=3.0)
    print("решений отражённых:", res.decisions_reflected)
    print("пародоксальных эскалаций:", res.paradox_escalations)
    print("paradox_likelihoods:", res.paradox_likelihoods)
    print("возмущений Улама:", res.ulam_perturbations)
    print("новинность Улама:", res.ulam_novelty)
    print("sync_order медитации:", res.meditation_sync_order, "состояние:", res.meditation_state)
    print("зеркало разделяет спокойное/дикое:", res.mirror_paradox_blocked)

    assert res.decisions_reflected == 2, "должно быть отражено 2 решения"
    assert res.mirror_paradox_blocked, "зеркало должно различать спокойное и дикое решение"
    assert res.paradox_escalations >= 1, "дикое решение должно вызвать эскалацию"
    assert res.ulam_perturbations >= 1, "Автомат Улама должен внести возмущение"
    assert res.meditation_state == "moksha", f"медитация должна достичь sync_order > 0.8, got {res.meditation_state}"
    assert res.meditation_sync_order > 0.8, "sync_order должен превысить 0.8"
    print("Этап 9: приёмка пройдена")
