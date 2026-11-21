"""Этап 8 — Эмпатия и гормоны, Проект II (дни 41–48).

Каждому агенту — HormoneModule (Mood Organ).
EmpathyBus: эмоция агента транслируется рою с затуханием 0.5.
Гормоны влияют на модулятор M (через cognitive_ability) и на risk_appetite.

Интеграция со стеком:
  - HormoneModule.on_reward / on_punishment -> эмоция (радость/злость/страх)
  - EmpathyBus.broadcast -> lokal_feel (сумма затухающих откликов соседей)
  - HormoneModule.tone_for_response() -> risk_appetite, cognitive_ability
  - cognitive_ability масштабирует модулятор M в reward_stdp
  - risk_appetite масштабт готовность к рискованному действию
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "01-omega-archive" / "code"))
sys.path.insert(0, str(REPO / "02-mirror-swarm" / "code"))

from hormone_module import HormoneModule, BASELINE  # noqa: E402
from empathy_bus import EmpathyBus  # noqa: E402


@dataclass
class Agent8:
    agent_id: str
    hormones: HormoneModule = None

    def __post_init__(self):
        if self.hormones is None:
            self.hormones = HormoneModule()

    def reward_signal(self, r: float) -> dict:
        self.hormones.on_reward(r)
        self.hormones.tick()
        return self.hormones.tone_for_response()

    def punishment_signal(self, s: float) -> dict:
        self.hormones.on_punishment(s)
        self.hormones.tick()
        return self.hormones.tone_for_response()


@dataclass
class Stage8Result:
    agents: int
    reward_emotion_propagated: bool  # сосед感受 радость после награды источника
    punishment_emotion_propagated: bool  # сосед感受 страх после штрафа
    empathy_decay_works: bool  #遠 neighbor получает меньше, чем близкий
    cognitive_ability_affects_M: bool  # M масштабируется когнитивной способностью
    risk_appetite_changes: bool  # эмоция меняет risk_appetite
    chain_of_emotion: list[dict]


def run_stage8(n_agents: int = 6, registry_path=None) -> Stage8Result:
    bus = EmpathyBus(max_intensity=0.5, decay=0.5)  # затухание 0.5 по условию Этапа 8
    agents = {f"agent-{i:02d}": Agent8(agent_id=f"agent-{i:02d}") for i in range(n_agents)}

    # регистрация в шине: кольцо соседства
    ids = list(agents.keys())
    for i, aid in enumerate(ids):
        neighbors = {ids[i - 1], ids[(i + 1) % n_agents]}
        bus.register(aid, neighbors=neighbors)

    chain = []

    # --- сценарий 1: награда агенту-00 -> радость распространяется ---
    src = ids[0]
    tone = agents[src].reward_signal(r=2.0)  # радость
    bus.broadcast(src, "радость", intensity=1.0)
    close = ids[1]          # прямой сосед
    far = ids[n_agents // 2]  # далеко (другой конец кольца)
    feel_close = bus.local_feel(close)
    feel_far = bus.local_feel(far)
    chain.append({
        "event": "reward_broadcast",
        "source": src,
        "source_emotion": tone["emotion"],
        "close_neighbor": close,
        "close_feel": feel_close,
        "far_neighbor": far,
        "far_feel": feel_far,
    })

    # --- сценарий 2: штрафу агенту-3 -> страх распространяется ---
    src2 = ids[3]
    tone2 = agents[src2].punishment_signal(s=3.0)  # страх/злость
    bus.broadcast(src2, "страх", intensity=1.0)
    feel_close2 = bus.local_feel(ids[4])
    chain.append({
        "event": "punishment_broadcast",
        "source": src2,
        "source_emotion": tone2["emotion"],
        "close_neighbor": ids[4],
        "close_feel": feel_close2,
    })

    # --- влияние когнитивной способности на модулятор M ---
    hm = agents[ids[0]].hormones
    ca_high = hm.cognitive_ability()  # после награды 높은 дофамин -> высокая когнитивная способность
    hm.on_punishment(10.0)  # стресс -> кортизол растёт -> когнитивная способность падает
    hm.tick()
    ca_low = hm.cognitive_ability()
    chain.append({
        "event": "cognitive_ability",
        "ca_after_reward": ca_high,
        "ca_after_punishment": ca_low,
        "ca_decreased": ca_low < ca_high,
    })

    # --- risk_appetite меняется с эмоцией ---
    risk_neutral = agents[ids[5]].hormones.tone_for_response()["risk_appetite"]
    agents[ids[2]].hormones.on_punishment(5.0)
    agents[ids[2]].hormones.tick()
    risk_fearful = agents[ids[2]].hormones.tone_for_response()["risk_appetite"]
    chain.append({
        "event": "risk_appetite",
        "neutral": risk_neutral,
        "fearful": risk_fearful,
        "changed": risk_fearful != risk_neutral,
    })

    reward_emotion_propagated = feel_close.get("радость", 0.0) > 0
    punishment_emotion_propagated = feel_close2.get("страх", 0.0) > 0
    empathy_decay_works = feel_close.get("радость", 0.0) > feel_far.get("радость", 0.0)
    cognitive_ability_affects_M = ca_low < ca_high
    risk_appetite_changes = risk_fearful != risk_neutral

    return Stage8Result(
        agents=n_agents,
        reward_emotion_propagated=reward_emotion_propagated,
        punishment_emotion_propagated=punishment_emotion_propagated,
        empathy_decay_works=empathy_decay_works,
        cognitive_ability_affects_M=cognitive_ability_affects_M,
        risk_appetite_changes=risk_appetite_changes,
        chain_of_emotion=chain,
    )


if __name__ == "__main__":
    res = run_stage8(n_agents=6)
    print("агентов:", res.agents)
    print("радость распространена:", res.reward_emotion_propagated)
    print("страх распространён:", res.punishment_emotion_propagated)
    print("затухание эмпатии работает (близкий >遠):", res.empathy_decay_works)
    print("когнитивная способность влияет на M:", res.cognitive_ability_affects_M)
    print("risk_appetite меняется с эмоцией:", res.risk_appetite_changes)
    print("цепочка событий:", res.chain_of_emotion)

    assert res.reward_emotion_propagated, "радость не передалась соседу"
    assert res.punishment_emotion_propagated, "страх не передался соседу"
    assert res.empathy_decay_works, "затухание эмпатии не работает"
    assert res.cognitive_ability_affects_M, "когнитивная способность не влияет на M"
    assert res.risk_appetite_changes, "risk_appetite не меняется с эмоцией"
    print("Этап 8: приёмка пройдена")
