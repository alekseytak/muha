"""Принцип 6 — наблюдатель = то, что наблюдает (ни одной внешней оценки).

Не доказывает феноменологию. Операционализировано как архитектурный инвариант на
границе доверия: сигнал, которым обучается сеть, обязан идти из внутреннего канала
(собственное восприятие/действие агента), а не подаваться извне критиком. Когда
инвариант включён, внешняя инъекция оценки обязана быть ОТВЕРГНУТА.

Границы как у принципов 4/1:
- internal_observer_only по умолчанию False → путём нарушается ничего, замороженный
  контур побитово цел;
- инвариант обязан быть ВЫПОЛНИМ: обычный эпизод (награда из env.step, которую
  агент сам же и совершил) помечен "internal" и проходит → иначе это пустой запрет.

Тест, который обязан упасть, если инвариант бутафорский: внешняя подача оценки при
включённом инварианте → отказ; при выключенном → проходит; внутренний канал при
включённом → проходит И равен дефолтному поведению.
"""
from __future__ import annotations

import numpy as np
import pytest

from fly_connectome_agent.src.engineering.harness import symmetric_toy as toy
from fly_connectome_agent.src.engineering.harness.symmetric_toy import (
    ExternalValuationError,
    ToySettings,
    valuation_gate,
)


def _tiny():
    return ToySettings(window_steps=12, max_steps=6, track_length=9, cell_distance=3,
                       edge_margin=1, sigma=0.0, episodes=1)


# --- 1. сам страж на границе ----------------------------------------------

def test_gate_rejects_external_valuation_when_invariant_on():
    with pytest.raises(ExternalValuationError):
        valuation_gate("external_teacher", internal_only=True)


def test_gate_passes_internal_channel_when_invariant_on():
    # Не должно бросить: внутренний канал — единственный разрешённый источник.
    valuation_gate("internal", internal_only=True)


def test_gate_is_inert_when_invariant_off():
    # Дефолт выключен → внешняя оценка проходит (обратная совместимость, ноль утечек).
    valuation_gate("external_teacher", internal_only=False)


# --- 2. инвариант выполним на реальном цикле (не вакуумный запрет) ---------

def _run_agent(**kw):
    agent = toy.ToyAgent(learner_kind="stdp", init_regime="anti", seed=500, settings=_tiny(), **kw)
    for ep in range(3):
        r = agent.run_episode("right", ep)
    return r.weights_after


def test_invariant_on_episode_is_untouched():
    """Награда в цикле приходит из env.step самого агента → source 'internal' →
    инвариант включён, но эпизод проходит и веса побитово равны дефолту. Значит
    запрет выполним и не сдвигает замороженный путь."""
    default = _run_agent()
    guarded = _run_agent(internal_observer_only=True)
    assert np.array_equal(default, guarded)


# --- 3. внешняя инъекция в контур отказывается ----------------------------

def test_external_injection_refused_on_guarded_agent():
    agent = toy.ToyAgent(learner_kind="stdp", init_regime="anti", seed=500,
                         settings=_tiny(), internal_observer_only=True)
    with pytest.raises(ExternalValuationError):
        agent.inject_valuation(10.0, source="external_teacher")


def test_external_injection_allowed_when_off():
    # Без инварианта тот же вызов — легальная внешняя оценка: возвращает M.
    agent = toy.ToyAgent(learner_kind="stdp", init_regime="anti", seed=500,
                         settings=_tiny(), internal_observer_only=False)
    assert np.isfinite(agent.inject_valuation(10.0, source="external_teacher"))
