"""Structural plasticity: устранение синапсов «use-it-or-lose-it».

Отдельная от весовой R-STDP способность: сам набор связей меняется — синапс,
который долго проводит ниже порога, устраняется и больше не проводит ничего
(несъёмный 0 в переносимом графе). Реализуем только каноническую безопасную
половину (prune). Рост ребра сознательно НЕ делаем: добавление конкретного
pair'а нейрон→нейрон асимметрично и может инъектировать предпочтение стороны —
это ровно тот fraud-режим, от которого бережёт принцип 1.

Что под проверкой (и почему это направленно-нейтрально):
- критерий отбора зависит ТОЛЬКО от |веса| и порога ⇒ он эквариантен к любой
  перестановке слотов, в том числе к зеркалу left↔right, — предпочесть сторону
  структурно нечем;
- выключенный по дефолту канал (floor=0) обязан оставить побитово идентичным
  замороженный прогон plastic-arm — рабочее ядро и operating point P4.2 не
  двигаются;
- устранение монотонно (мёртвый синапс не воскрешается) и переносится в 0 только
  на СЛЕДУЮЩИХ эпизодах (устранение идёт forward, текущую траекторию не правит).

Интеграционные проверки на живость механизма/композицию обязаны краснеть только
если механизм реально не жив или ломает инвариант соседних каналов.
"""
from __future__ import annotations

import numpy as np
import pytest

from fly_connectome_agent.src.engineering.harness import symmetric_toy as toy
from fly_connectome_agent.src.engineering.harness.symmetric_toy import ToySettings
from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog


def _tiny():
    # sigma=0.0 → траектории детерминированы, сравнения побитово точны.
    return ToySettings(window_steps=12, max_steps=6, track_length=9, cell_distance=3,
                       edge_margin=1, sigma=0.0, episodes=1)


# plastic_slots=(0,1,2,3): локальный индекс 0=direct_L, 1=contra_L,
# 2=direct_R, 3=contra_R. Зеркало left↔right меняет их как (0 2)(1 3).

# --- чистая функция отбора -------------------------------------------------

def test_select_prunable_берёт_слабые_по_модулю():
    w = np.array([12.0, 70.0, -12.0, 70.0])
    assert toy.select_prunable(w, floor=20.0) == frozenset({0, 2})


def test_select_prunable_выключён_при_floor_nonpositive():
    w = np.array([0.0, 0.0, 70.0, 70.0])
    assert toy.select_prunable(w, floor=0.0) == frozenset()
    assert toy.select_prunable(w, floor=-5.0) == frozenset()


def test_select_prunable_игнорирует_знак_тормозных():
    # тормозной -10 по |w|<floor устраняется так же, как возбуждающий +10
    w = np.array([-10.0, 70.0, 10.0, 70.0])
    assert toy.select_prunable(w, floor=20.0) == frozenset({0, 2})


def test_select_prunable_перестановочно_эквариантен_то_есть_нейтрален():
    # Ключевой тест направленной нейтральности: зеркально переставленные веса
    # дают зеркально переставленное множество устранённых ⇒ механизм нечем
    # выбрать сторону. Проверяем на случайных весах и на зеркале (0 2)(1 3).
    rng = np.random.default_rng(7)
    perm = [2, 3, 0, 1]
    for _ in range(20):
        w = rng.uniform(0.0, 90.0, size=4)
        base = toy.select_prunable(w, floor=30.0)
        mirrored = toy.select_prunable(w[perm], floor=30.0)
        # new[i] = old[perm[i]] → прунутые new = { i : perm[i] ∈ base }
        expect = frozenset(i for i, src in enumerate(perm) if src in base)
        assert mirrored == expect


# --- default-off не двигает замороженный путь ------------------------------

def test_floor_nоль_побитово_равен_замороженному_arm():
    r = toy.run_arm("plastic", "right", 500, settings=_tiny(), episodes=3)
    agent = toy.ToyAgent(learner_kind="stdp", init_regime="symmetric", seed=500,
                         reward_coupled=True, settings=_tiny())
    traj = [agent.run_episode("right", ep) for ep in range(3)]
    for got, exp in zip(traj, r.episodes):
        assert np.array_equal(got.weights_after, exp.weights_after)
        assert got.reward == pytest.approx(exp.reward)
    assert agent._pruned == set()


# --- механика устранения ---------------------------------------------------

def test_устранение_монотонно_и_щадит_сильные():
    agent = toy.ToyAgent(learner_kind="stdp", init_regime="symmetric", seed=500,
                         reward_coupled=True, settings=_tiny(), structural_prune_floor=20.0)
    agent.learner.set_weights(np.array([12.0, 70.0, 12.0, 70.0]))
    agent._apply_structural_prune()
    assert agent._pruned == {0, 2}
    # даже если вес вернулся выше пола — мёртвый синапс не воскрешается
    agent.learner.set_weights(np.array([60.0, 70.0, 60.0, 70.0]))
    agent._apply_structural_prune()
    assert agent._pruned == {0, 2}
    assert not (agent._pruned & {1, 3})


def test_устранённый_синапс_не_проводит_в_переносимом_графе():
    agent = toy.ToyAgent(learner_kind="stdp", init_regime="symmetric", seed=500,
                         reward_coupled=True, settings=_tiny(), structural_prune_floor=20.0)
    agent.learner.set_weights(np.array([12.0, 70.0, 12.0, 70.0]))
    agent._apply_structural_prune()
    g = agent._carry_weights_to(toy.build_graph(agent.weights_init, agent.settings))
    conducted = g.weights[np.array(agent.settings.plastic_slots, dtype=np.int64)]
    assert conducted[0] == 0.0 and conducted[2] == 0.0
    assert conducted[1] != 0.0 and conducted[3] != 0.0


# --- механизм жив: устранённый синапс не проводит, а выученный вес не тронут --

def test_включённый_канал_выключает_синапс_который_жив_при_выключенном():
    off = toy.ToyAgent(learner_kind="stdp", init_regime="symmetric", seed=500,
                       reward_coupled=True, settings=_tiny())
    # floor=85: на задаче 'right' непрестижированный прямой путь direct_L не
    # получает drive (сенсор слева молчит) → вес остаётся стартовым 12 < 85 →
    # устраняется. Это ровно тот честный случай: поведение не меняется, меняется
    # СТРУКТУРА (синапс выключен), а не выученный вектор весов.
    on = toy.ToyAgent(learner_kind="stdp", init_regime="symmetric", seed=500,
                      reward_coupled=True, settings=_tiny(), structural_prune_floor=85.0)
    r_off = off.run_episode("right", 0)
    r_on = on.run_episode("right", 0)
    assert np.array_equal(r_off.weights_after, r_on.weights_after)   # ep0 побитово совпал
    assert 0 in on._pruned                                           # direct_L устранён
    slots = np.array(off.settings.plastic_slots, dtype=np.int64)
    w_off = off._carry_weights_to(toy.build_graph(off.weights_init, off.settings)).weights[slots]
    w_on = on._carry_weights_to(toy.build_graph(on.weights_init, on.settings)).weights[slots]
    assert w_on[0] == 0.0 and w_off[0] > 0.0                        # в on не проводит, в off жив
    # различие чисто структурное: выученный вектор (learner) у обоих одинаков
    assert on.plastic_weights[0] == pytest.approx(off.plastic_weights[0])


# --- композиция: не конфликтует с 4/1/6 -----------------------------------

def test_композиция_с_тремя_каналами_инварианты_держатся(tmp_path):
    prov = ProvenanceLog(path=str(tmp_path / "struct_comp.jsonl"))
    agent = toy.ToyAgent(learner_kind="stdp", init_regime="anti", seed=500,
                         settings=_tiny(), verb_feedback=0.4, self_obs_gain=0.5,
                         internal_observer_only=True, structural_prune_floor=40.0)
    for ep in range(3):
        r = agent.run_episode("mixed", ep, prov=prov)
        assert np.isfinite(r.reward) and r.steps > 0
    assert prov.count == 3
    assert prov.verify_chain() is True
    assert agent.prov_stats["governance_violations"] == 0
