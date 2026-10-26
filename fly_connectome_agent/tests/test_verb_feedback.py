"""Принцип 4 (обратная половина): действие → его глагол → обучающий сигнал.

Что здесь фальсифицируется, а что нет — честно.

Это ДЕМОНСТРАЦИЯ МЕХАНИЗМА, а не результат обучения и не claim про «поведение vs
мозг». Смысл ровно в утвержении из PHILOSOPHY.md: глагол, который сеть реально
совершила, обязан возвращаться в сеть как отдельный обучающий вход, отдельный от
task_reward. Поэтому:

- `verb_feedback` по умолчанию 0.0: при выключенной обратной связи контур
  побитово совпадает с прежним (ничего замороженного не трогается, ни один seed не
  пережигается);
- сигнал обязан быть НАПРАВЛЕННО-НЕЙТРАЛЬНЫМ (одинаково для left-left и right-right),
  иначе он вольётся в зеркальный тест P4.1 как встроенное предпочтение стороны —
  это единственный способ, которым такая фича могла бы стать научной ложью.

Тест, который обязан упасть, если фича притворяется: оборвать обратную связь
(verb_feedback=0) → вклад глагола в модулятор обязан обнулиться. Если он не
обнулился — сигнал был не про глагол, а про что-то ещё (или про награду дважды).
"""
from __future__ import annotations

import numpy as np
import pytest

from fly_connectome_agent.src.engineering.harness import symmetric_toy as toy
from fly_connectome_agent.src.engineering.harness.symmetric_toy import (
    ToySettings,
    verb_coherence,
    verb_for_action,
)


# --- 1. глагол рождается из совершённого действия --------------------------

def test_verb_is_derived_from_the_committed_action():
    """Глагол выводится из действия, а не берётся константой: три действия —
    три разных глагола. Если бы все сводились к одному «выбирает», обратной
    связи «действие→глагол» не существовало бы."""
    assert verb_for_action("move_left") != verb_for_action("move_right")
    assert verb_for_action("move_left") == verb_for_action("move_left")
    assert verb_for_action("stay") not in {verb_for_action("move_left"),
                                           verb_for_action("move_right")}


def test_unknown_action_is_rejected():
    with pytest.raises(ValueError, match="unknown action_type"):
        verb_for_action("fly_to_moscow")


# --- 2. сигнал направлен-нейтрален (иначе это отравление зеркального теста) -

def test_coherence_is_mirror_symmetric():
    """Повтор одного и того же глагола даёт одинаковый вклад и для лева, и для
    права. Это инвариант: как только left-left и right-right разойдутся, фича
    вольёт направление в сеть без всякого обучения."""
    same_left = verb_coherence("move_left", "move_left")
    same_right = verb_coherence("move_right", "move_right")
    assert same_left == pytest.approx(same_right)
    # Смена глагола (действие поменялось) — вклад строго меньше, чем устойчивое
    # повторение: сигнал про последовательность совершенных глаголов.
    switch = verb_coherence("move_left", "move_right")
    assert same_left > switch
    # stay-vs-move так же нейтрален по обе стороны.
    assert verb_coherence("stay", "move_left") == pytest.approx(verb_coherence("stay", "move_right"))


def test_first_action_has_a_defined_coherence():
    # Нет предыстории — определённое, воспроизводимое значение, не NaN.
    v = verb_coherence(None, "move_left")
    assert np.isfinite(v)


# --- 3. выученный глагол исчезает при обрыве обратной связи ----------------

def _modulator(epsilon):
    return toy.RewardModulator(params=toy.ModulatorParams(alpha=1.0, epsilon=epsilon, M_max=10.0))


def test_severing_feedback_zeroes_the_verb_contribution():
    """Главное: при epsilon=0 вклад глагола в M обязан исчезнуть — M совпадает
    с чисто наградным. При включённой связи тот же глагол сдвигает M.

    reward=5.0 заведомо ниже M_max=10: иначе оба значения клиппинга в 10 и
    разница исчезла бы по вине клиппинга, а не механизма.
    """
    reward = 5.0
    coh = 1.0

    m_off = _modulator(0.0).compute(task_reward=reward, coherence=coh)
    m_off_baseline = _modulator(0.0).compute(task_reward=reward, coherence=0.0)
    assert m_off == pytest.approx(m_off_baseline), "обратная связь выключена, но глагол всё ещё влияет"

    m_on = _modulator(0.5).compute(task_reward=reward, coherence=coh)
    m_on_zero = _modulator(0.5).compute(task_reward=reward, coherence=0.0)
    assert m_on != pytest.approx(m_on_zero), "включённая связь не даёт никакого вклада"


def test_verb_feedback_is_not_part_of_the_frozen_operating_point():
    """Механизм не протекает в замороженную рабочую точку P4.2.

    p4_2_protocol сверяет operating_point со ВСЕМИ полями ToySettings; новое поле
    молча изменило бы confirmatory-прогон, не трогая digest. Поэтому обратная связь
    — параметр агента (verb_feedback в __init__), а не поле settings.
    """
    import dataclasses
    assert "verb_feedback" not in dataclasses.asdict(ToySettings())
    # Дефолт механизма — выключен: term помножен на ноль, замороженный контур цел.
    m_default = toy.make_modulator(True, ToySettings())
    assert m_default.params.epsilon == 0.0


# --- 4. через реальный цикл, не только изолированный модулятор -------------

def _tiny():
    return ToySettings(window_steps=10, max_steps=6, track_length=9, cell_distance=3,
                       edge_margin=1, sigma=0.0, episodes=1)


def _final_weights(**kw):
    agent = toy.ToyAgent(learner_kind="stdp", init_regime="anti", seed=500, settings=_tiny(), **kw)
    for ep in range(4):
        r = agent.run_episode("right", ep)
    return r.weights_after


def test_default_agent_is_bitwise_identical_to_feedback_off():
    """Через весь цикл: агент без параметра == агент с verb_feedback=0.0. Значит
    механизм не сдвигает замороженный путь даже побитово."""
    default = _final_weights()
    explicit_off = _final_weights(verb_feedback=0.0)
    assert np.array_equal(default, explicit_off)


def test_live_feedback_changes_the_learning_trajectory():
    """Тот же seed, та же траектория, только feedback включён → веса обязаны
    разойтись (σ=0 → детерминированно). Если совпадают — сигнал глагола в цикл
    не доходит, и вся фича декоративная."""
    off = _final_weights(verb_feedback=0.0)
    on = _final_weights(verb_feedback=1.0)
    assert not np.array_equal(off, on)
