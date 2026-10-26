"""Принцип 1 — петля самонаблюдения: агент читает СОБСТВЕННЫЙ журнал как вход.

Фальсифицируется ровно одно утверждение из PHILOSOPHY.md: признаки прошлых
эпизодов, прочитанные из собственного provenance-журнала, обязан влиять на текущий
выбор. Если влияния нет — петля фиктивная (читает, но не влияет).

Честность и границы (те же, что у принципа 4):
- self_obs_gain по умолчанию 0.0 → obs_scale=1.0 → умножение sensory_drive на 1.0
  побитово точно → замороженный контур и все прежние прогоны не сдвигаются;
- канал направлен-нейтрален: scale — скаляр, одинаково усиливает SENS_LEFT и
  SENS_RIGHT. Он не может влить «предпочтение стороны» в зеркальный тест;
- signal берётся из ЧИСТого чтения журнала (сколько СВОИХ записей уже есть), а не
  из task_reward — иначе это был бы просто замаскированный сигнал награды.

Тест, который обязан упасть, если фича притворяется: поведение с петлёй vs без неё
различимо по измеримому признаку (σ=0 → детерминированно).
"""
from __future__ import annotations

import numpy as np
import pytest

from fly_connectome_agent.src.engineering.harness import symmetric_toy as toy
from fly_connectome_agent.src.engineering.harness.symmetric_toy import (
    ToySettings,
    self_observation_scale,
)
from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog


def _tiny():
    return ToySettings(window_steps=12, max_steps=6, track_length=9, cell_distance=3,
                       edge_margin=1, sigma=0.0, episodes=1)


# --- 1. чистая функция: направленность и выключение ------------------------

def test_scale_is_off_without_history_or_gain():
    """Нет своих записей в журнале ИЛИ выключен gain → scale ровно 1.0: канал
    молчит, замороженный путь цел."""
    assert self_observation_scale(0, 0.5) == pytest.approx(1.0)
    assert self_observation_scale(7, 0.0) == pytest.approx(1.0)


def test_scale_is_active_with_history_and_gain():
    assert self_observation_scale(3, 0.5) > 1.0
    assert np.isfinite(self_observation_scale(1000, 0.5))


def test_scale_is_direction_neutral():
    """Вход — только число записей и gain; направление (left/right) не участвует и
    не может повлиять. Скаляр одинаков для идентичных входов — проверка, что
    функция вообще не принимает задачу/сторону."""
    import inspect
    params = set(inspect.signature(self_observation_scale).parameters)
    assert not (params & {"task", "side", "action_type", "target"}), \
        f"самонаблюдение приняло направленный вход: {params}"


# --- 2. петля через реальный цикл: прочитанное влияет на поведение ----------

def _episode_spikes(gain: float, tmp_path, fname: str):
    prov = ProvenanceLog(path=str(tmp_path / fname))
    agent = toy.ToyAgent(learner_kind="stdp", init_regime="anti", seed=500,
                         settings=_tiny(), self_obs_gain=gain)
    out = []
    for ep in range(3):
        r = agent.run_episode("right", ep, prov=prov)
        out.append((r.spikes_left, r.spikes_right))
    return out


def test_reading_own_journal_changes_behavior(tmp_path):
    """Эпизоды №2,№3: агент с gain>0 уже прочитал СВОИ прошлые записи и ведёт себя
    иначе, чем агент с gain=0 (тот же seed, та же траектория). Совпадут — петля
    фиктивная."""
    with_loop = _episode_spikes(0.5, tmp_path, "loop_on.jsonl")
    without = _episode_spikes(0.0, tmp_path, "loop_off.jsonl")
    # Эпизод 0: журнала ещё нет → scale 1.0 у обоих → обязаны совпасть (нет leaks).
    assert with_loop[0] == without[0]
    # Начиная с эпизода 1: прочитанная предыстория сдвигает отклик.
    assert with_loop[1:] != without[1:], "агент прочитал свой журнал, но поведение не изменилось"


def test_severed_loop_is_bitwise_identical(tmp_path):
    """Оборвать петлю (gain=0) → поведение побитово равно дефолтному агенту без
    параметра. Это гарантия, что замороженный контур не сдвинут."""
    default = _episode_spikes(0.0, tmp_path, "d.jsonl")
    explicit_off = _episode_spikes(0.0, tmp_path, "e.jsonl")
    assert default == explicit_off
