"""Композиция принципов 4 + 1 + 6: три канала включены ОДНОВРЕМЕННО.

Каждый механизм по отдельности уже закрыт своим тестом (verb_feedback /
self_observation / internal_observer). Здесь проверяем ровно одно: что они
ДРУГ ДРУГА не ломают, когда живут на одном агенте одновременно.

Что реально может пойти не так и что тут под проверкой:
- принцип 1 читает журнал, принцип 4 мутирует M каждый шаг, принцип 6 стоит на
  входе оценки — вместе они пишут/читают один и тот же provenance-журнал. Если
  чтение прошлого смешивается с текущей записью — цепь портится;
- включённый watch-страж (6) не должен пропускать внешнюю оценку даже когда
  остальные каналы активны;
- выключенные по дефолту каналы не должны завязывать поведение на наличие
  журнала (только присутствие prov не должно ничего менять);
- ортогональность: эффект самонаблюдения обязан пережить соседство двух других
  включённых каналов (иначе они его маскируют).

Интеграционный тест, а не новый механизм: красным он обязан стать только если
композиция реально ломает инвариант — иначе он зелёный и это и есть результат.
"""
from __future__ import annotations

import numpy as np
import pytest

from fly_connectome_agent.src.engineering.harness import symmetric_toy as toy
from fly_connectome_agent.src.engineering.harness.symmetric_toy import (
    ExternalValuationError,
    ToySettings,
)
from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog


def _tiny():
    # sigma=0.0 → траектории детерминированы, сравнения побитово точны.
    return ToySettings(window_steps=12, max_steps=6, track_length=9, cell_distance=3,
                       edge_margin=1, sigma=0.0, episodes=1)


def _all_on(**over):
    """Агент со всеми тремя каналами; over позволяет глушить отдельный из них."""
    kw = dict(verb_feedback=0.4, self_obs_gain=0.5, internal_observer_only=True)
    kw.update(over)
    return toy.ToyAgent(learner_kind="stdp", init_regime="anti", seed=500,
                        settings=_tiny(), **kw)


def _signature(r):
    """Измеримый слепок эпизода — только наблюдаемое поведение пользователя."""
    return (r.spikes_left, r.spikes_right, r.reached, round(r.reward, 6), r.steps)


# --- 1. все три включены: управление чисто, цепь цела -----------------------

def test_all_three_on_governance_zero_and_chain_intact(tmp_path):
    prov = ProvenanceLog(path=str(tmp_path / "all_on.jsonl"))
    agent = _all_on()
    # Убедиться, что каналы реально живы на этом экземпляре (а не проглочены).
    assert agent.modulator.params.epsilon == pytest.approx(0.4)   # принцип 4
    assert agent.self_obs_gain == 0.5                              # принцип 1
    assert agent.internal_observer_only is True                    # принцип 6

    for ep in range(3):
        r = agent.run_episode("right", ep, prov=prov)
        assert np.isfinite(r.reward) and r.steps > 0

    # Три эпизода → три записи; самонаблюдение их читало, ничего не порвало.
    assert prov.count == 3
    assert prov.verify_chain() is True
    assert agent.prov_stats["governance_violations"] == 0


# --- 2. страж компонуется: внешняя оценка отвергается при соседях -----------

def test_guard_refuses_external_while_other_channels_live(tmp_path):
    agent = _all_on()  # verb_feedback и self_obs активны, guard включён
    with pytest.raises(ExternalValuationError):
        agent.inject_valuation(5.0, source="external")
    # Внутренний канал при том же guard'e остаётся легален.
    M = agent.inject_valuation(5.0, source="internal")
    assert np.isfinite(M)


# --- 3. выключенные каналы не завязывают поведение на журнал ----------------

def test_all_off_is_independent_of_journal_presence(tmp_path):
    """Отключённое самонаблюдение не должно даже замечать, что журнал есть:
    прогон с prov и прогон без prov обязан совпасть поведению эпизодов."""
    with_journal = []
    a = toy.ToyAgent(learner_kind="stdp", init_regime="anti", seed=500, settings=_tiny())
    prov = ProvenanceLog(path=str(tmp_path / "off.jsonl"))
    for ep in range(3):
        with_journal.append(_signature(a.run_episode("right", ep, prov=prov)))

    no_journal = []
    b = toy.ToyAgent(learner_kind="stdp", init_regime="anti", seed=500, settings=_tiny())
    for ep in range(3):
        no_journal.append(_signature(b.run_episode("right", ep, prov=None)))

    assert with_journal == no_journal
    assert a.prov_stats["governance_violations"] == 0


# --- 4. ортогональность: самонаблюдение заметно даже рядом с двумя другими ---

def test_self_obs_effect_survives_with_neighbors_on(tmp_path):
    """Соседи (глагол-обратная связь + страж) включены в обоих агентах; отличается
    только самонаблюдение. Эпизод 0 совпадает (истории ещё нет), эпизоды ≥1 — нет:
    канал не замаскирован композицией."""
    with_obs = _all_on(self_obs_gain=0.5)
    without_obs = _all_on(self_obs_gain=0.0)

    pa = ProvenanceLog(path=str(tmp_path / "a.jsonl"))
    pb = ProvenanceLog(path=str(tmp_path / "b.jsonl"))
    a_sig = [_signature(with_obs.run_episode("right", ep, prov=pa)) for ep in range(3)]
    b_sig = [_signature(without_obs.run_episode("right", ep, prov=pb)) for ep in range(3)]

    assert a_sig[0] == b_sig[0]          # нет истории → scale=1.0 → совпадают
    assert a_sig[1:] != b_sig[1:]        # прочитанное самонаблюдение сдвигает отклик
    # И соседи не разъехались: у обоих цепь цела и нарушений нет.
    assert pa.verify_chain() and pb.verify_chain()
    assert with_obs.prov_stats["governance_violations"] == 0
    assert without_obs.prov_stats["governance_violations"] == 0
