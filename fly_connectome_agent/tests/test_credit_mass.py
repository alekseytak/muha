"""Счётчики массы кредита внутри клетки (P4.6 v2, гейт механики).

Пилот v1 показал, что отношение `mean|Δw|` **между двумя разошёвшимися руками**
не может быть гейтом механизма: `delay_frozen` учится слабее `plastic`, поэтому
его собственная масса eligibility меньше, и отношение уезжает за полосу даже при
абсолютно правильном механизме (на seed'е 182: 0.2704 против предсказанных 0.3679).

Внутри одной клетки такой проблемы нет. Для каждого события доставки считаем две
величины:

- `credit_at_action` — какой dw дал бы этот кредит, доставь мы его в момент
  действия (тот же payload, тот же модуллятор, след **на момент действия**);
- `credit_delivered` — какой dw действительно вышел (след на момент доставки для
  `delay_live`, слепок, умноженный на `decay**D`, для `delay_frozen`).

Тогда честное предсказание механизма — это отношение **внутри руки**, и для
`delay_frozen` оно равно `exp(-D/tau)` с точностью численного расчёта, ни от
какой траектории не зависит. Для `delay_live` это отношение свободно: именно в
нём живёт адресность (наполненный задержкой след даёт больше, чем `exp(-D/tau)`).

Проверяется:

- при `D = 0` счётчики равны друг другу (задержки нет — доставлено ровно то, что
  доставалось бы сразу);
- `delay_frozen`: отношение `delivered / at_action` равно `exp(-D/tau)` при
  загрязнённом интервале задержки — то есть не зависит от поздних спайков;
- `delay_live`: то же отношение **не** равно `exp(-D/tau)` при загрязнении — оно
  больше, и это и есть набранная чужая масса;
- `credit_at_action` фиксируется в момент действия и не меняется от того, что
  происходит в интервале задержки;
- `credit_delivered` — это ровно сумма `|dw|` тех же событий, что применились к
  весам, а не вторая, независимая проверка;
- `dw_for` не трогает веса и совпадает с `apply_modulator` бит-в-бит;
- заморозка весов даёт нули в обоих счётчиках (гейт не делит на ноль);
- векторный payload тоже считается по рёбрам, а не как скаляр.
"""
import math

import numpy as np
import pytest

from fly_connectome_agent.tests.test_credit_delay import (
    drive,
    make_agent,
    pattern,
)
from fly_connectome_agent.src.engineering.harness.snn_corridor import DELAY_ARMS

DELAY = 20
TAU = 80.0
FACTOR = math.exp(-DELAY / TAU)


def _stats(agent) -> dict:
    return agent.prov_stats


def _one_credit(arm: str, *, contaminate: bool, delay: int = DELAY) -> dict:
    """Один кредит, затем отсчёт задержки; возвращает счётчики клетки."""
    agent = make_agent(arm, window_steps=max(10, delay), credit_delay_steps=delay)
    agent.learner.eligibility = pattern(agent)
    agent.schedule_credit(2.0)
    dws = drive(agent, delay, contaminate=contaminate)
    stats = _stats(agent)
    stats = {**stats, "_dw_sum": float(np.sum(np.abs(np.concatenate(dws))))}
    return stats


# --- D = 0: задержки нет, доставлено ровно то, что доставалось бы сразу -----
@pytest.mark.parametrize("arm", ["plastic", *DELAY_ARMS])
def test_pri_nulevoy_zaderzhke_schetchiki_ravny(arm):
    agent = make_agent(arm, window_steps=10, credit_delay_steps=0)
    agent.learner.eligibility = pattern(agent)
    dw = agent._apply_now(2.0)
    stats = _stats(agent)
    assert stats["credit_at_action"] == pytest.approx(
        float(np.sum(np.abs(dw))), rel=1e-12)
    assert stats["credit_delivered"] == pytest.approx(stats["credit_at_action"],
                                                      rel=1e-12)
    assert stats["credit_delivered"] > 0.0, (
        "иначе нулевое отношение было бы неотличимо от сломанного механизма")


# --- frozen: отношение внутри руки равно exp(-D/tau) -----------------------
@pytest.mark.parametrize("contaminate", [False, True])
def test_frozen_otnoshenie_ravno_exp_minus_D_over_tau(contaminate):
    stats = _one_credit("delay_frozen", contaminate=contaminate)
    got = stats["credit_delivered"] / stats["credit_at_action"]
    assert got == pytest.approx(FACTOR, rel=1e-9), (
        "замороженный контроль обязан гаситься собственной константой угасания "
        "независимо от того, что происходило в интервале задержки")


# --- live: отношение НЕ является константой механизма -----------------------
def test_live_otnoshenie_vyshe_predskazannogo_pri_zagryaznenii():
    pure = _one_credit("delay_live", contaminate=False)
    dirty = _one_credit("delay_live", contaminate=True)
    r_pure = pure["credit_delivered"] / pure["credit_at_action"]
    r_dirty = dirty["credit_delivered"] / dirty["credit_at_action"]
    assert r_pure == pytest.approx(FACTOR, rel=1e-9), (
        "без поздних спайков живой след — это тот же слепок, только угасший")
    assert r_dirty > r_pure + 1e-9, (
        "набранная за задержку чужая масса обязана сделать доставленный кредит "
        "больше, чем exp(-D/tau): это и есть адресность, а не амплитуда")
    assert r_dirty > FACTOR + 1e-9


# --- at_action фиксируется в момент действия -------------------------------
def test_massa_na_momente_deystviya_ne_zavisit_ot_intervala_zaderzhki():
    pure = _one_credit("delay_frozen", contaminate=False)
    dirty = _one_credit("delay_frozen", contaminate=True)
    assert pure["credit_at_action"] == pytest.approx(dirty["credit_at_action"],
                                                     rel=1e-12)
    assert pure["credit_delivered"] == pytest.approx(dirty["credit_delivered"],
                                                     rel=1e-12)


# --- delivered = сумма |dw| тех же событий, что применились ----------------
def test_dostavlennyy_counter_eto_tot_zhe_dw():
    for arm in DELAY_ARMS:
        stats = _one_credit(arm, contaminate=True)
        assert stats["credit_delivered"] == pytest.approx(stats["_dw_sum"],
                                                          rel=1e-12)


# --- dw_for: чистая функция, не трогает веса -------------------------------
def test_dw_for_ne_mutit_vesa_i_sovpadaet_s_apply():
    a = make_agent("plastic")
    a.learner.eligibility = pattern(a)
    before = a.learner.get_weights().copy()
    preview = a.learner.dw_for(2.0)
    assert np.array_equal(a.learner.get_weights(), before), (
        "предпросмотр кредита не имеет права двигать веса")
    applied = a.learner.apply_modulator(2.0)
    assert np.array_equal(preview, applied), (
        "предпросмотр обязан совпасть с применяемым значением бит-в-бит")

    b = make_agent("delay_frozen")
    b.learner.eligibility = pattern(b)
    snap = b.learner.eligibility.copy()
    prev_late = b.learner.dw_for(2.0, onto=snap, decay_steps=DELAY)
    late = make_agent("delay_frozen")
    late.learner.eligibility = pattern(late)
    applied_late = late.learner.apply_modulator(2.0, onto=snap,
                                               onto_delay_steps=DELAY)
    assert np.array_equal(prev_late, applied_late)


# --- заморозка весов: нули, и ноль в знаменателе виден ---------------------
def test_no_plasticity_daut_nulevye_schetchiki():
    agent = make_agent("no_plasticity", window_steps=10, credit_delay_steps=0)
    agent.learner.eligibility = pattern(agent)
    agent._apply_now(2.0)
    stats = _stats(agent)
    assert stats["credit_at_action"] == 0.0
    assert stats["credit_delivered"] == 0.0
    assert stats["deliveries"] == 1, (
        "событие доставки считается и при замороженных весах: иначе счётчик "
        "доставок расходится с решениями")


# --- векторный payload -----------------------------------------------------
def test_vektornyy_payload_schitaetsya_po_rebram():
    agent = make_agent("plastic", window_steps=10, credit_delay_steps=0)
    elig = pattern(agent)
    agent.learner.eligibility = elig
    n = int(elig.size)
    payload = np.linspace(-1.0, 2.0, n)
    dw = agent._apply_now(payload)
    expected = float(np.sum(np.abs(agent.learner.dw_for(payload))))
    assert agent.prov_stats["credit_at_action"] == pytest.approx(expected,
                                                                 rel=1e-12)
    assert agent.prov_stats["credit_delivered"] == pytest.approx(
        float(np.sum(np.abs(dw))), rel=1e-12)
