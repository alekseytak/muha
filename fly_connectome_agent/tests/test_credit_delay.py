"""Тесты времени доставки кредита (P4.6, руки `delay_live` и `delay_frozen`).

До сих пор модулятор приходил в тот же момент, что и действие: награда
прочитана, `apply_modulator` вызван, след eligibility ещё свежий. P4.6 разводит
эти два момента на `credit_delay_steps` шагов симуляции. Задержка меняет две
независимые вещи, и их надо разделять так же аккуратно, как в v5 разделяли
привязку и амплитуду:

- **амплитуда**: след угасает, `elig -> elig * exp(-D/tau)`, кредита приходит
  меньше — это эквивалентно меньшающей eta;
- **адресность**: в интервале задержки сеть продолжает стрелять, и кредит за
  прошлое действие ложится на след, уже наполненный поздними спайками, то есть
  на чужие рёбра.

Рука `delay_live` — честный случай: кредит доставляется в живой след через D
шагов. Рука `delay_frozen` — амплитудный контроль: кредит доставляется через
столько же шагов, но в слепок следа, взятый в момент действия и погашенный ровно
на `exp(-D/tau)`. Разница рук и есть цена адресности при равной амплитуде.

Проверяется:

- `D = 0` даёт бит-в-бит то же самое, что `plastic` (иначе сравнение смешано);
- задержка объявлена только в delay-руках: `plastic` с `D > 0` не существует;
- доставка случается ровно через D шагов симуляции, не раньше и не позже;
- если в интервале задержки спайков не было, `live` и `frozen` совпадают
  бит-в-бит: единственное их различие — набранные за задержку спайки;
- амплитуда `frozen` равна `exp(-D/tau)` от амплитуды того же кредита без
  задержки — тождество, а не тенденция;
- живой след после задержки содержит чужую массу: `sum|dw_live| > sum|dw_frozen|`;
- `onto` проверяет форму и конечность, `onto=None` сохраняет старое поведение;
- задержка отрицательная или длиннее окна запрещена конструкцией;
- недоставленный кредит не теряется на границе эпизода, а доезжает в следующем;
- правило v1: у delay-рук нет ни пульного, ни пер-нейронного режима,
  доставляется скаляр.
"""
import math

import numpy as np
import pytest

from fly_connectome_agent.src.engineering.harness.snn_corridor import (
    ARMS,
    DELAY_ARMS,
    NEURON_ARMS,
    POOL_ARMS,
    Channel,
    CorridorAgent,
    LoopSettings,
)

IN_LEFT = (0, 1)
IN_RIGHT = (2, 3)
OUT_LEFT = (4, 5)
OUT_RIGHT = (6, 7)


def make_agent(arm: str, **overrides) -> CorridorAgent:
    from fly_connectome_agent.tests.test_snn_corridor import make_graph

    point = dict(episodes=1, window_steps=20, i_drive=60.0, gain=6.0, sigma=4.0,
                 eta=0.02, max_episode_steps=40, width=5, height=5,
                 start=(1, 1), target=(3, 3), seed=11)
    point.update(overrides)
    settings = LoopSettings(**point)
    channel = Channel(inputs_left=IN_LEFT, inputs_right=IN_RIGHT,
                      outputs_left=OUT_LEFT, outputs_right=OUT_RIGHT)
    # weight=0.5: середина диапазона, чтобы границы [0, 1] не маскировали разницу
    return CorridorAgent(graph=make_graph(weight=0.5), channel=channel,
                         settings=settings, arm=arm)


def pattern(agent: CorridorAgent) -> np.ndarray:
    """След, в котором каждый пост-нейрон внес свою долю — без нулей."""
    post = agent.learner.edge_index[:, 1]
    elig = np.ones(agent.learner.n_plastic)
    for k, node in enumerate(OUT_LEFT + OUT_RIGHT, start=1):
        elig[post == int(node)] = k
    return elig


def quiet(agent: CorridorAgent) -> np.ndarray:
    return np.zeros(agent.base.n_nodes, dtype=bool)


def fire(agent: CorridorAgent, node: int) -> np.ndarray:
    s = quiet(agent)
    s[node] = True
    return s


def drive(agent: CorridorAgent, steps: int, *, contaminate: bool = False) -> list:
    """Отсчёт задержки без среды: шаги update с нулевой модуляцией.

    При contaminate чередуем вход (нарастает pre_trace) и выход (он умножается
    на этот след и добавляет массу рёбрам, которых в момент действия не было):
    именно так поздняя активность пачкает живой след.
    """
    out: list = []
    for k in range(steps):
        if contaminate and k % 2 == 0:
            pre, post = fire(agent, IN_LEFT[0]), quiet(agent)
        elif contaminate:
            pre, post = quiet(agent), fire(agent, OUT_LEFT[0])
        else:
            pre = post = quiet(agent)
        agent.learner.update(pre, post, modulator=0.0)
        out.extend(agent.advance_credit())
    return out


# --- объявление рук и границы настройки -------------------------------------
def test_ruleki_obyavleny_i_bez_chuzhih_rezhimov():
    assert {"delay_live", "delay_frozen"} <= set(ARMS)
    assert DELAY_ARMS == {"delay_live": "live", "delay_frozen": "frozen"}
    for arm in DELAY_ARMS:
        assert arm not in POOL_ARMS and arm not in NEURON_ARMS


def test_zaderzhka_otricatelnaya_i_dlinnee_okna_zapreshena():
    with pytest.raises(ValueError):
        LoopSettings(credit_delay_steps=-1)
    with pytest.raises(ValueError):
        LoopSettings(window_steps=20, credit_delay_steps=21)
    LoopSettings(window_steps=20, credit_delay_steps=0)
    LoopSettings(window_steps=20, credit_delay_steps=20)


def test_zaderzhka_tolko_v_delay_rulekah():
    make_agent("plastic")
    with pytest.raises(ValueError):
        make_agent("plastic", credit_delay_steps=10)
    make_agent("delay_live", window_steps=20, credit_delay_steps=10)


# --- D = 0 тождественна v1 -------------------------------------------------
def test_nulevaya_zaderzhka_bit_v_bit_sovpadaet_s_plastic():
    ref = make_agent("plastic").run(episodes=3)
    for arm in DELAY_ARMS:
        got = make_agent(arm, credit_delay_steps=0).run(episodes=3)
        assert [e.__dict__ for e in got.episodes] == [e.__dict__ for e in ref.episodes]


# --- отсчёт ---------------------------------------------------------------
def test_dostava_slushitsya_rovno_cherez_D_shagov(monkeypatch):
    agent = make_agent("delay_live", window_steps=10, credit_delay_steps=4)
    seen: list[int] = []
    real_apply = agent.learner.apply_modulator

    def spy(modulator, **kwargs):
        seen.append(agent.learner.step_count)
        return real_apply(modulator, **kwargs)

    monkeypatch.setattr(agent.learner, "apply_modulator", spy)
    agent.learner.eligibility = pattern(agent)
    at = agent.learner.step_count
    agent.schedule_credit(2.0)
    assert seen == [], "при D > 0 кредит не доставляется в момент действия"
    assert drive(agent, 3) == [], "раньше срока доставка не происходит"
    assert len(drive(agent, 1)) == 1
    assert seen == [at + 4]


# --- live против frozen: амплитуда и адресность ----------------------------
def test_bez_spaykov_v_intervale_live_i_frozen_sovpadayutsya():
    outs = {}
    for arm in DELAY_ARMS:
        agent = make_agent(arm, window_steps=10, credit_delay_steps=6)
        agent.learner.eligibility = pattern(agent)
        agent.schedule_credit(2.0)
        outs[arm] = np.concatenate(drive(agent, 6))
    assert np.array_equal(outs["delay_live"], outs["delay_frozen"])


def test_zaderzhka_gasit_amplitudu_rovno_v_exp_minus_D_over_tau():
    delay, tau = 20, 80.0
    now = make_agent("plastic")
    now.learner.eligibility = pattern(now)
    dw_now = now.learner.apply_modulator(2.0)

    late = make_agent("delay_frozen", credit_delay_steps=delay)
    late.learner.eligibility = pattern(late)
    snapshot = late.learner.eligibility.copy()
    dw_late = late.learner.apply_modulator(2.0, onto=snapshot,
                                          onto_delay_steps=delay)
    factor = math.exp(-delay / tau)

    assert float(np.sum(np.abs(dw_late))) == pytest.approx(
        factor * float(np.sum(np.abs(dw_now))), rel=1e-12)
    assert np.allclose(dw_late / factor, dw_now, rtol=1e-12, atol=0.0)
    bez_vozrasta = late.learner.apply_modulator(2.0, onto=snapshot,
                                                onto_delay_steps=0)
    assert np.array_equal(bez_vozrasta, dw_now), (
        "onto без возраста — это ровно тот же кредит, что и в момент действия")
    assert np.allclose(dw_late / factor, dw_now, rtol=1e-12, atol=0.0)


def test_zhivoy_sled_posle_zaderzhki_beret_chuzhee():
    outs = {}
    for arm in DELAY_ARMS:
        agent = make_agent(arm, window_steps=10, credit_delay_steps=6)
        agent.learner.eligibility = pattern(agent)
        agent.schedule_credit(2.0)
        outs[arm] = np.concatenate(drive(agent, 6, contaminate=True))
    live, frozen = outs["delay_live"], outs["delay_frozen"]
    assert not np.array_equal(live, frozen)
    assert float(np.sum(np.abs(live))) > float(np.sum(np.abs(frozen))), (
        "поздние спайки обязаны добавить в живой след массу, которой нет в слепке")


# --- контракт apply_modulator(onto=...) ------------------------------------
def test_onto_bez_parametra_sohranyaet_staryy_put():
    a = make_agent("plastic")
    a.learner.eligibility = pattern(a)
    base = a.learner.apply_modulator(2.0)
    b = make_agent("plastic")
    b.learner.eligibility = pattern(b)
    assert np.array_equal(b.learner.apply_modulator(2.0, onto=None), base)


def test_onto_proveryaet_formu_i_konechnost():
    agent = make_agent("delay_frozen", credit_delay_steps=5)
    good = np.abs(agent.learner.eligibility) + 1.0
    with pytest.raises(ValueError):
        agent.learner.apply_modulator(2.0, onto=good[:-1])
    bad = good.copy()
    bad[0] = float("nan")
    with pytest.raises(ValueError):
        agent.learner.apply_modulator(2.0, onto=bad)
    with pytest.raises(ValueError):
        agent.learner.apply_modulator(2.0, onto=good, onto_delay_steps=-1)


def test_no_plasticity_prinimaet_onto_i_daet_nuli():
    # Задержка как настройка руки здесь не нужна: проверяется именно контракт
    # обучателя — заморозка весов не зависит от того, на какой след кредит.
    agent = make_agent("no_plasticity")
    elig = np.ones(agent.learner.n_plastic)
    dw = agent.learner.apply_modulator(2.0, onto=elig, onto_delay_steps=5)
    assert dw.shape == elig.shape and float(np.sum(np.abs(dw))) == 0.0
    with pytest.raises(ValueError):
        agent.learner.apply_modulator(2.0, onto=elig[:-1])
    with pytest.raises(ValueError):
        agent.learner.apply_modulator(2.0, onto=elig, onto_delay_steps=-1)


# --- кредит не теряется на границе эпизода ---------------------------------
def test_nedostavlennyy_kredit_doezzhaet_v_sleduyushchem_epizode():
    agent = make_agent("delay_live", window_steps=5, credit_delay_steps=5,
                       max_episode_steps=2)
    summary = agent.run(episodes=2)
    assert len(summary.episodes) == 2
    decisions = agent.prov_stats["decisions"]
    assert decisions > 2, "игрушечный контур обязан сделать несколько решений"
    assert agent.prov_stats["deliveries"] == decisions - 1, (
        "доставляется всё, кроме хвоста последнего решения")
    assert agent.prov_stats["pending_at_end"] == 1


def test_skalyar_i_ochered_daut_odinakovye_schetchiki_dostavok():
    # Счётчик означает «кредит применён», а не «кредит прошёл через очередь»:
    # иначе гейт механики не отличил бы нулевую задержку от потери доставок.
    agent = make_agent("delay_live", window_steps=6, credit_delay_steps=0)
    agent.run(episodes=2)
    assert agent.prov_stats["deliveries"] == agent.prov_stats["decisions"] > 0
    assert agent.prov_stats["pending_at_end"] == 0


def test_delay_ruleke_dostaetsya_skalyar(monkeypatch):
    agent = make_agent("delay_live", window_steps=6, credit_delay_steps=2)
    got: list[object] = []
    real = CorridorAgent.advance_credit

    def wrap(self):
        got.extend(item["payload"] for item in self._pending)
        return real(self)

    monkeypatch.setattr(CorridorAgent, "advance_credit", wrap)
    agent.run(episodes=2)
    assert got, "в delay-руке кредит должен ставиться в очередь"
    assert all(isinstance(p, float) for p in got), (
        "правило v1: payload — скаляр, а не вектор по рёбрам")
