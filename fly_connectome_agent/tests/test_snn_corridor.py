"""Тесты контура ЭТАП 4 на игрушечном графе (реальный канал проверяется отдельно)."""
import numpy as np
import pytest

from fly_connectome_agent.src.engineering.harness.snn_corridor import (
    ARMS,
    Channel,
    CorridorAgent,
    LoopSettings,
)
from fly_connectome_agent.src.science.graph.models import ConnectomeGraph, Edge, Node

NEURONS = 8
IN_LEFT = (0, 1)
IN_RIGHT = (2, 3)
OUT_LEFT = (4, 5)
OUT_RIGHT = (6, 7)


def make_graph(weight: float = 1.0) -> ConnectomeGraph:
    nodes = tuple(Node(i, "cell", "toy", 0.0, 0.0, 0.0, "glutamate") for i in range(NEURONS))
    edges = []
    for src in (*IN_LEFT, *IN_RIGHT):
        for dst in (*OUT_LEFT, *OUT_RIGHT):
            edges.append(Edge(src, dst, 1.0, weight, 1.0, "glutamate", "excitatory"))
    weights = np.array([e.weight for e in edges], dtype=np.float64)
    return ConnectomeGraph(nodes=nodes, edges=tuple(edges), weights=weights)


def make_channel() -> Channel:
    return Channel(inputs_left=IN_LEFT, inputs_right=IN_RIGHT,
                   outputs_left=OUT_LEFT, outputs_right=OUT_RIGHT)


def make_agent(arm: str, **overrides) -> CorridorAgent:
    settings = LoopSettings(
        episodes=1, window_steps=20, i_drive=60.0, gain=6.0, sigma=4.0,
        eta=0.02, max_episode_steps=40, width=5, height=5,
        start=(1, 1), target=(3, 3), seed=11, **overrides,
    )
    return CorridorAgent(graph=make_graph(), channel=make_channel(),
                         settings=settings, arm=arm)


def test_oracle_dostigaet_celi():
    summary = make_agent("oracle").run()
    assert summary.episodes[0].reached is True
    assert summary.success_rate == 1.0
    assert summary.episodes[0].spikes == 0


def test_no_plasticity_ne_menyaet_vesa():
    agent = make_agent("no_plasticity")
    before = agent.learner.get_weights().copy()
    agent.run(episodes=3)
    assert np.array_equal(agent.learner.get_weights(), before)


def test_no_modulator_ne_menyaet_vesa():
    agent = make_agent("no_modulator")
    before = agent.learner.get_weights().copy()
    agent.run(episodes=2)
    assert np.array_equal(agent.learner.get_weights(), before)


def test_plastic_menyaet_vesa_pri_nagrade():
    agent = make_agent("plastic")
    before = agent.learner.get_weights().copy()
    summary = agent.run(episodes=2)
    after = agent.learner.get_weights()
    assert not np.array_equal(before, after)
    assert np.mean(np.abs(after - before)) > 0
    assert summary.episodes[-1].weights_delta > 0


def test_spaykovyy_byudzhet_schitaetsya_i_net_nan():
    summary = make_agent("plastic").run(episodes=3)
    assert summary.total_spikes > 0
    assert all(np.isfinite(e.reward) for e in summary.episodes)
    assert not summary.nan_seen


def test_vorota_propuskayut_i_nichego_ne_narushayut():
    summary = make_agent("plastic").run(episodes=2)
    assert summary.governance_violations == 0
    assert all(e.denied == 0 for e in summary.episodes)
    assert all(e.steps > 0 for e in summary.episodes)


def test_determinizm_po_sidu():
    a = make_agent("plastic").run(episodes=2)
    b = make_agent("plastic").run(episodes=2)
    assert [e.reward for e in a.episodes] == [e.reward for e in b.episodes]
    assert [e.spikes for e in a.episodes] == [e.spikes for e in b.episodes]


def test_vesa_obucheniya_perenosyatsya_v_simulyaciyu():
    agent = make_agent("plastic")
    agent.run(episodes=1)
    sim = agent._sim_graph()
    learned = agent.learner.get_weights()
    assert np.allclose(sim.weights[agent.plastic], learned * agent.settings.gain)
    assert not np.allclose(sim.weights[agent.plastic], agent.base.weights[agent.plastic] * agent.settings.gain)


def test_kanal_proveryaetsya():
    with pytest.raises(ValueError):
        Channel(inputs_left=(0,), inputs_right=(0,), outputs_left=(4,), outputs_right=(6,))
    with pytest.raises(ValueError):
        Channel(inputs_left=(), inputs_right=(2,), outputs_left=(4,), outputs_right=(6,))
    with pytest.raises(ValueError):
        make_agent("unknown-arm")


def test_nabor_ruk_obyavlen():
    assert ARMS == ("plastic", "pool", "no_plasticity", "no_modulator", "oracle")


def test_tochnost_glagola_izmerima_tolko_tam_gde_otvet_opredelen():
    summary = make_agent("oracle").run()
    outcome = summary.episodes[0]
    assert outcome.graded_decisions > 0
    assert outcome.graded_correct == outcome.graded_decisions
    assert summary.decision_accuracy_of(0, 1) == 1.0


def test_tochnost_glagola_u_plastichnoy_ruki_v_diapazone():
    summary = make_agent("plastic").run(episodes=2)
    accuracy = summary.decision_accuracy_of(0, 2)
    assert 0.0 <= accuracy <= 1.0
    assert sum(e.graded_decisions for e in summary.episodes) > 0


def test_tochnost_na_pustom_bloke_ne_vydaet_sebya_za_nol():
    summary = make_agent("oracle").run()
    assert np.isnan(summary.decision_accuracy_of(0, 0))
