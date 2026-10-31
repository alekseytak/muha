"""Тесты контрастной модуляции по пулам (P4.3 v2 и абляция v3).

Правило v2: проигравший пул учится в том же направлении, но медленнее
(`M_loser = (1−κ)·M`), а не с обратным знаком: пилот с `−κM` развернул чтение.
Абляция v3 делит это правило по знаку кредита: `pool_pos` ослабляет
проигравшего только на подкрепляющих шагах, `pool_neg` — только на
наказывающих. Поэтому проверяется ровно пять вещей:

- величина M на рёбрах пула-победителя и пула-проигравшего;
- знак кредита не инвертируется ни на одном ребре;
- каждая рука включает контраст только на своём знаке M;
- при kappa=0 все три руки обязаны быть бит-в-бит равны v1 (`plastic`), иначе
  сравнение правил смешивается с посторонней правкой;
- режим руки (`always`/`positive`/`negative`) объявлен в одном месте.
"""
import numpy as np
import pytest

from fly_connectome_agent.src.engineering.harness.snn_corridor import (
    ARMS,
    Channel,
    CorridorAgent,
    LoopSettings,
)
from fly_connectome_agent.src.science.graph.models import ConnectomeGraph, Edge, Node
from fly_connectome_agent.src.science.learning.plasticity_sparse import (
    EdgeSparseSTDP,
    STDPParams,
    select_plastic_edges,
)

NEURONS = 9
IN_LEFT = (0, 1)
IN_RIGHT = (2, 3)
OUT_LEFT = (4, 5)
OUT_RIGHT = (6, 7)
HIDDEN = 8


def make_graph() -> ConnectomeGraph:
    """Игрушечный канал: 16 рёбер в пулы выходов плюс 4 ребра в скрытый нейрон.

    Скрытые рёбра нужны, чтобы проверить, что контраст не переписывает
    модулятор для рёбер, не принадлежащих ни одному читаемому пулу.
    """
    nodes = tuple(Node(i, "cell", "toy", 0.0, 0.0, 0.0, "glutamate") for i in range(NEURONS))
    edges = []
    for src in (*IN_LEFT, *IN_RIGHT):
        for dst in (*OUT_LEFT, *OUT_RIGHT):
            edges.append(Edge(src, dst, 1.0, 1.0, 1.0, "glutamate", "excitatory"))
    for src in (*IN_LEFT, *IN_RIGHT):
        edges.append(Edge(src, HIDDEN, 1.0, 1.0, 1.0, "glutamate", "excitatory"))
    weights = np.array([e.weight for e in edges], dtype=np.float64)
    return ConnectomeGraph(nodes=nodes, edges=tuple(edges), weights=weights)


def make_learner(eta: float = 0.1) -> EdgeSparseSTDP:
    graph = make_graph()
    plastic = select_plastic_edges(graph)
    assert len(plastic) == 20
    return EdgeSparseSTDP(graph, plastic, params=STDPParams(eta=eta, M_max=10.0))


def make_agent(arm: str, **overrides) -> CorridorAgent:
    settings = LoopSettings(
        episodes=1, window_steps=20, i_drive=60.0, gain=6.0, sigma=4.0,
        eta=0.02, max_episode_steps=40, width=5, height=5,
        start=(1, 1), target=(3, 3), seed=11, **overrides,
    )
    channel = Channel(inputs_left=IN_LEFT, inputs_right=IN_RIGHT,
                      outputs_left=OUT_LEFT, outputs_right=OUT_RIGHT)
    return CorridorAgent(graph=make_graph(), channel=channel,
                         settings=settings, arm=arm)


def test_vektor_modulyatora_dvizhet_tolko_ukazannye_rebra():
    learner = make_learner()
    learner.eligibility = np.ones(learner.n_plastic)
    M = np.zeros(learner.n_plastic)
    M[:4] = 2.0

    dw = learner.apply_modulator(M)

    assert np.allclose(dw[:4], 0.1 * 2.0)
    assert np.array_equal(dw[4:], np.zeros(learner.n_plastic - 4))


def test_skalyar_i_ravnomernyy_vektor_sovpadayut_bityo_v_bit():
    a, b = make_learner(), make_learner()
    pattern = np.linspace(-1.5, 1.5, a.n_plastic)
    a.eligibility = pattern.copy()
    b.eligibility = pattern.copy()

    a.apply_modulator(0.7)
    b.apply_modulator(np.full(b.n_plastic, 0.7))

    assert np.array_equal(a.get_weights(), b.get_weights())


def test_vektor_modulyatora_klipaetsya_i_proveryaet_formu():
    learner = make_learner()
    learner.eligibility = np.ones(learner.n_plastic)

    clipped = learner.apply_modulator(np.full(learner.n_plastic, 99.0))
    plain = learner.apply_modulator(10.0)
    assert np.allclose(clipped, plain)

    with pytest.raises(ValueError):
        learner.apply_modulator(np.ones(learner.n_plastic - 1))
    with pytest.raises(ValueError):
        learner.apply_modulator(np.full(learner.n_plastic, np.nan))


def test_kontrast_oslablyaet_proigravshiy_pul_ne_menyaya_znak():
    agent = make_agent("pool", pool_contrast=0.5)
    post = agent.learner.edge_index[:, 1]
    left = np.isin(post, agent.channel.outputs_left)
    right = np.isin(post, agent.channel.outputs_right)
    hidden = ~(left | right)

    M = agent._edge_modulator(4.0, "move_left")

    assert M is not None
    assert np.all(M[left] == 4.0)
    assert np.all(M[right] == 2.0)
    assert np.all(M[hidden] == 4.0)

    swapped = agent._edge_modulator(4.0, "move_right")
    assert np.all(swapped[left] == 2.0)
    assert np.all(swapped[right] == 4.0)

    for value in (4.0, 0.0, -1.0, -10.0):
        for action in ("move_left", "move_right"):
            edge_m = agent._edge_modulator(value, action)
            assert np.all(np.sign(edge_m) * np.sign(value) >= 0), (value, action)


def test_ravenstvo_i_chuzhaya_ruka_ne_vklyuchayut_kontrast():
    pool = make_agent("pool", pool_contrast=0.5)
    assert pool._edge_modulator(4.0, "stay") is None

    plain = make_agent("plastic", pool_contrast=0.5)
    assert plain._edge_modulator(4.0, "move_left") is None


def test_kappa_nol_daet_te_zhe_vesa_chto_v1():
    v1 = make_agent("plastic")
    v1.run(episodes=1)

    for arm in ("pool", "pool_pos", "pool_neg"):
        agent = make_agent(arm, pool_contrast=0.0)
        agent.run(episodes=1)
        assert np.array_equal(v1.learner.get_weights(), agent.learner.get_weights()), arm


def test_polozhitelnyy_kontrast_ne_trogaet_otricatelnyy_kredit():
    agent = make_agent("pool_pos", pool_contrast=0.5)
    post = agent.learner.edge_index[:, 1]
    left = np.isin(post, agent.channel.outputs_left)
    right = np.isin(post, agent.channel.outputs_right)

    M = agent._edge_modulator(4.0, "move_left")
    assert M is not None
    assert np.all(M[left] == 4.0)
    assert np.all(M[right] == 2.0)

    assert agent._edge_modulator(-1.0, "move_left") is None
    assert agent._edge_modulator(0.0, "move_left") is None


def test_otricatelnyy_kontrast_zerkalno_polozhitelnomu():
    agent = make_agent("pool_neg", pool_contrast=0.5)
    post = agent.learner.edge_index[:, 1]
    left = np.isin(post, agent.channel.outputs_left)
    right = np.isin(post, agent.channel.outputs_right)

    assert agent._edge_modulator(4.0, "move_right") is None
    assert agent._edge_modulator(0.0, "move_right") is None

    M = agent._edge_modulator(-2.0, "move_right")
    assert M is not None
    assert np.all(M[right] == -2.0)
    assert np.all(M[left] == -1.0)
    assert np.all(np.sign(M) * np.sign(-2.0) >= 0)


def test_rezhim_kontrasta_obyavlen_dlya_kazhdoy_ruki():
    from fly_connectome_agent.src.engineering.harness.snn_corridor import POOL_ARMS

    assert POOL_ARMS == {"pool": "always", "pool_pos": "positive", "pool_neg": "negative"}
    for arm, mode in POOL_ARMS.items():
        assert make_agent(arm, pool_contrast=0.5).pool_when == mode
    assert make_agent("plastic").pool_when == "always"


def test_kazhdaya_ruka_kontrasta_est_v_arms():
    for arm in ("pool", "pool_pos", "pool_neg"):
        assert arm in ARMS


def test_kappa_ogranichena_diapazonom():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            LoopSettings(pool_contrast=bad)


def test_ruka_pool_obyavlena_i_uchyvaet_kontrast():
    assert "pool" in ARMS
    agent = make_agent("pool", pool_contrast=0.5)
    before = agent.learner.get_weights().copy()
    agent.run(episodes=1)
    assert not np.array_equal(before, agent.learner.get_weights())
