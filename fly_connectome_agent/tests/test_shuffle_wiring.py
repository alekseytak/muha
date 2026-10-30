"""Юнит-тесты null-модели «перемешанная проводка» (волна 4).

Зачем: контроль, который отличает «сеть так работает, потому что это реальная
проводка мухи» от «сеть так работает, потому что мы так подобрали веса».
Перемешивание уничтожает ТОПОЛОГИЮ (кто именно на кого влияет) и обязано
сохранить всё остальное, иначе сравнение не будет сравнением:

- исходящая и входящая степень каждого узла — ровно как были;
- мультимножество весов и synapse_count — тот же масштаб;
- согласованность знак↔edge_class: вес остаётся при своём ребре (источник не
  меняется), поэтому выводимый из класса знак не врёт;
- петель (ребро из узла в него же) и дублей не появляется;
- детерминизм по seed, исходный граф не мутируется.
"""
from __future__ import annotations

import numpy as np

from fly_connectome_agent.src.science.graph.models import ConnectomeGraph, Edge, Node
from fly_connectome_agent.src.science.graph.shuffle_wiring import shuffled_wiring
from fly_connectome_agent.src.science.snn.lif_sparse import LIFParams, simulate_lif
from fly_connectome_agent.src.science.snn.weights_policy import graph_with_signed_weights


def make_graph(specs):
    """specs = [(source, target, synapse_count, class_id), ...]; узлы 0..max."""
    n = max(max(s, t) for s, t, _, _ in specs) + 1
    nodes = tuple(Node(i, "unknown", "A", float(i), 0.0, 0.0, None) for i in range(n))
    edges = tuple(
        Edge(s, t, float(c), weight=None, delay_ms=0.0, class_id=cls)
        for s, t, c, cls in specs
    )
    return ConnectomeGraph(nodes=nodes, edges=edges)


def rich_graph(seed=3):
    """Случайный граф: несколько источников и целей на каждый узел."""
    rng = np.random.default_rng(seed)
    specs = []
    for s in range(12):
        for t in rng.choice(np.arange(12), size=3, replace=False):
            if int(t) == s:
                continue
            cls = "excitatory" if rng.random() < 0.5 else "inhibitory"
            specs.append((s, int(t), float(rng.integers(1, 30)), cls))
    return make_graph(specs)


def degrees(graph):
    out = np.bincount(graph.edge_index[:, 0], minlength=graph.n_nodes)
    inc = np.bincount(graph.edge_index[:, 1], minlength=graph.n_nodes)
    return out, inc


def test_stepeni_sohranyayutsya_rovno():
    g = rich_graph()
    shuffled = shuffled_wiring(g, seed=1)
    out_a, in_a = degrees(g)
    out_b, in_b = degrees(shuffled)
    assert np.array_equal(out_a, out_b)
    assert np.array_equal(in_a, in_b)


def test_multimnozhestvo_vesov_i_synapsov_tozhe_samoe():
    g = graph_with_signed_weights(rich_graph())
    shuffled = shuffled_wiring(g, seed=1)
    assert np.array_equal(np.sort(shuffled.weights), np.sort(g.weights))
    assert np.array_equal(np.sort(shuffled.synapse_counts), np.sort(g.synapse_counts))


def test_znak_ostaetsya_privyazannym_k_klassu():
    """Вес остаётся при своём ребре, поэтому класс и знак не расходятся."""
    g = graph_with_signed_weights(rich_graph())
    shuffled = shuffled_wiring(g, seed=2)
    cls = np.array([str(e.class_id).lower() for e in shuffled.edges])
    exc, inh = cls == "excitatory", cls == "inhibitory"
    assert np.all(shuffled.weights[exc] >= 0.0) and np.all(shuffled.weights[exc] <= 1.0)
    assert np.all(shuffled.weights[inh] <= 0.0) and np.all(shuffled.weights[inh] >= -1.0)


def test_net_petel_i_dublev():
    g = rich_graph()
    shuffled = shuffled_wiring(g, seed=5)
    pairs = list(zip(shuffled.edge_index[:, 0], shuffled.edge_index[:, 1]))
    assert all(s != t for s, t in pairs)
    assert len(pairs) == len(set(pairs))


def test_peremeshivka_realno_menyaet_topologiyu():
    g = rich_graph()
    shuffled = shuffled_wiring(g, seed=7)
    moved = sum(1 for a, b in zip(g.edges, shuffled.edges) if a.target != b.target)
    assert moved > 0
    assert shuffled.n_edges == g.n_edges


def test_determinizm_po_seed_i_razlichie_mezhdu_seeds():
    g = rich_graph()
    ta = [e.target for e in shuffled_wiring(g, seed=11).edges]
    tb = [e.target for e in shuffled_wiring(g, seed=11).edges]
    tc = [e.target for e in shuffled_wiring(g, seed=12).edges]
    assert ta == tb
    assert ta != tc


def test_ishodnyi_graf_ne_mutiruetsya():
    g = rich_graph()
    before = [(e.source, e.target, e.synapse_count) for e in g.edges]
    w_before = g.weights.copy()
    shuffled_wiring(g, seed=13)
    assert [(e.source, e.target, e.synapse_count) for e in g.edges] == before
    assert np.array_equal(g.weights, w_before)


def test_gde_peremenovat_nechego_graf_ostayotsya_kopiei():
    """По одному ребру на класс: законных пар для обмена нет, паники тоже нет."""
    g = make_graph([(0, 1, 5.0, "excitatory"), (2, 3, 7.0, "inhibitory")])
    shuffled = shuffled_wiring(g, seed=1)
    assert [e.target for e in shuffled.edges] == [1, 3]
    assert np.array_equal(shuffled.weights, g.weights)


def test_rebra_bez_klassa_ne_lomayut_sbornik():
    """class_id отсутствует (None): такие рёбра вне обмена, но не роняют модель."""
    g = make_graph(
        [(0, 1, 5.0, "excitatory"), (2, 3, 4.0, "inhibitory"), (4, 5, 6.0, None)]
    )
    shuffled = shuffled_wiring(g, seed=1)
    assert shuffled.n_edges == g.n_edges
    assert shuffled.edges[2].target == g.edges[2].target
    assert str(shuffled.edges[2].class_id) == str(g.edges[2].class_id)


def test_snn_na_peremeshannom_grafe_schitaetsya_bez_nan():
    g = graph_with_signed_weights(rich_graph())
    shuffled = shuffled_wiring(g, seed=17)
    real = simulate_lif(g, n_steps=200, params=LIFParams(sigma=3.0), seed=1, record=True)
    shuf = simulate_lif(shuffled, n_steps=200, params=LIFParams(sigma=3.0), seed=1, record=True)
    assert np.all(np.isfinite(shuf["V_hist"]))
    assert shuf["spikes_hist"].shape == real["spikes_hist"].shape
