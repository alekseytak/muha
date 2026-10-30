"""Юнит-тесты отбора plastic-рёбер для реального коннектома (волна 3).

Контракт: EdgeSparseSTDP определяет класс ребра по нейротрансмиттеру (NT),
а веса SNN берут знак из edge_class. В реальных FlyWire-данных эти две
колонки согласованы не везде (GABA|excitatory, glutamate|inhibitory), а
модуляторные NT (5-HT/DA/OA) вообще вне пластичности. Значит plastic-кандидат
= ребро, где NT известен И согласуется с edge_class; прочие остаются
структурными (замороженными) — отбор до EdgeSparseSTDP, чтобы его fail-fast
по unknown NT вообще не срабатывал.
"""
from __future__ import annotations

import numpy as np

from fly_connectome_agent.src.science.graph.models import ConnectomeGraph, Edge, Node
from fly_connectome_agent.src.science.learning.plasticity_sparse import (
    EdgeSparseSTDP,
    select_plastic_edges,
)


def make_graph(triples):
    """triples: [(synapse_count, nt, edge_class), ...]; рёбра 0 -> i+1."""
    n = len(triples) + 1
    nodes = tuple(Node(i, "unknown", "A", float(i), 0.0, 0.0, None) for i in range(n))
    edges = tuple(
        Edge(0, i + 1, float(c), weight=None, delay_ms=0.0, neurotransmitter=nt, class_id=cls)
        for i, (c, nt, cls) in enumerate(triples)
    )
    return ConnectomeGraph(nodes=nodes, edges=edges)


def test_vybiraet_soglasovannye_exc_i_inh():
    graph = make_graph([
        (3.0, "acetylcholine", "excitatory"),   # 0: согласовано (ACh возбужд.)
        (5.0, "gaba", "inhibitory"),            # 1: согласовано
        (2.0, "glutamate", "excitatory"),       # 2: согласовано
    ])
    idx = select_plastic_edges(graph)
    assert np.array_equal(idx, [0, 1, 2])


def test_isklyuchaet_nesovpadenie_nt_i_znaka():
    graph = make_graph([
        (3.0, "glutamate", "inhibitory"),       # 0: конфликт (NT возб., класс тормозной)
        (4.0, "gaba", "excitatory"),            # 1: конфликт
        (5.0, "gaba", "inhibitory"),            # 2: ок
    ])
    idx = select_plastic_edges(graph)
    assert np.array_equal(idx, [2])


def test_isklyuchaet_modulyatornye_i_neizvestnye_nt():
    graph = make_graph([
        (1.0, "dopamine", "excitatory"),        # вне MVP-пластичности
        (2.0, "serotonin", "excitatory"),
        (3.0, "octopamine", "excitatory"),
        (4.0, "somatostatin", "inhibitory"),    # неизвестный NT
        (5.0, None, "excitatory"),              # NT отсутствует
        (6.0, "gaba", "inhibitory"),            # единственный кандидат
    ])
    idx = select_plastic_edges(graph)
    assert np.array_equal(idx, [5])


def test_isklyuchaet_rebro_bez_edge_class():
    graph = make_graph([
        (3.0, "glutamate", None),
        (4.0, "glutamate", ""),
        (5.0, "glutamate", "excitatory"),
    ])
    idx = select_plastic_edges(graph)
    assert np.array_equal(idx, [2])


def test_normolizuet_registr_i_probely():
    graph = make_graph([
        (3.0, " GABA ", "Inhibitory"),
        (4.0, "AcetylCholine", "EXCITATORY"),
    ])
    idx = select_plastic_edges(graph)
    assert np.array_equal(idx, [0, 1])


def test_poryadok_i_type_vyhoda():
    graph = make_graph([
        (1.0, "dopamine", "excitatory"),
        (2.0, "glutamate", "excitatory"),
        (3.0, "gaba", "inhibitory"),
    ])
    idx = select_plastic_edges(graph)
    assert idx.dtype == np.int64
    assert np.all(np.diff(idx) > 0)  # строго возрастает
    assert np.all(idx < graph.n_edges)


def test_pustoy_vybor_kogda_kandidatov_net():
    graph = make_graph([(1.0, "dopamine", "excitatory")])
    idx = select_plastic_edges(graph)
    assert idx.size == 0


def test_edge_sparse_stdp_prinimaet_selekciyu_bez_fail_fast():
    """Интеграция: то, что пропустил селектор, EdgeSparseSTDP принимает молча.

    В граме есть конфликтные и модуляторные рёбра — если бы селектор их
    пропустил, UnknownNT-конструктор бросил бы ValueError.
    """
    graph = make_graph([
        (3.0, "glutamate", "excitatory"),
        (4.0, "glutamate", "inhibitory"),   # конфликт — не plastic
        (5.0, "dopamine", "excitatory"),    # модуляторный — не plastic
        (6.0, "gaba", "inhibitory"),
    ])
    weights = np.array([0.5, -0.5, 0.3, -0.7])
    graph = ConnectomeGraph(nodes=graph.nodes, edges=graph.edges, weights=weights)
    idx = select_plastic_edges(graph)
    stdp = EdgeSparseSTDP(graph, idx)  # не должен бросить
    assert stdp.n_plastic == 2
    assert np.array_equal(stdp.plastic_indices, [0, 3])
