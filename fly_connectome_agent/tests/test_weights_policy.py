"""Юнит-тесты политики подписанных весов для SNN (волна 2).

Контракт: знак берётся из ДАННЫХ (edge_class: excitatory/inhibitory),
магнитуда — из synapse_count с явной инженерной saturation-ной константой.
Итоговые веса обязаны лежать в тех же диапазонах, что и sign-constraints
пластичности (STDPParams: exc [0, w_exc_max], inh [-w_inh_max, 0]).
"""
from __future__ import annotations

import numpy as np
import pytest

from fly_connectome_agent.src.science.graph.models import ConnectomeGraph, Edge, Node
from fly_connectome_agent.src.science.learning.plasticity_sparse import STDPParams
from fly_connectome_agent.src.science.snn.weights_policy import (
    graph_with_signed_weights,
    signed_structural_weights,
)


def make_graph(counts_and_classes):
    """Один источник 0 -> каждая цель i; класс ребра задаёт знак."""
    n = len(counts_and_classes) + 1
    nodes = tuple(
        Node(i, "unknown", "A", float(i), 0.0, 0.0, None) for i in range(n)
    )
    edges = tuple(
        Edge(0, i + 1, float(count), weight=None, delay_ms=0.0, class_id=cls)
        for i, (count, cls) in enumerate(counts_and_classes)
    )
    return ConnectomeGraph(nodes=nodes, edges=edges)


def test_excitatory_polozhitornye_inhibitornye_otricatelnye():
    graph = make_graph([(3.0, "excitatory"), (5.0, "inhibitory")])
    w = signed_structural_weights(graph)
    assert w[0] > 0.0
    assert w[1] < 0.0


def test_ves_v_dopustimyh_koronkah_plastichnosti():
    """Значения по умолчанию не выходят за ranges STDPParams (1.0/1.0)."""
    graph = make_graph(
        [(c, "excitatory" if i % 2 == 0 else "inhibitory") for i, c in enumerate([1, 7, 20, 2405])]
    )
    w = signed_structural_weights(graph)
    params = STDPParams()
    exc = w[::2]
    inh = w[1::2]
    assert np.all(exc >= 0.0) and np.all(exc <= params.w_exc_max)
    assert np.all(inh >= -params.w_inh_max) and np.all(inh <= 0.0)


def test_monotonnost_do_saturacii():
    graph = make_graph([(1.0, "excitatory"), (2.0, "excitatory"), (4.0, "excitatory")])
    w = signed_structural_weights(graph, w_syn_per_count=0.05, count_saturation=10.0)
    assert w[0] < w[1] < w[2]
    assert np.isclose(w, [0.05, 0.10, 0.20]).all()


def test_hub_rebro_ne_rvanet_skalyu():
    """synapse_count=2405 (реальный хабы FlyWire) -> магнитуда = saturation * w_syn."""
    graph = make_graph([(2405.0, "excitatory")])
    w = signed_structural_weights(graph, w_syn_per_count=0.05, count_saturation=20.0)
    assert w[0] == pytest.approx(1.0)


def test_neizvestnyy_klass_rebra_fail_fast():
    graph = make_graph([(2.0, "excitatory"), (3.0, "modulatory")])
    with pytest.raises(ValueError, match="modulatory|edge_class|неизвест"):
        signed_structural_weights(graph)


def test_otsutstvie_klassa_rebra_fail_fast():
    graph = make_graph([(2.0, "excitatory"), (3.0, None)])
    with pytest.raises(ValueError):
        signed_structural_weights(graph)


def test_chistaya_funkciya_registr_ne_menyaetsya():
    graph = make_graph([(3.0, "excitatory"), (9.0, "inhibitory")])
    weights_before = graph.weights.copy()
    w1 = signed_structural_weights(graph)
    w2 = signed_structural_weights(graph)
    assert np.array_equal(w1, w2)
    assert np.array_equal(graph.weights, weights_before)
    assert np.array_equal(graph.synapse_counts, [3.0, 9.0])


def test_registr_s_podpisannymi_vesami_sobiraet_validnyy_graf():
    graph = make_graph([(3.0, "excitatory"), (9.0, "inhibitory")])
    signed = graph_with_signed_weights(graph, w_syn_per_count=0.05, count_saturation=20.0)
    assert signed.n_nodes == graph.n_nodes
    assert signed.n_edges == graph.n_edges
    assert signed.weights.shape == (2,)
    assert signed.weights[0] > 0 > signed.weights[1]
    # исходный граф не тронут
    assert graph.weights[0] == pytest.approx(3.0)


def test_registr_dyshit_vesami_a_ne_synapse_counts():
    """После policy у графа weights != synapse_counts (сырые counts больше не попадают в LIF)."""
    graph = make_graph([(2405.0, "excitatory")])
    signed = graph_with_signed_weights(graph)
    assert not np.array_equal(signed.weights, signed.synapse_counts)
    assert np.max(np.abs(signed.weights)) <= 1.0
