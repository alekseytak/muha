"""Юнит-тесты синаптического выигрыша (волна 5, снятие блокера ЭТАПа 4).

Проблема: policy-кап |w| <= 1.0 выбран ради законности весов для STDP
(STDPParams.w_exc_max/w_inh_max), но при таком масштабе рекуррентный ток
составляет ~2% от шума (fan-in 20 при ~3 Гц против sigma=3 мВ за шаг). Сеть
активна, однако проводка ни на что не влияет — именно это измерено в волне 4
(реальная и перемешанная проводки совпадают с точностью до 0.6%). Поднять же
кап нельзя: он одновременно верхняя граница правила пластичности, а без капа
обучение уходит за запрещённые значения.

Решение: разделить единицы. Вес остаётся НОРМИРОВАННЫМ ([-1, 1], их меняет
STDP), а в симуляцию входит вес × gain — явный физический масштаб синаптической
проводимости. Выигрыш не меняет знак, класс, мультимножество весов и не
трогает граф обучения.
"""
from __future__ import annotations

import numpy as np
import pytest

from fly_connectome_agent.src.science.graph.models import ConnectomeGraph, Edge, Node
from fly_connectome_agent.src.science.snn.weights_policy import (
    apply_synaptic_gain,
    graph_with_signed_weights,
)


_NTX_BY_CLASS = {"excitatory": "glutamate", "inhibitory": "gaba"}


def make_graph(counts_and_classes):
    """Как в реальных данных: NT и edge_class заданы оба (пластичность смотрит
    на NT, политика знака — на class_id)."""
    n = len(counts_and_classes) + 1
    nodes = tuple(Node(i, "unknown", "A", float(i), 0.0, 0.0, None) for i in range(n))
    edges = tuple(
        Edge(0, i + 1, float(count), weight=None, delay_ms=0.0,
             neurotransmitter=_NTX_BY_CLASS[cls], class_id=cls)
        for i, (count, cls) in enumerate(counts_and_classes)
    )
    return ConnectomeGraph(nodes=nodes, edges=edges)


def base_graph():
    return graph_with_signed_weights(
        make_graph([(1, "excitatory"), (4, "inhibitory"), (20, "excitatory"), (2405, "inhibitory")])
    )


def test_gain_umnozaet_ves_lineyno():
    scaled = apply_synaptic_gain(base_graph(), gain=3.0)
    assert np.array_equal(scaled.weights, base_graph().weights * 3.0)


def test_gain_edinitsa_vernyaet_tot_zhe_ves():
    graph = base_graph()
    scaled = apply_synaptic_gain(graph, gain=1.0)
    assert np.array_equal(scaled.weights, graph.weights)


def test_znak_i_klass_ne_menyayutsya():
    graph = base_graph()
    scaled = apply_synaptic_gain(graph, gain=7.5)
    cls = np.array([str(e.class_id).lower() for e in scaled.edges])
    exc, inh = cls == "excitatory", cls == "inhibitory"
    assert np.all(scaled.weights[exc] > 0.0)
    assert np.all(scaled.weights[inh] < 0.0)
    assert np.array_equal(scaled.edge_classes, graph.edge_classes)


def test_graf_obucheniya_ne_mutiruetsya():
    graph = base_graph()
    before = graph.weights.copy()
    apply_synaptic_gain(graph, gain=9.0)
    assert np.array_equal(graph.weights, before)
    # и границы нормированного графа остаются законными для STDP
    assert np.all(np.abs(before) <= 1.0)


def test_topologiya_i_synapse_count_ostayutsya():
    graph = base_graph()
    scaled = apply_synaptic_gain(graph, gain=4.0)
    assert scaled.edges == graph.edges
    assert np.array_equal(scaled.synapse_counts, graph.synapse_counts)
    assert scaled.n_nodes == graph.n_nodes


def test_usilennyi_ves_hodit_za_granicami_plastichnosti():
    """Выигрыш > 1 выводит веса за [-1, 1] — это ожидаемо и фиксируется:
    симуляция имеет право на них, правило пластичности — нет."""
    graph = base_graph()
    scaled = apply_synaptic_gain(graph, gain=4.0)
    assert float(np.max(np.abs(scaled.weights))) > 1.0
    assert float(np.max(np.abs(graph.weights))) <= 1.0


@pytest.mark.parametrize("bad_gain", [0.0, -1.0, float("nan"), float("inf")])
def test_nekorektnyi_gain_padaet_fail_fast(bad_gain):
    with pytest.raises(ValueError):
        apply_synaptic_gain(base_graph(), gain=bad_gain)


def test_obuchenie_na_normirovannom_simulyacia_na_usilennom():
    """Тот же граф, две роли: learner держит нормированный, сеть читает усиленный."""
    from fly_connectome_agent.src.science.learning.plasticity_sparse import (
        EdgeSparseSTDP,
        STDPParams,
    )
    from fly_connectome_agent.src.science.snn.lif_sparse import LIFParams, SparseLIF

    graph = base_graph()
    scaled = apply_synaptic_gain(graph, gain=8.0)
    stdp = EdgeSparseSTDP(graph, np.array([0, 1, 2, 3]))
    net = SparseLIF(scaled, LIFParams(sigma=3.0), seed=1)
    params = STDPParams()
    for _ in range(30):
        spikes = net.step()
        stdp.update(spikes, spikes, modulator=1.0)
    w = stdp.get_weights()
    assert np.all(np.isfinite(net.V))
    assert np.all(w >= -params.w_inh_max) and np.all(w <= params.w_exc_max)
