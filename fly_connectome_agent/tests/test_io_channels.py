"""Красные тесты на контракты входов/выходов (ЭТАП 4, снятие блокера).

Проблема: у реального подграфа нет ни чистых источников, ни чистых стоков
(297 из 300 узлов в одной сильносвязной компоненте), поэтому роль «вход» и
«выход» нельзя вывести из степени узла. В прошлой итерации я подставлял каналы
по координате — это инженерная выдумка, а не данные. Роль берётся из аннотации
FlyWire: super_class уже лежит в Node.class_id (см. ingest.build_node), и в
полном датасете он даёт настоящие концы: 16 351 sensory, 2 317 ascending,
1 299 descending, 106 motor.

Правило должно молчать честно: если роли пустые — падать с цензусом, а не
возвращать пустой канал, на котором потом «обучение» ничего не измеряет.
"""
from __future__ import annotations

import pytest

from fly_connectome_agent.src.science.graph.io_channels import (
    INPUT_CLASSES,
    OUTPUT_CLASSES,
    io_channels,
)
from fly_connectome_agent.src.science.graph.models import ConnectomeGraph, Edge, Node


def make_graph(roles):
    """roles — список super_class по одному на узел; рёбра полное кольцо."""
    n = len(roles)
    nodes = tuple(
        Node(
            node_id=i,
            cell_type=f"type{i}",
            region="R",
            x=float(i),
            y=0.0,
            z=0.0,
            neurotransmitter="glutamate",
            class_id=role,
        )
        for i, role in enumerate(roles)
    )
    edges = tuple(
        Edge(
            source=i,
            target=(i + 1) % n,
            synapse_count=2,
            weight=None,
            delay_ms=0.0,
            neurotransmitter="glutamate",
            class_id="excitatory",
        )
        for i in range(n)
    )
    return ConnectomeGraph(nodes=nodes, edges=edges)


@pytest.fixture
def graph():
    return make_graph(
        [
            "sensory", "central", "descending", "ascending",
            "motor", "optic", "sensory", "descending",
        ]
    )


def test_rol_birutsya_iz_class_id_uzla(graph):
    channels = io_channels(graph)
    assert channels.inputs == (0, 3, 6)
    assert channels.outputs == (2, 7)


def test_indeksy_eto_pozicii_v_grafe_a_ne_id_uzlov(graph):
    # в этом графе node_id совпадают с позициями, поэтому проверяем способ
    # обхода: роль считается по порядку graph.nodes, а не по node_id
    channels = io_channels(graph)
    assert all(0 <= i < graph.n_nodes for i in channels.inputs + channels.outputs)
    assert graph.nodes[channels.inputs[0]].class_id in INPUT_CLASSES
    assert graph.nodes[channels.outputs[0]].class_id in OUTPUT_CLASSES


def test_poryadok_indeksov_stabilen(graph):
    first = io_channels(graph)
    second = io_channels(graph)
    assert first.inputs == tuple(sorted(first.inputs))
    assert second.outputs == first.outputs


def test_registr_i_probely_ne_vlyayut_na_rol():
    graph = make_graph([" Sensory ", "DESCENDING", "central"])
    channels = io_channels(graph)
    assert channels.inputs == (0,)
    assert channels.outputs == (1,)


def test_kastomnye_roli_peredayutsya_parametrom(graph):
    channels = io_channels(graph, inputs=("optic",), outputs=("motor",))
    assert channels.inputs == (5,)
    assert channels.outputs == (4,)


def test_uzel_bez_class_id_ne_popadaet_v_roli():
    roles = ["sensory", "descending", None, ""]
    nodes = tuple(
        Node(
            node_id=i, cell_type=f"t{i}", region="R", x=0.0, y=0.0, z=0.0,
            neurotransmitter="glutamate", class_id=r,
        )
        for i, r in enumerate(roles)
    )
    edges = tuple(
        Edge(i, (i + 1) % len(roles), 1, None, 0.0, "glutamate", "excitatory")
        for i in range(len(roles))
    )
    channels = io_channels(ConnectomeGraph(nodes=nodes, edges=edges))
    assert channels.inputs == (0,)
    assert channels.outputs == (1,)


def test_pustaya_rol_padaet_fail_fast_s_cenzusom(graph):
    with pytest.raises(ValueError) as err:
        io_channels(make_graph(["central", "optic", "central"]))
    text = str(err.value)
    # сообщение обязано сказать, что реально есть в графе, иначе оператор
    # будет гадать, пустая ли роль или сломанная склейка
    assert "central" in text and "optic" in text
    assert "sensory" in text  # какой класс искали и не нашли


def test_pustoy_vyhod_padaet_otdelno(graph):
    with pytest.raises(ValueError) as err:
        io_channels(make_graph(["sensory", "central"]))
    assert "descending" in str(err.value)


def test_scheta_soglasovany_s_indeksami(graph):
    channels = io_channels(graph)
    assert channels.n_inputs == len(channels.inputs) == 3
    assert channels.n_outputs == len(channels.outputs) == 2
