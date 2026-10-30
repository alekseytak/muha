"""Юнит-тесты отбора подграфа по сенсоромоторному пути (волна 5, разблокировка ЭТАПа 4).

Смысл правила: в подграф берутся только те входы, у которых есть путь к выходу
длиной не больше max_hops, вместе с промежуточными узлами этих путей и самими
выходами. Степенное правило ЭТАПа 1 (топ-N по числу синапсов) систематически
выбрасывало сенсорные узлы: их медиана — 42 синапса против порога 12 726, а
задаче нужны именно они. Ранг здесь — число РАЗЛИЧНЫХ выходов, достижимых за
max_hops, то есть специфичность пути, а не сила узла.
"""
import dataclasses

import pytest

from fly_connectome_agent.src.science.graph.path_selection import IOSubgraph, select_io_subgraph

# Крошечная ручная графовая схема.
# 10 sensory -> 20 central -> 30 descending (путь длины 2)
# 11 sensory -> 31 descending (путь длины 1)
# 12 sensory -> 22 central (к выходу не приходит никогда)
# 13 sensory -> 23 -> 33 (два шага) -> 34 (три шага): на горизонте 2 даёт один
# выход, на горизонте 3 — два, и потому является более специфичным входом
NODE_IDS = [10, 11, 12, 13, 20, 22, 23, 30, 31, 33, 34]
PRE = [10, 20, 11, 12, 22, 13, 23, 33]
POST = [20, 30, 31, 22, 22, 23, 33, 34]
CLASSES = {
    10: "sensory", 11: "sensory", 12: "sensory", 13: "sensory",
    20: "central", 22: "central", 23: "central",
    30: "descending", 31: "descending", 33: "descending", 34: "descending",
}
OPTIONS = dict(
    node_ids=NODE_IDS, pre_ids=PRE, post_ids=POST, classes=CLASSES,
    input_classes=("sensory",), output_classes=("descending",), max_hops=2,
)


def test_vhodiat_tolko_vhody_s_putiom_k_vykhodu():
    got = select_io_subgraph(**OPTIONS, max_inputs=10)
    assert got.inputs == (10, 11, 13)
    assert got.outputs == (30, 31, 33)
    assert got.intermediates == (20, 23)


def test_polnoe_mnozhestvo_uzlov_otsortirovano():
    got = select_io_subgraph(**OPTIONS, max_inputs=10)
    assert got.node_ids == (10, 11, 13, 20, 23, 30, 31, 33)
    assert got.node_ids == tuple(sorted(got.node_ids))


def test_uzel_bez_puti_k_vykhodu_ne_popadaet():
    got = select_io_subgraph(**OPTIONS, max_inputs=100)
    assert 12 not in got.node_ids and 22 not in got.node_ids


def test_gorizont_shaga_menyaet_sostav():
    got = select_io_subgraph(**{**OPTIONS, "max_hops": 2}, max_inputs=100)
    assert 34 not in got.outputs and got.reach[13] == 1
    hop3 = select_io_subgraph(**{**OPTIONS, "max_hops": 3}, max_inputs=100)
    assert 34 in hop3.outputs and hop3.reach[13] == 2


def test_byudzhet_beret_bolee_specifichnyy_vkhod_i_deterniniran():
    # на горизонте 3 у 13 два разных выхода (33 и 34), у 10 и 11 по одному
    got = select_io_subgraph(**{**OPTIONS, "max_hops": 3}, max_inputs=1)
    assert got.inputs == (13,)
    again = select_io_subgraph(**{**OPTIONS, "max_hops": 3}, max_inputs=1)
    assert got.node_ids == again.node_ids


def test_bez_vkhodnykh_uzlov_oshibka_s_perepisyu_klassov():
    with pytest.raises(ValueError) as excinfo:
        select_io_subgraph(**{**OPTIONS, "input_classes": ("glia",)}, max_inputs=5)
    message = str(excinfo.value)
    assert "sensory" in message and "descending" in message


def test_esli_ni_odin_vkhod_ne_dostigaet_vykoda_eto_oshibka():
    dead = dict(OPTIONS, post_ids=[20, 22, 22, 22, 22, 22, 22, 22])
    with pytest.raises(ValueError, match="не достигает"):
        select_io_subgraph(**dead, max_inputs=5)


def test_nulevoy_byudzhet_otkazyvaetsya():
    with pytest.raises(ValueError, match="max_inputs"):
        select_io_subgraph(**OPTIONS, max_inputs=0)


def test_rebro_bez_uzla_eto_oshibka_dannyh_a_ne_sdvignutyy_indeks():
    broken = dict(OPTIONS, pre_ids=[10, 20, 11, 12, 22, 13, 23, 999])
    with pytest.raises(ValueError, match="вне node_ids"):
        select_io_subgraph(**broken, max_inputs=5)


def test_tupik_ne_tyanet_uzly_bez_puti_k_vykhodu():
    # 50 sensory -> 60 выход (канал есть) и 50 -> 70 -> 80 тупик (к выходу не
    # приходит никогда). Без нижней границы dist тупики попадали в подграф:
    # на реальном датасете это дало 965 фантомных «промежуточных» при hops=1.
    got = select_io_subgraph(
        node_ids=[50, 60, 70, 80],
        pre_ids=[50, 50, 70],
        post_ids=[60, 70, 80],
        classes={50: "sensory", 60: "descending", 70: "central", 80: "central"},
        input_classes=("sensory",), output_classes=("descending",),
        max_hops=2, max_inputs=1,
    )
    assert got.inputs == (50,)
    assert got.outputs == (60,)
    assert got.intermediates == ()
    assert got.node_ids == (50, 60)


def test_result_neizmenyaemyy_snimok_s_chislom_ikhodov():
    got = select_io_subgraph(**OPTIONS, max_inputs=10)
    assert isinstance(got, IOSubgraph)
    assert got.reach == {10: 1, 11: 1, 13: 1}
    with pytest.raises(dataclasses.FrozenInstanceError):
        got.inputs = (99,)
