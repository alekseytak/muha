"""Волна 1 ingest FlyWire 783 → контракт nodes.csv/edges.csv.

Тесты синтетические: работают без 133MB данных и без pyarrow, чтобы `check.sh`
оставался зелёным на любой машине. Проверка на реальных файлах — отдельный
skipif-тест (test_flywire_real_data.py).
"""
from __future__ import annotations

import json
import pathlib

import pytest

from fly_connectome_agent.src.science.ingest.flywire import (
    build_connectome_manifest,
    build_edge,
    build_node,
    filter_induced,
    local_reindex,
    pick_coordinates,
    select_degree_induced,
    voxel_to_micron,
)

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "schemas" / "connectome_manifest.schema.json"
)


def _ann(**over):
    """Минимальная строка file1 (воксельные координаты 4x4x40nm)."""
    base = {
        "soma_x": 1000, "soma_y": 2000, "soma_z": 3000,
        "pos_x": 900, "pos_y": 950, "pos_z": 960,
        "cell_type": "MBV01", "hemibrain_type": "MBV1",
        "super_class": "intrinsic", "cell_class": "central",
        "top_nt": "acetylcholine",
    }
    base.update(over)
    return base


def test_voxel_to_micron_учитывает_анизотропию():
    x, y, z = voxel_to_micron(1000, 1000, 1000)
    assert x == pytest.approx(4.0)      # 1000 * 0.004 µm
    assert y == pytest.approx(4.0)
    assert z == pytest.approx(40.0)     # 1000 * 0.04 µm (шаг z в 10 раз крупнее)


def test_pick_coordinates_предпочитает_soma_и_помечает_источник():
    x, y, z, source = pick_coordinates(_ann())
    assert source == "soma"
    assert (x, y, z) == (pytest.approx(4.0), pytest.approx(8.0), pytest.approx(120.0))


def test_pick_coordinates_fallback_na_anchor_pri_otsutstvii_soma():
    ann = _ann(soma_z=None)  # soma неполная → берём pos (anchor)
    x, y, z, source = pick_coordinates(ann)
    assert source == "anchor"
    assert (x, y, z) == (pytest.approx(3.6), pytest.approx(3.8), pytest.approx(38.4))


def test_pick_coordinates_brosaet_esli_koordinat_net_vovse():
    ann = _ann(soma_x=None, soma_y=None, soma_z=None, pos_x=None, pos_y=None, pos_z=None)
    with pytest.raises(ValueError):
        pick_coordinates(ann)


def test_build_node_region_hranitsya_doslovno_bez_parsovki():
    node = build_node(root_id=720575940628857210, ann=_ann(), nps="AL_LH_MB_CA")
    assert node["neuron_id"] == 720575940628857210
    assert node["region"] == "AL_LH_MB_CA"      # дословно, не разбито по подчёркиваниям
    assert node["cell_type"] == "MBV01"
    assert node["neurotransmitter"] == "acetylcholine"
    assert node["coord_source"] == "soma"
    assert node["x"] == pytest.approx(4.0)


def test_build_node_pustoy_region_dlya_ostalnyh_i_nt_unknown():
    node = build_node(root_id=1, ann=_ann(top_nt=""), nps=None)
    assert node["region"] == ""                  # не выдумываем регион
    assert node["neurotransmitter"] is None       # "" → unknown, не ""
    assert node["cell_type"] == "MBV01"


def test_build_edge_sign_iz_kolki_synapse_count_izmerenie_weight_nicht():
    e = build_edge(source=10, target=20, synapse_count=7, sign=-1, pre_nt="GABA")
    assert e["source"] == 10 and e["target"] == 20
    assert e["synapse_count"] == 7
    assert e["edge_class"] == "inhibitory"       # знак из колонки Excitatory
    assert e["weight"] is None                    # connectome НЕ даёт вес (scientific_scope §2)
    assert e["delay_ms"] == 0.0                   # задержка — будущая модель, не данные
    assert e["synapse_type"] == "GABA"


def test_build_edge_tormoznyy_klass_i_neizvestnyy_nt():
    e = build_edge(source=1, target=2, synapse_count=1, sign=1, pre_nt=None)
    assert e["edge_class"] == "excitatory"
    assert e["synapse_type"] is None              # неизвестный NT остаётся unknown


def test_select_degree_induced_deterniniran_i_ogranichen():
    # total-degree (вход+выход): 9=18, 5=10, 1=4, 4=4 (tie на четвёрке).
    edges = [
        {"source": 5, "target": 9, "synapse_count": 10},
        {"source": 1, "target": 9, "synapse_count": 4},
        {"source": 4, "target": 9, "synapse_count": 4},
    ]
    chosen = select_degree_induced(edges, target_nodes=2)
    assert chosen == {9, 5}
    again = select_degree_induced(edges, target_nodes=2)
    assert again == chosen                        # воспроизводимо
    # tie-break при равном весе: меньший id раньше → 1 перед 4
    chosen3 = select_degree_induced(edges, target_nodes=3)
    assert chosen3 == {9, 5, 1}


def test_filter_induced_ostavlyaet_tolko_rebra_vnutri_mnozhestva():
    edges = [
        {"source": 1, "target": 2}, {"source": 2, "target": 3}, {"source": 1, "target": 3},
    ]
    kept = filter_induced(edges, node_ids={1, 2})
    assert kept == [{"source": 1, "target": 2}]


def test_local_reindex_dae_0_based_i_ne_dopiskaet_float_kolliziy():
    # FlyWire root_id ~7.2e17 > 2**53: float() схлопнул бы соседние id.
    big = [720575940628857211, 720575940628857210, 720575940596125868]
    m = local_reindex(big)
    assert set(m.values()) == {0, 1, 2}              # компактные 0..N-1
    assert m[720575940596125868] == 0                # sorted asc → наименьший id = 0
    assert len(set(m.values())) == len(m)             # все разные
    assert local_reindex(big) == m                    # детерминированно


def _minimal_provenance():
    return {
        "source_url": "https://github.com/philshiu/Drosophila_brain_model",
        "retrieved_at": "2026-10-27T00:17:00Z",
        "license": "Graph MIT; annotations flywire_annotations v2.1.0 license UNVERIFIED.",
    }


def test_build_connectome_manifest_valid_po_skhe():
    jsonschema = pytest.importorskip("jsonschema")
    manifest = build_connectome_manifest(
        dataset_id="flywire.fafb.v783",
        dataset_version="783",
        node_count=12,
        edge_count=34,
        subgraph_rule={
            "rule_id": "degree-induced-subgraph",
            "description": "top-N neurons by total synapse count; induced edge set.",
            "target_node_count": 2000,
        },
        provenance=_minimal_provenance(),
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)  # не должно бросить
    assert manifest["edge_semantics"] == "pre_to_post"
    assert manifest["node_count"] == 12 and manifest["edge_count"] == 34
    assert manifest["unknown_annotation_policy"] == "separate_class"
    assert manifest["sex"] == "female"
    assert "UNVERIFIED" in manifest["provenance"]["license"]


def test_manifest_bez_license_shema_otbrasyvaet():
    jsonschema = pytest.importorskip("jsonschema")
    bad = build_connectome_manifest(
        dataset_id="d", dataset_version="1", node_count=1, edge_count=0,
        subgraph_rule={"rule_id": "r", "description": "d"},
        provenance=_minimal_provenance(),
    )
    del bad["provenance"]["license"]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
