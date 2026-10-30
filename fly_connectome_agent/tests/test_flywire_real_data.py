"""Гейт ЭТАП 1 на реальных данных FlyWire 783 (вариант A).

 Скипается без pyarrow или без 133MB (они в gitignore) — поэтому `check.sh`
 остаётся зелёным на машине без данных. На машине с данными (как сейчас)
 прогоняется по-настоящему: декодирует parquet, режет degree-induced подграф,
 собирает ConnectomeGraph через load_connectome и валидирует манифест по схеме.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

from fly_connectome_agent.src.science.graph.load_connectome import load_connectome
from fly_connectome_agent.src.science.ingest.flywire import run_ingest
from fly_connectome_agent.src.science.snn.lif_sparse import LIFParams, simulate_lif
from fly_connectome_agent.src.science.snn.weights_policy import graph_with_signed_weights

AGENT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = AGENT / "data"
SCHEMA_PATH = AGENT / "schemas" / "connectome_manifest.schema.json"
GRAPH_FILE = DATA_DIR / "Connectivity_783.parquet"

requires_env = pytest.mark.skipif(
    not (importlib.util.find_spec("pyarrow") and GRAPH_FILE.exists()),
    reason="нет pyarrow или файлов FlyWire (data/ в gitignore)",
)


@requires_env
def test_real_flywire_ingest_sobiraet_validnyy_graf(tmp_path):
    stats = run_ingest(DATA_DIR, tmp_path, target_nodes=800)
    assert stats["node_count"] > 0
    assert stats["edge_count"] > 0

    graph = load_connectome(
        tmp_path / "nodes.csv",
        tmp_path / "edges.csv",
        manifest_path=tmp_path / "connectome_manifest.json",
    )
    # отчёт валидации уже прошёл (load_connectome бросил бы на ошибке)
    assert graph.n_nodes == stats["node_count"]
    assert graph.n_edges == stats["edge_count"]
    assert all(d == 0.0 for d in graph.delays_ms)  # задержки — не данные, держим нулём
    import numpy as np

    assert np.all(np.isfinite(graph.weights))


@requires_env
def test_real_reindex_ne_dopiskaet_float_kolliziy(tmp_path):
    run_ingest(DATA_DIR, tmp_path, target_nodes=400)
    id_map = json.loads((tmp_path / "flywire_id_map.json").read_text(encoding="utf-8"))
    # оригинальные FlyWire root_id различны...
    assert len(set(id_map)) == len(id_map)
    # ...а локальные id — ровно компактные 0..N-1 (без дыр от float-склейки)
    locals_ = sorted(id_map.values())
    assert locals_ == list(range(len(locals_)))


@requires_env
def test_real_manifest_valid_po_skhe_konnektoma(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    run_ingest(DATA_DIR, tmp_path, target_nodes=300)
    manifest = json.loads((tmp_path / "connectome_manifest.json").read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)
    assert manifest["sex"] == "female"
    assert manifest["cns_scope"] == "brain_only"
    assert "UNVERIFIED" in manifest["provenance"]["license"]
    # манифест честно отражает фактический размер подграфа
    assert manifest["node_count"] > 0 and manifest["edge_count"] > 0


@requires_env
def test_real_podgraf_gonit_snn_bez_nan_seyzima_i_vsyo_bez_molchania(tmp_path):
    """Волна 2: реальный FlyWire-подграф реально крутит SparseLIF (5 seeds).

    Утверждаем ТОЛЬКО safety-инварианты из validation_plan §6 и инженерную
    полосу частот [1, 20] Hz (engineering target, не биологический факт).
    Знак весов — из данных (edge_class); масштаб и шум sigma=3 — модельные
    инженерные константы (см. weights_policy). Пороги — грубые инженерные
    границы, а не подгонка под измеренное (меры 2.9-3.2 Hz на пяти seeds).
    """
    import numpy as np

    run_ingest(DATA_DIR, tmp_path, target_nodes=300)
    graph = load_connectome(
        tmp_path / "nodes.csv",
        tmp_path / "edges.csv",
        manifest_path=tmp_path / "connectome_manifest.json",
    )
    signed = graph_with_signed_weights(graph)
    # знак не потерян: оба класса дают нетто-веса (в сырых counts его нет)
    assert np.any(signed.weights > 0) and np.any(signed.weights < 0)
    assert np.max(np.abs(signed.weights)) <= 1.0  # границы пластичности STDPParams

    n_steps = 1000
    total_time_s = n_steps * 1.0 / 1000.0
    # validation_plan: для утверждений нужно >=5 seeds — держим нижнюю границу
    for seed in [7, 13, 21, 42, 99]:
        result = simulate_lif(
            signed, n_steps=n_steps, params=LIFParams(sigma=3.0), seed=seed, record=True
        )
        # 1) конечность: ни NaN, ни inf нигде в состояниях
        assert np.all(np.isfinite(result["V_hist"])), f"seed={seed}"
        assert np.all(np.isfinite(result["I_hist"])), f"seed={seed}"

        spikes = result["spikes_hist"]
        rates_hz = spikes.sum(axis=0) / total_time_s

        # 2) не «всё молчит»: есть хотя бы один спайк и популяционная активность
        assert spikes.sum() > 0, f"seed={seed}"
        assert 1.0 <= float(rates_hz.mean()) <= 20.0, f"seed={seed}"  # engineering target

        # 3) no runaway: ни один нейрон не уходит в припадочный режим
        assert float(rates_hz.max()) < 100.0, f"seed={seed}"
