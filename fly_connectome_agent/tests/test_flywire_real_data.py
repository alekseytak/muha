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
from fly_connectome_agent.src.science.graph.shuffle_wiring import shuffled_wiring
from fly_connectome_agent.src.science.ingest.flywire import run_ingest
from fly_connectome_agent.src.science.learning.plasticity_sparse import (
    EdgeSparseSTDP,
    select_plastic_edges,
)
from fly_connectome_agent.src.science.snn.lif_sparse import LIFParams, SparseLIF, simulate_lif
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


@requires_env
def test_bazlayn_syrie_counts_vyvodit_seizim_policy_derzhit_polosu(tmp_path):
    """Волна 2, сравнительный базлайн: тот же граф, те же seeds, два arms.

    Сырые synapse_counts (наивный фолбэк ConnectomeGraph без weights_policy)
    садят сеть на рефрактерный потолок: mean ~240 Hz, max ~325 Hz на всех
    пяти seeds. Полоса [1, 20] Hz у policy-весов при этом держится.
    Тест ловит ИМЕННО scale-cap политики; знак при таких магнитудах
    измеримого эффекта на safety-статистику не дал (null-результат, см.
    FLYWIRE_DATA.md) — специально не утверждаем обратного.

    Уровень: engineering_performance (парное сравнение двух модельных весов).
    """
    import numpy as np

    run_ingest(DATA_DIR, tmp_path, target_nodes=300)
    graph = load_connectome(
        tmp_path / "nodes.csv",
        tmp_path / "edges.csv",
        manifest_path=tmp_path / "connectome_manifest.json",
    )
    # наивный путь: веса = сырые counts (фолбэк models.py), знак теряется сам собой
    assert np.array_equal(graph.weights, graph.synapse_counts)
    signed = graph_with_signed_weights(graph)

    for seed in [7, 13, 21, 42, 99]:
        params = LIFParams(sigma=3.0)
        res_policy = simulate_lif(signed, n_steps=1000, params=params, seed=seed, record=True)
        res_raw = simulate_lif(graph, n_steps=200, params=params, seed=seed, record=True)

        # численная устойчивость симулятора держится в обоих arms (ловим здесь, а не в весах)
        assert np.all(np.isfinite(res_raw["V_hist"])), f"seed={seed}"
        assert np.all(np.isfinite(res_policy["V_hist"])), f"seed={seed}"

        rp = res_policy["spikes_hist"].sum(axis=0) / (1000 * 1.0 / 1000.0)
        rr = res_raw["spikes_hist"].sum(axis=0) / (200 * 1.0 / 1000.0)

        # policy — в инженерной полосе; сырые counts — припадочная сатурация
        assert 1.0 <= float(rp.mean()) <= 20.0, f"seed={seed}: policy {rp.mean()}"
        assert float(rr.mean()) > 100.0, f"seed={seed}: raw {rr.mean()}"
        # разрыв измерен: минимум десятикратный (240 vs ~3 Hz) — не пограничный случай
        assert float(rr.mean()) > 10.0 * float(rp.mean()), f"seed={seed}"


@requires_env
def test_stdp_na_realnym_podgrafe_derzhit_granicy_i_repleyayetsya(tmp_path):
    """Волна 3: reward-modulated STDP в замкнутом цикле на реальном подграфе.

    Уровень simulation_result: пластичность крутится на реальной структуре,
    знаки/границы соблюдены, replay детерминирован, после обучения сеть
    остаётся в инженерной полосе. Утверждения «обучение улучшает задачу»
    здесь НЕТ — это ЭТАП 4 с полигоном и базлайнами.
    """
    import numpy as np

    run_ingest(DATA_DIR, tmp_path, target_nodes=300)
    graph = load_connectome(
        tmp_path / "nodes.csv",
        tmp_path / "edges.csv",
        manifest_path=tmp_path / "connectome_manifest.json",
    )
    signed = graph_with_signed_weights(graph)

    plastic_idx = select_plastic_edges(signed)
    # в реальных данных есть конфликтные NT↔edge_class и модуляторные NT —
    # значит plastic-подмножество строгое, но непустое
    assert 0 < len(plastic_idx) < graph.n_edges
    # fail-fast EdgeSparseSTDP не срабатывает: селектор пропустил только согласованные
    cls_plastic = np.array([str(signed.edge_classes[i]).lower() for i in plastic_idx])
    assert np.all(np.isin(cls_plastic, ["excitatory", "inhibitory"]))

    def closed_loop(seed: int, n_steps: int) -> np.ndarray:
        stdp = EdgeSparseSTDP(signed, plastic_idx)
        net = SparseLIF(signed, LIFParams(sigma=3.0), seed=seed)
        for _ in range(n_steps):
            spikes = net.step()
            stdp.update(spikes, spikes, modulator=1.0)
        assert np.all(np.isfinite(net.V))
        return stdp.get_weights()

    w_final = closed_loop(seed=7, n_steps=500)
    w_init = signed.weights[plastic_idx]

    # конечность и границы: знак каждого plastic-ребра сохранён, |w| <= 1.0
    assert np.all(np.isfinite(w_final))
    exc = cls_plastic == "excitatory"
    inh = ~exc
    assert np.all(w_final[exc] >= 0.0) and np.all(w_final[exc] <= 1.0)
    assert np.all(w_final[inh] <= 0.0) and np.all(w_final[inh] >= -1.0)
    # обучение реально произошло: часть сдвигов не нулевая (замер: >50% рёбер)
    assert np.any(w_final != w_init)
    # детерминизм: повторный прогон с тем же seed даёт побитово те же веса
    assert np.array_equal(w_final, closed_loop(seed=7, n_steps=500))

    # послеобученная устойчивость: выученные веса обратно в граф → §6-инварианты
    full = signed.weights.copy()
    full[plastic_idx] = w_final
    learned = signed.copy_with_edges(signed.edges, weights=full)
    result = simulate_lif(learned, n_steps=500, params=LIFParams(sigma=3.0), seed=11, record=True)
    assert np.all(np.isfinite(result["V_hist"]))
    rates_hz = result["spikes_hist"].sum(axis=0) / 0.5
    assert 1.0 <= float(rates_hz.mean()) <= 20.0
    assert float(rates_hz.max()) < 100.0


def _mean_hz(result, n_steps: int) -> float:
    """Средняя частота сети в Гц (dt = 1 мс)."""
    return float(result["spikes_hist"].sum() / (n_steps / 1000.0) / result["spikes_hist"].shape[1])


@requires_env
def test_null_model_lichaet_sebya_pri_silnyh_sinapsah(tmp_path):
    """Контроль не пустой: если снять кап магнитуды, реальная проводка и её
    degree-preserving null расходятся в разы (замер: 22 Hz против 3 Hz).

    Без этого теста вывод «на штатной планке топология не видна» был бы
    неотличим от сломанного контроля.
    """
    import numpy as np

    run_ingest(DATA_DIR, tmp_path, target_nodes=300)
    graph = load_connectome(
        tmp_path / "nodes.csv",
        tmp_path / "edges.csv",
        manifest_path=tmp_path / "connectome_manifest.json",
    )
    strong = graph_with_signed_weights(
        graph, w_syn_per_count=1.0, w_exc_max=20.0, w_inh_max=20.0
    )
    assert float(np.mean(np.abs(strong.weights))) > 5.0

    means = {}
    for arm in ("real", "null"):
        values = []
        for seed in (7, 13, 42):
            g_arm = strong if arm == "real" else shuffled_wiring(strong, seed=seed)
            result = simulate_lif(
                g_arm, n_steps=1000, params=LIFParams(sigma=3.0), seed=seed, record=True
            )
            assert np.all(np.isfinite(result["V_hist"]))
            values.append(_mean_hz(result, 1000))
        means[arm] = float(np.mean(values))
    assert means["null"] > 0.0
    # запас порядокный: измерено 6.6-кратное расхождение при пороге 2-кратное
    assert means["real"] >= 2.0 * means["null"]


@requires_env
def test_na_shtatnoy_planke_topologiya_ne_vlyaet_chastotu(tmp_path):
    """Честный null: на штатном масштабе policy (кап 1.0) реальная проводка и
    перемешанная с теми же степенями и весами неразличимы.

    Смысл для предыдущих волн: инженерная полоса [1, 20] Hz из волн 2–3
    следует из масштаба весов и шума, а НЕ из структуры мозга мухи. Претензии
    «дело в топологии» на этой планке не обоснованы (замер: ≤0.6% расхождения
    на пяти повторах с разными seed).
    """
    import numpy as np

    run_ingest(DATA_DIR, tmp_path, target_nodes=300)
    graph = load_connectome(
        tmp_path / "nodes.csv",
        tmp_path / "edges.csv",
        manifest_path=tmp_path / "connectome_manifest.json",
    )
    signed = graph_with_signed_weights(graph)

    deltas = []
    for seed in (7, 13, 21, 42, 99):
        shuffled = shuffled_wiring(signed, seed=seed)
        moved = sum(1 for a, b in zip(signed.edges, shuffled.edges) if a.target != b.target)
        # перемешано реально (замер: 99.4% рёбер), а не «почти тот же граф»
        assert moved > 0.9 * signed.n_edges
        arms = {}
        for arm, g_arm in (("real", signed), ("null", shuffled)):
            result = simulate_lif(
                g_arm, n_steps=1000, params=LIFParams(sigma=3.0), seed=seed, record=True
            )
            hz = result["spikes_hist"].sum(axis=0)
            assert np.all(np.isfinite(result["V_hist"]))
            assert 1.0 <= float(hz.mean()) <= 20.0
            assert float(hz.max()) < 100.0
            arms[arm] = float(hz.mean())
        deltas.append(abs(arms["null"] - arms["real"]) / arms["real"])
    assert max(deltas) < 0.05


@requires_env
def test_stdp_na_peremeshannom_grafe_dvigaetsya_kak_na_realnom(tmp_path):
    """Парный контроль обучения: объём сдвигов пластичности на штатной планке
    не зависит от топологии (замер: от -2% до +5% на пяти перемешиваниях).

    Шум фиксирован одним seed на обеих ветках — сравнение попарное, а не
    «среднее против среднего». Знак и границы весов обязаны держаться на обеих
    ветках: null-модель сохраняет edge_class, поэтому правила пластичности те же.
    """
    import numpy as np

    run_ingest(DATA_DIR, tmp_path, target_nodes=300)
    graph = load_connectome(
        tmp_path / "nodes.csv",
        tmp_path / "edges.csv",
        manifest_path=tmp_path / "connectome_manifest.json",
    )
    signed = graph_with_signed_weights(graph)

    def sum_dw(base):
        idx = select_plastic_edges(base)
        assert 0 < len(idx) < base.n_edges
        cls = np.array([str(base.edge_classes[i]).lower() for i in idx])
        exc = cls == "excitatory"
        w_init = base.weights[idx].copy()
        stdp = EdgeSparseSTDP(base, idx)
        net = SparseLIF(base, LIFParams(sigma=3.0), seed=7)
        for _ in range(500):
            spikes = net.step()
            stdp.update(spikes, spikes, modulator=1.0)
        w_final = stdp.get_weights()
        assert np.all(np.isfinite(net.V))
        assert np.all(w_final[exc] >= 0.0) and np.all(w_final[exc] <= 1.0)
        assert np.all(w_final[~exc] <= 0.0) and np.all(w_final[~exc] >= -1.0)
        assert np.any(w_final != w_init)
        return float(np.abs(w_final - w_init).sum())

    baseline = sum_dw(signed)
    ratios = []
    for seed in (7, 13, 21, 42, 99):
        shuffled = shuffled_wiring(signed, seed=seed)
        ratios.append(abs(sum_dw(shuffled) - baseline) / baseline)
    # измерено <=5.2% при пороге 15% — запас троекратный
    assert max(ratios) < 0.15
