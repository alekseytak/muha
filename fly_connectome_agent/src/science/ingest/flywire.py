"""Ingest FlyWire 783 (вариант A) → контракт ЭТАП 1: nodes.csv / edges.csv.

Чистые функции (маппинг, отбор подграфа, манифест) работают на dict/tuple из
stdlib и покрываются синтетическими тестами без 133MB данных и без pyarrow.
Тяжёлый ввод-вывод (parquet/tsv) — лениво, только в run_ingest.

Честные рамки (docs/scientific_scope.md, docs/FLYWIRE_DATA.md):
- connectome НЕ даёт synaptic weight/delay → weight=None, delay_ms=0.0 (модель позже);
- `region` = строка `nps` дословно, НЕ парсится (символ `_` встречается внутри
  имён нейропилов, порядок алфавитный) → для 84.6% без file4 остаётся "";
- знак ребра берём из авторской колонки `Excitatory` (per-edge), а не из
  per-neuron `top_nt` (тот лишь предсказание нейромедиатора);
- FlyWire root_id ~7.2e17 > 2**53 → на выходе reindex в 0..N-1 (float-collision
  в models._as_int иначе ломает id), оригиналы — доп-колонка и sidecar-карта.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..graph.path_selection import select_io_subgraph

__all__ = [
    "voxel_to_micron",
    "pick_coordinates",
    "build_node",
    "build_edge",
    "select_degree_induced",
    "filter_induced",
    "local_reindex",
    "build_connectome_manifest",
    "load_annotations_tsv",
    "load_regions_csv",
    "run_ingest",
    "run_role_ingest",
]

# Воксельная сетка FAFB 4x4x40 nm → µm (анизотропия: шаг z в 10 раз крупнее).
VOXEL_XY_UM = 0.004
VOXEL_Z_UM = 0.040

NODE_COLUMNS = (
    "neuron_id", "cell_type", "region", "x", "y", "z",
    "neurotransmitter", "class_id", "coord_source", "flywire_id",
)
EDGE_COLUMNS = (
    "source", "target", "synapse_count", "weight", "delay_ms",
    "synapse_type", "edge_class",
)

FILE_CORES = {
    "graph": "Connectivity_783.parquet",
    "completeness": "Completeness_783.csv",
    "annotations": "Supplemental_file1_neuron_annotations.tsv",
    "regions": "Supplemental_file4_hemilineages_clustering.csv",
}

DEFAULT_PROVENANCE = {
    "source_url": "https://github.com/philshiu/Drosophila_brain_model",
    "retrieved_at": "2026-10-27T00:17:00Z",
    "license": (
        "Graph: MIT (Shiu & Spiller 2023). Annotations: flywire_annotations v2.1.0 "
        "per-file license UNVERIFIED (не подтверждена в закреплённом дереве) — "
        "не коммерциализировать и не распространять в Git; цитировать "
        "Schlegel et al. 2024 (10.1038/s41586-024-07686-5), "
        "Dorkenwald et al. 2024 (10.1038/s41586-024-07558-y)."
    ),
}


def voxel_to_micron(vx: float, vy: float, vz: float) -> tuple[float, float, float]:
    """Конвертация воксельных координат FAFB (4x4x40 nm) в µm."""
    return (float(vx) * VOXEL_XY_UM, float(vy) * VOXEL_XY_UM, float(vz) * VOXEL_Z_UM)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _as_exact_int(value: Any) -> int | None:
    """Точный int из строки/числа. НЕ через float: FlyWire root_id ~7.2e17 > 2**53,
    float() исказил бы последние разряды, и ключи аннотаций разъехались бы с
    точными int64, что приходят из parquet. int(str) точен произвольно."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text)) if ("." in text or "e" in text.lower()) else int(text)
    except (TypeError, ValueError):
        return None


def pick_coordinates(ann: Mapping[str, Any]) -> tuple[float, float, float, str]:
    """soma (тело клетки) приоритетна; при неполной — anchor pos. Источник — 4-м.

    Не смешиваем soma и anchor безмаркерно: это разные эталоны, и distance/delay
    по ним несопоставимы. coord_source обязан дойти до потребителя.
    """
    soma = (_as_float(ann.get("soma_x")), _as_float(ann.get("soma_y")), _as_float(ann.get("soma_z")))
    if all(v is not None for v in soma):
        return (*voxel_to_micron(*soma), "soma")  # type: ignore[arg-type]
    pos = (_as_float(ann.get("pos_x")), _as_float(ann.get("pos_y")), _as_float(ann.get("pos_z")))
    if all(v is not None for v in pos):
        return (*voxel_to_micron(*pos), "anchor")  # type: ignore[arg-type]
    raise ValueError("annotation row has neither complete soma nor pos coordinates")


def build_node(root_id: int, ann: Mapping[str, Any], nps: str | None) -> dict[str, Any]:
    """Строка nodes.csv. Регион храним дословно; cell_type/NT/class — из file1."""
    x, y, z, coord_source = pick_coordinates(ann)
    cell_type = _clean(ann.get("cell_type")) or _clean(ann.get("hemibrain_type"))
    nt = _clean(ann.get("top_nt")) or None
    class_id = _clean(ann.get("super_class")) or _clean(ann.get("cell_class")) or None
    return {
        "neuron_id": int(root_id),
        "cell_type": cell_type,
        "region": _clean(nps),                 # дословно; "" когда региона нет
        "x": x, "y": y, "z": z,
        "neurotransmitter": nt,
        "class_id": class_id,
        "coord_source": coord_source,
    }


def build_edge(source: int, target: int, synapse_count: int, sign: int, pre_nt: str | None) -> dict[str, Any]:
    """Строка edges.csv. synapse_count — измерение; weight/delay — модель (пока пусто/0)."""
    return {
        "source": int(source),
        "target": int(target),
        "synapse_count": int(synapse_count),
        "weight": None,                        # connectome не даёт синаптический вес
        "delay_ms": 0.0,                       # задержка — будущая модельная надстройка
        "synapse_type": (_clean(pre_nt) or None),
        "edge_class": "excitatory" if int(sign) > 0 else "inhibitory",
    }


def select_degree_induced(edges: Iterable[Mapping[str, Any]], *, target_nodes: int) -> set[int]:
    """Top-N нейронов по суммарному числу синапсов (вход+выход). Индуцированное ядро.

    Опирается только на граф (MIT, чистый: 0 self-loops / 0 параллельных / 0
    смешанных знаков) → воспроизводимо и не зависит от разрежённых аннотаций.
    Связи решаются по возрастанию id. Таргет-региональный выбор отложен за
    neuropil-парсером (не умеем честно задать `region`).
    """
    if target_nodes <= 0:
        return set()
    degree: dict[int, int] = defaultdict(int)
    for e in edges:
        w = int(e.get("synapse_count", 0) or 0)
        degree[int(e["source"])] += w
        degree[int(e["target"])] += w
    if len(degree) <= target_nodes:
        return set(degree)
    ordered = sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))
    return {node_id for node_id, _ in ordered[:target_nodes]}


def filter_induced(edges: Iterable[Mapping[str, Any]], *, node_ids: set[int]) -> list[dict[str, Any]]:
    """Рёбра, у которых оба конца внутри множества (индуцированный подграф)."""
    keep = set(node_ids)
    return [dict(e) for e in edges if int(e["source"]) in keep and int(e["target"]) in keep]


def local_reindex(flywire_ids: Iterable[int]) -> dict[int, int]:
    """FlyWire root_id → компактный 0..N-1 (sorted asc → наименьший id = 0).

    Нужен потому, что root_id > 2**53, а models._as_int парсит через float() и
    схлопнул бы соседние id в один. Оригиналы сохраняем рядом (колонка/sidecar).
    """
    return {fid: i for i, fid in enumerate(sorted({int(x) for x in flywire_ids}))}


def build_connectome_manifest(
    *,
    dataset_id: str,
    dataset_version: str,
    node_count: int,
    edge_count: int,
    subgraph_rule: Mapping[str, Any],
    provenance: Mapping[str, Any],
    species: str = "Drosophila melanogaster",
    sex: str = "female",
    cns_scope: str = "brain_only",
    unknown_annotation_policy: str = "separate_class",
    node_classes: list[dict[str, Any]] | None = None,
    edge_classes: list[dict[str, Any]] | None = None,
    io_channel: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Connectome-манифест по schemas/connectome_manifest.schema.json.

    node_classes по умолчанию честные: уровень нейрона как excitatory/inhibitory
    в этих данных не измерен → одна категория unknown. edge_classes несут знак
    (из колонки Excitatory) и structural=True, plastic=False (пластичность —
    свойство модели, не коннектома).
    """
    if node_classes is None:
        node_classes = [{"class_id": "unclassified", "sign": "unknown", "count": int(node_count)}]
    if edge_classes is None:
        edge_classes = [
            {"class_id": "excitatory", "structural": True, "plastic": False, "sign_constraint": "positive"},
            {"class_id": "inhibitory", "structural": True, "plastic": False, "sign_constraint": "negative"},
        ]
    manifest = {
        "manifest_version": "1.0.0",
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "species": species,
        "sex": sex,
        "cns_scope": cns_scope,
        "subgraph_rule": dict(subgraph_rule),
        "node_count": int(node_count),
        "edge_count": int(edge_count),
        "edge_semantics": "pre_to_post",
        "edge_policy": {"allow_self_loops": False, "merge_parallel": True, "merge_method": "sum"},
        "unknown_annotation_policy": unknown_annotation_policy,
        "node_classes": list(node_classes),
        "edge_classes": list(edge_classes),
        "provenance": dict(provenance),
    }
    if io_channel is not None:
        # Канал кладём на верхний уровень, а не в subgraph_rule: это контракт
        # для симулятора (какие узлы стимулировать и какие читать), а не
        # описание того, как резали подграф. Пустой канал сюда не доезжает —
        # select_io_subgraph падает раньше.
        manifest["io_channel"] = dict(io_channel)
    return manifest


# --------------------------------------------------------------------------- IO

def load_annotations_tsv(path: str | Path) -> dict[int, dict[str, str]]:
    """file1: root_id → строка аннотации (139k строк, stdlib csv, табуляция)."""
    out: dict[int, dict[str, str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            root = _as_exact_int(row.get("root_id"))
            if root is None:
                continue
            out[root] = row
    return out


def load_regions_csv(path: str | Path) -> dict[int, str]:
    """file4: root_id → сырой `nps` (только FlyWire-строки; hemibrain отброшен)."""
    out: dict[int, str] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if _clean(row.get("side")) == "hemibrain":
                continue
            root = _as_exact_int(row.get("root_id"))
            if root is None:
                continue
            nps = _clean(row.get("nps"))
            if nps:
                out[root] = nps
    return out


def _read_connectome_arrays(data_dir: Path):
    """pyarrow-чтение графа в numpy-массивы (только нужные колонки)."""
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(
        str(data_dir / FILE_CORES["graph"]),
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"],
    )
    pre = table["Presynaptic_ID"].to_numpy()
    post = table["Postsynaptic_ID"].to_numpy()
    conn = table["Connectivity"].to_numpy(zero_copy_only=False)
    exc = table["Excitatory"].to_numpy(zero_copy_only=False)
    return np, pre, post, conn, exc


def _top_degree_ids(np, pre, post, conn, target_nodes: int):
    """Векторизованный top-N по суммарной степени (синапсы), tie → id возр."""
    ids = np.concatenate([pre, post])
    weights = np.concatenate([conn, conn])
    uniq, inverse = np.unique(ids, return_inverse=True)
    degree = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(degree, inverse, weights.astype(np.int64))
    order = np.lexsort((uniq, -degree))  # primary=-degree, secondary=uniq(asc)
    take = min(target_nodes, len(uniq))
    return uniq[order[:take]]


def _materialize(
    out_dir: Path,
    *,
    np,
    ann: Mapping[int, dict[str, str]],
    regions: Mapping[int, str],
    pre,
    post,
    conn,
    exc,
    chosen_ids,
    subgraph_rule: Mapping[str, Any],
    io_channel: Mapping[str, Any] | None,
    dataset_id: str,
    dataset_version: str,
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Пишет nodes/edges/manifest для уже отобранного множества узлов.

    Общий хвост двух правил отбора: degree-induced (ЭТАП 1) и role-path
    (волна 6). Канал, если задан, приходит в flywire-id и здесь же
    переиндексируется: иначе симулятор читал бы id, которых нет в nodes.csv.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen_set = {int(i) for i in chosen_ids}
    chosen = np.fromiter(sorted(chosen_set), dtype=np.int64, count=len(chosen_set))

    in_pre = np.isin(pre, chosen)
    in_post = np.isin(post, chosen)
    mask = in_pre & in_post

    # reindex 0..N-1 для устойчивости к float-collision в моделях
    id_map = local_reindex(chosen_set)

    edges_out: list[dict[str, Any]] = []
    pre_sel = pre[mask]
    post_sel = post[mask]
    conn_sel = conn[mask]
    exc_sel = exc[mask]
    for s, t, c, x in zip(pre_sel.tolist(), post_sel.tolist(), conn_sel.tolist(), exc_sel.tolist()):
        raw = build_edge(source=s, target=t, synapse_count=c, sign=x,
                         pre_nt=_clean(ann.get(int(s), {}).get("top_nt")) or None)
        raw["source"] = id_map[int(s)]
        raw["target"] = id_map[int(t)]
        edges_out.append(raw)

    nodes_out: list[dict[str, Any]] = []
    for fid in sorted(chosen_set):
        row = ann.get(fid)
        if row is None:
            continue  # без аннотации нет координат → узел не строим (честно, не выдумка)
        node = build_node(root_id=fid, ann=row, nps=regions.get(fid))
        node["neuron_id"] = id_map[fid]
        node["flywire_id"] = fid
        nodes_out.append(node)

    if io_channel is not None:
        # Канал — контракт стимула и считывания; выпавший узел означает, что
        # манифест обещает не то, что лежит в nodes.csv. Молча урезанный канал
        # сделал бы гейт «успех вырос» измерением ни о чём.
        present = {n["flywire_id"] for n in nodes_out}
        wanted = {int(i) for i in io_channel["input_ids"]} | {int(i) for i in io_channel["output_ids"]}
        absent = sorted(wanted - present)
        if absent:
            raise ValueError(
                f"узлы канала не доехали до nodes.csv ({len(absent)}): {absent[:5]} — "
                "канал нельзя записать частично"
            )
        io_channel = {
            **io_channel,
            "input_ids": [id_map[int(i)] for i in io_channel["input_ids"]],
            "output_ids": [id_map[int(i)] for i in io_channel["output_ids"]],
        }

    _write_nodes(out_dir / "nodes.csv", nodes_out)
    _write_edges(out_dir / "edges.csv", edges_out)
    (out_dir / "flywire_id_map.json").write_text(
        json.dumps({str(k): v for k, v in id_map.items()}, ensure_ascii=False), encoding="utf-8"
    )

    manifest = build_connectome_manifest(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        node_count=len(nodes_out),
        edge_count=len(edges_out),
        subgraph_rule=dict(subgraph_rule),
        io_channel=io_channel,
        provenance=dict(provenance or DEFAULT_PROVENANCE),
    )
    (out_dir / "connectome_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with_region = sum(1 for n in nodes_out if n["region"])
    return {
        "node_count": len(nodes_out),
        "edge_count": len(edges_out),
        "nodes_with_region": with_region,
        "region_coverage": round(with_region / len(nodes_out), 4) if nodes_out else 0.0,
        "out_dir": str(out_dir),
    }


def run_ingest(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    target_nodes: int = 2000,
    dataset_id: str = "flywire.fafb.v783",
    dataset_version: str = "783",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Декодирует вариант A, режет degree-induced подграф, пишет nodes/edges/manifest.

    Возвращает сводку (node_count, edge_count, регион-покрытие, пути). Большие
    файлы остаются в data/ (gitignore); производные — в out_dir (var/, gitignore).
    """
    data_dir = Path(data_dir)
    ann = load_annotations_tsv(data_dir / FILE_CORES["annotations"])
    regions = load_regions_csv(data_dir / FILE_CORES["regions"])
    np, pre, post, conn, exc = _read_connectome_arrays(data_dir)

    stats = _materialize(
        Path(out_dir),
        np=np, ann=ann, regions=regions, pre=pre, post=post, conn=conn, exc=exc,
        chosen_ids=_top_degree_ids(np, pre, post, conn, target_nodes),
        subgraph_rule={
            "rule_id": "degree-induced-subgraph",
            "description": "top-N нейронов по суммарному числу синапсов; индуцированный набор рёбер.",
            "target_node_count": int(target_nodes),
        },
        io_channel=None,
        dataset_id=dataset_id, dataset_version=dataset_version, provenance=provenance,
    )
    stats["requested_target_nodes"] = int(target_nodes)
    return stats


def run_role_ingest(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    input_classes: Sequence[str] = ("sensory", "ascending"),
    output_classes: Sequence[str] = ("descending",),
    max_hops: int = 1,
    max_inputs: int = 300,
    dataset_id: str = "flywire.fafb.v783",
    dataset_version: str = "783",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Подграф по пути вход → выход (ЭТАП 4, волна 6).

    Не degree-топ: сенсорные узлы в нём не выживают (медиана 42 синапса, лучший
    ранг 5704). Здесь семена — входы, у которых есть проводка к выходам; бюджет
    считает ВХОДЫ, а не узлы. Канал (input_ids/output_ids в переиндексации
    0..N-1) фиксируется в манифесте и замораживается до обучения: переопределить
    его после прогона — значит померить другой эксперимент.

    `max_hops=2` на этих данных насыщается (все 1299 нисходящих достижимы,
    подграф ~13k узлов) — замер в docs/FLYWIRE_DATA.md, волна 6.
    """
    data_dir = Path(data_dir)
    ann = load_annotations_tsv(data_dir / FILE_CORES["annotations"])
    regions = load_regions_csv(data_dir / FILE_CORES["regions"])
    np, pre, post, conn, exc = _read_connectome_arrays(data_dir)

    ids = np.unique(np.concatenate([pre, post]))
    classes = {
        int(node_id): (_clean(ann.get(int(node_id), {}).get("super_class")) or None)
        for node_id in ids.tolist()
    }
    channel = select_io_subgraph(
        node_ids=ids.tolist(),
        pre_ids=pre,
        post_ids=post,
        classes=classes,
        input_classes=input_classes,
        output_classes=output_classes,
        max_hops=max_hops,
        max_inputs=max_inputs,
    )

    in_roles = sorted({str(value).strip().lower() for value in input_classes if str(value).strip()})
    out_roles = sorted({str(value).strip().lower() for value in output_classes if str(value).strip()})
    stats = _materialize(
        Path(out_dir),
        np=np, ann=ann, regions=regions, pre=pre, post=post, conn=conn, exc=exc,
        chosen_ids=channel.node_ids,
        subgraph_rule={
            "rule_id": "role-path-subgraph",
            "description": (
                f"входы ролей {'/'.join(in_roles)} с путём к выходам {'/'.join(out_roles)} "
                f"за <= {int(max_hops)} шаг(ов); бюджет {int(max_inputs)} входов, узлы индуцированы"
            ),
        },
        io_channel={
            "rule_id": "role-path",
            "input_classes": in_roles,
            "output_classes": out_roles,
            "max_hops": int(max_hops),
            "max_inputs": int(max_inputs),
            "input_ids": [int(i) for i in channel.inputs],
            "output_ids": [int(i) for i in channel.outputs],
            "intermediate_count": len(channel.intermediates),
        },
        dataset_id=dataset_id, dataset_version=dataset_version, provenance=provenance,
    )
    stats.update({
        "selection": "role-path-subgraph",
        "requested_max_inputs": int(max_inputs),
        "max_hops": int(max_hops),
        "io_inputs": len(channel.inputs),
        "io_outputs": len(channel.outputs),
        "io_intermediates": len(channel.intermediates),
    })
    return stats


def _write_nodes(path: Path, nodes: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(NODE_COLUMNS))
        writer.writeheader()
        for n in nodes:
            writer.writerow({k: n.get(k) for k in NODE_COLUMNS})


def _write_edges(path: Path, edges: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(EDGE_COLUMNS))
        writer.writeheader()
        for e in edges:
            writer.writerow({k: e.get(k) for k in EDGE_COLUMNS})
