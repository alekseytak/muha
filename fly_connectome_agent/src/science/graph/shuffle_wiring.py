"""Null-модель «перемешанная проводка» для сравнений на реальном коннектопе.

Идея контроля (та же, что у кнопки «перемешать проводку» в объяснителях про
муху в Doom): оставить масштаб и распределение всего, что мы и так подобрали,
и уничтожить только топологию — кто конкретно на кого влияет. Если реальная
проводка и перемешанная дают статистически неотличимое поведение, никакие
претензии «дело в структуре мозга мухи» не обоснованы.

Что сохраняется строго:
- исходящая степень каждого узла (источники рёбер не трогаем);
- входящая степень каждого узла (обмен целями меняет их местами);
- мультимножество весов и synapse_count (вес остаётся при своём ребре);
- согласованность знак↔edge_class (ребро не меняет источник и класс);
- задержки и остальные свойства ребра.

Что уничтожается: конкретные пары source→target, то есть сама цепочка
возбуждение/торможение между конкретными нейронами.

Обмен — двойной (Maslov–Sneppen для ориентированных графов): берём два ребра
ОДНОГО класса и меняем их цели местами. Обмен отклоняется, если появилась петля
(ребро из узла в него же) или дубль пары. Рёбра без class_id в обмене не
участвуют: их класс неизвестен, а значит нельзя гарантировать согласованность
знака и класса.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from .models import ConnectomeGraph

__all__ = ["shuffled_wiring"]


def _class_key(edge) -> str | None:
    """Нормализованный класс ребра или None, если класс неизвестен."""
    raw = edge.class_id
    if raw is None:
        return None
    key = str(raw).strip().lower()
    return key or None


def shuffled_wiring(
    graph: ConnectomeGraph,
    *,
    seed: int,
    n_swaps: int | None = None,
) -> ConnectomeGraph:
    """Копия графа с перемешанной топологией при сохранённых степенях и весах.

    n_swaps — число ПОПЫТОК обмена (не гарантированных успешных); по умолчанию
    10 на ребро, чего хватает, чтобы перемешать связный граф целиком. Успешные
    обмены отбрасываются, если создали бы петлю или дубль, поэтому при малом
    числе рёбер данного класса граф может остаться копией оригинала — это
    легитимный ответ «перемешивать нечего», а не ошибка.
    """
    edges = list(graph.edges)
    if len(edges) < 2:
        return graph.copy_with_edges(tuple(edges), weights=graph.weights.copy())

    groups: dict[str, list[int]] = defaultdict(list)
    for i, edge in enumerate(edges):
        key = _class_key(edge)
        if key is not None:
            groups[key].append(i)
    swappable = [np.asarray(idx, dtype=np.int64) for idx in groups.values() if len(idx) >= 2]
    if not swappable:
        return graph.copy_with_edges(tuple(edges), weights=graph.weights.copy())

    pairs = {(int(e.source), int(e.target)) for e in edges}
    rng = np.random.default_rng(seed)
    attempts = 10 * len(edges) if n_swaps is None else int(n_swaps)

    for _ in range(attempts):
        group = swappable[int(rng.integers(len(swappable)))]
        a, b = group[rng.integers(len(group), size=2)]
        a, b = int(a), int(b)
        if a == b:
            continue
        ea, eb = edges[a], edges[b]
        ta, tb = int(ea.target), int(eb.target)
        sa, sb = int(ea.source), int(eb.source)
        if ta == tb:
            continue  # обмен целей одного и того же узла ничего не меняет
        if sa == tb or sb == ta:
            continue  # петля
        if (sa, tb) in pairs or (sb, ta) in pairs:
            continue  # дубль существующей пары

        edges[a] = replace(ea, target=tb)
        edges[b] = replace(eb, target=ta)
        pairs.discard((sa, ta))
        pairs.discard((sb, tb))
        pairs.add((sa, tb))
        pairs.add((sb, ta))

    return graph.copy_with_edges(tuple(edges), weights=graph.weights.copy())
