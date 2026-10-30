"""Отбор подграфа по сенсоромоторному пути (ЭТАП 4).

Зачем: правило ЭТАПа 1 резало подграф как топ-N по суммарному числу синапсов.
Измерено (docs/FLYWIRE_DATA.md, волна 5): порог топ-300 равен 12 726 синапсам,
а медиана sensory — 42 синапса, то есть самый нужный задаче класс отбирается
хуже всего. Степень узла говорит о мощности, а задаче нужен ПУТЬ: вход, у
которого есть проводка к выходу.

Здесь чистая функция без графа в памяти: на полном датасете 15 091 983 рёбер
строить объекты нельзя, поэтому вход — плоские массивы id и концов рёбер, а
память — две CSR-строки (вперёд и назад). Порядок детерминирован: ties
разрешаются возрастанием id, ничего не зависит от обхода словаря.

Роли берутся из аннотации (super_class → Node.class_id), а не из степени и не
из координат: см. io_channels.py о том, почему структурное правило не работает.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = ["IOSubgraph", "select_io_subgraph"]


@dataclass(frozen=True)
class IOSubgraph:
    """Узлы, лежащие на пути вход → выход длиной не больше max_hops.

    inputs — отобранные (учтён бюджет), outputs — те, до которых реально дошли,
    intermediates — всё остальное на этих путях. reach — число РАЗЛИЧНЫХ выходов,
    достижимых из входа: это ранжир, а не сила.
    """

    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    intermediates: tuple[int, ...]
    node_ids: tuple[int, ...]
    reach: Mapping[int, int]
    max_hops: int

    def __len__(self) -> int:
        return len(self.node_ids)


def _census(classes: Mapping[int, str | None]) -> str:
    """Перепись классов для сообщения об ошибке — без неё провал нечитаем."""
    counter = Counter(
        str(value).strip().lower() if value not in (None, "") else "(без аннотации)"
        for value in classes.values()
    )
    return " | ".join(f"{name}:{count}" for name, count in counter.most_common(12))


def _normalized(values: Sequence[str]) -> frozenset[str]:
    return frozenset(str(value).strip().lower() for value in values if str(value).strip())


def _csr(keys: np.ndarray, values: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """(indptr, indices) по отсортированным ключам строк: соседи rows[i] rows[i+1]."""
    counts = np.bincount(keys, minlength=n_rows)
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    order = np.argsort(keys, kind="stable")
    return indptr, values[order]


def _gather(indptr: np.ndarray, indices: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Соседи строк rows одной склейкой. Цикл по фронтиру, а не по рёбрам:
    на полном датасете фронтир — десятки тысяч узлов, рёбер — миллионы, и
    идиомы вроде arange-смещения тут легко дают неверный разброс (поймано
    тестом)."""
    if rows.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([indices[indptr[row]:indptr[row + 1]] for row in rows])


def _hops_to_outputs(
    reverse_indptr: np.ndarray, reverse_indices: np.ndarray, seeds: np.ndarray, max_hops: int
) -> np.ndarray:
    """Для каждого узла — минимальное число шагов до ближайшего выхода (или -1).

    Волной назад от выходов: dist[v] = h, если v за h шагов попадает в выход.
    Дальше max_hops не идём — именно это и есть смысл ограничения.
    """
    n = reverse_indptr.size - 1
    dist = np.full(n, -1, dtype=np.int8)
    front = np.unique(seeds)
    dist[front] = 0
    for hop in range(1, max_hops + 1):
        neighbors = np.unique(_gather(reverse_indptr, reverse_indices, front))
        fresh = neighbors[dist[neighbors] < 0]
        if fresh.size == 0:
            break
        dist[fresh] = hop
        front = fresh
    return dist


def _locate(ids: np.ndarray, values, *, what: str) -> np.ndarray:
    """Позиции значений в отсортированном ids. Векторно: на 15 млн рёбер
    python-цикл с dict смотрел бы 15 млн раз и жрал память.

    Ребро, ссылающееся на узел вне node_ids, — ошибка данных, а не повод
    молча сдвинуть индексы.
    """
    arr = values if isinstance(values, np.ndarray) else np.fromiter(
        (int(value) for value in values), dtype=np.int64
    )
    arr = arr.astype(np.int64, copy=False)
    pos = np.searchsorted(ids, arr)
    inside = pos < ids.size
    matched = np.zeros(arr.size, dtype=bool)
    matched[inside] = ids[pos[inside]] == arr[inside]
    if not matched.all():
        missing = np.unique(arr[~matched])[:5]
        raise ValueError(
            f"ребро ссылается на узел вне node_ids ({what}): {[int(v) for v in missing]}"
        )
    return pos.astype(np.int64)


def select_io_subgraph(
    *,
    node_ids: Sequence[int],
    pre_ids: Sequence[int],
    post_ids: Sequence[int],
    classes: Mapping[int, str | None],
    input_classes: Sequence[str],
    output_classes: Sequence[str],
    max_hops: int,
    max_inputs: int,
) -> IOSubgraph:
    """Входы с реальным путём к выходам + промежуточные узлы + сами выходы.

    Бюджет max_inputs ограничивает число входов, а не узлов: ранг — число
    различных достижимых выходов (специфичность канала), ties по возрастанию id.
    Пустая роль или нулевой бюджет — падение с переписью классов: молча
    вернуть пустой канал нельзя, на нём потом «обучение» ничего не измеряет.
    """
    if max_inputs < 1:
        raise ValueError(f"max_inputs должен быть >= 1, получено {max_inputs}")
    if max_hops < 1:
        raise ValueError(f"max_hops должен быть >= 1, получено {max_hops}")

    wanted_in = _normalized(input_classes)
    wanted_out = _normalized(output_classes)
    if not wanted_in or not wanted_out:
        raise ValueError("роли не могут быть пустыми списком классов")

    ids = np.unique(np.asarray(list(node_ids), dtype=np.int64))
    pre_idx = _locate(ids, pre_ids, what="pre_ids")
    post_idx = _locate(ids, post_ids, what="post_ids")
    n = ids.size
    out_indptr, out_indices = _csr(pre_idx, post_idx, n)
    rev_indptr, rev_indices = _csr(post_idx, pre_idx, n)

    labels = np.asarray([
        str(classes.get(int(node_id), "") or "").strip().lower() for node_id in ids
    ])
    input_mask = np.isin(labels, list(wanted_in))
    output_mask = np.isin(labels, list(wanted_out))
    if not input_mask.any():
        raise ValueError(
            f"нет узлов входной роли {sorted(wanted_in)}. Доступные классы: [{_census(classes)}]"
        )
    if not output_mask.any():
        raise ValueError(
            f"нет узлов выходной роли {sorted(wanted_out)}. Доступные классы: [{_census(classes)}]"
        )

    outputs_at = np.flatnonzero(output_mask)
    dist = _hops_to_outputs(rev_indptr, rev_indices, outputs_at, max_hops)

    candidates = np.flatnonzero(input_mask & (dist > 0))
    if candidates.size == 0:
        raise ValueError(
            f"ни один вход роли {sorted(wanted_in)} не достигает выход роли "
            f"{sorted(wanted_out)} за {max_hops} шага(ов)"
        )

    # Обход вперёд от каждого кандидата: считаем РАЗЛИЧНЫЕ выходы и узлы на путях.
    reach: dict[int, int] = {}
    visited_by_input: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for at in candidates:
        # обход идёт по индексам, а не по id: id здесь ~7.2e17 и в качестве
        # индекса улетел бы за границу indptr (поймано тестом)
        reached_outputs, on_path = _forward(int(at), out_indptr, out_indices, dist, output_mask, max_hops)
        if reached_outputs.size == 0:
            continue
        reach[int(ids[at])] = int(reached_outputs.size)
        visited_by_input[int(ids[at])] = (reached_outputs, on_path)

    if not reach:
        raise ValueError(
            f"ни один вход роли {sorted(wanted_in)} не достигает выход роли "
            f"{sorted(wanted_out)} за {max_hops} шага(ов)"
        )

    ranked = sorted(reach.items(), key=lambda kv: (-kv[1], kv[0]))[:max_inputs]
    chosen_ids = tuple(node_id for node_id, _ in ranked)
    outputs_found: set[int] = set()
    path_nodes: set[int] = set()
    for node_id in chosen_ids:
        reached_outputs, on_path = visited_by_input[node_id]
        outputs_found.update(int(ids[i]) for i in reached_outputs)
        path_nodes.update(int(ids[i]) for i in on_path)
        path_nodes.add(node_id)

    inputs_kept = set(chosen_ids)
    intermediates = tuple(sorted(path_nodes - inputs_kept - outputs_found))
    all_nodes = tuple(sorted(inputs_kept | outputs_found | path_nodes))

    return IOSubgraph(
        inputs=chosen_ids,
        outputs=tuple(sorted(outputs_found)),
        intermediates=intermediates,
        node_ids=all_nodes,
        reach={node_id: reach[node_id] for node_id in chosen_ids},
        max_hops=max_hops,
    )


def _forward(
    start: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    dist: np.ndarray,
    output_mask: np.ndarray,
    max_hops: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Выходы, достижимые из start за <= max_hops, и узлы, лежащие на таких путях.

    Идём только по рёбрам, которые уменьшают dist до выхода, — иначе обход
    собирал бы блуждания мимо цели и раздувал подграф.
    """
    frontier = np.array([start], dtype=np.int64)
    seen = {start}
    outputs: set[int] = set()
    for remaining in range(max_hops, 0, -1):
        neighbors = _gather(indptr, indices, frontier)
        if neighbors.size == 0:
            break
        # dist == -1 значит «до выхода не доходит никогда», а не «уже рядом»:
        # без нижней границы тупики-соседи проходили как dist <= remaining-1 и
        # раздували подграф (на реальном датасете — 965 фантомных узлов).
        neighbor_dist = dist[neighbors]
        keep = (neighbor_dist >= 0) & (neighbor_dist <= remaining - 1)
        neighbors = np.unique(neighbors[keep])
        if neighbors.size == 0:
            break
        seen.update(int(v) for v in neighbors)
        outputs.update(int(v) for v in neighbors if output_mask[v])
        frontier = neighbors
    return np.array(sorted(outputs), dtype=np.int64), np.array(sorted(seen), dtype=np.int64)
