"""Каналы ввода/вывода из аннотации FlyWire (ЭТАП 4).

Задача-полигон требует, чтобы у сети были назначены входные и выходные узлы.
Из топологии архивного подграфа их вывести нельзя: в нём 297 из 300 узлов
сидят в одной сильносвязной компоненте, минимальная входная степень 2,
минимальная выходная 1 — то есть чистых источников и чистых стоков нет.
Подставлять каналы по координате — инженерная выдумка: ревью не примет правило,
которое само сеть придумала, а потом на нём же измеряла успех.

Роль берётся из данных. ingest.build_node кладёт super_class в Node.class_id,
а super_class в релизе 783 — это и есть функциональные концы: sensory,
ascending (вход), descending, motor (выход). Замерено по полному parquet:
16 351 sensory, 2 317 ascending, 1 299 descending, 106 motor; у 106 из 106
моторных и 1288 из 1299 нисходящих узлов есть прямой сенсорный или восходящий
вход, стороны 54/52 и 646/645 — двусторонняя симметрия настоящая, не смоделированная.

Пустая роль — ошибка, а не пустой кортеж. Молчаливый пустой канал превращает
гейм «улучшился ли успех» в измеряемую ни на чём величину, поэтому цензус
доступных классов выводится прямо в текст исключения.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import ConnectomeGraph

__all__ = ["INPUT_CLASSES", "OUTPUT_CLASSES", "IOChannels", "io_channels"]

# Вход по умолчанию: узлы, приносящие в мозг информацию извне и от тела.
INPUT_CLASSES = ("sensory", "ascending")
# Выход по умолчанию: собственные выходы мозга на брюшно-нервную цепочку.
# motor (периферия, 106 узлов) передаётся явно, когда задаче нужен привод
# напрямую у мышцы, а не команда, идущая в цепочку.
OUTPUT_CLASSES = ("descending",)

_UNLABELLED = "(без аннотации)"


@dataclass(frozen=True)
class IOChannels:
    """Позиции узлов в graph.nodes, сыгравшие роль входа и выхода."""

    inputs: tuple[int, ...]
    outputs: tuple[int, ...]

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    @property
    def n_outputs(self) -> int:
        return len(self.outputs)


def _normalize(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _check_names(names: Sequence[str], *, role: str) -> tuple[str, ...]:
    cleaned = tuple(_normalize(name) for name in names)
    if not cleaned or any(name == "" for name in cleaned):
        raise ValueError(f"роль {role} задана пустым или бессмысленным списком классов: {names!r}")
    return cleaned


def _indices(graph: ConnectomeGraph, wanted: Iterable[str]) -> tuple[int, ...]:
    targets = set(wanted)
    return tuple(
        position
        for position, node in enumerate(graph.nodes)
        if _normalize(node.class_id) in targets
    )


def _census(graph: ConnectomeGraph) -> str:
    counter: Counter[str] = Counter(
        _normalize(node.class_id) or _UNLABELLED for node in graph.nodes
    )
    return ", ".join(f"{name}: {count}" for name, count in counter.most_common())


def io_channels(
    graph: ConnectomeGraph,
    *,
    inputs: Sequence[str] = INPUT_CLASSES,
    outputs: Sequence[str] = OUTPUT_CLASSES,
) -> IOChannels:
    """Индексы входных и выходных узлов по super_class из данных аннотации.

    Индексы — позиции в graph.nodes (то, чем их адресует SparseLIF), а не
    node_id: в реальных данных node_id это FlyWire root_id ~7.2e17.
    Отсортированы и детерминированы: один и тот же граф даёт один и тот же
    канал, поэтому сравнение руки plastic с рукой null честно.
    """
    in_classes = _check_names(inputs, role="вход")
    out_classes = _check_names(outputs, role="выход")

    selected_inputs = _indices(graph, in_classes)
    selected_outputs = _indices(graph, out_classes)

    missing = []
    if not selected_inputs:
        missing.append("вход: " + " | ".join(in_classes))
    if not selected_outputs:
        missing.append("выход: " + " | ".join(out_classes))
    if missing:
        raise ValueError(
            "в графе нет узлов для роли ("
            + " ; ".join(missing)
            + f"). Доступные классы: [{_census(graph)}]"
        )

    return IOChannels(inputs=selected_inputs, outputs=selected_outputs)
