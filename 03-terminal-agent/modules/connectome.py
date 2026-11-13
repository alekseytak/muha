"""Загрузка коннектома дрозофилы (FlyWire role-subgraph).

Данные: `var/flywire-role/` — nodes.csv (1515 узлов), edges.csv (58674 связи),
connectome_manifest.json, flywire_id_map.json.

Загрузка ленивая: файлы читаются при первом обращении, а не при импорте,
поэтому отсутствие датасета не ломает агента.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .filesystem import ToolError


def _parse_csv_line(line: str) -> List[str]:
    """Разбор одной строки CSV по RFC4180-lite (кавычки, удвоенные кавычки)."""
    fields: List[str] = []
    current: List[str] = []
    in_quotes = False
    index = 0
    length = len(line)
    while index < length:
        char = line[index]
        if in_quotes:
            if char == '"':
                if index + 1 < length and line[index + 1] == '"':
                    current.append('"')
                    index += 2
                    continue
                in_quotes = False
                index += 1
                continue
            current.append(char)
            index += 1
            continue
        if char == '"':
            in_quotes = True
            index += 1
            continue
        if char == ",":
            fields.append("".join(current))
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    fields.append("".join(current))
    return fields


def _read_csv_dicts(path: Path) -> Iterator[Dict[str, str]]:
    """Итерирует CSV как словари. Чистый Python: не требует C-расширения _csv."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        header: Optional[List[str]] = None
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            values = _parse_csv_line(line)
            if header is None:
                header = values
                continue
            if len(values) < len(header):
                values = values + [""] * (len(header) - len(values))
            yield dict(zip(header, values))


@dataclass
class Neuron:
    """Узел коннектома."""

    id: int
    cell_type: str
    region: str
    neurotransmitter: str
    class_id: str
    flywire_id: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def short(self) -> str:
        region = f", регион {self.region}" if self.region else ""
        return (
            f"#{self.id} {self.cell_type or '(без типа)'} "
            f"[{self.class_id}]{region}, медиатор: {self.neurotransmitter or '?'}"
        )


@dataclass
class Connection:
    """Связь между узлами (pre -> post)."""

    target: int
    synapses: int
    edge_class: str
    synapse_type: str


@dataclass
class Connectome:
    """Доступ к графу без внешних зависимостей."""

    directory: Path
    manifest: Dict = field(default_factory=dict)
    nodes: Dict[int, Neuron] = field(default_factory=dict)
    outgoing: Dict[int, List[Connection]] = field(default_factory=dict)
    incoming: Dict[int, List[Connection]] = field(default_factory=dict)
    flywire_to_local: Dict[str, int] = field(default_factory=dict)
    loaded: bool = False
    load_error: Optional[str] = None

    # ---------- загрузка ----------

    @classmethod
    def open(cls, directory: str | Path) -> "Connectome":
        return cls(directory=Path(directory).expanduser())

    def _require(self, name: str) -> Path:
        path = self.directory / name
        if not path.exists():
            raise ToolError(f"нет файла коннектома: {path}")
        return path

    def load(self) -> "Connectome":
        """Читает датасет. Повторный вызов — no-op."""
        if self.loaded:
            return self
        try:
            self.manifest = json.loads(self._require("connectome_manifest.json").read_text("utf-8"))
            for row in _read_csv_dicts(self._require("nodes.csv")):
                neuron = Neuron(
                    id=int(row["neuron_id"]),
                    cell_type=row.get("cell_type", ""),
                    region=row.get("region", ""),
                    neurotransmitter=row.get("neurotransmitter", ""),
                    class_id=row.get("class_id", ""),
                    flywire_id=row.get("flywire_id", ""),
                    x=float(row.get("x") or 0.0),
                    y=float(row.get("y") or 0.0),
                    z=float(row.get("z") or 0.0),
                )
                self.nodes[neuron.id] = neuron
                if neuron.flywire_id:
                    self.flywire_to_local[neuron.flywire_id] = neuron.id

            for row in _read_csv_dicts(self._require("edges.csv")):
                src = int(row["source"])
                dst = int(row["target"])
                synapses = int(float(row.get("synapse_count") or 0))
                edge_class = row.get("edge_class", "")
                synapse_type = row.get("synapse_type", "")
                self.outgoing.setdefault(src, []).append(
                    Connection(dst, synapses, edge_class, synapse_type)
                )
                self.incoming.setdefault(dst, []).append(
                    Connection(src, synapses, edge_class, synapse_type)
                )

            for table in (self.outgoing, self.incoming):
                for key in table:
                    table[key].sort(key=lambda c: -c.synapses)

            self.loaded = True
            self.load_error = None
        except ToolError as exc:
            self.load_error = str(exc)
        except (OSError, ValueError, KeyError) as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
        return self

    @property
    def available(self) -> bool:
        self.load()
        return self.loaded

    # ---------- запросы ----------

    def _node(self, ident: str | int) -> Neuron:
        self.load()
        if not self.loaded:
            raise ToolError(f"коннектом недоступен: {self.load_error}")
        text = str(ident).strip().lstrip("#")
        if text in self.flywire_to_local:
            return self.nodes[self.flywire_to_local[text]]
        try:
            return self.nodes[int(text)]
        except (KeyError, ValueError):
            pass
        matches = [n for n in self.nodes.values() if n.cell_type.lower() == text.lower()]
        if matches:
            return matches[0]
        raise ToolError(f"узел не найден: {ident}")

    def stats(self) -> Dict:
        self.load()
        if not self.loaded:
            return {"available": False, "error": self.load_error, "directory": str(self.directory)}
        classes: Dict[str, int] = {}
        for neuron in self.nodes.values():
            classes[neuron.class_id] = classes.get(neuron.class_id, 0) + 1
        return {
            "available": True,
            "directory": str(self.directory),
            "dataset_id": self.manifest.get("dataset_id", "?"),
            "nodes": len(self.nodes),
            "edges": sum(len(v) for v in self.outgoing.values()),
            "classes": classes,
        }

    def neuron_report(self, ident: str | int) -> str:
        neuron = self._node(ident)
        out_edges = self.outgoing.get(neuron.id, [])
        in_edges = self.incoming.get(neuron.id, [])
        total_out = sum(c.synapses for c in out_edges)
        total_in = sum(c.synapses for c in in_edges)
        lines = [
            neuron.short(),
            f"flywire_id: {neuron.flywire_id or '?'}",
            f"координаты: ({neuron.x:.1f}, {neuron.y:.1f}, {neuron.z:.1f})",
            f"исходящих связей: {len(out_edges)} (синапсов {total_out})",
            f"входящих связей: {len(in_edges)} (синапсов {total_in})",
        ]
        if out_edges:
            lines.append("топ исходящих:")
            for conn in out_edges[:5]:
                target = self.nodes.get(conn.target)
                label = target.cell_type if target else str(conn.target)
                lines.append(
                    f"  -> {label} #{conn.target}: {conn.synapses} синапсов, {conn.edge_class}"
                )
        if in_edges:
            lines.append("топ входящих:")
            for conn in in_edges[:5]:
                source = self.nodes.get(conn.target)
                label = source.cell_type if source else str(conn.target)
                lines.append(
                    f"  <- {label} #{conn.target}: {conn.synapses} синапсов, {conn.edge_class}"
                )
        return "\n".join(lines)

    def neighbors(self, ident: str | int, direction: str = "out", limit: int = 10) -> str:
        neuron = self._node(ident)
        direction = direction if direction in ("out", "in", "both") else "out"
        lines = [f"соседи {neuron.short()} (направление: {direction})"]
        if direction in ("out", "both"):
            all_edges = self.outgoing.get(neuron.id, [])
            lines.append(f"исходящие ({len(all_edges)}):")
            rows = [
                f"  -> #{c.target} "
                f"{self.nodes[c.target].cell_type if c.target in self.nodes else '?'}: "
                f"{c.synapses} синапсов"
                for c in all_edges[:limit]
            ]
            lines.extend(rows if rows else ["  (нет)"])
        if direction in ("in", "both"):
            all_edges = self.incoming.get(neuron.id, [])
            lines.append(f"входящие ({len(all_edges)}):")
            rows = [
                f"  <- #{c.target} "
                f"{self.nodes[c.target].cell_type if c.target in self.nodes else '?'}: "
                f"{c.synapses} синапсов"
                for c in all_edges[:limit]
            ]
            lines.extend(rows if rows else ["  (нет)"])
        return "\n".join(lines)

    def find(self, text: str, limit: int = 15) -> str:
        """Поиск узлов по подстроке в cell_type/регионе/классе."""
        self.load()
        if not self.loaded:
            raise ToolError(f"коннектом недоступен: {self.load_error}")
        needle = (text or "").strip().lower()
        if not needle:
            raise ToolError("пустой запрос поиска по коннектому")
        found: List[Neuron] = [
            n for n in self.nodes.values()
            if needle in n.cell_type.lower()
            or needle in n.region.lower()
            or needle in n.class_id.lower()
        ]
        if not found:
            raise ToolError(f"в коннектоме нет узлов по запросу {text!r}")
        found.sort(key=lambda n: n.cell_type)
        lines = [f"найдено {len(found)} узлов по {text!r} (показано до {limit}):"]
        lines.extend(f"  {n.short()}" for n in found[:limit])
        return "\n".join(lines)

    def classes_report(self) -> str:
        """Распределение узлов по ролям/классам."""
        self.load()
        if not self.loaded:
            raise ToolError(f"коннектом недоступен: {self.load_error}")
        counts: Dict[str, int] = {}
        for neuron in self.nodes.values():
            counts[neuron.class_id] = counts.get(neuron.class_id, 0) + 1
        lines = [f"классы узлов ({self.manifest.get('dataset_id', '?')}):"]
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name}: {count}")
        io_channel = self.manifest.get("io_channel", {})
        if io_channel:
            lines.append(
                "канал ввода-вывода: входы "
                f"{io_channel.get('input_classes')} -> выходы {io_channel.get('output_classes')} "
                f"(<= {io_channel.get('max_hops')} шагов)"
            )
        return "\n".join(lines)
