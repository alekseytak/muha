"""Data contracts for the validated connectome graph used by P1."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Node:
    node_id: int
    cell_type: str
    region: str
    x: float
    y: float
    z: float
    neurotransmitter: str | None = None
    class_id: str | None = None
    reconstruction_confidence: float | None = None

    @property
    def type(self) -> str:
        return self.cell_type

    @property
    def id(self) -> int:
        return self.node_id

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Node":
        return cls(
            node_id=_as_int(_first(value, "node_id", "neuron_id", "id"), "node_id"),
            cell_type=str(_first(value, "cell_type", "type", default="")),
            region=str(_first(value, "region", default="")),
            x=float(_first(value, "x", default=0.0)),
            y=float(_first(value, "y", default=0.0)),
            z=float(_first(value, "z", default=0.0)),
            neurotransmitter=_optional_str(
                _first(value, "neurotransmitter", "neurotransmitter_type", "synapse_type")
            ),
            class_id=_optional_str(_first(value, "class_id", "neuron_class", "cell_class")),
            reconstruction_confidence=_optional_float(
                _first(value, "reconstruction_confidence", "confidence")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "cell_type": self.cell_type,
            "region": self.region,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "neurotransmitter": self.neurotransmitter,
            "class_id": self.class_id,
            "reconstruction_confidence": self.reconstruction_confidence,
        }


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    synapse_count: float
    weight: float | None = None
    delay_ms: float = 0.0
    neurotransmitter: str | None = None
    class_id: str | None = None
    reconstruction_confidence: float | None = None

    @property
    def src(self) -> int:
        return self.source

    @property
    def tgt(self) -> int:
        return self.target

    @property
    def pre(self) -> int:
        return self.source

    @property
    def post(self) -> int:
        return self.target

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Edge":
        synapse_count = _first(value, "synapse_count", "synapse_count_raw", "count", default=1.0)
        return cls(
            source=_as_int(_first(value, "source", "src", "pre", "pre_synaptic_neuron"), "source"),
            target=_as_int(_first(value, "target", "tgt", "post", "post_synaptic_neuron"), "target"),
            synapse_count=float(synapse_count),
            weight=_optional_float(_first(value, "weight", "weight_value")),
            delay_ms=float(_first(value, "delay_ms", "delay", default=0.0)),
            neurotransmitter=_optional_str(
                _first(value, "neurotransmitter", "neurotransmitter_type", "synapse_type")
            ),
            class_id=_optional_str(_first(value, "class_id", "edge_class")),
            reconstruction_confidence=_optional_float(
                _first(value, "reconstruction_confidence", "confidence")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "synapse_count": self.synapse_count,
            "weight": self.weight,
            "delay_ms": self.delay_ms,
            "neurotransmitter": self.neurotransmitter,
            "class_id": self.class_id,
            "reconstruction_confidence": self.reconstruction_confidence,
        }


@dataclass
class GraphValidationReport:
    node_count: int
    edge_count: int
    self_loops: int = 0
    duplicate_edges: int = 0
    missing_nodes: int = 0
    invalid_weights: int = 0
    invalid_delays: int = 0
    unknown_annotations: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise GraphValidationError("; ".join(self.errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "self_loops": self.self_loops,
            "duplicate_edges": self.duplicate_edges,
            "missing_nodes": self.missing_nodes,
            "invalid_weights": self.invalid_weights,
            "invalid_delays": self.invalid_delays,
            "unknown_annotations": self.unknown_annotations,
            "valid": self.valid,
            "errors": list(self.errors),
        }


class GraphValidationError(ValueError):
    """Raised when a connectome CSV/manifest violates the P1 contract."""


@dataclass
class ConnectomeGraph:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    weights: np.ndarray | Sequence[float] | None = None
    manifest: dict[str, Any] | None = None
    source_manifest: dict[str, Any] | None = None
    source_paths: tuple[str, str] | None = None
    _index: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.nodes = tuple(self.nodes)
        self.edges = tuple(self.edges)
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique")
        self._index = {node_id: i for i, node_id in enumerate(ids)}
        missing = [edge for edge in self.edges if edge.source not in self._index or edge.target not in self._index]
        if missing:
            first = missing[0]
            raise ValueError(f"edge references unknown node: {first.source}->{first.target}")

        self.node_ids = np.asarray(ids, dtype=np.int64)
        self.edge_index = np.asarray(
            [[self._index[edge.source], self._index[edge.target]] for edge in self.edges],
            dtype=np.int64,
        ).reshape(-1, 2) if self.edges else np.empty((0, 2), dtype=np.int64)
        self.synapse_counts = np.asarray(
            [edge.synapse_count for edge in self.edges], dtype=np.float64
        )
        self.delays_ms = np.asarray([edge.delay_ms for edge in self.edges], dtype=np.float64)
        if self.weights is None:
            self.weights = self.synapse_counts.copy()
        self.weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if self.weights.shape != (len(self.edges),):
            raise ValueError("weights must align with edges")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights must be finite")
        if not np.all(np.isfinite(self.delays_ms)):
            raise ValueError("delays must be finite")

        self.edge_neurotransmitters = np.asarray(
            [edge.neurotransmitter for edge in self.edges], dtype=object
        )
        self.edge_classes = np.asarray([edge.class_id for edge in self.edges], dtype=object)
        self.edge_reconstruction_confidence = np.asarray(
            [edge.reconstruction_confidence for edge in self.edges], dtype=np.float64
        )

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def node_count(self) -> int:
        return self.n_nodes

    @property
    def edge_count(self) -> int:
        return self.n_edges

    @property
    def index(self) -> dict[int, int]:
        return dict(self._index)

    @property
    def regions(self) -> np.ndarray:
        return np.asarray([node.region for node in self.nodes], dtype=object)

    @property
    def cell_types(self) -> np.ndarray:
        return np.asarray([node.cell_type for node in self.nodes], dtype=object)

    @property
    def neurotransmitters(self) -> np.ndarray:
        return np.asarray([node.neurotransmitter for node in self.nodes], dtype=object)

    @property
    def node_classes(self) -> np.ndarray:
        return np.asarray([node.class_id for node in self.nodes], dtype=object)

    @property
    def edge_weights(self) -> np.ndarray:
        return self.weights

    @property
    def weight_matrix(self) -> csr_matrix:
        if self.n_edges == 0:
            return csr_matrix((self.n_nodes, self.n_nodes), dtype=np.float64)
        return csr_matrix(
            (self.weights, (self.edge_index[:, 0], self.edge_index[:, 1])),
            shape=(self.n_nodes, self.n_nodes),
        )

    @property
    def sparse_weight_matrix(self) -> csr_matrix:
        return self.weight_matrix

    @property
    def adjacency_matrix(self) -> csr_matrix:
        return self.weight_matrix

    @property
    def incoming_weight_matrix(self) -> csr_matrix:
        return self.weight_matrix.transpose().tocsr()

    @property
    def provenance(self) -> dict[str, Any] | None:
        return None if self.manifest is None else self.manifest.get("provenance")

    def subgraph(self, node_ids: Sequence[int], *, reindex: bool = False) -> "ConnectomeGraph":
        from .extract_subgraph import extract_subgraph

        return extract_subgraph(self, node_ids=node_ids, reindex=reindex)

    def copy_with_edges(self, edges: tuple[Edge, ...], weights: np.ndarray | Sequence[float] | None = None) -> "ConnectomeGraph":
        return ConnectomeGraph(
            nodes=self.nodes,
            edges=tuple(edges),
            weights=self.synapse_counts.copy() if weights is None else weights,
            manifest=self.manifest,
            source_manifest=self.source_manifest,
            source_paths=self.source_paths,
        )

    def copy_with_nodes(self, nodes: Sequence[Node]) -> "ConnectomeGraph":
        return ConnectomeGraph(
            nodes=tuple(nodes),
            edges=self.edges,
            weights=self.weights.copy(),
            manifest=self.manifest,
            source_manifest=self.source_manifest,
            source_paths=self.source_paths,
        )


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] not in (None, ""):
            return mapping[name]
    return default


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _as_int(value: Any, field_name: str) -> int:
    if value is None or value == "":
        raise ValueError(f"{field_name} is required")
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"{field_name} must be an integer")
    return int(number)