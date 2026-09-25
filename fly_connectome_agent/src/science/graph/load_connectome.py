"""Validated connectome graph loader for P1."""
from __future__ import annotations

import csv
import json

import numpy as np
from pathlib import Path
from typing import Any

from .models import (
    ConnectomeGraph,
    Edge,
    GraphValidationError,
    GraphValidationReport,
    Node,
)
from .validate_nodes_edges import validate_nodes_edges


def load_connectome(
    nodes_path: str | Path,
    edges_path: str | Path,
    manifest_path: str | Path | None = None,
    manifest: dict[str, Any] | None = None,
    dt_ms: float = 1.0,
) -> ConnectomeGraph:
    """Load a connectome from CSV files and optional manifest.

    Args:
        nodes_path: Path to nodes CSV (neuron_id,type,region,x,y,z,...)
        edges_path: Path to edges CSV (source,target,synapse_type,weight,delay_ms,...)
        manifest_path: Optional path to connectome manifest JSON
        manifest: Optional manifest dict (used if manifest_path not provided)
        dt_ms: Time step in ms for delay validation

    Returns:
        ConnectomeGraph with validated nodes, edges, and sparse weights

    Raises:
        GraphValidationError: If validation fails
    """
    nodes = _load_nodes(nodes_path)
    edges = _load_edges(edges_path)

    if manifest_path is not None and manifest is None:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

    report = validate_nodes_edges(nodes, edges, manifest, dt_ms)
    report.raise_for_errors()

    # Веса обязаны дойти до графа: без этого ConnectomeGraph подставляет
    # synapse_count (у CSV его нет, поэтому 1.0), и модель считает все синапсы
    # единичными, теряя колонку weight.
    weights = np.array(
        [edge.weight if edge.weight is not None else edge.synapse_count for edge in edges],
        dtype=np.float64,
    )

    graph = ConnectomeGraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        weights=weights,
        manifest=manifest,
        source_manifest=manifest,
        source_paths=(str(nodes_path), str(edges_path)),
    )
    return graph


def _load_nodes(path: str | Path) -> list[Node]:
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [Node.from_mapping(row) for row in reader]


def _load_edges(path: str | Path) -> list[Edge]:
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [Edge.from_mapping(row) for row in reader]

