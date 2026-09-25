"""Subgraph extraction for P1."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .models import ConnectomeGraph, Edge, Node


def extract_subgraph(
    graph: ConnectomeGraph,
    node_ids: Sequence[int] | None = None,
    regions: Sequence[str] | None = None,
    neuron_classes: Sequence[str] | None = None,
    max_nodes: int | None = None,
    reindex: bool = False,
) -> ConnectomeGraph:
    """Extract a subgraph from a ConnectomeGraph.

    Args:
        graph: Source ConnectomeGraph
        node_ids: Specific node IDs to include
        regions: Region names to include (all nodes in these regions)
        neuron_classes: Neuron class IDs to include
        max_nodes: Maximum number of nodes (deterministic selection if exceeded)
        reindex: If True, reindex node IDs to 0..N-1

    Returns:
        New ConnectomeGraph containing only selected nodes and their edges
    """
    if not any([node_ids, regions, neuron_classes]):
        if max_nodes is None or max_nodes >= graph.n_nodes:
            return graph.copy_with_nodes(graph.nodes)
        node_ids = tuple(sorted(graph._index.keys()))[:max_nodes]

    selected = set()
    if node_ids:
        selected.update(node_ids)
    if regions:
        region_set = set(regions)
        for node in graph.nodes:
            if node.region in region_set:
                selected.add(node.node_id)
    if neuron_classes:
        class_set = set(neuron_classes)
        for node in graph.nodes:
            if node.class_id in class_set:
                selected.add(node.node_id)

    unknown = selected - set(graph._index.keys())
    if unknown:
        raise ValueError(f"unknown node_ids: {sorted(unknown)}")

    if max_nodes is not None and len(selected) > max_nodes:
        selected = set(sorted(selected)[:max_nodes])

    new_nodes = [n for n in graph.nodes if n.node_id in selected]
    new_node_ids = {n.node_id for n in new_nodes}
    new_edges = [e for e in graph.edges if e.source in new_node_ids and e.target in new_node_ids]

    if reindex:
        id_map = {old: i for i, old in enumerate(sorted(new_node_ids))}
        new_nodes = tuple(
            Node(
                node_id=id_map[n.node_id],
                cell_type=n.cell_type,
                region=n.region,
                x=n.x,
                y=n.y,
                z=n.z,
                neurotransmitter=n.neurotransmitter,
                class_id=n.class_id,
                reconstruction_confidence=n.reconstruction_confidence,
            )
            for n in new_nodes
        )
        new_edges = tuple(
            Edge(
                source=id_map[e.source],
                target=id_map[e.target],
                synapse_count=e.synapse_count,
                weight=e.weight,
                delay_ms=e.delay_ms,
                neurotransmitter=e.neurotransmitter,
                class_id=e.class_id,
                reconstruction_confidence=e.reconstruction_confidence,
            )
            for e in new_edges
        )
    else:
        new_nodes = tuple(new_nodes)
        new_edges = tuple(new_edges)

    old_weights = np.asarray(graph.weights, dtype=np.float64)
    edge_mask = np.array([e.source in new_node_ids and e.target in new_node_ids for e in graph.edges])
    new_weights = old_weights[edge_mask]

    new_manifest = dict(graph.manifest) if graph.manifest else None
    if new_manifest:
        new_manifest = dict(new_manifest)
        new_manifest["node_count"] = len(new_nodes)
        new_manifest["edge_count"] = len(new_edges)

    return ConnectomeGraph(
        nodes=new_nodes,
        edges=new_edges,
        weights=new_weights,
        manifest=new_manifest,
        source_manifest=graph.source_manifest,
        source_paths=graph.source_paths,
    )