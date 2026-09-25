"""Graph validation utilities for P1."""
from __future__ import annotations

from typing import Any

from .models import GraphValidationError, GraphValidationReport, Node, Edge


def validate_nodes_edges(
    nodes: list[Node] | tuple[Node, ...],
    edges: list[Edge] | tuple[Edge, ...],
    manifest: dict[str, Any] | None = None,
    dt_ms: float = 1.0,
) -> GraphValidationReport:
    """Validate nodes and edges against the P1 contract.

    Args:
        nodes: Sequence of Node objects
        edges: Sequence of Edge objects
        manifest: Optional connectome manifest for count/provenance checks
        dt_ms: Time step in ms for delay quantization check

    Returns:
        GraphValidationReport with counts and any errors
    """
    report = GraphValidationReport(node_count=len(nodes), edge_count=len(edges))

    node_ids = {n.node_id for n in nodes}
    if len(node_ids) != len(nodes):
        report.errors.append("duplicate node_ids found")
        report.missing_nodes = len(nodes) - len(node_ids)

    edge_pairs = {}
    for edge in edges:
        if edge.source == edge.target:
            report.self_loops += 1
        pair = (edge.source, edge.target)
        if pair in edge_pairs:
            report.duplicate_edges += 1
        edge_pairs[pair] = edge_pairs.get(pair, 0) + 1

        if edge.source not in node_ids:
            report.missing_nodes += 1
        if edge.target not in node_ids:
            report.missing_nodes += 1

        if edge.weight is not None and not (edge.weight == edge.weight and abs(edge.weight) != float("inf")):
            report.invalid_weights += 1

        if edge.delay_ms is not None:
            if edge.delay_ms < 0 or edge.delay_ms != edge.delay_ms:
                report.invalid_delays += 1
            elif dt_ms > 0 and abs(edge.delay_ms / dt_ms - round(edge.delay_ms / dt_ms)) > 1e-9:
                report.invalid_delays += 1

        if edge.neurotransmitter is None:
            report.unknown_annotations += 1

    if manifest:
        if "node_count" in manifest and manifest["node_count"] != len(nodes):
            report.errors.append(f"manifest node_count {manifest['node_count']} != actual {len(nodes)}")
        if "edge_count" in manifest and manifest["edge_count"] != len(edges):
            report.errors.append(f"manifest edge_count {manifest['edge_count']} != actual {len(edges)}")
        if manifest.get("edge_semantics") != "pre_to_post":
            report.errors.append("edge_semantics must be 'pre_to_post'")

        edge_policy = manifest.get("edge_policy", {})
        allow_self_loops = edge_policy.get("allow_self_loops", False)
        merge_parallel = edge_policy.get("merge_parallel", True)

        if report.self_loops and not allow_self_loops:
            report.errors.append("self-loops are forbidden by edge policy")
        if report.duplicate_edges and not merge_parallel:
            report.errors.append("parallel edges are forbidden by edge policy")

    # Structural fatal errors (always fatal, independent of manifest)
    if report.missing_nodes:
        report.errors.append("edges reference unknown node ids")
    if report.invalid_weights:
        report.errors.append("non-finite edge weights found")
    if report.invalid_delays:
        report.errors.append("invalid or non-quantized edge delays found")

    return report


def validate_and_raise(
    nodes: list[Node] | tuple[Node, ...],
    edges: list[Edge] | tuple[Edge, ...],
    manifest: dict[str, Any] | None = None,
    dt_ms: float = 1.0,
) -> GraphValidationReport:
    """Validate and raise GraphValidationError if invalid."""
    report = validate_nodes_edges(nodes, edges, manifest, dt_ms)
    report.raise_for_errors()
    return report