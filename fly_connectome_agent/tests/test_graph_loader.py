"""Unit tests for graph loader and validation."""
import tempfile
import csv
from pathlib import Path

import numpy as np
import pytest

from fly_connectome_agent.src.science.graph.models import (
    ConnectomeGraph,
    Node,
    Edge,
    GraphValidationError,
)
from fly_connectome_agent.src.science.graph.load_connectome import load_connectome, validate_nodes_edges
from fly_connectome_agent.src.science.graph.extract_subgraph import extract_subgraph


def make_test_nodes_csv(path: Path):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["neuron_id", "type", "region", "x", "y", "z", "neurotransmitter", "class_id"])
        writer.writerow([1, "exc", "A", 0.0, 0.0, 0.0, "glutamate", "E1"])
        writer.writerow([2, "inh", "A", 1.0, 0.0, 0.0, "GABA", "I1"])
        writer.writerow([3, "exc", "B", 0.0, 1.0, 0.0, "glutamate", "E1"])


def make_test_edges_csv(path: Path):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "target", "synapse_type", "weight", "delay_ms"])
        writer.writerow([1, 2, "glutamate", 0.5, 1.0])
        writer.writerow([2, 3, "GABA", -0.3, 2.0])
        writer.writerow([1, 3, "glutamate", 0.4, 1.0])


def test_load_connectome_basic():
    with tempfile.TemporaryDirectory() as tmp:
        nodes_path = Path(tmp) / "nodes.csv"
        edges_path = Path(tmp) / "edges.csv"
        make_test_nodes_csv(nodes_path)
        make_test_edges_csv(edges_path)

        graph = load_connectome(nodes_path, edges_path)
        assert isinstance(graph, ConnectomeGraph)
        assert graph.n_nodes == 3
        assert graph.n_edges == 3
        assert graph.weight_matrix.shape == (3, 3)


def test_validate_nodes_edges_valid():
    nodes = [
        Node(1, "exc", "A", 0, 0, 0, "glutamate", "E1"),
        Node(2, "inh", "A", 1, 0, 0, "GABA", "I1"),
    ]
    edges = [
        Edge(1, 2, 1.0, 0.5, 1.0, "glutamate", "E1"),
    ]
    report = validate_nodes_edges(nodes, edges)
    assert report.valid
    assert report.node_count == 2
    assert report.edge_count == 1


def test_validate_nodes_edges_self_loop():
    nodes = [Node(1, "exc", "A", 0, 0, 0)]
    edges = [Edge(1, 1, 1.0, 0.5, 1.0)]
    report = validate_nodes_edges(nodes, edges)
    assert report.self_loops == 1


def test_validate_nodes_edges_missing_node():
    nodes = [Node(1, "exc", "A", 0, 0, 0)]
    edges = [Edge(1, 2, 1.0, 0.5, 1.0)]
    report = validate_nodes_edges(nodes, edges)
    assert report.missing_nodes >= 1


def test_load_connectome_with_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        nodes_path = Path(tmp) / "nodes.csv"
        edges_path = Path(tmp) / "edges.csv"
        manifest_path = Path(tmp) / "manifest.json"
        make_test_nodes_csv(nodes_path)
        make_test_edges_csv(edges_path)

        manifest = {
            "node_count": 3,
            "edge_count": 3,
            "edge_semantics": "pre_to_post",
        }
        import json
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh)

        graph = load_connectome(nodes_path, edges_path, manifest_path=manifest_path)
        assert graph.manifest is not None
        assert graph.manifest["node_count"] == 3


def test_extract_subgraph_by_node_ids():
    with tempfile.TemporaryDirectory() as tmp:
        nodes_path = Path(tmp) / "nodes.csv"
        edges_path = Path(tmp) / "edges.csv"
        make_test_nodes_csv(nodes_path)
        make_test_edges_csv(edges_path)
        graph = load_connectome(nodes_path, edges_path)

        sub = extract_subgraph(graph, node_ids=[1, 2])
        assert sub.n_nodes == 2
        assert sub.n_edges == 1  # only edge 1->2
        assert set(sub._index.keys()) == {1, 2}


def test_extract_subgraph_by_region():
    with tempfile.TemporaryDirectory() as tmp:
        nodes_path = Path(tmp) / "nodes.csv"
        edges_path = Path(tmp) / "edges.csv"
        make_test_nodes_csv(nodes_path)
        make_test_edges_csv(edges_path)
        graph = load_connectome(nodes_path, edges_path)

        sub = extract_subgraph(graph, regions=["A"])
        assert sub.n_nodes == 2
        assert set(sub._index.keys()) == {1, 2}


def test_extract_subgraph_reindex():
    with tempfile.TemporaryDirectory() as tmp:
        nodes_path = Path(tmp) / "nodes.csv"
        edges_path = Path(tmp) / "edges.csv"
        make_test_nodes_csv(nodes_path)
        make_test_edges_csv(edges_path)
        graph = load_connectome(nodes_path, edges_path)

        sub = extract_subgraph(graph, node_ids=[1, 3], reindex=True)
        assert sub.n_nodes == 2
        assert set(sub._index.keys()) == {0, 1}


def test_extract_subgraph_reindex_alignment():
    """Test that weights/edges alignment is preserved after reindex."""
    with tempfile.TemporaryDirectory() as tmp:
        nodes_path = Path(tmp) / "nodes.csv"
        edges_path = Path(tmp) / "edges.csv"
        make_test_nodes_csv(nodes_path)
        make_test_edges_csv(edges_path)
        graph = load_connectome(nodes_path, edges_path)

        # Original edges: 1->2 (w=0.5), 2->3 (w=-0.3), 1->3 (w=0.4)
        # Select nodes 1 and 3, reindex to 0,1
        sub = extract_subgraph(graph, node_ids=[1, 3], reindex=True)
        assert sub.n_nodes == 2
        assert sub.n_edges == 1  # only edge 1->3
        # The remaining edge should be 0->1 with weight 0.4
        edge_idx = sub.edge_index
        assert edge_idx.shape == (1, 2)
        assert edge_idx[0, 0] == 0  # source
        assert edge_idx[0, 1] == 1  # target
        assert np.isclose(sub.weights[0], 0.4)


def test_validation_raises_on_error():
    nodes = [Node(1, "exc", "A", 0, 0, 0)]
    edges = [Edge(1, 2, 1.0, 0.5, 1.0)]
    with pytest.raises(GraphValidationError):
        validate_nodes_edges(nodes, edges).raise_for_errors()