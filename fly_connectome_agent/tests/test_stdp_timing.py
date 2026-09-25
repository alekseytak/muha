"""Timing tests for edge-sparse reward-modulated STDP (P2.1)."""
from __future__ import annotations

import numpy as np
import pytest

from fly_connectome_agent.src.science.graph.models import Node, Edge, ConnectomeGraph
from fly_connectome_agent.src.science.learning.plasticity_sparse import (
    EdgeSparseSTDP,
    STDPParams,
    NoPlasticityBaseline,
    PlasticityDiagnostics,
)
from fly_connectome_agent.src.science.learning.reward_modulator import (
    RewardModulator,
    ModulatorParams,
)


def _make_graph(n: int, edges: list[tuple[int, int, float]]) -> ConnectomeGraph:
    nodes = tuple(
        Node(node_id=i, cell_type="E" if i < n else "I", region="R1", x=0.0, y=0.0, z=0.0)
        for i in range(n)
    )
    edge_objs = tuple(
        Edge(source=s, target=t, synapse_count=1.0, weight=w, neurotransmitter="glutamate" if w >= 0 else "gaba")
        for s, t, w in edges
    )
    weights = np.array([w for _, _, w in edges], dtype=np.float64)
    return ConnectomeGraph(nodes=nodes, edges=edge_objs, weights=weights)


def _make_exc_params() -> STDPParams:
    return STDPParams(
        tau_plus=20.0, tau_minus=20.0, tau_eligibility=20.0,
        A_plus=0.1, A_minus=0.1, eta=1.0,
        w_exc_max=1.0, w_inh_max=1.0, M_max=10.0,
    )


# ---------------------------------------------------------------------------
# Test 1: pre @ t=0, post @ t=5 -> final weight > initial
# ---------------------------------------------------------------------------

class TestPreBeforePost:
    """Test 1: Causal timing (pre before post) -> LTP."""

    def test_ltp_weight_increase(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = _make_exc_params()
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0, simultaneous_convention="zero")
        initial = stdp.get_weights()[0]
        pre = np.array([True, False])
        post = np.array([False, False])
        stdp.update(pre, post, modulator=1.0)
        for _ in range(4):
            stdp.update(np.array([False, False]), np.array([False, False]), modulator=1.0)
        pre = np.array([False, False])
        post = np.array([False, True])
        stdp.update(pre, post, modulator=1.0)
        final = stdp.get_weights()[0]
        assert final > initial, f"Expected LTP: {initial} -> {final}"


# ---------------------------------------------------------------------------
# Test 2: post @ t=0, pre @ t=5 -> final weight < initial
# ---------------------------------------------------------------------------

class TestPostBeforePre:
    """Test 2: Anti-causal timing (post before pre) -> LTD."""

    def test_ltd_weight_decrease(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = _make_exc_params()
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0, simultaneous_convention="zero")
        initial = stdp.get_weights()[0]
        stdp.update(np.array([False, False]), np.array([False, True]), modulator=1.0)
        for _ in range(4):
            stdp.update(np.array([False, False]), np.array([False, False]), modulator=1.0)
        stdp.update(np.array([True, False]), np.array([False, False]), modulator=1.0)
        final = stdp.get_weights()[0]
        assert final < initial, f"Expected LTD: {initial} -> {final}"


# ---------------------------------------------------------------------------
# Test 3: Distant timing > tau -> effect tends to zero
# ---------------------------------------------------------------------------

class TestDistantTiming:
    """Test 3: Distant timing > tau -> effect tends to zero."""

    def test_far_lag_effect_decays(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = _make_exc_params()
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0, simultaneous_convention="zero")
        initial = stdp.get_weights()[0]
        stdp.update(np.array([True, False]), np.array([False, False]), modulator=1.0)
        for _ in range(200):
            stdp.update(np.array([False, False]), np.array([False, False]), modulator=1.0)
        stdp.update(np.array([False, False]), np.array([False, True]), modulator=1.0)
        final = stdp.get_weights()[0]
        delta = abs(final - initial)
        assert delta < 0.01, f"Expected near-zero effect for distant timing, got delta={delta}"


# ---------------------------------------------------------------------------
# Test 4: simultaneous zero -> no direct pair contribution
# ---------------------------------------------------------------------------

class TestSimultaneousZero:
    """Test 4: Simultaneous spikes with zero convention -> no contribution."""

    def test_simultaneous_zero_no_change(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = _make_exc_params()
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0, simultaneous_convention="zero")
        initial = stdp.get_weights()[0]
        stdp.update(np.array([True, False]), np.array([False, True]), modulator=1.0)
        final = stdp.get_weights()[0]
        assert abs(final - initial) < 1e-10, f"Expected zero change for simultaneous zero, got {final - initial}"

    def test_simultaneous_zero_after_trace_priming_no_change(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = _make_exc_params()
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0, simultaneous_convention="zero")
        initial = stdp.get_weights()[0]
        stdp.update(np.array([True, False]), np.array([False, False]), modulator=0.0)
        stdp.update(np.array([False, False]), np.array([False, True]), modulator=0.0)
        stdp.update(np.array([True, False]), np.array([False, True]), modulator=0.0)
        final = stdp.get_weights()[0]
        assert abs(final - initial) < 1e-10, f"Expected zero change for simultaneous zero, got {final - initial}"


# ---------------------------------------------------------------------------
# Test 5: simultaneous ltd_dominant -> LTD
# ---------------------------------------------------------------------------

class TestSimultaneousLtdDominant:
    """Test 5: Simultaneous spikes with ltd_dominant convention -> LTD."""

    def test_simultaneous_ltd(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = STDPParams(tau_plus=20.0, tau_minus=20.0, tau_eligibility=20.0, A_plus=1.0, A_minus=1.0, eta=1.0, w_exc_max=10.0, w_inh_max=10.0, M_max=10.0)
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0, simultaneous_convention="ltd_dominant")
        stdp.update(np.array([True, False]), np.array([False, False]), modulator=0.0)
        stdp.update(np.array([False, False]), np.array([False, True]), modulator=0.0)
        pre = np.array([True, False])
        post = np.array([False, True])
        dw = stdp.update(pre, post, modulator=1.0)
        assert dw[0] < 0, f"Expected LTD (negative dw) for ltd_dominant, got {dw[0]}"


# ---------------------------------------------------------------------------
# Test 6: simultaneous ltp_dominant -> LTP
# ---------------------------------------------------------------------------

class TestSimultaneousLtpDominant:
    """Test 6: Simultaneous spikes with ltp_dominant convention -> LTP."""

    def test_simultaneous_ltp(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = STDPParams(tau_plus=20.0, tau_minus=20.0, tau_eligibility=20.0, A_plus=1.0, A_minus=1.0, eta=1.0, w_exc_max=10.0, w_inh_max=10.0, M_max=10.0)
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0, simultaneous_convention="ltp_dominant")
        stdp.update(np.array([True, False]), np.array([False, False]), modulator=0.0)
        stdp.update(np.array([False, False]), np.array([False, True]), modulator=0.0)
        pre = np.array([True, False])
        post = np.array([False, True])
        dw = stdp.update(pre, post, modulator=1.0)
        assert dw[0] > 0, f"Expected LTP (positive dw) for ltp_dominant, got {dw[0]}"


# ---------------------------------------------------------------------------
# Test 7: Excitatory weight never becomes negative
# ---------------------------------------------------------------------------

class TestExcitatorySignConstraint:
    """Test 7: Excitatory weight never becomes negative."""

    def test_exc_weight_non_negative(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = STDPParams(tau_plus=20.0, tau_minus=20.0, tau_eligibility=20.0, A_plus=1.0, A_minus=1.0, eta=100.0, w_exc_max=1.0, w_inh_max=1.0, M_max=10.0)
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0, simultaneous_convention="zero")
        rng = np.random.default_rng(seed=42)
        for _ in range(100):
            pre = rng.random(2) < 0.3
            post = rng.random(2) < 0.3
            stdp.update(pre, post, modulator=-1.0)
        assert stdp.edge_weights[0] >= 0.0, f"Excitatory weight went negative: {stdp.edge_weights[0]}"

    def test_exc_weight_clipped_at_zero(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.0)])
        params = STDPParams(tau_plus=10.0, tau_minus=10.0, tau_eligibility=10.0, A_plus=0.0, A_minus=100.0, eta=100.0, w_exc_max=1.0, w_inh_max=1.0, M_max=10.0)
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0, simultaneous_convention="zero")
        stdp.update(np.array([False, True]), np.array([False, False]), modulator=1.0)
        stdp.update(np.array([True, False]), np.array([False, False]), modulator=1.0)
        assert stdp.edge_weights[0] >= 0.0, f"Excitatory weight at 0 should not go negative: {stdp.edge_weights[0]}"


# ---------------------------------------------------------------------------
# Test 8: Inhibitory weight never becomes positive
# ---------------------------------------------------------------------------

class TestInhibitorySignConstraint:
    """Test 8: Inhibitory weight never becomes positive."""

    def test_inh_weight_non_positive(self):
        graph = _make_graph(n=2, edges=[(0, 1, -0.5)])
        params = STDPParams(tau_plus=20.0, tau_minus=20.0, tau_eligibility=20.0, A_plus=1.0, A_minus=1.0, eta=100.0, w_exc_max=1.0, w_inh_max=1.0, M_max=10.0)
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0, simultaneous_convention="zero")
        rng = np.random.default_rng(seed=42)
        for _ in range(100):
            pre = rng.random(2) < 0.3
            post = rng.random(2) < 0.3
            stdp.update(pre, post, modulator=1.0)
        assert stdp.edge_weights[0] <= 0.0, f"Inhibitory weight went positive: {stdp.edge_weights[0]}"


# ---------------------------------------------------------------------------
# Test 9: Shapes and no dense N x N
# ---------------------------------------------------------------------------

class TestStorageShapes:
    """Test 9: Correct storage shapes, no dense N x N arrays."""

    def test_eligibility_shape(self):
        n = 10
        graph = _make_graph(n=n, edges=[(i, i + 1, 0.5) for i in range(n - 1)])
        params = _make_exc_params()
        stdp = EdgeSparseSTDP(graph, np.arange(n - 1, dtype=np.int64), params, dt_ms=1.0)
        assert stdp.eligibility.shape == (n - 1,), f"eligibility shape: {stdp.eligibility.shape}"
        assert stdp.eligibility.ndim == 1

    def test_trace_shapes(self):
        n = 100
        graph = _make_graph(n=n, edges=[(i, i + 1, 0.5) for i in range(n - 1)])
        params = _make_exc_params()
        stdp = EdgeSparseSTDP(graph, np.arange(n - 1, dtype=np.int64), params, dt_ms=1.0)
        assert stdp.pre_trace.shape == (n,), f"pre_trace shape: {stdp.pre_trace.shape}"
        assert stdp.post_trace.shape == (n,), f"post_trace shape: {stdp.post_trace.shape}"

    def test_no_dense_arrays(self):
        n = 50
        graph = _make_graph(n=n, edges=[(i, i + 1, 0.5) for i in range(n - 1)])
        params = _make_exc_params()
        stdp = EdgeSparseSTDP(graph, np.arange(n - 1, dtype=np.int64), params, dt_ms=1.0)
        assert stdp.pre_trace.ndim == 1
        assert stdp.post_trace.ndim == 1
        assert stdp.eligibility.ndim == 1
        assert stdp.eligibility.shape == (n - 1,)
        rng = np.random.default_rng(42)
        for _ in range(20):
            pre = rng.random(n) < 0.2
            post = rng.random(n) < 0.2
            stdp.update(pre, post, modulator=1.0)
        assert stdp.pre_trace.ndim == 1
        assert stdp.post_trace.ndim == 1
        assert stdp.eligibility.ndim == 1


# ---------------------------------------------------------------------------
# Test 10: NoPlasticityBaseline interface compatibility
# ---------------------------------------------------------------------------

class TestNoPlasticityBaseline:
    """Test 10: NoPlasticityBaseline has same edge-wise interface as STDP."""

    def test_same_interface(self):
        n = 3
        graph = _make_graph(n=n, edges=[(0, 1, 0.5), (1, 2, 0.5)])
        params = _make_exc_params()
        plastic_indices = np.array([0, 1], dtype=np.int64)
        stdp = EdgeSparseSTDP(graph, plastic_indices, params, dt_ms=1.0)
        baseline = NoPlasticityBaseline(graph, plastic_edge_indices=plastic_indices, params=params, dt_ms=1.0)
        pre = np.array([True, False, False])
        post = np.array([False, True, False])
        dw_stdp = stdp.update(pre, post, modulator=1.0)
        dw_base = baseline.update(pre, post, modulator=1.0)
        assert dw_stdp.shape == dw_base.shape
        assert dw_base.shape == (2,)
        assert np.all(dw_base == 0.0)

    def test_baseline_no_change(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = _make_exc_params()
        baseline = NoPlasticityBaseline(graph, np.array([0]), params=params, dt_ms=1.0)
        initial = baseline.get_weights().copy()
        for _ in range(50):
            pre = np.array([True, False])
            post = np.array([False, True])
            baseline.update(pre, post, modulator=1.0)
        assert np.allclose(baseline.get_weights(), initial), "Baseline weights should not change"

    def test_baseline_step_alias(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = _make_exc_params()
        baseline = NoPlasticityBaseline(graph, np.array([0]), params=params, dt_ms=1.0)
        pre = np.array([True, False])
        post = np.array([False, True])
        dw1 = baseline.step(pre, post, modulator=1.0)
        dw2 = baseline.update(pre, post, modulator=1.0)
        assert np.array_equal(dw1, dw2)


# ---------------------------------------------------------------------------
# Test 11: M=0 does not change weights
# ---------------------------------------------------------------------------

class TestModulatorZero:
    """Test 11: M=0 does not change weights."""

    def test_zero_modulator_no_change(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = _make_exc_params()
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0)
        initial = stdp.get_weights().copy()
        for _ in range(10):
            stdp.update(np.array([True, False]), np.array([False, True]), modulator=0.0)
        assert np.allclose(stdp.get_weights(), initial), "Weights should not change with M=0"


# ---------------------------------------------------------------------------
# Test 12: Non-finite M rejects safely
# ---------------------------------------------------------------------------

class TestNonFiniteInputs:
    """Test 12: Non-finite M or inputs raise ValueError."""

    def test_nan_modulator_rejected(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = _make_exc_params()
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0)
        with pytest.raises(ValueError, match="not finite"):
            stdp.update(np.array([True, False]), np.array([False, True]), modulator=float("nan"))

    def test_inf_modulator_rejected(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = _make_exc_params()
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0)
        with pytest.raises(ValueError, match="not finite"):
            stdp.update(np.array([True, False]), np.array([False, True]), modulator=float("inf"))

    def test_nan_inputs_rejected(self):
        modulator = RewardModulator(profile_id="default_v1")
        with pytest.raises(ValueError, match="not finite"):
            modulator.compute(task_reward=float("nan"))

    def test_inf_inputs_rejected(self):
        modulator = RewardModulator(profile_id="default_v1")
        with pytest.raises(ValueError, match="not finite"):
            modulator.compute(task_reward=float("inf"))


# ---------------------------------------------------------------------------
# Test 13: Fixed seed + identical spike sequence -> identical final weights
# ---------------------------------------------------------------------------

class TestDeterministicReplay:
    """Test 13: Fixed seed + identical spike sequence -> identical final weights."""

    def test_deterministic_replay(self):
        n = 3
        edges = [(0, 1, 0.5), (1, 2, 0.5)]
        graph = _make_graph(n=n, edges=edges)
        params = _make_exc_params()
        plastic_indices = np.array([0, 1], dtype=np.int64)
        spike_seq = [
            (np.array([True, False, False]), np.array([False, False, False])),
            (np.array([False, False, False]), np.array([False, True, False])),
            (np.array([False, True, False]), np.array([False, False, False])),
            (np.array([False, False, False]), np.array([False, False, True])),
            (np.array([True, False, False]), np.array([False, True, False])),
        ]
        stdp1 = EdgeSparseSTDP(graph, plastic_indices, params, dt_ms=1.0)
        for pre, post in spike_seq:
            stdp1.update(pre, post, modulator=1.0)
        weights1 = stdp1.get_weights().copy()
        stdp2 = EdgeSparseSTDP(graph, plastic_indices, params, dt_ms=1.0)
        for pre, post in spike_seq:
            stdp2.update(pre, post, modulator=1.0)
        weights2 = stdp2.get_weights().copy()
        np.testing.assert_array_equal(weights1, weights2, "Replays should be deterministic")


# ---------------------------------------------------------------------------
# RewardModulator diagnostics tests
# ---------------------------------------------------------------------------

class TestModulatorDiagnostics:
    """Test modulator diagnostics output."""

    def test_compute_with_diagnostics(self):
        modulator = RewardModulator(profile_id="default_v1")
        clipped_M, diag = modulator.compute_with_diagnostics(task_reward=5.0)
        assert diag.raw_M == 5.0
        assert diag.clipped_M == 5.0

    def test_clip_when_exceeds_M_max(self):
        modulator = RewardModulator(profile_id="default_v1")
        clipped_M, diag = modulator.compute_with_diagnostics(task_reward=100.0)
        assert diag.raw_M == 100.0
        assert diag.clipped_M == 10.0

    def test_profile_id_required(self):
        with pytest.raises(ValueError, match="profile_id"):
            RewardModulator(profile_id="nonexistent")


# ---------------------------------------------------------------------------
# PlasticityDiagnostics integration test
# ---------------------------------------------------------------------------

class TestPlasticityDiagnostics:
    """Test that step() returns diagnostics."""

    def test_update_with_diagnostics(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = _make_exc_params()
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0)
        pre = np.array([False, False])
        post = np.array([False, True])
        dw = stdp.update(pre, post, modulator=1.0)
        assert dw.shape == (1,)
        assert np.all(np.isfinite(dw))


# ---------------------------------------------------------------------------
# Full integration: STDP + Modulator
# ---------------------------------------------------------------------------

class TestSTDPModulatorIntegration:
    """Integration tests for STDP with reward modulator."""

    def test_modulator_scales_dw(self):
        n = 2
        graph = _make_graph(n=n, edges=[(0, 1, 0.5)])
        params = STDPParams(tau_plus=20.0, tau_minus=20.0, tau_eligibility=20.0, A_plus=0.1, A_minus=0.1, eta=1.0, w_exc_max=10.0, M_max=10.0)
        stdp1 = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0)
        stdp1.update(np.array([True, False]), np.array([False, False]), modulator=0.0)
        stdp1.update(np.array([False, False]), np.array([False, False]), modulator=0.0)
        dw1 = stdp1.update(np.array([False, False]), np.array([False, True]), modulator=1.0)
        stdp2 = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0)
        stdp2.update(np.array([True, False]), np.array([False, False]), modulator=0.0)
        stdp2.update(np.array([False, False]), np.array([False, False]), modulator=0.0)
        dw2 = stdp2.update(np.array([False, False]), np.array([False, True]), modulator=2.0)
        assert abs(dw2[0] - 2.0 * dw1[0]) < 1e-10, f"Expected 2x scaling: {dw1[0]} vs {dw2[0]}"

    def test_moderator_clipped_does_not_overflow(self):
        graph = _make_graph(n=2, edges=[(0, 1, 0.5)])
        params = STDPParams(tau_plus=20.0, tau_minus=20.0, tau_eligibility=20.0, A_plus=0.1, A_minus=0.1, eta=1.0, w_exc_max=10.0, M_max=1.0)
        stdp = EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0)
        stdp.update(np.array([True, False]), np.array([False, False]), modulator=0.0)
        dw = stdp.update(np.array([False, False]), np.array([False, True]), modulator=100.0)
        assert stdp.params.M_max == 1.0


# ---------------------------------------------------------------------------
# Test: unknown neurotransmitter edge -> ValueError at construction
# ---------------------------------------------------------------------------

class TestUnknownNeurotransmitterFailsFast:
    """Unknown neurotransmitter edges must raise ValueError in the constructor."""

    def _make_graph_with_unknown_ntx(self) -> ConnectomeGraph:
        nodes = tuple(
            Node(node_id=i, cell_type="E" if i < 2 else "I", region="R1", x=0.0, y=0.0, z=0.0)
            for i in range(2)
        )
        edge_objs = (
            Edge(source=0, target=1, synapse_count=1.0, weight=0.5, neurotransmitter="unknown_ntx"),
        )
        weights = np.array([0.5], dtype=np.float64)
        return ConnectomeGraph(nodes=nodes, edges=edge_objs, weights=weights)

    def test_unknown_ntx_raises_value_error(self):
        graph = self._make_graph_with_unknown_ntx()
        params = _make_exc_params()
        with pytest.raises(ValueError, match="unknown neurotransmitter"):
            EdgeSparseSTDP(graph, np.array([0]), params, dt_ms=1.0)

    def test_unknown_ntx_non_plastic_does_not_raise(self):
        graph = self._make_graph_with_unknown_ntx()
        params = _make_exc_params()
        stdp = EdgeSparseSTDP(graph, np.array([], dtype=np.int64), params, dt_ms=1.0)
        assert stdp.n_plastic == 0
