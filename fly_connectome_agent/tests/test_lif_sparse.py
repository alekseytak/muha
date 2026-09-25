"""Unit tests for sparse LIF simulator."""
import numpy as np
import pytest

from fly_connectome_agent.src.science.graph.models import ConnectomeGraph, Node, Edge
from fly_connectome_agent.src.science.snn.lif_sparse import (
    LIFParams,
    LIFState,
    SparseLIF,
    simulate_lif,
)


def make_simple_graph() -> ConnectomeGraph:
    nodes = (
        Node(0, "exc", "A", 0, 0, 0, "glutamate"),
        Node(1, "inh", "A", 1, 0, 0, "GABA"),
    )
    edges = (
        Edge(0, 1, 1.0, 0.5, 1.0, "glutamate"),
    )
    return ConnectomeGraph(nodes=nodes, edges=edges, weights=np.array([0.5]))


def test_lif_params_defaults():
    params = LIFParams()
    assert params.E_L == -70.0
    assert params.V_th == -50.0
    assert params.V_reset == -70.0
    assert params.tau_m == 20.0
    assert params.R_m == 1.0
    assert params.sigma == 0.0
    assert params.t_refrac == 2.0


def test_sparse_lif_initialization():
    graph = make_simple_graph()
    net = SparseLIF(graph, dt_ms=1.0, synapse_tau_ms=5.0, seed=42)
    assert net.n == 2
    assert net.step_count == 0
    assert net.t == 0.0
    assert np.allclose(net.V, net.E_L)


def test_sparse_lif_step_no_spike():
    graph = make_simple_graph()
    net = SparseLIF(graph, seed=123)
    net.V = np.array([-70.0, -70.0])
    spikes = net.step()
    assert not np.any(spikes)
    assert net.step_count == 1


def test_sparse_lif_spike_and_refractory():
    graph = make_simple_graph()
    net = SparseLIF(graph, seed=42)
    net.V = np.array([-49.0, -70.0])  # neuron 0 above threshold
    spikes = net.step()
    assert spikes[0] == True
    assert spikes[1] == False
    assert net.V[0] == net.V_reset[0]
    assert net.refractory_until[0] > net.step_count


def test_sparse_lif_deterministic_replay():
    graph = make_simple_graph()
    params = LIFParams(sigma=1.0)
    net1 = SparseLIF(graph, params, seed=999)
    net2 = SparseLIF(graph, params, seed=999)
    for _ in range(10):
        s1 = net1.step()
        s2 = net2.step()
        assert np.array_equal(s1, s2)


def test_sparse_lif_noise_scaling():
    """Test that noise scales with sqrt(dt) in isolation (no leak, no reset)."""
    graph = make_simple_graph()
    # Use very large tau_m so leak is negligible, very high threshold so no spikes
    params = LIFParams(tau_m=1e9, V_th=1e9, sigma=1.0)
    dt_values = [0.1, 0.5, 1.0, 2.0]
    variances = []
    for dt in dt_values:
        net = SparseLIF(graph, params, dt_ms=dt, seed=42)
        net.V = np.array([-70.0, -70.0])
        dvs = []
        for _ in range(5000):
            old_V = net.V.copy()
            net.step()
            dvs.append(net.V[0] - old_V[0])
        variances.append(np.var(dvs))
    for i in range(len(dt_values) - 1):
        ratio = variances[i + 1] / variances[i]
        expected = dt_values[i + 1] / dt_values[i]
        assert abs(ratio - expected) / expected < 0.15


def test_sparse_lif_no_nan():
    graph = make_simple_graph()
    params = LIFParams(sigma=10.0)  # large noise
    net = SparseLIF(graph, params, seed=1)
    for _ in range(1000):
        net.step()
        assert np.all(np.isfinite(net.V))
        assert np.all(np.isfinite(net.I_syn))


def test_sparse_lif_refractory_period():
    graph = make_simple_graph()
    params = LIFParams(t_refrac=3.0)  # 3ms refractory -> exactly 3 steps blocked
    net = SparseLIF(graph, params, dt_ms=1.0, seed=42)
    net.V = np.array([-49.0, -70.0])
    # First spike at step 0
    spikes = net.step()
    assert spikes[0] == True
    # Should be refractory for exactly 3 subsequent steps (steps 1, 2, 3)
    for _ in range(3):
        net.V = np.array([-49.0, -70.0])
        spikes = net.step()
        assert not spikes[0]
    # Active again at step 4
    net.V = np.array([-49.0, -70.0])
    spikes = net.step()
    assert spikes[0]


def test_simulate_lif_convenience():
    graph = make_simple_graph()
    result = simulate_lif(graph, n_steps=10, seed=42, record=True)
    assert "spike_mask" in result
    assert "V" in result
    assert "I_syn" in result
    assert "t" in result
    assert "V_hist" in result
    assert "spikes_hist" in result
    assert result["V_hist"].shape == (10, 2)
    assert result["spikes_hist"].shape == (10, 2)


def test_lif_state_copy():
    graph = make_simple_graph()
    net = SparseLIF(graph, seed=1)
    net.step()
    state = net.get_state()
    assert isinstance(state, LIFState)
    assert state.V.shape == (2,)
    assert state.I_syn.shape == (2,)
    assert state.refractory_until.shape == (2,)
    assert state.spike_mask.shape == (2,)


def test_sparse_lif_reset():
    graph = make_simple_graph()
    net = SparseLIF(graph, seed=42)
    for _ in range(5):
        net.step()
    net.reset(seed=42)
    assert net.step_count == 0
    assert net.t == 0.0
    assert np.allclose(net.V, net.E_L)
    assert np.all(net.I_syn == 0)
    assert np.all(net.refractory_until == 0)


def test_sparse_lif_delay_delivery():
    """Test that delayed spikes are delivered correctly."""
    graph = make_simple_graph()
    net = SparseLIF(graph, dt_ms=1.0, synapse_tau_ms=5.0, seed=42)
    # Set up a spike at time 0 with 2ms delay
    net.V = np.array([-49.0, -70.0])
    net.step()  # spike at step 0, delay 1ms -> delivered at step 1
    # Before delivery, I_syn should be 0
    assert np.all(net.I_syn == 0)
    # Step 1: deliver delayed spike
    net.step()
    assert net.I_syn[1] > 0


def test_sparse_lif_per_neuron_params():
    graph = make_simple_graph()
    params = [
        LIFParams(E_L=-65.0, V_th=-55.0),
        LIFParams(E_L=-75.0, V_th=-50.0),
    ]
    net = SparseLIF(graph, params, seed=1)
    assert net.E_L[0] == -65.0
    assert net.E_L[1] == -75.0
    assert net.V_th[0] == -55.0
    assert net.V_th[1] == -50.0