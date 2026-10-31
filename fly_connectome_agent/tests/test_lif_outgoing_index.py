"""Эквивалентность исходящей индексации в SparseLIF старому полному скану.

Контракт: ускорение шага не имеет права менять ни одного спайка. Здесь лежит
эталонная (медленная) реализация ровно в том виде, в каком она была до правки:
на каждый спайк — маска по всем рёбрам. Оба класса крутятся на одном графе с
одним сидом и обязаны совпасть побитово, включая порядок доставки задержек.
"""
import numpy as np

from fly_connectome_agent.src.science.graph.models import ConnectomeGraph, Edge, Node
from fly_connectome_agent.src.science.snn.lif_sparse import LIFParams, SparseLIF


class NaiveLIF(SparseLIF):
    """Эталон: O(спайки x E) на шаг, как до оптимизации."""

    def step(self, I_ext=None):
        if I_ext is None:
            I_ext = np.zeros(self.n, dtype=np.float64)
        else:
            I_ext = np.asarray(I_ext, dtype=np.float64)
        self._deliver_delayed_spikes()
        self.I_syn *= self.decay

        is_refractory = self.step_count < self.refractory_until
        active = ~is_refractory
        V_before = self.V.copy()

        dv = (-(self.V - self.E_L) + self.R_m * (I_ext + self.I_syn)) * (self.dt / self.tau_m)
        if np.any(self.sigma > 0):
            noise = self.rng.standard_normal(self.n)
            dv += self.sigma * np.sqrt(self.dt) * noise
        self.V += dv
        self.V[is_refractory] = self.V_reset[is_refractory]

        supra = (V_before >= self.V_th) | (self.V >= self.V_th)
        self.spike_mask = supra & active
        spiking = np.where(self.spike_mask)[0]

        if len(spiking) > 0:
            self.V[self.spike_mask] = self.V_reset[self.spike_mask]
            self.refractory_until[self.spike_mask] = (
                self.step_count + self.refrac_steps[self.spike_mask] + 1
            )
            for pre in spiking:
                post_mask = self.edge_index[:, 0] == pre
                if not np.any(post_mask):
                    continue
                posts = self.edge_index[post_mask, 1]
                weights = self.edge_weights[post_mask]
                delays = self.delay_steps[post_mask]
                for post, w, d in zip(posts, weights, delays):
                    queue_idx = (self.step_count + d) % (self.max_delay + 1)
                    self.delay_queues[queue_idx].append((post, w))

        self.t += self.dt
        self.step_count += 1
        return self.spike_mask.copy()


def make_dense_graph() -> ConnectomeGraph:
    """8 нейронов, кратные рёбра, петли, задержки 0..3 мс, E и I."""
    rng = np.random.default_rng(5)
    nodes = tuple(Node(i, "cell", "test", 0.0, 0.0, 0.0, "glutamate") for i in range(8))
    edges = []
    for src in range(8):
        for dst in range(8):
            if src == dst and src % 3:
                continue  # петли только у 0 и 3
            if rng.random() < 0.45:
                continue
            ntx = "gaba" if dst in (2, 5) else "glutamate"
            sign = -1.0 if ntx == "gaba" else 1.0
            edges.append(Edge(src, dst, float(rng.integers(1, 6)),
                              float(sign * rng.uniform(0.2, 0.9)),
                              float(rng.integers(0, 4)), ntx,
                              "inhibitory" if ntx == "gaba" else "excitatory"))
    weights = np.array([e.weight for e in edges], dtype=np.float64)
    return ConnectomeGraph(nodes=nodes, edges=tuple(edges), weights=weights)


def test_fast_path_bit_identical_to_full_scan():
    graph = make_dense_graph()
    params = LIFParams(sigma=2.0, t_refrac=2.0)
    drive = np.zeros(graph.n_nodes)
    drive[:3] = 26.0

    fast = SparseLIF(graph, params, seed=4242)
    slow = NaiveLIF(graph, params, seed=4242)
    assert len(fast.edge_index) == len(slow.edge_index) > 20

    total = 0
    for step in range(400):
        a = fast.step(drive)
        b = slow.step(drive)
        total += int(a.sum())
        assert np.array_equal(a, b), f"спайки разошлись на шаге {step}"
        assert np.array_equal(fast.refractory_until, slow.refractory_until)
        assert np.array_equal(fast.I_syn, slow.I_syn), f"I_syn разошёлся на шаге {step}"
        assert np.array_equal(fast.V, slow.V), f"V разошёлся на шаге {step}"
        assert fast.delay_queues == slow.delay_queues, f"очередь задержек разошлась на шаге {step}"

    assert total > 0, "тест бессмыслен без спайков"


def test_fast_path_uses_current_weights_not_cached():
    """Веса читаются в момент спайка, а не запоминаются при сборке сети.

    Очередь доставки хранит (post, w) с момента выброса — это существующая
    семантика задержки, не кэш. Проверяем именно чтение edge_weights: вес
    меняется ДО спайка, и в доставке обязан оказаться новый.
    """
    nodes = (Node(0, "a", "t", 0.0, 0.0, 0.0, "glutamate"),
             Node(1, "b", "t", 0.0, 0.0, 0.0, "glutamate"))
    edges = (Edge(0, 1, 1.0, 0.5, 0.0, "glutamate", "excitatory"),)
    graph = ConnectomeGraph(nodes=nodes, edges=edges, weights=np.array([0.5]))

    net = SparseLIF(graph, seed=1)
    net.edge_weights[0] = 2.0
    net.V = np.array([-49.0, -70.0])
    net.step()          # спайк с новым весом попал в очередь доставки
    net.step()          # доставка, затем утечка тока

    assert np.isclose(net.I_syn[1], 2.0 * net.decay)
    assert not np.isclose(net.I_syn[1], 0.5 * net.decay)


def test_fast_path_handles_empty_and_isolated_nodes():
    nodes = tuple(Node(i, "a", "t", 0.0, 0.0, 0.0, "glutamate") for i in range(3))
    graph = ConnectomeGraph(nodes=nodes, edges=(), weights=np.zeros(0))
    net = SparseLIF(graph, LIFParams(sigma=0.0), seed=3)
    net.V = np.array([-49.0, -70.0, -70.0])
    spikes = net.step()
    assert spikes[0] and not spikes[1:].any()
