"""Sparse current-based LIF simulator for P1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from ..graph.models import ConnectomeGraph


@dataclass(frozen=True)
class LIFParams:
    """Per-neuron or per-class LIF parameters."""
    E_L: float = -70.0
    V_th: float = -50.0
    V_reset: float = -70.0
    tau_m: float = 20.0
    R_m: float = 1.0
    sigma: float = 0.0
    t_refrac: float = 2.0


@dataclass
class LIFState:
    """State of the LIF network at a given timestep."""
    V: np.ndarray
    I_syn: np.ndarray
    refractory_until: np.ndarray
    spike_mask: np.ndarray
    t: float
    step_count: int
    delay_queues: list[list[tuple[int, float]]]

    def copy(self) -> "LIFState":
        return LIFState(
            V=self.V.copy(),
            I_syn=self.I_syn.copy(),
            refractory_until=self.refractory_until.copy(),
            spike_mask=self.spike_mask.copy(),
            t=self.t,
            step_count=self.step_count,
            delay_queues=[list(q) for q in self.delay_queues],
        )


class SparseLIF:
    """Sparse current-based LIF network with Euler-Maruyama integration."""

    def __init__(
        self,
        graph: ConnectomeGraph,
        params: LIFParams | np.ndarray | dict[int, LIFParams] | None = None,
        dt_ms: float = 1.0,
        synapse_tau_ms: float = 5.0,
        seed: int | None = None,
    ):
        self.graph = graph
        self.dt = dt_ms
        self.synapse_tau = synapse_tau_ms
        self.n = graph.n_nodes

        if params is None:
            self.params = np.array([LIFParams() for _ in range(self.n)])
        elif isinstance(params, LIFParams):
            self.params = np.array([params for _ in range(self.n)])
        elif isinstance(params, dict):
            self.params = np.array([params.get(i, LIFParams()) for i in range(self.n)])
        else:
            self.params = np.asarray(params)

        self.E_L = np.array([p.E_L for p in self.params], dtype=np.float64)
        self.V_th = np.array([p.V_th for p in self.params], dtype=np.float64)
        self.V_reset = np.array([p.V_reset for p in self.params], dtype=np.float64)
        self.tau_m = np.array([p.tau_m for p in self.params], dtype=np.float64)
        self.R_m = np.array([p.R_m for p in self.params], dtype=np.float64)
        self.sigma = np.array([p.sigma for p in self.params], dtype=np.float64)
        self.t_refrac = np.array([p.t_refrac for p in self.params], dtype=np.float64)

        self.refrac_steps = np.ceil(self.t_refrac / self.dt).astype(np.int64)
        self.decay = np.exp(-self.dt / self.synapse_tau)

        self.rng = np.random.default_rng(seed)

        W = graph.weight_matrix
        delays = graph.delays_ms
        if delays is not None and len(delays) == graph.n_edges:
            max_delay = int(np.ceil(np.max(delays) / self.dt))
        else:
            max_delay = 0
            delays = np.zeros(graph.n_edges, dtype=np.float64)

        self.max_delay = max_delay
        self.delay_steps = np.ceil(delays / self.dt).astype(np.int64)
        self.edge_index = graph.edge_index
        self.edge_weights = np.asarray(graph.weights, dtype=np.float64)

        self.delay_queues: list[list[tuple[int, float]]] = [[] for _ in range(max_delay + 1)]
        self.V = self.E_L.copy()
        self.I_syn = np.zeros(self.n, dtype=np.float64)
        self.refractory_until = np.zeros(self.n, dtype=np.int64)
        self.spike_mask = np.zeros(self.n, dtype=bool)
        self.t = 0.0
        self.step_count = 0

    def step(self, I_ext: np.ndarray | None = None) -> np.ndarray:
        """Advance one timestep. Returns spike mask."""
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

        # Возбуждение: нейрон стреляет, если он был выше порога на входе шага ИЛИ
        # пересёк порог внутри шага. Утечка не может «отменить» уже достигнутый
        # порог — иначе нейрон, поставленный выше порога (например, при
        # инициализации состояния), терял бы спайк, сползая к потенциалу покоя.
        supra = (V_before >= self.V_th) | (self.V >= self.V_th)
        self.spike_mask = supra & active
        spiking = np.where(self.spike_mask)[0]

        if len(spiking) > 0:
            self.V[self.spike_mask] = self.V_reset[self.spike_mask]
            # Заблокированных шагов ровно ceil(t_refrac / dt): спайк занимает шаг s,
            # дальше идут s+1 .. s+R, поэтому первый свободный шаг — s+R+1.
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

    def _deliver_delayed_spikes(self) -> None:
        queue_idx = self.step_count % (self.max_delay + 1)
        for post, w in self.delay_queues[queue_idx]:
            self.I_syn[post] += w
        self.delay_queues[queue_idx].clear()

    def run(
        self,
        n_steps: int,
        I_ext: np.ndarray | None = None,
        record: bool = False,
    ) -> dict[str, Any]:
        """Run simulation for n_steps."""
        if record:
            V_hist = np.zeros((n_steps, self.n), dtype=np.float64)
            I_hist = np.zeros((n_steps, self.n), dtype=np.float64)
            spikes_hist = np.zeros((n_steps, self.n), dtype=bool)

        for i in range(n_steps):
            if I_ext is not None and I_ext.ndim == 2:
                step_I = I_ext[i]
            else:
                step_I = I_ext
            self.step(step_I)
            if record:
                V_hist[i] = self.V
                I_hist[i] = self.I_syn
                spikes_hist[i] = self.spike_mask

        result = {"spike_mask": self.spike_mask, "V": self.V, "I_syn": self.I_syn, "t": self.t}
        if record:
            result["V_hist"] = V_hist
            result["I_hist"] = I_hist
            result["spikes_hist"] = spikes_hist
        return result

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.V = self.E_L.copy()
        self.I_syn = np.zeros(self.n, dtype=np.float64)
        self.refractory_until = np.zeros(self.n, dtype=np.int64)
        self.spike_mask = np.zeros(self.n, dtype=bool)
        self.delay_queues = [[] for _ in range(self.max_delay + 1)]
        self.t = 0.0
        self.step_count = 0

    def get_state(self) -> LIFState:
        return LIFState(
            V=self.V.copy(),
            I_syn=self.I_syn.copy(),
            refractory_until=self.refractory_until.copy(),
            spike_mask=self.spike_mask.copy(),
            t=self.t,
            step_count=self.step_count,
            delay_queues=[list(q) for q in self.delay_queues],
        )

    def set_state(self, state: LIFState) -> None:
        self.V = state.V.copy()
        self.I_syn = state.I_syn.copy()
        self.refractory_until = state.refractory_until.copy()
        self.spike_mask = state.spike_mask.copy()
        self.t = state.t
        self.step_count = state.step_count
        self.delay_queues = [list(q) for q in state.delay_queues]


def simulate_lif(
    graph: ConnectomeGraph,
    n_steps: int,
    params: LIFParams | np.ndarray | dict[int, LIFParams] | None = None,
    dt_ms: float = 1.0,
    synapse_tau_ms: float = 5.0,
    I_ext: np.ndarray | None = None,
    seed: int | None = None,
    record: bool = False,
) -> dict[str, Any]:
    """Convenience function to run a sparse LIF simulation."""
    net = SparseLIF(graph, params, dt_ms, synapse_tau_ms, seed)
    return net.run(n_steps, I_ext, record)