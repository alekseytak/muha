"""Edge-sparse reward-modulated STDP for P2.

Implements the Sparse R-STDP contract:
- Neuron-level traces pre_trace/post_trace: O(N)
- Edge-level eligibility: O(E_plastic)
- No dense N×N arrays
- Modulator M clipped to [-M_max, M_max]
- Per-edge-class sign constraints from edge_class (neurotransmitter)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..graph.models import ConnectomeGraph


@dataclass(frozen=True)
class STDPParams:
    """STDP parameters.

    w_exc_max: max weight for excitatory edges (constraint: 0 <= w <= w_exc_max)
    w_inh_max: max magnitude for inhibitory edges (constraint: -w_inh_max <= w <= 0)
    M_max: clipping bound for modulator M
    """
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    tau_eligibility: float = 20.0
    A_plus: float = 0.1
    A_minus: float = 0.1
    eta: float = 0.01
    w_exc_max: float = 1.0
    w_inh_max: float = 1.0
    M_max: float = 10.0


_VALID_CONVENTIONS = {"zero", "ltd_dominant", "ltp_dominant"}

_EXC_NTX = {"glutamate", "acetylcholine"}
_INH_NTX = {"gaba", "glycine"}


@dataclass
class PlasticityDiagnostics:
    """Diagnostics output from a single STDP step."""
    raw_M: float
    clipped_M: float
    mean_abs_dw: float
    fraction_weights_at_bound: float


@dataclass
class PlasticityState:
    """State of the STDP plasticity engine."""
    pre_trace: np.ndarray
    post_trace: np.ndarray
    eligibility: np.ndarray
    edge_weights: np.ndarray
    step_count: int


class EdgeSparseSTDP:
    """Edge-sparse reward-modulated STDP.

    Storage:
    - pre_trace: O(N) — presynaptic neuron traces
    - post_trace: O(N) — postsynaptic neuron traces
    - eligibility: O(E_plastic) — edge-eligibility traces
    - edge_weights: O(E_plastic) — plastic edge weights

    No dense N×N arrays are allocated at any point.

    Sign constraints are determined from edge neurotransmitter annotation
    (glutamate/acetylcholine -> excitatory, GABA -> inhibitory), NOT from
    the sign of the initial weight.
    """

    def __init__(
        self,
        graph: ConnectomeGraph,
        plastic_edge_indices: np.ndarray,
        params: STDPParams | None = None,
        dt_ms: float = 1.0,
        simultaneous_convention: str = "zero",
    ):
        self.graph = graph
        self.plastic_indices = np.asarray(plastic_edge_indices, dtype=np.int64)
        self.params = params if params is not None else STDPParams()
        self.dt = dt_ms

        if simultaneous_convention not in _VALID_CONVENTIONS:
            raise ValueError(
                f"simultaneous_convention must be one of {_VALID_CONVENTIONS}, "
                f"got '{simultaneous_convention}'"
            )
        self.simultaneous = simultaneous_convention

        self.n = graph.n_nodes
        self.n_plastic = len(self.plastic_indices)
        self.edge_index = graph.edge_index[self.plastic_indices]
        self.edge_weights = np.asarray(graph.weights[self.plastic_indices], dtype=np.float64)

        # Determine edge class from neurotransmitter annotation
        edge_ntx = graph.edge_neurotransmitters[self.plastic_indices]
        self.edge_is_exc = np.array(
            [ntx in _EXC_NTX if ntx is not None else False for ntx in edge_ntx],
            dtype=bool,
        )
        self.edge_is_inh = np.array(
            [ntx in _INH_NTX if ntx is not None else False for ntx in edge_ntx],
            dtype=bool,
        )
        self.edge_is_unknown = ~(self.edge_is_exc | self.edge_is_inh)

        # Fail-fast: unknown edges cannot be plastic in MVP
        if np.any(self.edge_is_unknown):
            unknown_indices = self.plastic_indices[self.edge_is_unknown]
            raise ValueError(
                f"unknown neurotransmitter edges cannot be plastic in MVP. "
                f"Unknown plastic edge indices: {unknown_indices.tolist()}"
            )

        # Neuron-level traces O(N)
        self.pre_trace = np.zeros(self.n, dtype=np.float64)
        self.post_trace = np.zeros(self.n, dtype=np.float64)

        # Edge-level eligibility O(E_plastic)
        self.eligibility = np.zeros(self.n_plastic, dtype=np.float64)

        # Decay factors
        self._decay_pre = np.exp(-self.dt / self.params.tau_plus)
        self._decay_post = np.exp(-self.dt / self.params.tau_minus)
        self._decay_elig = np.exp(-self.dt / self.params.tau_eligibility)

        self.step_count = 0

    def update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        modulator: float = 1.0,
    ) -> np.ndarray:
        """Apply one STDP update step.

        Steps:
        1. Decay eligibility traces.
        2. Handle simultaneous spike convention (zero appropriate flags).
        3. Compute eligibility using traces from PREVIOUS timestep.
        4. Apply modulator: dw = eta * clip(M, -M_max, M_max) * eligibility
        5. Update weights with sign constraints.
        6. Update neuron-level traces with current spikes.

        Args:
            pre_spikes: boolean array of size N
            post_spikes: boolean array of size N
            modulator: scalar modulator value M

        Returns:
            Weight changes dw for plastic edges, shape (E_plastic,)
        """
        pre_spikes = np.asarray(pre_spikes, dtype=bool)
        post_spikes = np.asarray(post_spikes, dtype=bool)
        s_pre = pre_spikes.astype(np.float64)
        s_post = post_spikes.astype(np.float64)

        edge_pre = self.edge_index[:, 0]
        edge_post = self.edge_index[:, 1]

        # Get per-edge spike flags (copies to allow modification)
        s_pre_edge = s_pre[edge_pre].copy()
        s_post_edge = s_post[edge_post].copy()

        # Step 1: Decay eligibility traces
        self.eligibility *= self._decay_elig

        # Step 2: Handle simultaneous spike convention BEFORE eligibility
        sim_mask = pre_spikes[edge_pre] & post_spikes[edge_post]
        if np.any(sim_mask):
            if self.simultaneous == "zero":
                s_pre_edge[sim_mask] = 0.0
                s_post_edge[sim_mask] = 0.0
            elif self.simultaneous == "ltd_dominant":
                s_post_edge[sim_mask] = 0.0
            else:  # ltp_dominant
                s_pre_edge[sim_mask] = 0.0

        # Step 3: Compute eligibility using traces from PREVIOUS timestep
        pre_trace_edge = self.pre_trace[edge_pre]
        post_trace_edge = self.post_trace[edge_post]

        self.eligibility += self.params.A_plus * pre_trace_edge * s_post_edge
        self.eligibility -= self.params.A_minus * s_pre_edge * post_trace_edge

        # Step 4: Apply rewards modulator with clipping
        clipped_M = self._clip_modulator(modulator)
        dw = self.params.eta * clipped_M * self.eligibility

        # Step 5: Update weights with sign constraints
        new_weights = self.edge_weights + dw
        new_weights = self._apply_sign_constraints(new_weights)
        self.edge_weights = new_weights

        # Step 6: Update neuron-level traces AFTER computing eligibility
        self.pre_trace = self.pre_trace * self._decay_pre + s_pre
        self.post_trace = self.post_trace * self._decay_post + s_post
        self.step_count += 1

        return dw

    def update_with_diagnostics(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        modulator: float = 1.0,
    ) -> tuple[np.ndarray, PlasticityDiagnostics]:
        """Update with diagnostics.

        Returns:
            Tuple of (dw, diagnostics)
        """
        pre_spikes = np.asarray(pre_spikes, dtype=bool)
        post_spikes = np.asarray(post_spikes, dtype=bool)
        s_pre = pre_spikes.astype(np.float64)
        s_post = post_spikes.astype(np.float64)

        edge_pre = self.edge_index[:, 0]
        edge_post = self.edge_index[:, 1]

        s_pre_edge = s_pre[edge_pre].copy()
        s_post_edge = s_post[edge_post].copy()

        sim_mask = pre_spikes[edge_pre] & post_spikes[edge_post]
        if np.any(sim_mask):
            if self.simultaneous == "zero":
                s_pre_edge[sim_mask] = 0.0
                s_post_edge[sim_mask] = 0.0
            elif self.simultaneous == "ltd_dominant":
                s_post_edge[sim_mask] = 0.0
            else:  # ltp_dominant
                s_pre_edge[sim_mask] = 0.0

        # Step 1: Decay eligibility
        self.eligibility *= self._decay_elig

        pre_trace_edge = self.pre_trace[edge_pre]
        post_trace_edge = self.post_trace[edge_post]

        self.eligibility += self.params.A_plus * pre_trace_edge * s_post_edge
        self.eligibility -= self.params.A_minus * s_pre_edge * post_trace_edge

        # Step 4: Apply modulator
        raw_M = modulator
        clipped_M = self._clip_modulator(modulator)
        dw = self.params.eta * clipped_M * self.eligibility

        # Step 5: Update weights with sign constraints
        new_weights = self.edge_weights + dw
        new_weights = self._apply_sign_constraints(new_weights)

        # Compute diagnostics
        mean_abs_dw = float(np.mean(np.abs(dw)))
        at_bound = np.zeros(self.n_plastic, dtype=bool)
        exc = self.edge_is_exc
        inh = self.edge_is_inh
        at_bound[exc] = (
            np.isclose(new_weights[exc], 0.0)
            | np.isclose(new_weights[exc], self.params.w_exc_max)
        )
        at_bound[inh] = (
            np.isclose(new_weights[inh], 0.0)
            | np.isclose(new_weights[inh], -self.params.w_inh_max)
        )
        fraction_at_bound = float(np.sum(at_bound)) / self.n_plastic if self.n_plastic > 0 else 0.0

        self.edge_weights = new_weights

        # Step 6: Update traces
        self.pre_trace = self.pre_trace * self._decay_pre + s_pre
        self.post_trace = self.post_trace * self._decay_post + s_post
        self.step_count += 1

        diagnostics = PlasticityDiagnostics(
            raw_M=float(raw_M),
            clipped_M=float(clipped_M),
            mean_abs_dw=mean_abs_dw,
            fraction_weights_at_bound=fraction_at_bound,
        )

        return dw, diagnostics

    def _clip_modulator(self, M: float) -> float:
        """Clip modulator to [-M_max, M_max], reject non-finite."""
        if not np.isfinite(M):
            raise ValueError(f"modulator M is not finite: {M}")
        return float(np.clip(M, -self.params.M_max, self.params.M_max))

    def _apply_sign_constraints(self, new_weights: np.ndarray) -> np.ndarray:
        """Apply per-edge-class sign constraints.

        Excitatory edges (glutamate/acetylcholine): clip to [0, w_exc_max]
        Inhibitory edges (GABA): clip to [-w_inh_max, 0]
        Unknown edges: not clipped (MVP — should be excluded upstream)
        """
        clipped = new_weights.copy()
        exc_mask = self.edge_is_exc
        inh_mask = self.edge_is_inh
        clipped[exc_mask] = np.clip(clipped[exc_mask], 0.0, self.params.w_exc_max)
        clipped[inh_mask] = np.clip(clipped[inh_mask], -self.params.w_inh_max, 0.0)
        return clipped

    def reset(self) -> None:
        """Reset all state to initial values."""
        self.pre_trace = np.zeros(self.n, dtype=np.float64)
        self.post_trace = np.zeros(self.n, dtype=np.float64)
        self.eligibility = np.zeros(self.n_plastic, dtype=np.float64)
        self.edge_weights = np.asarray(self.graph.weights[self.plastic_indices], dtype=np.float64)
        self.step_count = 0

    def get_weights(self) -> np.ndarray:
        """Return a copy of edge weights."""
        return self.edge_weights.copy()

    def set_weights(self, weights: np.ndarray) -> None:
        """Set edge weights."""
        self.edge_weights = np.asarray(weights, dtype=np.float64)

    def get_state(self) -> PlasticityState:
        """Get current plasticity state."""
        return PlasticityState(
            pre_trace=self.pre_trace.copy(),
            post_trace=self.post_trace.copy(),
            eligibility=self.eligibility.copy(),
            edge_weights=self.edge_weights.copy(),
            step_count=self.step_count,
        )

    def set_state(self, state: PlasticityState) -> None:
        """Restore plasticity state."""
        self.pre_trace = state.pre_trace.copy()
        self.post_trace = state.post_trace.copy()
        self.eligibility = state.eligibility.copy()
        self.edge_weights = state.edge_weights.copy()
        self.step_count = state.step_count


class NoPlasticityBaseline:
    """No-plasticity baseline with same interface as EdgeSparseSTDP.

    Returns zero weight changes for all edges, matching the
    (E_plastic,) shape contract.
    """

    def __init__(
        self,
        graph: ConnectomeGraph,
        plastic_edge_indices: np.ndarray | None = None,
        params: STDPParams | None = None,
        dt_ms: float = 1.0,
        simultaneous_convention: str = "zero",
    ):
        self.graph = graph
        self.params = params if params is not None else STDPParams()

        if plastic_edge_indices is not None:
            self.plastic_indices = np.asarray(plastic_edge_indices, dtype=np.int64)
            self.n_plastic = len(self.plastic_indices)
            self.edge_index = graph.edge_index[self.plastic_indices]
            self.edge_weights = np.asarray(graph.weights[self.plastic_indices], dtype=np.float64).copy()
        else:
            self.plastic_indices = np.arange(graph.n_edges, dtype=np.int64)
            self.n_plastic = graph.n_edges
            self.edge_index = graph.edge_index
            self.edge_weights = np.asarray(graph.weights, dtype=np.float64).copy()

        self.n = graph.n_nodes
        self.simultaneous = simultaneous_convention
        self.step_count = 0

    def update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        modulator: float = 1.0,
    ) -> np.ndarray:
        """Return zero weight changes, same shape as plastic weights.

        Args:
            pre_spikes: boolean array of size N
            post_spikes: boolean array of size N
            modulator: scalar (ignored, always 0 dw)

        Returns:
            Zero array of shape (E_plastic,)
        """
        pre_spikes = np.asarray(pre_spikes, dtype=bool)
        post_spikes = np.asarray(post_spikes, dtype=bool)
        self.step_count += 1
        return np.zeros(self.n_plastic, dtype=np.float64)

    def step(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        modulator: float = 1.0,
    ) -> np.ndarray:
        """Alias for update()."""
        return self.update(pre_spikes, post_spikes, modulator)

    def get_weights(self) -> np.ndarray:
        """Return a copy of edge weights."""
        return self.edge_weights.copy()

    def reset(self) -> None:
        """Reset to initial weights."""
        self.edge_weights = np.asarray(self.graph.weights[self.plastic_indices], dtype=np.float64).copy()
        self.step_count = 0