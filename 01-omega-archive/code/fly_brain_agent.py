"""FlyBrain Agent — SNN-ядро на подграфе коннектома дрозофилы.

Данные: Male CNS v1.0 (166 700 нейронов, Janelia/Google DeepMind, Sep 2026).
Для MVP берётся подграф 1k-50k нейронов:
  optic_lobe -> central_complex -> motor_output
  antennal_lobe -> mushroom_body -> reward_modulation

Формат входа (контракт, см. AGENT-BUILD-INSTRUCTIONS.md ЭТАП 1):
  nodes.csv: neuron_id,type,region,x,y,z
  edges.csv: source,target,synapse_type,weight,delay_ms

Нейрон: Leaky Integrate-and-Fire. Пластичность: reward_stdp.RewardModulatedSTDP.
Каждое моторное решение выходит наружу ТОЛЬКО как verb-act (OS Глаголов).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np


@dataclass
class LIFParams:
    v_rest: float = -65.0
    v_th: float = -50.0
    v_reset: float = -70.0
    tau_ms: float = 20.0
    r_mohm: float = 10.0
    dt_ms: float = 1.0
    noise_sd: float = 0.5


class FlyBrain:
    """Подграф коннектома как спайковая сеть LIF-нейронов (numpy-MVP)."""

    REGIONS = ("optic_lobe", "antennal_lobe", "central_complex",
               "mushroom_body", "motor_output")

    def __init__(self, nodes_csv: str, edges_csv: str,
                 params: LIFParams | None = None):
        self.p = params or LIFParams()
        self.ids: list[int] = []
        self.region_of: dict[int, str] = {}
        self.type_of: dict[int, str] = {}
        self._load_nodes(nodes_csv)
        self.n = len(self.ids)
        if self.n == 0:
            raise ValueError("пустой nodes.csv")
        self.idx = {nid: i for i, nid in enumerate(self.ids)}
        self.W = np.zeros((self.n, self.n))
        self.delay = np.zeros((self.n, self.n), dtype=int)
        self._load_edges(edges_csv)
        self._init_state()

    # ---------- загрузка ----------
    def _load_nodes(self, path: str) -> None:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                nid = int(row["neuron_id"])
                self.ids.append(nid)
                self.region_of[nid] = row["region"]
                self.type_of[nid] = row["type"]

    def _load_edges(self, path: str) -> None:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("synapse_type", "chemical") != "chemical":
                    continue  # электрические/модуляторные — ЭТАП 10
                src, tgt = int(row["source"]), int(row["target"])
                if src not in self.idx or tgt not in self.idx:
                    continue
                i, j = self.idx[src], self.idx[tgt]
                self.W[i, j] = float(row.get("weight", 0.1))
                d_ms = float(row.get("delay_ms", 1.0))
                self.delay[i, j] = max(1, int(d_ms / self.p.dt_ms))

    def _init_state(self) -> None:
        self.V = np.full(self.n, self.p.v_rest)
        self.spikes = np.zeros(self.n, dtype=bool)
        self.I_syn = np.zeros(self.n)
        self._max_delay = int(self.delay.max()) if self.n else 1
        self.spike_history = np.zeros((self._max_delay + 1, self.n), dtype=bool)
        self._h_idx = 0
        self.t = 0.0
        self.spike_budget_used = 0

    # ---------- симуляция ----------
    def step(self, input_current: np.ndarray) -> np.ndarray:
        """Один шаг: входной ток сенсоров -> спайки всех нейронов."""
        p = self.p
        I = np.asarray(input_current, dtype=float) + self.I_syn
        dV = (-(self.V - p.v_rest) + p.r_mohm * I) / p.tau_ms * p.dt_ms
        self.V = self.V + dV + np.random.normal(0.0, p.noise_sd, self.n)
        self.spikes = self.V >= p.v_th
        self.V[self.spikes] = p.v_reset

        self.spike_history[self._h_idx] = self.spikes
        self._h_idx = (self._h_idx + 1) % self.spike_history.shape[0]

        self.I_syn = np.zeros(self.n)
        for d in range(1, self._max_delay + 1):
            hist = self.spike_history[(self._h_idx - d) % self.spike_history.shape[0]]
            if not hist.any():
                continue
            mask = (self.delay == d)
            if mask.any():
                self.I_syn += (hist[:, None] * self.W * mask).sum(axis=0)

        self.t += p.dt_ms
        self.spike_budget_used += int(self.spikes.sum())
        return self.spikes

    # ---------- выходы ----------
    def region_rates(self, window: np.ndarray) -> dict[str, float]:
        """Средние частоты по регионам за окно (window: bool-матрица T x n)."""
        out: dict[str, float] = {}
        for region in self.REGIONS:
            mask = np.array([self.region_of[i] == region for i in self.ids])
            if mask.any():
                out[region] = float(window[:, mask].mean())
        return out

    def motor_vector(self, rates: dict[str, float]) -> np.ndarray:
        """Декодирование моторной команды: [вперёд, поворот, стабилизация]."""
        m = rates.get("motor_output", 0.0)
        cc = rates.get("central_complex", 0.0)
        return np.array([m, (cc - m) * 0.5, min(m, cc) * 0.25])

    def energy_cost(self) -> float:
        """Штраф за энергию: стоимость каждого спайка."""
        return self.spike_budget_used * 1e-5


if __name__ == "__main__":
    # Свободный прогон на синтетическом подграфе (самопроверка)
    import tempfile, os
    nodes = "neuron_id,type,region,x,y,z\n"
    edges = "source,target,synapse_type,weight,delay_ms\n"
    plan = [(0, "optic_lobe", 5), (5, "central_complex", 5),
            (10, "mushroom_body", 5), (15, "motor_output", 5)]
    start = 1
    for region, count in plan:
        for k in range(count):
            nodes += f"{start + k},interneuron,{region},0,0,0\n"
        start += count
    for i in range(15):
        edges += f"{i+1},{i+2},chemical,0.35,1\n" if i < 4 else ""
    with tempfile.TemporaryDirectory() as d:
        np_, ep = os.path.join(d, "n.csv"), os.path.join(d, "e.csv")
        open(np_, "w").write(nodes)
        open(ep, "w").write(edges)
        brain = FlyBrain(np_, ep)
        cur = np.zeros(brain.n)
        cur[0] = 3.0
        fired = 0
        for _ in range(200):
            fired += int(brain.step(cur).sum())
        print(f"нейронов: {brain.n}, спайков за 200 шагов: {fired}")
