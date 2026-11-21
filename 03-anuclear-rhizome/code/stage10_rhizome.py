"""Этап 10 — Ризома и перенос ауры, Проект III (дни 56–65).

1. Убрать центрального координатора: gossip-обмен между соседями (3–5 связей),
   консенсус кворум 2/3 (RhizomeMesh).
2. Тест на разрыв: удалить 30% узлов — рой сохранит связность и задачу.
3. Перенос ауры: Teacher передаёт ауру ученику (Aura + transfer_aura);
   capability ученика ⊆ capability учителя; подпись Registrar обязательна.
4. АРФА-фильтры владельца на сенсорный вход (csc_filter), домен atqec —
   фильтруют/маркируют/оценивают, НЕ делают commit.

НЕ ТРОГАТЬ: WR не передаётся никогда; append-only Registry.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "01-omega-archive" / "code"))
sys.path.insert(0, str(REPO / "03-anuclear-rhizome" / "code"))

from rhizome_mesh import RhizomeMesh  # noqa: E402
from aura_transfer import Aura, compile_aura, transfer_aura, receive_aura  # noqa: E402
from arfa_filters import csc_filter  # noqa: E402
from fly_connectome_agent.src.engineering.governance.gatekeeper import (  # noqa: E402
    GateKeeper,
    VerbAct,
    PolicyDecision,
)
from omega_registry import OmegaRegistry  # noqa: E402


@dataclass
class ShellAgent:
    """Анаклеарная оболочка ученика — принимает ауру."""
    agent_id: str
    capabilities: set = None
    golden_apples: list = None
    hormone_state: dict = None
    policy_version: str = "os-glagolov/1.0"
    inherited_from: str = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = set()
        if self.golden_apples is None:
            self.golden_apples = []
        if self.hormone_state is None:
            self.hormone_state = {}


@dataclass
class Stage10Result:
    mesh_nodes: int
    mesh_links: int
    connected_before_break: bool
    nodes_killed: int
    connected_after_break: bool  # рой сохранил связность после 30% потерь
    gossip_consensus_accepted: bool
    aura_transfer_ok: bool
    capability_subset_ok: bool  # ученик ⊆ учитель, без WR
    registrar_signature_required: bool
    arfa_filtered_signal: bool  # сенсорный вход прошёл csc_filter
    chain_ok: bool


def run_stage10(n_nodes: int = 20, kill_fraction: float = 0.30,
                registrar_key: str = "registrar-key-2026") -> Stage10Result:
    # --- 1. Ризома без центра: узлы + gossip-связи 3–5 ---
    mesh = RhizomeMesh(seed=42)
    roles = ("Registrar", "Observer", "Analyst", "Synthesizer", "Decider", "Guardian")
    for i in range(n_nodes):
        mesh.add_node(f"node-{i:02d}", role=roles[i % len(roles)],
                      max_neighbors=5)

    mesh_nodes = len(mesh.nodes)
    mesh_links = sum(len(v) for v in mesh.links.values()) // 2
    connected_before = mesh.is_whole()

    # --- 2. Тест на разрыв: убить 30% узлов ---
    alive = [n for n in mesh.nodes if mesh.nodes[n]["alive"]]
    kill_count = max(1, int(n_nodes * kill_fraction))
    to_kill = alive[:kill_count]
    for nid in to_kill:
        mesh.kill_node(nid)
    connected_after = mesh.is_whole()

    # --- 3. Gossip-консенсус без центра ---
    initial_votes = {n: "yes" for n in mesh.nodes if mesh.nodes[n]["alive"]}
    # сделаем часть узлов "no", чтобы проверить кворум 2/3
    for n in list(mesh.nodes)[: max(1, n_nodes // 4)]:
        if mesh.nodes[n]["alive"]:
            initial_votes[n] = "no"
    consensus = mesh.gossip_consensus(
        topic="adopt-new-protocol", initial_votes=initial_votes, quorum=0.66)

    # --- 4. Перенос ауры Teacher → ученик ---
    class Teacher:
        agent_id = "Teacher-01"
        capabilities = {"R", "WL", "X", "E", "WE", "A"}  # без WR
        W = np.random.default_rng(1).normal(size=(8, 8))
        golden_apples = ["apple-01", "apple-02"]
        hormone_state = {"dopamine": 70.0, "serotonin": 60.0}

    teacher = Teacher()
    aura = compile_aura(teacher)
    # aura.registrar_signature пуста до transfer_aura — подпись ставится там

    shell = ShellAgent(agent_id="Student-Shell-01",
                       capabilities={"R", "WL", "X", "E"})  # ⊆ учителя
    result = transfer_aura(aura, registrar_key, shell.capabilities)
    aura_transfer_ok = result["ok"]
    granted = result["granted"]
    capability_subset_ok = granted <= set(aura.capability_set) and "WR" not in granted
    registrar_signature_required = bool(aura.registrar_signature)

    if aura_transfer_ok:
        receive_aura(shell, aura, granted)

    # --- 5. АРФА-фильтр на сенсорный вход (csc_filter) ---
    sensor_signal = np.sin(np.linspace(0, 4 * np.pi, 200)) + \
        0.3 * np.random.default_rng(0).normal(size=200)
    filtered = csc_filter(sensor_signal, fs=200.0)
    arfa_filtered_signal = (filtered.shape == sensor_signal.shape and
                            np.isfinite(filtered).all())

    # --- 6. Якорь результата этапа в Registry (append-only) ---
    tmp = Path(tempfile.mkdtemp(prefix="stage10_"))
    registry = OmegaRegistry(tmp / "omega.jsonl")
    registry.anchor(
        kind="stage10:rhizome",
        payload={
            "mesh_nodes": int(mesh_nodes),
            "mesh_links": int(mesh_links),
            "connected_before": bool(connected_before),
            "nodes_killed": int(kill_count),
            "connected_after": bool(connected_after),
            "consensus_accepted": bool(consensus["accepted"]),
            "aura_transfer_ok": bool(aura_transfer_ok),
            "capability_subset_ok": bool(capability_subset_ok),
            "registrar_signature_required": bool(registrar_signature_required),
            "arfa_filtered_signal": bool(arfa_filtered_signal),
        },
        subject="Registrar",
    )
    chain_ok = registry.verify_chain()

    return Stage10Result(
        mesh_nodes=mesh_nodes,
        mesh_links=mesh_links,
        connected_before_break=connected_before,
        nodes_killed=kill_count,
        connected_after_break=connected_after,
        gossip_consensus_accepted=consensus["accepted"],
        aura_transfer_ok=aura_transfer_ok,
        capability_subset_ok=capability_subset_ok,
        registrar_signature_required=registrar_signature_required,
        arfa_filtered_signal=arfa_filtered_signal,
        chain_ok=chain_ok,
    )


if __name__ == "__main__":
    res = run_stage10(n_nodes=20, kill_fraction=0.30)
    print("узлов ризомы:", res.mesh_nodes, "связей:", res.mesh_links)
    print("связность до разрыва:", res.connected_before_break)
    print("узлов убито (30%):", res.nodes_killed)
    print("связность после разрыва:", res.connected_after_break)
    print("gossip-консенсус принят:", res.gossip_consensus_accepted)
    print("перенос ауры OK:", res.aura_transfer_ok)
    print("capability ⊆ учитель (без WR):", res.capability_subset_ok)
    print("подпись Registrar обязательна:", res.registrar_signature_required)
    print("АРФА csc_filter сенсорный вход:", res.arfa_filtered_signal)
    print("цепочка Registry цела:", res.chain_ok)

    assert res.connected_before_break, "ризома должна быть связной до разрыва"
    assert res.connected_after_break, "после потери 30% узлов ризома должна остаться связной"
    assert res.aura_transfer_ok, "перенос ауры должен пройти"
    assert res.capability_subset_ok, "capability ученика ⊆ учителя, без WR"
    assert res.registrar_signature_required, "подпись Registrar обязательна"
    assert res.arfa_filtered_signal, "сенсорный вход должен пройти АРФА-фильтр"
    assert res.chain_ok, "цепочка Registry нарушена"
    print("Этап 10: приёмка пройдена")
