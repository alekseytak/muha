"""Ризома (Ж. Делез, Ф. Гваттари) — меш роя без центра.

Свойства ризомы (применительно к рою):
  1. связь: любой узел может связаться с любым (через цепочку);
  2. гетерогенность: разные роли агентов в одной сети;
  3. множественность: нет единственного центра;
  4. разрыв: сеть восстанавливается после потери узлов;
  5. картография: карта роя строится из локальных следов, а не копии.

Консенсус: gossip-протокол (горизонтально), кворум 2/3.
Единственная вертикаль — Registrar (см. rhizome_policy.yaml).
"""
from __future__ import annotations

import random
from collections import deque


class RhizomeMesh:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.nodes: dict = {}      # node_id -> {"role", "alive"}
        self.links: dict = {}      # node_id -> set(node_id)

    # ---------- рост ризомы ----------
    def add_node(self, node_id: str, role: str = "Observer",
                 max_neighbors: int = 5) -> None:
        self.nodes[node_id] = {"role": role, "alive": True}
        self.links.setdefault(node_id, set())
        alive = [n for n in self.nodes
                 if n != node_id and self.nodes[n]["alive"]]
        self.rng.shuffle(alive)
        for other in alive[:max_neighbors]:
            self.links[node_id].add(other)
            self.links[other].add(node_id)

    def kill_node(self, node_id: str) -> None:
        """Принцип разрыва: узел умирает, связи перестраиваются."""
        if node_id not in self.nodes:
            return
        self.nodes[node_id]["alive"] = False
        former = list(self.links[node_id])
        for other in former:
            self.links[other].discard(node_id)
        self.links[node_id] = set()
        self._heal(former)

    def _heal(self, affected: list) -> None:
        """Исцеление: одинокие узлы reconnect к живым соседям соседей."""
        alive = [n for n in self.nodes if self.nodes[n]["alive"]]
        for node in affected:
            if not self.nodes.get(node, {}).get("alive"):
                continue
            if len(self.links[node]) >= 2:
                continue
            candidates = set()
            for nb in self.links[node]:
                candidates |= self.links[nb]
            candidates -= {node}
            candidates = {c for c in candidates
                          if self.nodes[c]["alive"]} or set(alive) - {node}
            for cand in list(candidates)[:2]:
                self.links[node].add(cand)
                self.links[cand].add(node)

    # ---------- связность ----------
    def connected_components(self) -> list:
        alive = {n for n in self.nodes if self.nodes[n]["alive"]}
        seen, comps = set(), []
        for start in alive:
            if start in seen:
                continue
            comp, q = set(), deque([start])
            while q:
                cur = q.popleft()
                if cur in seen:
                    continue
                seen.add(cur)
                comp.add(cur)
                q.extend(self.links[cur] - seen)
            comps.append(comp)
        return comps

    def is_whole(self) -> bool:
        return len(self.connected_components()) <= 1

    # ---------- gossip-консенсус ----------
    def gossip_consensus(self, topic: str, initial_votes: dict,
                         quorum: float = 0.66,
                         max_rounds: int = 10) -> dict:
        """Горизонтальное голосование без центра.

        initial_votes: {node_id: "yes"/"no"}. Мнение расходится по соседям;
        узел меняет мнение, если локальное большинство против.
        """
        opinions = dict(initial_votes)
        alive = [n for n in self.nodes if self.nodes[n]["alive"]]
        for _ in range(max_rounds):
            changed = False
            self.rng.shuffle(alive)
            for node in alive:
                if node not in opinions:
                    continue
                neighbors = [nb for nb in self.links[node]
                             if nb in opinions]
                if not neighbors:
                    continue
                yes = sum(1 for nb in neighbors
                          if opinions[nb] == "yes")
                local_yes = (yes + (opinions[node] == "yes")) / (
                    len(neighbors) + 1)
                new = "yes" if local_yes >= 0.5 else "no"
                if new != opinions[node]:
                    opinions[node] = new
                    changed = True
            if not changed:
                break
        yes_total = sum(1 for n in alive if opinions.get(n) == "yes")
        ratio = yes_total / max(1, len(alive))
        return {
            "topic": topic,
            "accepted": ratio >= quorum,
            "yes_ratio": round(ratio, 3),
            "rounds_used": _,
            "opinions": opinions,
        }
