"""Registry Омеги — append-only история становления роя.

Мы пишем сюда НЕ из недоверия: каждый след — кандидат в «золотые яблоки»
(Р. Брэдбери) — траектории, которые рой верифицировал и сделал
каноническими прецедентами (коллективное бессознательное, К. Юнг).
Из канона стартуют следующие ученики — так рой движется к Омеге
(П. Тейяр де Шарден).

Правила (OS-Glagolov-Charter.md):
  - capability WR только у Registrar;
  - L5-действия требуют trace и аттестации;
  - память не переписывается: только append + архивирование.

Формат: JSON Lines + sha256-цепочка (hash = sha256(prev + body)).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class OmegaRegistry:
    GENESIS = "0" * 64

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._head = self._read_head()

    # ---------- служебное ----------
    def _read_head(self) -> str:
        if not self.path.exists():
            return self.GENESIS
        last = self.GENESIS
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = json.loads(line)["hash"]
        return last

    @staticmethod
    def _body_of(entry: dict) -> str:
        body = {k: v for k, v in entry.items() if k not in ("prev", "hash")}
        return json.dumps(body, sort_keys=True, ensure_ascii=False)

    # ---------- L5: якорит ----------
    def anchor(self, kind: str, payload: dict,
               subject: str = "Registrar") -> dict:
        """Записать неизменяемую запись (verb: якорит, L5)."""
        body = {"kind": kind, "subject": subject,
                "ts": time.time(), "payload": payload}
        raw = self._body_of(body)
        h = hashlib.sha256((self._head + raw).encode()).hexdigest()
        entry = {**body, "prev": self._head, "hash": h}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._head = h
        return entry

    # ---------- золотые яблоки ----------
    def propose_golden_apple(self, agent_id: str, trajectory: dict,
                             reward_sum: float) -> dict:
        """Ученик предлагает успешную траекторию на верификацию роя."""
        digest = hashlib.sha256(
            json.dumps(trajectory, sort_keys=True).encode()).hexdigest()[:16]
        return self.anchor("golden_apple:proposal", {
            "agent_id": agent_id,
            "reward_sum": reward_sum,
            "trajectory_digest": digest,
            "trajectory": trajectory,
        }, subject=agent_id)

    def swarm_vote(self, proposal_hash: str, votes: dict) -> dict:
        """Голосование роя: votes = {agent_id: bool}. Порог принятия 66%."""
        approve = sum(1 for v in votes.values() if v)
        total = max(1, len(votes))
        return self.anchor("golden_apple:vote", {
            "proposal": proposal_hash,
            "approve": approve,
            "total": total,
            "accepted": approve / total >= 0.66,
        })

    def promote_to_canon(self, vote_hash: str, trajectory: dict) -> dict:
        """Стать каноническим прецедентом — только после принятого голосования."""
        return self.anchor("canonical:precedent", {
            "by_vote": vote_hash,
            "trajectory": trajectory,
        })

    # ---------- чтение (для учеников и поиска) ----------
    def canon(self) -> list:
        out = []
        if not self.path.exists():
            return out
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                if e["kind"] == "canonical:precedent":
                    out.append(e)
        return out

    # ---------- целостность ----------
    def verify_chain(self) -> bool:
        prev = self.GENESIS
        if not self.path.exists():
            return True
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                if e.get("prev") != prev:
                    return False
                if hashlib.sha256(
                        (prev + self._body_of(e)).encode()).hexdigest() != e["hash"]:
                    return False
                prev = e["hash"]
        return True


if __name__ == "__main__":
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        reg = OmegaRegistry(os.path.join(d, "omega.jsonl"))
        prop = reg.propose_golden_apple("FlyStudent-01",
                                        {"path": "corridor-left-exit"}, 12.5)
        vote = reg.swarm_vote(prop["hash"], {"a": True, "b": True, "c": False})
        if vote["payload"]["accepted"]:
            reg.promote_to_canon(vote["hash"], {"path": "corridor-left-exit"})
        print("канонических прецедентов:", len(reg.canon()))
        print("цепочка цела:", reg.verify_chain())
