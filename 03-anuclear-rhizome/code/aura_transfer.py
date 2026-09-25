"""Перенос «ауры» (Пирс Энтони, «Кластер»).

Аура = скомпилированная личность агента:
  политика + capability + дайджест весов + золотые яблоки +
  гормональный профиль.

Аура передаётся Учителем ученику (или анаклеарной оболочке)
и ПОДПИСЫВАЕТСЯ Registrar'ом, чтобы ученик не получил capability,
которых у него нет (OS-Glagolov-Charter.md п.6: WR не передаётся НИКОГДА).

Перенос ауры — это verb-act Registrar'а «версирует» (L5) с аттестацией.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict


@dataclass
class Aura:
    identity: str
    capability_set: set
    policy_version: str = "os-glagolov/1.0"
    weights_digest: str = ""
    golden_apples: list = field(default_factory=list)
    hormone_profile: dict = field(default_factory=dict)
    teacher_signature: str = ""
    registrar_signature: str = ""
    created_ts: float = field(default_factory=time.time)

    # ---------- сериализация ----------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["capability_set"] = sorted(self.capability_set)
        return d

    def digest(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def sign(self, key: str) -> str:
        return hashlib.sha256((self.digest() + "|" + key).encode()).hexdigest()


def compile_aura(agent) -> Aura:
    """Собрать ауру из живого агента (учителя/выпускника)."""
    w = getattr(agent, "W", None)
    digest = ""
    if w is not None:
        import numpy as np
        digest = hashlib.sha256(
            np.round(np.asarray(w), 6).tobytes()).hexdigest()[:32]
    return Aura(
        identity=agent.agent_id,
        capability_set=set(agent.capabilities),
        weights_digest=digest,
        golden_apples=list(getattr(agent, "golden_apples", [])),
        hormone_profile=dict(getattr(agent, "hormone_state", {})),
    )


def transfer_aura(aura: Aura, registrar_key: str,
                  student_capabilities: set) -> dict:
    """Перенос ауры в оболочку ученика.

    Правила (НЕ ТРОГАТЬ):
      1) WR не передаётся никогда;
      2) capability ученика ⊆ capability учителя;
      3) аура должна быть подписана Registrar'ом.
    """
    granted = set(aura.capability_set) & set(student_capabilities)
    granted.discard("WR")

    if not granted <= set(aura.capability_set):
        return {"ok": False, "reason": "нарушение правила подмножества"}

    aura.registrar_signature = aura.sign(registrar_key)

    verb_act = {
        "subject": "Registrar",
        "role": "Registrar",
        "verb": "версирует",
        "object": f"aura:{aura.identity}",
        "domain": "registry",
        "context": {"mode": "aura_transfer", "risk_level": "medium"},
        "result": {"granted_capabilities": sorted(granted),
                   "aura_digest": aura.digest()},
        "trace": {"why": "teacher_graduation",
                  "policy_version": aura.policy_version},
    }
    return {"ok": True, "aura": aura, "granted": granted,
            "registry_verb_act": verb_act}


def receive_aura(shell, aura: Aura, granted: set) -> None:
    """Анаклеарная оболочка принимает ауру (встраивание в экосистему)."""
    if not aura.registrar_signature:
        raise PermissionError("аура без подписи Registrar не принимается")
    shell.capabilities = set(granted)
    shell.golden_apples = list(aura.golden_apples)
    shell.hormone_state = dict(aura.hormone_profile)
    shell.policy_version = aura.policy_version
    shell.inherited_from = aura.identity
