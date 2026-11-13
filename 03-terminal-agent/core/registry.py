"""Registry — append-only журнал с SHA256 цепью (память агента)."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import time


@dataclass
class RegistryEntry:
    """Одна запись в реестре."""
    id: str
    timestamp: float
    action: str
    actor: str
    result: Optional[str]
    success: bool
    previous_hash: str
    hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        action: str,
        actor: str,
        result: Optional[str],
        success: bool,
        previous_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RegistryEntry":
        now = time.time()
        entry_id = f"{int(now * 1000)}:{action[:8]}"
        content = f"{entry_id}:{now}:{action}:{actor}:{result}:{success}:{previous_hash}"
        entry_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return cls(
            id=entry_id,
            timestamp=now,
            action=action,
            actor=actor,
            result=result,
            success=success,
            previous_hash=previous_hash,
            hash=entry_hash,
            metadata=metadata or {},
        )


class Registry:
    """Append-only журнал — память агента, нельзя удалить, только дописать."""

    def __init__(self, path: str = "./memory/registry.jsonl") -> None:
        self.path = path
        self.entries: List[RegistryEntry] = []
        self._last_hash = self._genesis_hash()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._load()

    def _genesis_hash(self) -> str:
        return hashlib.sha256(b"genesis:muha-registry").hexdigest()

    def _load(self) -> None:
        """Загружает существующие записи из файла."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    entry = RegistryEntry(**data)
                    self.entries.append(entry)
                    self._last_hash = entry.hash
        except (json.JSONDecodeError, OSError):
            self.entries = []
            self._last_hash = self._genesis_hash()

    def record(
        self,
        action: str,
        actor: str,
        result: Optional[str] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RegistryEntry:
        """Добавляет новую запись (append-only)."""
        entry = RegistryEntry.create(
            action=action,
            actor=actor,
            result=result,
            success=success,
            previous_hash=self._last_hash,
            metadata=metadata,
        )
        self.entries.append(entry)
        self._last_hash = entry.hash
        self._append_to_file(entry)
        return entry

    def _append_to_file(self, entry: RegistryEntry) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def verify_chain(self) -> bool:
        """Проверяет целостность SHA256 цепи."""
        expected = self._genesis_hash()
        for entry in self.entries:
            if entry.previous_hash != expected:
                return False
            # Пересчитываем hash для проверки
            content = f"{entry.id}:{entry.timestamp}:{entry.action}:{entry.actor}:{entry.result}:{entry.success}:{entry.previous_hash}"
            actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if actual_hash != entry.hash:
                return False
            expected = entry.hash
        return True

    def last_n(self, n: int = 10) -> List[RegistryEntry]:
        return self.entries[-n:]

    def search(self, query: str) -> List[RegistryEntry]:
        """Поиск по записям."""
        q = query.lower()
        return [
            e for e in self.entries
            if q in e.action.lower() or q in e.actor.lower()
            or (e.result and q in e.result.lower())
        ]

    def stats(self) -> Dict[str, Any]:
        total = len(self.entries)
        success = sum(1 for e in self.entries if e.success)
        return {
            "total_entries": total,
            "successful": success,
            "failed": total - success,
            "chain_valid": self.verify_chain(),
            "last_hash": self._last_hash[:16] + "...",
        }
