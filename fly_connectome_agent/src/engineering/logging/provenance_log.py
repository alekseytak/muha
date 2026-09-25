"""Local tamper-evident provenance log (P3)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone


def _canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ProvenanceEvent:
    event_id: str
    timestamp_utc: str
    previous_hash: str
    payload: dict
    payload_hash: str = ""
    entry_hash: str = ""

    def __post_init__(self):
        if not self.payload_hash:
            object.__setattr__(self, "payload_hash", _sha256(_canonical_json(self.payload)))
        if not self.entry_hash:
            chain_input = self.previous_hash + ":" + self.payload_hash
            object.__setattr__(self, "entry_hash", _sha256(chain_input))

    def to_log_entry(self) -> dict:
        return {"event_id": self.event_id, "timestamp_utc": self.timestamp_utc, "previous_hash": self.previous_hash, "payload": self.payload, "payload_hash": self.payload_hash, "entry_hash": self.entry_hash}


class ProvenanceLog:
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(tempfile.gettempdir(), "p3_provenance.jsonl")
        self._lock = threading.Lock()
        self._entries: list[ProvenanceEvent] = []
        self._loaded = False
        self._last_hash = ""

    def _load_if_needed(self):
        if self._loaded:
            return
        self._loaded = True
        self._entries = []
        self._last_hash = ""
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    event = ProvenanceEvent(event_id=entry["event_id"], timestamp_utc=entry["timestamp_utc"], previous_hash=entry["previous_hash"], payload=entry["payload"], payload_hash=entry["payload_hash"], entry_hash=entry["entry_hash"])
                    self._entries.append(event)
                    self._last_hash = event.entry_hash

    @property
    def last_hash(self) -> str:
        self._load_if_needed()
        return self._last_hash

    @property
    def count(self) -> int:
        self._load_if_needed()
        return len(self._entries)

    def append(self, payload: dict) -> ProvenanceEvent:
        self._load_if_needed()
        with self._lock:
            event = ProvenanceEvent(event_id=str(uuid.uuid4()), timestamp_utc=datetime.now(timezone.utc).isoformat(), previous_hash=self._last_hash, payload=dict(payload))
            entry_json = _canonical_json(event.to_log_entry())
            with open(self.path, "a") as f:
                f.write(entry_json + "\n")
            self._entries.append(event)
            self._last_hash = event.entry_hash
            return event

    def verify_chain(self) -> bool:
        self._load_if_needed()
        expected_prev = ""
        for event in self._entries:
            if event.previous_hash != expected_prev:
                return False
            if event.payload_hash != _sha256(_canonical_json(event.payload)):
                return False
            chain_input = event.previous_hash + ":" + event.payload_hash
            if event.entry_hash != _sha256(chain_input):
                return False
            expected_prev = event.entry_hash
        return True

    def get_events(self) -> list[ProvenanceEvent]:
        self._load_if_needed()
        return list(self._entries)

    def clear(self):
        with self._lock:
            self._entries = []
            self._last_hash = ""
            self._loaded = True
            if os.path.exists(self.path):
                os.remove(self.path)
