"""Local tamper-evident provenance log (P3).

Threat model (honest scope):
- tamper-EVIDENT, not tamper-proof: edits to the file are detectable via the
  hash chain, but nothing stops a writer who can also recompute hashes;
- appends are serialized with an in-process threading.Lock AND an OS-level
  fcntl.flock (POSIX), so concurrent processes cannot interleave/corrupt the
  chain; on non-POSIX platforms only the threading lock applies;
- verify_chain() re-reads the file from disk (no cached-trust) and also fails
  if the log shrank below what this instance has already written;
- there is no external anchoring/Registry — an attacker with full disk access
  and the ability to rewrite the whole file is out of scope.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

try:  # POSIX file locking gives cross-process safety; degrade gracefully elsewhere.
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None
    _HAVE_FCNTL = False


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
        for event in self._read_disk_events():
            self._entries.append(event)
            self._last_hash = event.entry_hash

    @contextmanager
    def _file_lock(self, mode: str):
        """Open the log holding an exclusive OS-level lock when available.

        threading.Lock guards this process; fcntl.flock guards other processes
        writing the same file, so the read-hash -> append -> fsync sequence in
        append() is atomic across processes on POSIX systems.
        """
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.path, mode) as f:
            if _HAVE_FCNTL:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield f
            finally:
                if _HAVE_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _parse_entry(entry: dict) -> ProvenanceEvent:
        return ProvenanceEvent(
            event_id=entry["event_id"],
            timestamp_utc=entry["timestamp_utc"],
            previous_hash=entry["previous_hash"],
            payload=entry["payload"],
            payload_hash=entry["payload_hash"],
            entry_hash=entry["entry_hash"],
        )

    def _read_disk_events(self) -> list[ProvenanceEvent]:
        """Parse the on-disk log fresh, under the file lock."""
        if not os.path.exists(self.path):
            return []
        events: list[ProvenanceEvent] = []
        with self._file_lock("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(self._parse_entry(json.loads(line)))
        return events

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
        with self._lock, self._file_lock("a+") as f:
            # The chain tail is read from the file while the exclusive lock is
            # held, so a second process appending concurrently cannot fork it.
            disk_events = self._read_disk_events_unlocked(f)
            previous_hash = disk_events[-1].entry_hash if disk_events else ""
            # Resync with anything other processes wrote before appending ours.
            self._entries = list(disk_events)
            event = ProvenanceEvent(event_id=str(uuid.uuid4()), timestamp_utc=datetime.now(timezone.utc).isoformat(), previous_hash=previous_hash, payload=dict(payload))
            f.write(_canonical_json(event.to_log_entry()) + "\n")
            f.flush()
            os.fsync(f.fileno())
            self._entries.append(event)
            self._last_hash = event.entry_hash
            return event

    @staticmethod
    def _read_disk_events_unlocked(f) -> list[ProvenanceEvent]:
        f.seek(0)
        events: list[ProvenanceEvent] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(ProvenanceLog._parse_entry(json.loads(line)))
        return events

    def verify_chain(self) -> bool:
        """Re-verify the chain against the bytes on disk, not the memory cache.

        Three failure modes are caught: a broken/reordered hash chain on disk,
        a log that shrank below what this instance already wrote, and a cache
        that drifted from disk (someone edited an event object in memory).
        """
        events = self._read_disk_events()
        if self._loaded and len(events) < len(self._entries):
            return False  # log shrank: entries deleted after we wrote them
        if self._loaded and len(events) >= len(self._entries):
            for cached, on_disk in zip(self._entries, events):
                if cached.entry_hash != on_disk.entry_hash:
                    return False  # what we remember writing is not what is stored
            if any(
                cached.payload_hash != _sha256(_canonical_json(cached.payload))
                for cached in self._entries
            ):
                return False  # in-memory payload edited after it was hashed
        expected_prev = ""
        for event in events:
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
