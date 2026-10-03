"""Local tamper-evident provenance log (P3).

Threat model (honest scope):
- tamper-EVIDENT, not tamper-proof: edits are detectable through the hash chain and
  the head witness, but a writer who can rewrite BOTH files and recompute every hash
  wins — there is no external anchoring/registry here;
- appends are serialized with an in-process threading.Lock AND an OS-level
  fcntl.flock (POSIX), so concurrent processes cannot interleave or fork the chain;
  on non-POSIX platforms only the threading lock applies;
- append() is O(1) in the length of the log: the chain tail comes from the head
  witness file, and only bytes written past the witnessed offset are re-read (which
  is nothing, unless another process appended in the meantime). The whole file is
  never parsed per append — that version made an 86 400-episode confirmatory run
  quadratic and cost P4.2.v1 its abort (see var/aborted/p4_2_v1_infrastructure_abort);
- verify_chain() is the O(N) audit operation, meant for startups, recovery and
  reporting — never for the hot append path;
- a fresh reader with no witness and no memory of having written cannot prove that
  the log was never longer (a shorter, internally valid chain looks fine). The head
  witness closes that gap for files it has seen; deleting the witness restores the
  old limitation.
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

HEAD_VERSION = 1
HEAD_SUFFIX = ".head.json"


class ProvenanceIntegrityError(RuntimeError):
    """Журнал противоречит собственному witness: дописывать цепь нельзя."""


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
        self.head_path = self.path + HEAD_SUFFIX
        self._lock = threading.Lock()
        # Полный список событий существует только после явного сканирования
        # (get_events/verify_chain). В горячем пути append его нет намеренно:
        # держать 86k событий в памяти — это вторая половина той же квадратичной
        # истории (GC + рост resident set), а не только перечитывание файла.
        self._entries: list[ProvenanceEvent] = []
        self._scanned = False
        self._written = 0
        self._last_hash = ""

    # ------------------------------------------------------------------ locking
    @contextmanager
    def _file_lock(self, mode: str):
        """Open the log holding an exclusive OS-level lock when available.

        threading.Lock guards this process; fcntl.flock guards other processes
        writing the same file, so the read-head -> resync -> append -> fsync ->
        update-head sequence is atomic across processes on POSIX systems.
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

    # ------------------------------------------------------------------- parsing
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

    @staticmethod
    def _events_from(data: bytes) -> list[ProvenanceEvent]:
        events: list[ProvenanceEvent] = []
        for line in data.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            events.append(ProvenanceLog._parse_entry(json.loads(line)))
        return events

    def _read_disk_events(self) -> list[ProvenanceEvent]:
        """Полное чтение журнала с диска, под эксклюзивным локом. O(N)."""
        if not os.path.exists(self.path):
            return []
        with self._file_lock("rb") as f:
            return self._events_from(f.read())

    # ---------------------------------------------------------------- head file
    def read_head(self) -> dict | None:
        """Компактный witness цепи: {entry_count, last_hash, log_byte_offset}."""
        try:
            with open(self.head_path, encoding="utf-8") as fh:
                head = json.load(fh)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as exc:
            raise ProvenanceIntegrityError(f"witness {os.path.basename(self.head_path)} не читается: {exc}")
        for key in ("entry_count", "last_hash", "log_byte_offset"):
            if key not in head:
                raise ProvenanceIntegrityError(f"witness без поля {key}: {head}")
        return head

    def _write_head(self, entry_count: int, last_hash: str, offset: int) -> None:
        """Атомарная замена witness ПОСЛЕ того, как сам журнал уже fsync'нут.

        Порядок важен: падение между fsync журнала и заменой witness оставляет
        журнал длиннее witness — это дочитывается хвостовым сканом при следующем
        append. Обратный порядок (witness впереди журнала) означал бы, что
        восстановление приписало бы цепи событие, которого на диске нет.
        """
        payload = {
            "version": HEAD_VERSION,
            "log_file": os.path.basename(self.path),
            "entry_count": entry_count,
            "last_hash": last_hash,
            "log_byte_offset": offset,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self.head_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, separators=(",", ":"))
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.head_path)
        self._last_hash = last_hash
        d = os.path.dirname(self.head_path) or "."
        try:  # durable rename: best effort, не все FS это дают
            dir_fd = os.open(d, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    # ----------------------------------------------------------------- recovery
    def _resync_locked(self, f) -> tuple[int, str]:
        """(entry_count, previous_hash) для следующего события. Ожидает лок.

        С witness: дочитываются только байты после witnessed offset — обычно ноль
        байт. Без witness (первый запуск или legacy-файл до P4.2a): один проход по
        всему файлу, затем witness заводится.
        """
        size = f.seek(0, os.SEEK_END)
        head = self.read_head()
        if head is None:
            if size == 0:
                return 0, ""
            events = self._events_from(self._read_all(f))
            return len(events), events[-1].entry_hash
        if size < head["log_byte_offset"]:
            raise ProvenanceIntegrityError(
                f"журнал укорочен ({size} байт) против witness ({head['log_byte_offset']}) — "
                "записи удалены после того, как их видели; дописывать цепь нельзя")
        f.seek(head["log_byte_offset"])
        tail = f.read()
        if tail and not tail.endswith(b"\n"):
            raise ProvenanceIntegrityError(
                "хвост журнала обрывается посреди строки (порванная запись) — "
                "нужен осознанный ремонт, а не молчаливое дописывание")
        last_hash = head["last_hash"]
        count = head["entry_count"]
        for event in self._events_from(tail):
            if event.previous_hash != last_hash:
                raise ProvenanceIntegrityError(
                    "события, дописанные после witness, не продолжаются от его last_hash — "
                    "цепь уже сломана, дописывание её усугубит")
            last_hash, count = event.entry_hash, count + 1
        return count, last_hash

    @staticmethod
    def _read_all(f) -> bytes:
        f.seek(0)
        return f.read()

    # ------------------------------------------------------------------ public
    def append(self, payload: dict) -> ProvenanceEvent:
        with self._lock, self._file_lock("a+b") as f:
            count, previous_hash = self._resync_locked(f)
            event = ProvenanceEvent(
                event_id=str(uuid.uuid4()),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                previous_hash=previous_hash,
                payload=dict(payload),
            )
            line = (_canonical_json(event.to_log_entry()) + "\n").encode("utf-8")
            f.seek(0, os.SEEK_END)
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
            self._write_head(count + 1, event.entry_hash, f.tell())
            self._written += 1
            if self._scanned:
                self._entries.append(event)
            return event

    @property
    def last_hash(self) -> str:
        head = self.read_head()
        if head is not None:
            return head["last_hash"]
        events = self._read_disk_events()  # legacy-путь: witness ещё не заведён
        return events[-1].entry_hash if events else self._last_hash

    @property
    def count(self) -> int:
        if not os.path.exists(self.path):
            return 0
        with self._file_lock("rb") as f:
            count, _ = self._resync_locked(f)
        return count

    def get_events(self) -> list[ProvenanceEvent]:
        events = self._read_disk_events()
        self._entries = events
        self._scanned = True
        return list(events)

    def verify_chain(self) -> bool:
        """Полной стоимости аудит: перечитать, пересобрать цепь, сверить witness.

        Ловит: разрыв/перестановку цепи на диске, подмену payload или hash в памяти
        писавшего экземпляра, сокращение журнала против того, что записал этот
        экземпляр, и любое расхождение журнала с его witness (длина, счётчик, хэш,
        смещение). Не ловит: переписанный целиком файл, для которого честно
        пересчитали и журнал, и witness.
        """
        events = self._read_disk_events()
        if self._written and len(events) < self._written:
            return False  # журнал сжался ниже того, что мы в него записали
        if self._scanned:
            if len(events) < len(self._entries):
                return False
            for cached, on_disk in zip(self._entries, events):
                if cached.entry_hash != on_disk.entry_hash:
                    return False  # то, что помнится записанным, лежит не так
            if any(cached.payload_hash != _sha256(_canonical_json(cached.payload))
                   for cached in self._entries):
                return False  # payload в памяти подредактирован после хэширования
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
        try:
            head = self.read_head()
        except ProvenanceIntegrityError:
            # Нечитаемый witness — это провал проверки, а не падение: аудит обязан
            # возвращать False там, где свидетель повреждён или подправлен наживо.
            return False
        if head is None:
            return True  # witness ещё не заводился (legacy-файл): цепь сама по себе сходится
        tail_hash = events[-1].entry_hash if events else ""
        if head["last_hash"] != tail_hash:
            return False
        if head["entry_count"] != len(events):
            return False
        try:
            if head["log_byte_offset"] != os.path.getsize(self.path):
                return False
        except OSError:
            return False
        return True

    def clear(self):
        with self._lock:
            self._entries = []
            self._scanned = False
            self._written = 0
            self._last_hash = ""
            for path in (self.path, self.head_path, self.head_path + ".tmp"):
                if os.path.exists(path):
                    os.remove(path)
