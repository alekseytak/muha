"""Очередь задач для автономного режима.

Хранение — append-only JSONL: каждая строка это снимок задачи целиком,
актуальным считается последний снимок с данным `id`. Так файл можно читать
после падения процесса и не терять историю изменений: строки только
добавляются, ничего не переписывается на месте.

Почему не «одна строка на задачу с перезаписью файла»: перезапись рвётся
посреди записи, и очередь остаётся пустой. Дописывание одной строки — нет.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PENDING = "pending"
DONE = "done"
FAILED = "failed"
BLOCKED = "blocked"

VALID_STATUSES = (PENDING, DONE, FAILED, BLOCKED)


@dataclass
class Task:
    """Одна задача автономного цикла."""

    text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = PENDING
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    steps: int = 0
    result: str = ""
    stop_reason: str = ""
    llm_calls: int = 0
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    @property
    def is_open(self) -> bool:
        return self.status == PENDING


class TaskQueue:
    """Очередь задач с сохранением на диск."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(os.path.abspath(str(path)))
        self._tasks: Dict[str, Task] = {}
        self._order: List[str] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load()

    # ---------- чтение ----------

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # обрезанная при падении строка не должна ломать всю очередь
            task_id = data.get("id")
            if not task_id:
                continue
            if task_id not in self._tasks:
                self._order.append(task_id)
            self._tasks[task_id] = Task.from_dict(data)

    def all(self) -> List[Task]:
        return [self._tasks[tid] for tid in self._order]

    def by_status(self, status: str) -> List[Task]:
        return [t for t in self.all() if t.status == status]

    def pending(self) -> Optional[Task]:
        """Следующая задача в работе: самая старая незакрытая."""
        for task in self.all():
            if task.is_open:
                return task
        return None

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def stats(self) -> Dict[str, int]:
        counts = {status: 0 for status in VALID_STATUSES}
        for task in self.all():
            counts[task.status] = counts.get(task.status, 0) + 1
        counts["total"] = len(self._tasks)
        return counts

    # ---------- запись ----------

    def _append(self, task: Task) -> None:
        task.updated = time.time()
        line = json.dumps(task.to_dict(), ensure_ascii=False)
        # Дописываем строку целиком за один вызов: снимок либо есть, либо нет.
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if task.id not in self._tasks:
            self._order.append(task.id)
        self._tasks[task.id] = task

    def add(self, text: str) -> Task:
        text = (text or "").strip()
        if not text:
            raise ValueError("пустая задача: нужен текст")
        task = Task(text=text)
        self._append(task)
        return task

    def save(self, task: Task) -> Task:
        if task.status not in VALID_STATUSES:
            raise ValueError(f"неизвестный статус задачи: {task.status!r}")
        self._append(task)
        return task

    def close(self, task: Task, status: str, result: str = "", stop_reason: str = "") -> Task:
        task.status = status
        task.result = result[:2000]
        task.stop_reason = stop_reason
        return self.save(task)

    def reset(self, task_id: str) -> Optional[Task]:
        """Вернуть задачу в очередь (например, после ручного разбора)."""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.status = PENDING
        task.result = ""
        task.stop_reason = ""
        return self.save(task)