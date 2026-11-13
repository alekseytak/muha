"""Инструменты файловой системы — реальный ввод-вывод.

Все операции ограничены корнем (`sandbox.root`): путь, выходящий за пределы
корня, отвергается. Запись выполняется атомарно (temp-файл + rename).
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class ToolError(Exception):
    """Ошибка инструмента: сообщение пригодно для показа пользователю."""


@dataclass
class FsSandbox:
    """Корень и лимиты файловых операций."""

    root: Path
    max_read_bytes: int = 65536
    max_write_bytes: int = 262144

    def resolve(self, raw_path: str, *, must_exist: bool = False) -> Path:
        """Возвращает путь внутри root или бросает ToolError."""
        if not raw_path or not str(raw_path).strip():
            raise ToolError("пустой путь")
        candidate = Path(os.path.expanduser(str(raw_path).strip()))
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except OSError as exc:  # битый symlink и т.п.
            raise ToolError(f"не удалось разобрать путь {raw_path!r}: {exc}") from exc

        root = self.root.resolve(strict=False)
        if resolved != root and root not in resolved.parents:
            raise ToolError(
                f"путь вне песочницы: {raw_path!r} (корень: {root})"
            )
        if must_exist and not resolved.exists():
            raise ToolError(f"файл не найден: {raw_path}")
        return resolved


def _read_text(path: Path, limit: int) -> tuple[str, bool]:
    """Читает текст с ограничением по размеру. Возвращает (текст, обрезано)."""
    size = path.stat().st_size
    raw = path.read_bytes()
    truncated = False
    if size > limit:
        raw = raw[:limit]
        truncated = True
    for encoding in ("utf-8", "cp1251"):
        try:
            return raw.decode(encoding), truncated
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), truncated


def read_file(path: str, sandbox: FsSandbox) -> str:
    """Читает файл. Возвращает содержимое с шапкой-метаданными."""
    target = sandbox.resolve(path, must_exist=True)
    if target.is_dir():
        raise ToolError(f"это каталог, а не файл: {path} (используйте list)")
    text, truncated = _read_text(target, sandbox.max_read_bytes)
    lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    head = f"[{target.name}] {target.stat().st_size} байт, {lines} строк"
    if truncated:
        head += f", показаны первые {sandbox.max_read_bytes} байт"
    return f"{head}\n{text}"


def write_file(path: str, content: str, sandbox: FsSandbox) -> str:
    """Атомарно пишет текст в файл внутри песочницы."""
    if content is None:
        raise ToolError("не передано содержимое (parameters.content)")
    data = str(content).encode("utf-8")
    if len(data) > sandbox.max_write_bytes:
        raise ToolError(
            f"слишком большой объём записи: {len(data)} > {sandbox.max_write_bytes}"
        )
    target = sandbox.resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".muha-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    verb = "перезаписан" if existed else "создан"
    return f"{verb}: {target} ({len(data)} байт)"


def list_dir(path: str, sandbox: FsSandbox, limit: int = 200) -> str:
    """Список содержимого каталога."""
    target = sandbox.resolve(path or ".", must_exist=True)
    if not target.is_dir():
        raise ToolError(f"это файл, а не каталог: {path}")
    entries: Iterable[Path] = sorted(
        target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
    )
    lines = []
    for item in list(entries)[:limit]:
        try:
            size = item.stat().st_size if item.is_file() else 0
        except OSError:
            size = -1
        mark = "DIR " if item.is_dir() else "FILE"
        size_txt = "" if item.is_dir() else f"{size:>9} байт  "
        lines.append(f"{mark}  {size_txt}{item.name}")
    if not lines:
        return f"{target}: (пусто)"
    if len(lines) == limit:
        lines.append(f"... (показано {limit}, есть ещё)")
    return f"{target}:\n" + "\n".join(lines)


def stat_file(path: str, sandbox: FsSandbox) -> str:
    """Метаданные файла/каталога."""
    target = sandbox.resolve(path, must_exist=True)
    st = target.stat()
    kind = "каталог" if target.is_dir() else "файл"
    return (
        f"{target}\n"
        f"тип: {kind}\n"
        f"размер: {st.st_size} байт\n"
        f"изменён: {st.st_mtime:.0f} (unix)\n"
        f"права: {oct(st.st_mode & 0o777)}"
    )
