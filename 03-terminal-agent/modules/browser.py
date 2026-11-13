"""Рендер страниц настоящим браузером через external-бинарь (Obscura).

Зачем: urllib получает HTML как есть, а страницы на React/Vue отдают пустой
каркас — текст появляется только после исполнения JS. Scrapling умеет это
через свои fetchers, но они тянут Chromium на сотни мегабайт; у нас уже есть
лёгкий бинарь Obscura, который считает то же самое.

Почему не subprocess: в этом окружении `import subprocess` виснет в dlopen
(см. AUDIT.md), поэтому запуск идёт через os.posix_spawn — без shell, только
argv-список, поэтому подстановка аргументов невозможна.
"""

import os
import select
import signal
import time
from typing import List, Optional, Tuple

DEFAULT_BINARY = os.path.expanduser("~/obscura-x86_64-macos/obscura")
DEFAULT_TIMEOUT = 30
MAX_OUTPUT = 4_000_000
# Обрезаем окружение: браузеру не нужны наши секреты.
SAFE_ENV_KEYS = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")


class BrowserUnavailable(RuntimeError):
    """Браузер недоступен: бинаря нет, он не исполняемый или упал."""


def binary_path(explicit: Optional[str] = None) -> str:
    return os.path.expanduser(explicit or os.environ.get("OBSCURA_BIN") or DEFAULT_BINARY)


def is_available(explicit: Optional[str] = None) -> bool:
    path = binary_path(explicit)
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _spawn_argv(path: str, argv: List[str], timeout: int) -> Tuple[int, bytes, bytes]:
    """Запускает бинарь и собирает stdout/stderr с жёстким дедлайном."""
    env = {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ}
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    try:
        pid = os.posix_spawn(
            path,
            [path] + argv,
            env,
            file_actions=[
                (os.POSIX_SPAWN_DUP2, out_w, 1),
                (os.POSIX_SPAWN_DUP2, err_w, 2),
            ],
        )
    except OSError as exc:
        os.close(out_r), os.close(out_w), os.close(err_r), os.close(err_w)
        raise BrowserUnavailable(f"не удалось запустить браузер {path}: {exc}") from exc

    os.close(out_w)
    os.close(err_w)
    out_chunks: List[bytes] = []
    err_chunks: List[bytes] = []
    total = 0
    deadline = time.monotonic() + timeout
    open_reads = {out_r: out_chunks, err_r: err_chunks}
    try:
        while open_reads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.kill(pid, signal.SIGKILL)
                _, status = os.waitpid(pid, 0)
                raise BrowserUnavailable(
                    f"браузер не уложился в {timeout}s, убит (статус {status})"
                )
            ready, _, _ = select.select(list(open_reads), [], [], min(remaining, 0.5))
            for fd in ready:
                block = os.read(fd, 65536)
                if not block:
                    os.close(fd)
                    open_reads.pop(fd, None)
                    continue
                bucket = open_reads[fd]
                if bucket is out_chunks:
                    total += len(block)
                    if total > MAX_OUTPUT:
                        os.kill(pid, signal.SIGKILL)
                        raise BrowserUnavailable(
                            f"страница больше предела {MAX_OUTPUT} байт"
                        )
                bucket.append(block)
        _, status = os.waitpid(pid, 0)
    finally:
        for fd in list(open_reads):
            try:
                os.close(fd)
            except OSError:
                pass
        for fd in (out_r, err_r):
            try:
                os.close(fd)
            except OSError:
                pass
    return status, b"".join(out_chunks), b"".join(err_chunks)


def render(url: str, timeout: int = DEFAULT_TIMEOUT, fmt: str = "text",
           explicit_binary: Optional[str] = None) -> Tuple[str, dict]:
    """Возвращает (текст страницы после JS, метаданные).

    fmt: text | markdown | html | links — что именно попросить у браузера.
    Вызывающий обязан сам проверить URL (SSRF) до этой функции.
    """
    path = binary_path(explicit_binary)
    if not is_available(path):
        raise BrowserUnavailable(f"браузер не найден или не исполняемый: {path}")
    argv = ["fetch", "--dump", fmt, url]
    status, out, err = _spawn_argv(path, argv, timeout)
    text = out.decode("utf-8", "replace")
    meta = {
        "backend": "obscura",
        "binary": path,
        "exit_status": status,
        "bytes": len(out),
        "stderr": err.decode("utf-8", "replace")[:400],
        "format": fmt,
    }
    if status != 0:
        raise BrowserUnavailable(
            f"браузер завершился со статусом {status}: {meta['stderr'][:200] or 'без вывода'}"
        )
    return text, meta