"""Инструменты системы — безопасное исполнение команд.

Модель безопасности:
  * `shell=False` — шелл не участвует в запуске;
  * команда разбирается через `shlex`, метасимволы шелла запрещены;
  * первое слово обязано быть в allowlist, иначе отказ;
  * запрещённые подстроки (denied_patterns) режутся до запуска;
  * жёсткий таймаут, вывод обрезается.

Два бэкенда запуска:

  * `posix` (по умолчанию) — `os.fork` + `os.execvp` + неблокирующие pipe.
    Не импортирует `subprocess`/`_posixsubprocess`, поэтому работает даже там,
    где загрузка этого C-расширения зависает в `dlopen` (держится GIL, и
    `join(timeout)` в потоке не спасает — встаёт весь интерпретатор);
  * `subprocess` — привычный `subprocess.run`, если он доступен и работает.

Бэкенд выбирается ключом `tools.commands_backend` в конфиге.
"""
from __future__ import annotations

import os
import shlex
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .filesystem import ToolError

SHELL_META = set(";|&$`<>\n\r")

# Глобальное состояние на процесс: флаг включения и выбранный бэкенд.
STATE = {"enabled": True, "backend": "posix", "kill_grace": 0.5}


def set_enabled(flag: bool) -> None:
    """Включает или выключает исполнение команд (tools.commands_enabled)."""
    STATE["enabled"] = bool(flag)


def is_enabled() -> bool:
    return bool(STATE["enabled"])


def set_backend(name: str) -> None:
    """Выбирает бэкенд: 'posix' или 'subprocess'."""
    chosen = str(name or "").strip().lower()
    if chosen not in ("posix", "subprocess"):
        raise ToolError(f"неизвестный бэкенд исполнения: {name!r} (posix|subprocess)")
    STATE["backend"] = chosen


def get_backend() -> str:
    return str(STATE["backend"])


@dataclass
class SystemSandbox:
    """Правила исполнения команд.

    Команды делятся на два класса:

    * `allowed_commands` — наблюдательные (`uname`, `pwd`, `ls`, `cat`…):
      они ничего не исполняют, кроме себя;
    * `privileged_commands` — интерпретаторы и инструменты с побочными
      каналами запуска (`python3`, `git`, `node`…). Формально проходя
      allowlist, они дают произвольное исполнение кода, поэтому требуют
      отдельного разрешения и роли из `privileged_roles`.

    `denied_patterns` — дополнительный фильтр, а не граница безопасности:
    это литеральные подстроки, их легко обойти конкатенацией.
    """

    allowed_commands: tuple[str, ...] = ()
    denied_patterns: tuple[str, ...] = ()
    timeout: int = 10
    cwd: Path = field(default_factory=Path.cwd)
    max_output: int = 8192
    privileged_commands: tuple[str, ...] = ()
    privileged_roles: tuple[str, ...] = ("guardian", "registrar")

    def is_privileged(self, executable: str) -> bool:
        return executable in self.privileged_commands

    def check(self, command: str, role: Optional[str] = None) -> list[str]:
        """Возвращает argv или бросает ToolError с причиной отказа."""
        text = (command or "").strip()
        if not text:
            raise ToolError("пустая команда")

        bad = sorted(SHELL_META & set(text))
        if bad:
            raise ToolError(
                "метасимволы шелла запрещены: " + " ".join(bad)
            )

        lowered = text.lower()
        for pattern in self.denied_patterns:
            if pattern.lower() in lowered:
                raise ToolError(f"запрещённый шаблон в команде: {pattern!r}")

        try:
            argv = shlex.split(text)
        except ValueError as exc:
            raise ToolError(f"не удалось разобрать команду: {exc}") from exc
        if not argv:
            raise ToolError("пустая команда после разбора")

        executable = Path(argv[0]).name
        known = tuple(self.allowed_commands) + tuple(self.privileged_commands)
        if executable not in known:
            raise ToolError(
                f"команда {executable!r} не в allowlist ({', '.join(known) or 'список пуст'})"
            )
        if self.is_privileged(executable):
            # python3/git и подобные — это не «одна команда», а канал запуска
            # произвольного кода: `python3 -c "print(__import__('os')...)"`
            # проходит любой фильтр подстрок.
            if role is None or role not in self.privileged_roles:
                raise ToolError(
                    f"команда {executable!r} привилегированная: даёт исполнение "
                    f"произвольного кода и требует роли из "
                    f"{', '.join(self.privileged_roles)} (текущая: {role or 'нет'})"
                )
        return argv


# ---------- бэкенд posix (без subprocess) ----------

def _drain(fd: int, sink: list, budget: int, deadline: float) -> bool:
    """Читает из неблокирующего fd до EOF. Возвращает True, если fd закрыт."""
    while True:
        try:
            chunk = os.read(fd, 4096)
        except BlockingIOError:
            return False
        except OSError:
            return True
        if not chunk:
            return True
        if len(sink[0]) < budget:
            sink[0] += chunk
        if time.time() > deadline:
            return True


def _run_posix(
    argv: list[str], cwd: Path, timeout: float, max_output: int
) -> tuple[int, str, str]:
    """Запускает argv через fork/exec. Не требует модуля subprocess."""
    if not hasattr(os, "fork"):  # pragma: no cover — только Windows
        raise ToolError("бэкенд posix недоступен: в этой ОС нет os.fork")

    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    try:
        pid = os.fork()
    except OSError as exc:
        for fd in (out_r, out_w, err_r, err_w):
            os.close(fd)
        raise ToolError(f"не удалось создать процесс: {exc}") from exc

    if pid == 0:  # дочерний процесс
        try:
            os.close(out_r)
            os.close(err_r)
            os.dup2(out_w, 1)
            os.dup2(err_w, 2)
            devnull = os.open(os.devnull, os.O_RDONLY)
            os.dup2(devnull, 0)
            try:
                os.chdir(str(cwd))
            except OSError:
                pass
            os.execvp(argv[0], argv)
        except BaseException:
            pass
        os._exit(127)

    os.close(out_w)
    os.close(err_w)
    os.set_blocking(out_r, False)
    os.set_blocking(err_r, False)

    stdout, stderr = [b""], [b""]
    budget = max_output * 4
    open_fds = {out_r: stdout, err_r: stderr}
    deadline = time.time() + timeout
    status = None
    timed_out = False

    while True:
        for fd in list(open_fds):
            if _drain(fd, open_fds[fd], budget, deadline):
                os.close(fd)
                del open_fds[fd]

        if status is None:
            done, raw = os.waitpid(pid, os.WNOHANG)
            if done == pid:
                status = raw
                if not open_fds:
                    break
                # Ребёнок вышел, но трубу может держать его потомок
                # (`sh -c "sleep 5 &"`). Ждать полный таймаут из-за него нельзя:
                # дочитываем короткий остаток и закрываем дескрипторы сами.
                grace = time.time() + STATE["kill_grace"]
                while open_fds and time.time() < grace:
                    for fd in list(open_fds):
                        if _drain(fd, open_fds[fd], budget, grace):
                            os.close(fd)
                            del open_fds[fd]
                    if open_fds:
                        time.sleep(0.002)
                break
        elif not open_fds:
            break

        if time.time() > deadline:
            if status is None:
                timed_out = True
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
                done, raw = os.waitpid(pid, 0)
                status = raw
                # дочитываем то, что успело попасть в трубу
                drain_deadline = time.time() + STATE["kill_grace"]
                while open_fds and time.time() < drain_deadline:
                    for fd in list(open_fds):
                        if _drain(fd, open_fds[fd], budget, drain_deadline):
                            os.close(fd)
                            del open_fds[fd]
            break
        time.sleep(0.004)

    for fd in list(open_fds):
        os.close(fd)
    if status is None:
        try:
            _, status = os.waitpid(pid, 0)
        except OSError:
            status = 0

    try:
        code = os.waitstatus_to_exitcode(status)
    except ValueError:
        code = -1
    if timed_out:
        raise ToolError(
            f"таймаут {timeout:g}s: команда прервана "
            f"({stdout[0].decode('utf-8', 'replace').strip()[:200]})"
        )
    return (
        code,
        stdout[0].decode("utf-8", "replace"),
        stderr[0].decode("utf-8", "replace"),
    )


# ---------- бэкенд subprocess (если он работает) ----------

def _run_subprocess(
    argv: list[str], cwd: Path, timeout: float, max_output: int
) -> tuple[int, str, str]:
    import subprocess  # ленивый импорт: _posixsubprocess грузится только здесь

    try:
        proc = subprocess.run(
            argv,
            shell=False,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"таймаут {timeout:g}s: {' '.join(argv)}") from None
    except FileNotFoundError as exc:
        raise ToolError(f"исполняемый файл не найден: {argv[0]} ({exc})") from exc
    except OSError as exc:
        raise ToolError(f"не удалось запустить {argv[0]}: {exc}") from exc
    return proc.returncode, proc.stdout or "", proc.stderr or ""


# ---------- общий вход ----------

def run_command(command: str, sandbox: SystemSandbox, role: Optional[str] = None) -> str:
    """Выполняет разрешённую команду и возвращает отчёт."""
    argv = sandbox.check(command, role=role)
    if not is_enabled():
        raise ToolError(
            "Исполнение команд выключено: tools.commands_enabled = false."
        )

    runner = _run_posix if get_backend() == "posix" else _run_subprocess
    code, stdout, stderr = runner(
        argv, sandbox.cwd, float(sandbox.timeout), sandbox.max_output
    )

    stdout = stdout.strip()
    stderr = stderr.strip()
    if len(stdout) > sandbox.max_output:
        stdout = stdout[: sandbox.max_output] + "\n... (вывод обрезан)"
    if len(stderr) > sandbox.max_output:
        stderr = stderr[: sandbox.max_output] + "\n... (stderr обрезан)"

    parts = [f"$ {' '.join(argv)}", f"код возврата: {code}"]
    if stdout:
        parts.append("--- stdout ---\n" + stdout)
    if stderr:
        parts.append("--- stderr ---\n" + stderr)
    if not stdout and not stderr:
        parts.append("(нет вывода)")

    report = "\n".join(parts)
    if code != 0:
        # Не 0 — это провал инструмента: пусть агент запишет success=False.
        raise ToolError(report)
    return report