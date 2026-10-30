#!/usr/bin/env python3
"""Гейт письменности: в русскоязычный текст не должны просачиваться иероглифы.

Повод конкретный: за одну сессию в прозу попали два иероглифа, латинская R
посреди кириллического слова и латинское e внутри него же. Тесты такое не ловят
— файл компилируется и читается человеком, который уже знает, что там должно
быть.

Запуск: scripts/check_prose.py [файл ...] — без аргументов берёт git ls-files.
"""
from __future__ import annotations

import subprocess
import sys
import unicodedata


def script_of(char: str) -> str | None:
    if len(char) != 1:
        return None                    # пустой сосед на краях токена
    name = unicodedata.name(char, "")
    if "CYRILLIC" in name:
        return "cyr"
    if any(tag in name for tag in ("CJK", "HIRAGANA", "KATAKANA", "HALFWIDTH", "FULLWIDTH")):
        return "cjk"
    if "LATIN" in name:
        return "lat"
    return None


def offending_words(line: str) -> list[str]:
    """Сигнатура дефекта: соседние символы двух письменностей без разделителя
    (кириллическое слово, склеенное с латинской гласной, или латинская буква
    вместо похожей кириллической) и любой иероглиф. Разделитель спасает:
    f"средняя — это f-строка, seed'ов и toy-сеть — легальный термин, а не
    опечатка. Сами примеры здесь не приводятся — этот файл сам под гейтом.
    """
    hits = []
    for raw in line.split():
        for i in range(len(raw)):
            if script_of(raw[i]) == "cjk":
                hits.append(raw)
                break
            nxt = raw[i + 1] if i + 1 < len(raw) else ""
            if {script_of(raw[i]), script_of(nxt)} != {"cyr", "lat"}:
                continue
            # escape-последовательность (\n, \t) — не проза
            if raw[i] in "ntabfrsu" and i and raw[i - 1] == "\\":
                continue
            hits.append(raw)
            break
    return hits


def tracked_files() -> list[str]:
    """Что под git: и уже закоммиченное, и ещё не добавленное, но не игноряемое.
    Неотслеживаемое важно — именно новые файлы попадают в коммит, и именно в них
    дефекты просачиваются чаще всего."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard",
         "*.py", "*.md", "*.yaml", "*.yml", "*.json"],
        capture_output=True, text=True, check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def main(argv: list[str]) -> int:
    paths = argv[1:] or tracked_files()
    problems = 0
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    for word in offending_words(line):
                        print(f"{path}:{number}: смешение письменностей в '{word}'")
                        problems += 1
        except (UnicodeDecodeError, OSError):
            continue
    if problems:
        print(f"найдено подозрений: {problems}")
        return 1
    print(f"проза чиста: проверено файлов {len(paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
