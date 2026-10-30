#!/usr/bin/env python3
"""Гейт схемы журнала решений: шесть колонок, ни одной пустой, внятный штамп времени.

Журнал (var/decisions.tsv) — единственное место, где решение записано вместе с
причиной, проверкой и итогом. Он под git, чтобы его читали другие агенты, а не
только тот, кто его писал. Строка с пятью полями человеку незаметна, а следующему
агенту ломает смысл: «чем проверено» читается как «почему так решили».

Запуск: scripts/check_journal.py [путь] — без аргумента var/decisions.tsv.
"""
from __future__ import annotations

import re
import sys

COLUMNS = ["ts", "phase", "decision", "why", "evidence", "result"]
HEADER = "\t".join(COLUMNS)
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}Z)?$")
DEFAULT_PATH = "var/decisions.tsv"


def validate_lines(lines: list[str]) -> list[str]:
    """Ошибки построчно; пустой список — схема целая."""
    errors: list[str] = []
    if not lines:
        return ["журнал пуст — заголовка нет"]
    if lines[0] != HEADER:
        errors.append(f"строка 1: заголовок не «{HEADER}», а «{lines[0]}»")
    for number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != len(COLUMNS):
            errors.append(
                f"строка {number}: полей {len(fields)} вместо {len(COLUMNS)} "
                f"({COLUMNS})"
            )
            continue
        if not TS_RE.match(fields[0]):
            errors.append(f"строка {number}: штамп времени не дата и не ISO-секунда: {fields[0]!r}")
        for name, value in zip(COLUMNS[1:], fields[1:]):
            if not value.strip():
                errors.append(f"строка {number}: колонка {name} пустая")
    return errors


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else DEFAULT_PATH
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError as exc:
        print(f"журнал не читается: {path}: {exc}")
        return 1
    errors = validate_lines(lines)
    for error in errors:
        print(f"  {error}")
    if errors:
        print(f"журнал вне схемы: {len(errors)} дефектов в {path}")
        return 1
    print(f"журнал в схеме: строк {len(lines) - 1}, колонок {len(COLUMNS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
