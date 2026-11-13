"""Минимальный YAML-парсер на чистом Python (подмножество для конфигурации).

Зачем свой: на этой машине импорт C-расширений (`yaml`, `csv`, `subprocess`)
периодически зависает в `dlopen` — процесс висит бесконечно, без исключения.
Конфиг агента — простой YAML, поэтому его дешевле разобрать здесь и не тянуть
внешнюю зависимость.

Поддерживается:
  key: value
  key:            # вложенный словарь по отступу
    nested: 1
  list: [a, b, "c, d"]
  block:          # блочный список
    - a
    - b
  # комментарии вне кавычек
  строки в 'одинарных' и "двойных" кавычках, true/false/null, int, float

НЕ поддерживается (осознанно): якоря/алиасы, теги, многострочные блоки `|`/`>`.
При встрече такой конструкции поднимается YamlError с номером строки.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

__all__ = ["YamlError", "loads", "load_file"]


class YamlError(ValueError):
    """Ошибка разбора YAML-подмножества."""


QUOTES = ("'", '"')


def _strip_comment(line: str) -> str:
    """Убирает `#…` вне кавычек."""
    quote = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in QUOTES:
            quote = char
            continue
        if char == "#":
            return line[:index]
    return line


def _split_inline_list(text: str) -> List[str]:
    """Режет `[a, "b, c"]` на элементы, уважая кавычки."""
    body = text.strip()[1:-1]
    if not body.strip():
        return []
    items: List[str] = []
    current: List[str] = []
    quote = None
    escaped = False
    for char in body:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\" and quote == '"':
            current.append(char)
            escaped = True
            continue
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in QUOTES:
            quote = char
            current.append(char)
            continue
        if char == ",":
            items.append("".join(current))
            current = []
            continue
        current.append(char)
    remainder = "".join(current)
    if remainder.strip():
        items.append(remainder)
    return [item.strip() for item in items]


def _parse_scalar(text: str) -> Any:
    """Приводит скаляр YAML к типу Python."""
    value = text.strip()
    if not value:
        return None
    if value.startswith("[") and value.endswith("]"):
        return [_parse_scalar(item) for item in _split_inline_list(value)]
    if len(value) >= 2 and value[0] in QUOTES and value[-1] == value[0]:
        inner = value[1:-1]
        if value[0] == '"':
            return (
                inner.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
        return inner.replace("''", "'")
    lowered = value.lower()
    if lowered in ("null", "~", "none"):
        return None
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _split_key(line: str) -> Tuple[str, str]:
    """Делит `key: value` по двоеточию вне кавычек."""
    quote = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in QUOTES:
            quote = char
            continue
        if char == ":":
            key = line[:index].strip()
            rest = line[index + 1:].strip()
            if len(key) >= 2 and key[0] in QUOTES and key[-1] == key[0]:
                key = key[1:-1]
            return key, rest
    raise YamlError(f"нет двоеточия: {line!r}")


def loads(text: str) -> Any:
    """Разбирает YAML-подмножество в словари/списки/скаляры."""
    root: dict = {}
    # стек контейнеров: (отступ, контейнер)
    stack: List[Tuple[int, Any]] = [(-1, root)]

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if stripped.startswith("\t") or "\t" in line[: indent + 1]:
            raise YamlError(f"строка {lineno}: табуляция в отступе не поддерживается")
        if stripped.startswith(("|", ">")) and stripped.rstrip("+- ") in ("|", ">"):
            raise YamlError(
                f"строка {lineno}: многострочные блоки '|' и '>' не поддерживаются"
            )
        if stripped.startswith("&") or stripped.startswith("*"):
            raise YamlError(f"строка {lineno}: якоря и алиасы не поддерживаются")

        # закрываем контейнеры, чей отступ больше либо равен текущему
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]

        if stripped.startswith("- ") or stripped == "-":
            if not isinstance(container, list):
                raise YamlError(f"строка {lineno}: элемент списка вне списка")
            item_text = stripped[2:].strip() if stripped != "-" else ""
            if ":" in item_text and not item_text.startswith(QUOTES):
                # элемент-словарь: - key: value
                key, rest = _split_key(item_text)
                item: dict = {}
                item[key] = _parse_scalar(rest)
                container.append(item)
                stack.append((indent, item))
            else:
                container.append(_parse_scalar(item_text))
            continue

        key, rest = _split_key(stripped)
        if not isinstance(container, dict):
            raise YamlError(f"строка {lineno}: ключ {key!r} внутри списка")
        if rest.rstrip("+-") in ("|", ">"):
            raise YamlError(
                f"строка {lineno}: многострочные блоки '|' и '>' не поддерживаются"
            )
        if rest.startswith(("&", "*")):
            raise YamlError(f"строка {lineno}: якоря и алиасы не поддерживаются")
        if rest:
            container[key] = _parse_scalar(rest)
            continue

        # пустое значение: словарь или список — решит следующая строка
        probe = _next_content(lines_cache=text.splitlines(), start=lineno)
        child: Any = [] if (probe or "").lstrip().startswith("- ") else {}
        container[key] = child
        stack.append((indent, child))

    return root


def _next_content(lines_cache: List[str], start: int) -> str:
    """Следующая непустая строка после `start` (1-based номер текущей)."""
    for raw in lines_cache[start:]:
        line = _strip_comment(raw).rstrip()
        if line.strip():
            return line
    return ""


def load_file(path: str | Path) -> Any:
    """Читает и разбирает YAML-файл."""
    target = Path(path)
    return loads(target.read_text(encoding="utf-8"))


def dumps(data: Any, indent: int = 0) -> str:  # pragma: no cover — для отладки
    """Простейшая сериализация (не для round-trip сложных структур)."""
    pad = " " * indent
    if isinstance(data, dict):
        out = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                out.append(f"{pad}{key}:")
                out.append(dumps(value, indent + 2))
            else:
                out.append(f"{pad}{key}: {value}")
        return "\n".join(out)
    if isinstance(data, list):
        return "\n".join(f"{pad}- {item}" for item in data)
    return f"{pad}{data}"
