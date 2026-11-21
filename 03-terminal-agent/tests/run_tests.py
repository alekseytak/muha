#!/usr/bin/env python3
"""Минимальный раннер тестов — без unittest/pytest.

Не тянет `unittest` (а значит и `heapq`/`difflib`), поэтому работает даже там,
где импорт stdlib-расширений в `dlopen` деградировал. Каждый тест печатается
до запуска, чтобы место зависания было видно.

Запуск:  python3 tests/run_tests.py [подстрока]
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def _case_call(case_class, method_name: str):
    """Обёртка вокруг метода TestCase: setUp, тест, tearDown.

    Без setUp тест падал на отсутствии полей, которые он готовит, — и это
    выглядело как провал логики, хотя ломался раннер.
    """

    def run() -> None:
        case = case_class(method_name)
        case.setUp()
        try:
            getattr(case, method_name)()
        finally:
            case.tearDown()

    return run


def main(argv: list[str]) -> int:
    needle = argv[1] if len(argv) > 1 else ""
    started = time.time()
    # Наборы подключаются списком: каждый модуль даёт свои test_*-функции.
    # Модуль на unittest.TestCase раскрывается в bound-методы.
    import unittest

    modules = []
    for module_name in ("test_agent", "test_zaloop", "test_retina"):
        try:
            modules.append(__import__(module_name))
        except Exception:
            print(f"НЕ УДАЛОСЬ ИМПОРТИРОВАТЬ {module_name}:")
            traceback.print_exc()
            return 2

    tests = []
    low = needle.lower()
    for suite in modules:
        for name, obj in vars(suite).items():
            # Классы TestCase: имя класса НЕ начинается с test_, поэтому
            # проверяется отдельно, до фильтра по префиксу.
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                for method in sorted(dir(obj)):
                    if not method.startswith("test_"):
                        continue
                    if low and low not in method.lower() and low not in name.lower():
                        continue
                    tests.append((f"{name}.{method}",
                                  _case_call(obj, method)))
                continue
            if not name.startswith("test_") or not callable(obj):
                continue
            if isinstance(obj, type):
                continue
            if low and low not in name.lower():
                continue
            tests.append((name, obj))
    tests.sort(key=lambda item: getattr(
        getattr(item[1], "__code__", None) or getattr(item[1], "__func__", None),
        "co_firstlineno", 0))

    passed, failed, skipped = 0, [], 0
    for name, func in tests:
        print(f"  {name} ... ", end="", flush=True)
        test_started = time.time()
        try:
            func()
            print(f"ok ({time.time() - test_started:.2f}s)", flush=True)
            passed += 1
        except AssertionError as exc:
            print("ПРОВАЛ", flush=True)
            failed.append((name, f"AssertionError: {exc}", traceback.format_exc()))
        except Exception as exc:  # noqa: BLE001 — хотим видеть любой сбой
            print(f"ОШИБКА ({type(exc).__name__})", flush=True)
            failed.append((name, f"{type(exc).__name__}: {exc}", traceback.format_exc()))

    total_time = time.time() - started
    print()
    if failed:
        print("=" * 70)
        for name, summary, tb in failed:
            print(f"\n[ПРОВАЛ] {name}: {summary}")
            print(tb)
        print("=" * 70)

    print(
        f"ИТОГО: {len(tests)} тестов, успешно {passed}, провалено {len(failed)}, "
        f"пропущено {skipped}, время {total_time:.2f}s"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
