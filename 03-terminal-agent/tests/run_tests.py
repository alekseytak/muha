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


def main(argv: list[str]) -> int:
    needle = argv[1] if len(argv) > 1 else ""
    started = time.time()
    try:
        import test_agent as suite
    except Exception:
        print("НЕ УДАЛОСЬ ИМПОРТИРОВАТЬ НАБОР ТЕСТОВ:")
        traceback.print_exc()
        return 2

    tests = [
        (name, obj) for name, obj in vars(suite).items()
        if name.startswith("test_") and callable(obj) and needle in name
    ]
    tests.sort(key=lambda item: item[1].__code__.co_firstlineno)

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
