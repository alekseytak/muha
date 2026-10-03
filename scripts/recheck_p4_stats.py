#!/usr/bin/env python3
"""Пересчёт заявленных p-value из сырых per-seed CSV.

Назначение: число в отчёте обязано быть выводимым из строк, а не из памяти прогона.
Скрипт берёт var/p4_results.csv (20 seeds, одиночные зеркала, критерий 1) и
var/p4_conf60_{a,b}.csv (mixed, 60 seeds, критерий 3), считает двусторонний exact
sign test по seed'ам и печатает W/L/T. Ссылки на var/ в отчёте — единственный
способ проверить «p<0.0001» не на глаз.

Запуск: .venv/bin/python scripts/recheck_p4_stats.py
Код возврата: 0 — заявленное сходится; 1 — расходится; 2 — нет сырых CSV.
"""
from __future__ import annotations

import csv
import pathlib
import sys

from scipy.stats import binomtest

ROOT = pathlib.Path(__file__).resolve().parent.parent
VAR = ROOT / "var"

# Заявления из docs/p4_validation.md, которые тут сверяются: (подпись, W, n_inf).
SINGLE_TASK = (
    ("right", "dw[S_R→C_R]", "dw[S_L→C_L]", "критерий 1, task A"),
    ("left", "dw[S_L→C_L]", "dw[S_R→C_R]", "критерий 1, task B"),
)
EXPECTED = {
    "критерий 1, task A": (19, 20),
    "критерий 1, task B": (18, 20),
    "mixed vs no_plasticity (fresh 20-59)": (37, 39),
    "mixed vs m_zero (fresh 20-59)": (37, 39),
    "mixed vs weight_shuffled_frozen (fresh 20-59)": (31, 38),
    "mixed vs direction_shuffled_frozen (fresh 20-59)": (40, 40),
}


def load(*names: str) -> list[dict]:
    rows: list[dict] = []
    for name in names:
        path = VAR / name
        if not path.exists():
            print(f"нет сырого CSV: {path}", file=sys.stderr)
            raise SystemExit(2)
        rows += list(csv.DictReader(path.open(encoding="utf-8")))
    return rows


def sign_test(pairs: list[tuple[float, float]]) -> tuple[int, int, int, float]:
    w = sum(1 for a, b in pairs if a > b)
    l = sum(1 for a, b in pairs if a < b)
    t = sum(1 for a, b in pairs if a == b)
    n = w + l
    p = binomtest(w, n, 0.5).pvalue if n else float("nan")
    return w, l, t, p


def report(label: str, pairs: list[tuple[float, float]]) -> bool:
    w, l, t, p = sign_test(pairs)
    n = w + l
    exp = EXPECTED.get(label)
    ok = exp == (w, n)
    print(f"  {'ок  ' if ok else 'РАЗБИЕНИЕ'} {label}")
    print(f"       W={w} L={l} T={t} n_informative={n} (пар всего {len(pairs)})")
    print(f"       exact two-sided binomtest p = {p:.6g}")
    if exp:
        print(f"       заявлено в отчёте: {exp[0]}/{exp[1]}" + ("" if ok else "  ← НЕ СХОДИТСЯ"))
    return ok


def main() -> int:
    single = load("p4_results.csv")
    conf = load("p4_conf60_a.csv", "p4_conf60_b.csv")
    ok = True

    print("критерий 1 — направление Δw по правилу награды (plastic, 20 seeds)")
    for task, hi, lo, label in SINGLE_TASK:
        pairs = [
            (float(r[hi]), float(r[lo]))
            for r in single if r["arm"] == "plastic" and r["task"] == task
        ]
        ok &= report(label, pairs)

    print("\nкритерий 3 — mixed min-half против базлайнов, ТОЛЬКО fresh seeds 20-59")
    cells: dict[tuple[str, int], float] = {}
    for r in conf:
        if r["task"] == "mixed":
            cells[(r["arm"], int(r["seed"]))] = float(r["success_mirror_min"])
    for arm in ("no_plasticity", "m_zero", "weight_shuffled_frozen", "direction_shuffled_frozen"):
        pairs = [
            (cells[("plastic", s)], cells[(arm, s)])
            for s in range(20, 60)
            if ("plastic", s) in cells and (arm, s) in cells
        ]
        ok &= report(f"mixed vs {arm} (fresh 20-59)", pairs)

    print("\n" + ("заявленные числа воспроизводятся из сырых CSV" if ok
                  else "ЕСТЬ РАЗБИЕНИЕ — числа в отчёте надо править, а не пересчитывать устно"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
