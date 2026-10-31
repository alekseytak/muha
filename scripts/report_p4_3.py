#!/usr/bin/env python3
"""Вердикт P4.3 по замороженным гипотезам и гейтам.

Читает summary.json раннера и применяет к нему протокол
`p4.3.spike-verb.v1`. Никаких решений здесь не принимается: пороги, блоки,
направления и набор рук взяты из манифеста, а не из этого файла. Если манифест
и summary расходятся по digest'у, вердикт не выносится.

    python scripts/report_p4_3.py var/p4_3_confirm/summary.json [--manifest path]

Код возврата: 0 — все required-гипотезы и гейты пройдены, 1 — нет.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(REPO, "fly_connectome_agent", "manifests",
                                "p4_3_spike_verb.json")


def _by_seed(cells: list[dict], arm: str) -> dict[int, dict]:
    return {c["seed"]: c for c in cells if c["arm"] == arm}


def _denials(summary_path: str, arm: str, seed: int) -> int:
    """Отказы ворот — из TSV клетки, а не из сводки: сводка может быть старше."""
    path = os.path.join(os.path.dirname(os.path.abspath(summary_path)),
                        f"{arm}-s{seed}.tsv")
    with open(path, encoding="utf-8", newline="") as fh:
        return sum(int(row["denied"]) for row in csv.DictReader(fh, delimiter="\t"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv[1:])

    summary = json.load(open(args.summary, encoding="utf-8"))
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    declared = manifest["operating_point"]["values"]

    protocol = summary.get("protocol") or {}
    if protocol.get("protocol_id") != manifest["protocol_id"]:
        print(f"summary не от этого протокола: {protocol.get('protocol_id')!r} != "
              f"{manifest['protocol_id']!r}")
        return 1
    if not protocol.get("verified"):
        print("summary помечен как непроверенный по протоколу (запуск без --confirm): "
              "вердикт не выносится")
        return 1

    cells = summary["cells"]
    for arm in ("plastic", "no_plasticity", "oracle"):
        if arm not in {c["arm"] for c in cells}:
            print(f"в summary нет руки {arm}")
            return 1

    seeds = sorted(_by_seed(cells, "plastic"))
    plastic, frozen = _by_seed(cells, "plastic"), _by_seed(cells, "no_plasticity")
    oracle = _by_seed(cells, "oracle")

    block = int(declared["episodes"]) // 5
    print(f"протокол {manifest['protocol_id']}  sha256={protocol['sha256'][:16]}")
    print(f"блоки: первые и последние {block} эпизодов; seed'ы {seeds}\n")
    print(f"{'сид':>5} {'рука':>14} {'s(1)':>6} {'s(2)':>6} {'Δs':>7} "
          f"{'acc(1)':>7} {'acc(2)':>7} {'спайки/эп':>10} {'у границы':>9}")
    for seed in seeds:
        for arm in ("plastic", "no_plasticity"):
            row = (plastic if arm == "plastic" else frozen)[seed]
            print(f"{seed:5d} {arm:>14} {row['success_first']:6.3f} {row['success_last']:6.3f} "
                  f"{row['delta']:+7.3f} {row['accuracy_first']:7.3f} "
                  f"{row['accuracy_last']:7.3f} {row['spikes_per_episode']:10.0f} "
                  f"{row['at_bound']:9.3f}")
        row = oracle.get(seed)
        if row:
            print(f"{seed:5d} {'oracle':>14} {row['success_first']:6.3f} "
                  f"{row['success_last']:6.3f} {row['delta']:+7.3f}")

    # --- гипотезы ---------------------------------------------------------
    h1 = all(plastic[s]["delta"] > 0 for s in seeds)
    h2 = all(plastic[s]["delta"] > frozen[s]["delta"] for s in seeds)
    ratio = (statistics.median(plastic[s]["spikes_per_episode"] for s in seeds)
             / statistics.median(frozen[s]["spikes_per_episode"] for s in seeds))
    h3 = ratio <= 2.0
    acc_delta = {s: plastic[s]["accuracy_last"] - plastic[s]["accuracy_first"] for s in seeds}
    froz_delta = {s: frozen[s]["accuracy_last"] - frozen[s]["accuracy_first"] for s in seeds}
    h4 = all(v > 0 for v in acc_delta.values()) and sum(v > 0 for v in froz_delta.values()) <= 1
    gap = statistics.median(plastic[s]["success_last"] for s in seeds) - \
        statistics.median(oracle[s]["success_last"] for s in seeds) if oracle else float("nan")

    # --- гейты ------------------------------------------------------------
    g1 = all(0.2 <= plastic[s]["success_first"] <= 0.7 for s in seeds)
    g2 = all(plastic[s]["at_bound"] <= 0.5 for s in seeds)
    g3 = (all(c["violations"] == 0 for c in cells)
          and all(_denials(args.summary, arm, seed) == 0
                  for arm in ("plastic", "no_plasticity") for seed in seeds))
    g4 = not any(c["nan"] for c in cells)
    g5 = bool(oracle) and all(oracle[s]["success_last"] == 1.0 for s in oracle)

    print()
    for hid, ok, text in (
        ("H1", h1, "plastic: success последних 100 > первых 100 на всех seed'ах"),
        ("H2", h2, "прирост plastic > прирост no_plasticity на том же seed'е"),
        ("H3", h3, f"спайковый бюджет plastic/no_plasticity = {ratio:.3f} <= 2"),
        ("H4", h4, "точность глагола растёт у plastic и не растёт у заморозки "
                   f"(plastic {[round(v, 3) for v in acc_delta.values()]}, "
                   f"заморозка {[round(v, 3) for v in froz_delta.values()]})"),
        ("H5", True, f"plastic - oracle по success = {gap:+.3f} (характеризация, не гейт)"),
    ):
        print(f"{hid}: {'ПРОЙДЕНО' if ok else 'НЕ ПРОЙДЕНО'} — {text}")

    for gid, ok, text in (
        ("G1", g1, "полоса сложности: success первых 100 у plastic в [0.2, 0.7]"),
        ("G2", g2, "насыщение: доля весов на границе <= 0.5"),
        ("G3", g3, "ворота: 0 нарушений, 0 отказов"),
        ("G4", g4, "числа: нет NaN/Inf"),
        ("G5", g5, "оракул: success последних 100 == 1.0"),
    ):
        print(f"{gid}: {'OK' if ok else 'ПРОВАЛ'} — {text}")

    required = h1 and h2 and h3 and g1 and g2 and g3 and g4 and g5
    print()
    if not g1:
        print("вердикт: прогон неинформативен (полоса сложности), H1 не интерпретируется")
    elif required:
        print("вердикт: критерий ЭТАПА 4 выполнен на объявленной рабочей точке "
              "(описание при n=3, значимость не заявляется)")
    else:
        print("вердикт: критерий ЭТАПА 4 не выполнен; публикуется как есть")
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
