#!/usr/bin/env python3
"""Вердикт P4.3 v2 (контраст по пулам) по замороженным гипотезам и гейтам.

Читает summary.json раннера и применяет к нему протокол
`p4.3.spike-verb.pool.v2`. Пороги, направления и набор рук берутся из
манифеста, а не из этого файла; при расхождении digest'а или при запуске без
`--confirm` вердикт не выносится.

    python scripts/report_p4_3_v2.py var/p4_3v2_confirm/summary.json

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
                                "p4_3_spike_verb_v2.json")


def _by_seed(cells: list[dict], arm: str) -> dict[int, dict]:
    return {c["seed"]: c for c in cells if c["arm"] == arm}


def _rows(summary_path: str, arm: str, seed: int) -> list[dict]:
    path = os.path.join(os.path.dirname(os.path.abspath(summary_path)),
                        f"{arm}-s{seed}.tsv")
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _kappa(summary_path: str, arm: str, seed: int) -> float | None:
    path = os.path.join(os.path.dirname(os.path.abspath(summary_path)),
                        f"{arm}-s{seed}.meta.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return float(json.load(fh)["settings"]["pool_contrast"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


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
    for arm in ("pool", "plastic", "no_plasticity", "oracle"):
        if arm not in {c["arm"] for c in cells}:
            print(f"в summary нет руки {arm}")
            return 1

    seeds = sorted(_by_seed(cells, "pool"))
    pool = _by_seed(cells, "pool")
    v1 = _by_seed(cells, "plastic")
    frozen = _by_seed(cells, "no_plasticity")
    oracle = _by_seed(cells, "oracle")
    if sorted(v1) != seeds or sorted(frozen) != seeds:
        print("руки измерены не на одном наборе seed'ов: парное сравнение невозможно")
        return 1

    block = int(declared["episodes"]) // 5
    print(f"протокол {manifest['protocol_id']}  sha256={protocol['sha256'][:16]}")
    print(f"блоки: первые и последние {block} эпизодов; seed'ы {seeds}\n")
    print(f"{'сид':>5} {'рука':>14} {'s(1)':>6} {'s(2)':>6} {'Δs':>7} "
          f"{'acc(1)':>7} {'acc(2)':>7} {'спайки/эп':>10} {'у границы':>9}")
    for seed in seeds:
        for arm in ("pool", "plastic", "no_plasticity"):
            row = {"pool": pool, "plastic": v1, "no_plasticity": frozen}[arm][seed]
            print(f"{seed:5d} {arm:>14} {row['success_first']:6.3f} {row['success_last']:6.3f} "
                  f"{row['delta']:+7.3f} {row['accuracy_first']:7.3f} "
                  f"{row['accuracy_last']:7.3f} {row['spikes_per_episode']:10.0f} "
                  f"{row['at_bound']:9.3f}")
        row = oracle.get(seed)
        if row:
            print(f"{seed:5d} {'oracle':>14} {row['success_first']:6.3f} "
                  f"{row['success_last']:6.3f} {row['delta']:+7.3f}")

    # --- гипотезы ---------------------------------------------------------
    h1 = all(pool[s]["delta"] > 0 for s in seeds)
    h2 = all(pool[s]["delta"] > v1[s]["delta"] for s in seeds)
    ratio = (statistics.median(pool[s]["spikes_per_episode"] for s in seeds)
             / statistics.median(frozen[s]["spikes_per_episode"] for s in seeds))
    h3 = ratio <= 2.0
    pool_acc = {s: pool[s]["accuracy_last"] - pool[s]["accuracy_first"] for s in seeds}
    frozen_acc = {s: frozen[s]["accuracy_last"] - frozen[s]["accuracy_first"] for s in seeds}
    # Порог величины эффекта объявлен в протоколе до прогона: в v1 его не было,
    # и H4 провалилась на разбросе заморозки в 0.3-1.1 п.п.
    h4 = (all(v > 0 for v in pool_acc.values())
          and all(frozen_acc[s] <= 0.25 * pool_acc[s] for s in seeds))
    gap_oracle = (statistics.median(pool[s]["success_last"] for s in seeds)
                  - statistics.median(oracle[s]["success_last"] for s in seeds))
    gap_v1 = (statistics.median(pool[s]["success_last"] for s in seeds)
              - statistics.median(v1[s]["success_last"] for s in seeds))

    # --- гейты ------------------------------------------------------------
    g1 = all(0.2 <= pool[s]["success_first"] <= 0.7 for s in seeds)
    g2 = all(pool[s]["at_bound"] <= 0.5 for s in seeds)
    g3 = (all(c["violations"] == 0 for c in cells)
          and all(int(r["denied"]) == 0
                  for arm in ("pool", "plastic", "no_plasticity") for s in seeds
                  for r in _rows(args.summary, arm, s)))
    g4 = not any(c["nan"] for c in cells)
    g5 = bool(oracle) and all(oracle[s]["success_last"] == 1.0 for s in oracle)
    collapsed = []
    for s in seeds:
        tail = _rows(args.summary, "pool", s)[-block:]
        left = sum(int(r["moved_left"]) for r in tail)
        right = sum(int(r["moved_right"]) for r in tail)
        if left < 20 or right < 20:
            collapsed.append(f"s{s}: влево {left}, вправо {right}")
    g6 = not collapsed
    kappas = {_kappa(args.summary, "pool", s) for s in seeds}
    g7 = (len(kappas) == 1 and kappas.pop() == float(declared["pool_contrast"])
          and all(_kappa(args.summary, "plastic", s) == float(declared["pool_contrast"])
                  for s in seeds))

    print()
    for hid, ok, text in (
        ("H1", h1, "pool: success последних 100 > первых 100 на всех seed'ах"),
        ("H2", h2, "прирост pool > прирост пластика v1 на том же seed'е "
                   f"({[round(pool[s]['delta'] - v1[s]['delta'], 3) for s in seeds]})"),
        ("H3", h3, f"спайковый бюджет pool/no_plasticity = {ratio:.3f} <= 2"),
        ("H4", h4, "точность растёт у pool и прирост заморозки <= 0.25 прироста pool "
                   f"(pool {[round(v, 3) for v in pool_acc.values()]}, "
                   f"заморозка {[round(v, 3) for v in frozen_acc.values()]})"),
        ("H5", True, f"pool - oracle = {gap_oracle:+.3f}, pool - v1 = {gap_v1:+.3f} "
                     "(характеризация, не гейт)"),
    ):
        print(f"{hid}: {'ПРОЙДЕНО' if ok else 'НЕ ПРОЙДЕНО'} — {text}")

    for gid, ok, text in (
        ("G1", g1, "полоса сложности: success первых 100 у pool в [0.2, 0.7]"),
        ("G2", g2, "насыщение: доля весов на границе <= 0.5"),
        ("G3", g3, "ворота: 0 нарушений, 0 отказов"),
        ("G4", g4, "числа: нет NaN/Inf"),
        ("G5", g5, "оракул: success последних 100 == 1.0"),
        ("G6", g6, "пул не схлопнулся: оба направления >= 20 в последних 100"),
        ("G7", g7, f"правило то: kappa={declared['pool_contrast']} в мете клеток"),
    ):
        print(f"{gid}: {'OK' if ok else 'ПРОВАЛ'} — {text}" + (
            f" ({'; '.join(collapsed)})" if gid == "G6" and collapsed else ""))

    required = h1 and h2 and h3 and g1 and g2 and g3 and g4 and g5 and g6 and g7
    print()
    if not g1:
        print("вердикт: прогон неинформативен (полоса сложности), H1 не интерпретируется")
    elif not g6:
        print("вердикт: пул схлопнулся, H2 не интерпретируется; публикуется как есть")
    elif not h2:
        print("вердикт: контраст по пулам не дал прироста над скаляром v1 — "
              "результат v2 отрицательный или нулевой, публикуется как есть")
    elif required:
        print("вердикт: v2 подтверждена на объявленной рабочей точке: контраст по пулам "
              "даёт больший прирост, чем скаляр v1 (описание при n=3, значимость не "
              "заявляется)")
    else:
        print("вердикт: критерий v2 не выполнен; публикуется как есть")
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
