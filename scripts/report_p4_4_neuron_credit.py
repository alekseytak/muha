#!/usr/bin/env python3
"""Вердикт P4.4 (кредит на нейрон вместо пула) по объявленным правилам.

Читает summary.json раннера и применяет к нему протокол
`p4.4.spike-verb.neuron-credit.v1`. Пороги, режим каждой руки и набор строк
таблицы исходов берутся из манифеста, а не из этого файла; при расхождении
protocol_id или при запуске без `--confirm` вердикт не выносится.

    python scripts/report_p4_4_neuron_credit.py var/p4_4_confirm/summary.json

Инструмент написан до подтверждающего прогона и проверен на синтетических
фикстурах — иначе вердикт подгонялся бы под уже увиденные числа.

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
                                "p4_4_neuron_credit.json")
TREATMENT = "neuron"
NULL_ARM = "neuron_shuffled"
REQUIRED_ARMS = (TREATMENT, NULL_ARM, "plastic", "no_plasticity")


def _by_seed(cells: list[dict], arm: str) -> dict[int, dict]:
    return {c["seed"]: c for c in cells if c["arm"] == arm}


def _rows(summary_path: str, arm: str, seed: int) -> list[dict]:
    path = os.path.join(os.path.dirname(os.path.abspath(summary_path)),
                        f"{arm}-s{seed}.tsv")
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _meta(summary_path: str, arm: str, seed: int) -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(summary_path)),
                        f"{arm}-s{seed}.meta.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _last_block(rows: list[dict], block: int) -> list[dict]:
    return rows[-block:]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv[1:])

    summary = json.load(open(args.summary, encoding="utf-8"))
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    declared = manifest["operating_point"]["values"]
    declared_mode = {a["arm_id"]: a.get("neuron_credit", "none")
                     for a in manifest["arms"]}

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
    have = {c["arm"] for c in cells}
    for arm in REQUIRED_ARMS:
        if arm not in have:
            print(f"в summary нет руки {arm}")
            return 1

    seeds = sorted(_by_seed(cells, TREATMENT))
    treat = _by_seed(cells, TREATMENT)
    shuffled = _by_seed(cells, NULL_ARM)
    v1 = _by_seed(cells, "plastic")
    frozen = _by_seed(cells, "no_plasticity")
    oracle = _by_seed(cells, "oracle")
    if any(sorted(d) != seeds for d in (shuffled, v1, frozen)):
        print("руки измерены не на одном наборе seed'ов: парное сравнение невозможно")
        return 1

    block = int(declared["episodes"]) // 5
    print(f"протокол {manifest['protocol_id']}  sha256={protocol['sha256'][:16]}")
    print(f"блоки: первые и последние {block} эпизодов; seed'ы {seeds}\n")
    print(f"{'сид':>5} {'рука':>16} {'s(1)':>6} {'s(2)':>6} {'Δs':>7} "
          f"{'acc(1)':>7} {'acc(2)':>7} {'спайки/эп':>10} {'у границы':>9}")
    for seed in seeds:
        for arm, table in ((TREATMENT, treat), (NULL_ARM, shuffled),
                           ("plastic", v1), ("no_plasticity", frozen)):
            row = table[seed]
            print(f"{seed:5d} {arm:>16} {row['success_first']:6.3f} {row['success_last']:6.3f} "
                  f"{row['delta']:+7.3f} {row['accuracy_first']:7.3f} "
                  f"{row['accuracy_last']:7.3f} {row['spikes_per_episode']:10.0f} "
                  f"{row['at_bound']:9.3f}")
        row = oracle.get(seed)
        if row:
            print(f"{seed:5d} {'oracle':>16} {row['success_first']:6.3f} "
                  f"{row['success_last']:6.3f} {row['delta']:+7.3f}")

    # --- гипотезы ---------------------------------------------------------
    treat_delta = {s: treat[s]["delta"] for s in seeds}
    shuffled_delta = {s: shuffled[s]["delta"] for s in seeds}
    v1_delta = {s: v1[s]["delta"] for s in seeds}
    treat_acc = {s: treat[s]["accuracy_last"] - treat[s]["accuracy_first"] for s in seeds}
    frozen_acc = {s: frozen[s]["accuracy_last"] - frozen[s]["accuracy_first"] for s in seeds}

    def dw_ratio(seed: int) -> float:
        mine = statistics.mean(float(r["mean_abs_dw"])
                               for r in _last_block(_rows(args.summary, TREATMENT, seed), block))
        base = statistics.mean(float(r["mean_abs_dw"])
                               for r in _last_block(_rows(args.summary, "plastic", seed), block))
        return mine / base if base else float("nan")

    def stays(arm: str, seed: int) -> int:
        return sum(int(r["stayed"]) for r in _last_block(_rows(args.summary, arm, seed), block))

    ratios = {s: dw_ratio(s) for s in seeds}

    h4_threshold = next(float(h["relative_threshold"])
                        for h in manifest["hypotheses"] if h["id"] == "H4")
    h1 = all(treat_delta[s] > 0 for s in seeds)
    h2 = all(treat_delta[s] > v1_delta[s] for s in seeds)
    h3 = all(treat_delta[s] > shuffled_delta[s] for s in seeds)
    h4 = (all(v > 0 for v in treat_acc.values())
          and all(ratios[s] >= h4_threshold for s in seeds))
    gap_v1 = (statistics.median(treat[s]["success_last"] for s in seeds)
              - statistics.median(v1[s]["success_last"] for s in seeds))
    gap_shuffled = (statistics.median(treat[s]["success_last"] for s in seeds)
                    - statistics.median(shuffled[s]["success_last"] for s in seeds))
    gap_oracle = (statistics.median(treat[s]["success_last"] for s in seeds)
                  - statistics.median(oracle[s]["success_last"] for s in seeds)) if oracle else float("nan")

    # --- гейты ------------------------------------------------------------
    g1 = all(0.2 <= treat[s]["success_first"] <= 0.7 for s in seeds)
    g2 = all(tbl[s]["at_bound"] <= 0.5 for tbl in (treat, shuffled) for s in seeds)
    g3 = (all(c["violations"] == 0 for c in cells)
          and all(int(r["denied"]) == 0
                  for arm in REQUIRED_ARMS for s in seeds
                  for r in _rows(args.summary, arm, s)))
    g4 = not any(c["nan"] for c in cells)
    g5 = bool(oracle) and all(row["success_last"] == 1.0 for row in oracle.values())
    collapsed = []
    for s in seeds:
        tail = _last_block(_rows(args.summary, TREATMENT, s), block)
        left = sum(int(r["moved_left"]) for r in tail)
        right = sum(int(r["moved_right"]) for r in tail)
        if left < 20 or right < 20:
            collapsed.append(f"s{s}: влево {left}, вправо {right}")
    g6 = not collapsed
    rules = {(_meta(args.summary, arm, s).get("settings", {}).get("pool_contrast"),
              _meta(args.summary, arm, s).get("settings", {}).get("neuron_credit"))
             for arm in REQUIRED_ARMS for s in seeds}
    g7 = (len(rules) == 1
          and rules.pop() == (float(declared["pool_contrast"]),
                              float(declared["neuron_credit"])))
    modes = {arm: {_meta(args.summary, arm, s).get("neuron_credit") for s in seeds}
             for arm in REQUIRED_ARMS}
    g8 = all(modes[arm] == {declared_mode[arm]} for arm in REQUIRED_ARMS)

    print()
    for hid, ok, text in (
        ("H1", h1, "neuron: success последних 100 > первых 100 на всех seed'ах "
                   f"({[round(treat_delta[s], 3) for s in seeds]})"),
        ("H2", h2, "прирост neuron > прирост пластика v1 на том же seed'е "
                   f"({[round(treat_delta[s] - v1_delta[s], 3) for s in seeds]})"),
        ("H3", h3, "прирост neuron > прирост neuron_shuffled на том же seed'е "
                   f"({[round(treat_delta[s] - shuffled_delta[s], 3) for s in seeds]})"),
        ("H4", h4, "точность растёт у neuron и |dw| последних 100 >= 0.9 от v1 "
                   f"(acc {[round(v, 3) for v in treat_acc.values()]}, "
                   f"C {[round(ratios[s], 3) for s in seeds]}; "
                   f"заморозка {[round(v, 3) for v in frozen_acc.values()]})"),
        ("H5", True, f"neuron - plastic = {gap_v1:+.3f}, "
                         f"neuron - shuffled = {gap_shuffled:+.3f}, "
                         f"neuron - oracle = {gap_oracle:+.3f}; "
                         f"stay {[stays(TREATMENT, s) for s in seeds]} "
                         f"против {[stays('plastic', s) for s in seeds]} (характеризация)"),
    ):
        label = "ИЗМЕРЕНО" if hid == "H5" else ("ПРОЙДЕНО" if ok else "НЕ ПРОЙДЕНО")
        print(f"{hid}: {label} — {text}")

    for gid, ok, text in (
        ("G1", g1, "полоса сложности: success первых 100 у neuron в [0.2, 0.7]"),
        ("G2", g2, "насыщение: доля весов на границе <= 0.5 у neuron и neuron_shuffled"),
        ("G3", g3, "ворота: 0 нарушений, 0 отказов"),
        ("G4", g4, "числа: нет NaN/Inf"),
        ("G5", g5, "оракул: success последних 100 == 1.0"),
        ("G6", g6, "пул не схлопнулся: оба направления >= 20 в последних 100"),
        ("G7", g7, f"правило то: kappa={declared['pool_contrast']}, "
                   f"lambda={declared['neuron_credit']} в мете клеток"),
        ("G8", g8, "режим руки в мете совпадает с объявленным"),
    ):
        print(f"{gid}: {'OK' if ok else 'ПРОВАЛ'} — {text}" + (
            f" ({'; '.join(collapsed)})" if gid == "G6" and collapsed else ""))

    # --- вердикт по таблице исходов, объявленной до прогона -----------------
    a = h2
    b = h3
    required = h1 and h2 and h3 and h4 and g1 and g2 and g3 and g4 and g5 and g6 and g7 and g8

    print()
    if not g1:
        print("чтение: прогон неинформативен (полоса сложности), H1 не интерпретируется")
    elif not g6:
        print("чтение: пул схлопнулся, H2/H3 не интерпретируются")
    elif not g7 or not g8:
        print("чтение: рука считалась не тем режимом, что объявлен; "
              "сравнение недействительно")
    elif a and b:
        print("чтение: кредит по участию работает, и привязка к участию важна — "
              "рука становится рабочим правилом внутри этой физики")
    elif a and not b:
        print("чтение: выигрыш даёт сама неоднородность кредита, а не привязка к "
              "участию: рука принимается по числу, заявление о механизме не делается")
    elif not a and b:
        print("чтение: привязка к участию важна, но пер-нейронный кредит слабее "
              "скаляра v1 — рабочее правило остаётся v1")
    else:
        print("чтение: пер-нейронный кредит не помогает — гипотеза отвергнута, "
              "линия «форма третьей компоненты» закрыта, рабочее правило — v1")

    if required:
        print("вердикт: критерий v4 выполнен (описание при n=3, значимость не "
              "заявляется)")
    elif h1 and h2 and h3 and h4:
        print("вердикт: гипотезы пройдены, но гейты не закрыты — критерий v4 не выполнен")
    else:
        print("вердикт: критерий v4 не выполнен; публикуется как есть")
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
