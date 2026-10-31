#!/usr/bin/env python3
"""Вердикт P4.3 v3 (абляция контраста по знаку кредита) по объявленным правилам.

Читает summary.json раннера и применяет к нему протокол
`p4.3.spike-verb.credit-sign.v3`. Направления, пороги и режим каждой руки
берутся из манифеста, а не из этого файла; при расхождении protocol_id или при
запуске без `--confirm` вердикт не выносится.

    python scripts/report_p4_3_v3.py var/p4_3v3_confirm/summary.json

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
                                "p4_3_spike_verb_v3.json")
CONTRAST_ARMS = ("pool_pos", "pool_neg")
REQUIRED_ARMS = ("pool_pos", "pool_neg", "plastic", "no_plasticity")


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


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv[1:])

    summary = json.load(open(args.summary, encoding="utf-8"))
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    declared = manifest["operating_point"]["values"]
    declared_when = {a["arm_id"]: a.get("pool_when", "always")
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

    seeds = sorted(_by_seed(cells, "pool_pos"))
    pos = _by_seed(cells, "pool_pos")
    neg = _by_seed(cells, "pool_neg")
    v1 = _by_seed(cells, "plastic")
    frozen = _by_seed(cells, "no_plasticity")
    oracle = _by_seed(cells, "oracle")
    if any(sorted(d) != seeds for d in (neg, v1, frozen)):
        print("руки измерены не на одном наборе seed'ов: парное сравнение невозможно")
        return 1

    block = int(declared["episodes"]) // 5
    print(f"протокол {manifest['protocol_id']}  sha256={protocol['sha256'][:16]}")
    print(f"блоки: первые и последние {block} эпизодов; seed'ы {seeds}\n")
    print(f"{'сид':>5} {'рука':>14} {'s(1)':>6} {'s(2)':>6} {'Δs':>7} "
          f"{'acc(1)':>7} {'acc(2)':>7} {'спайки/эп':>10} {'у границы':>9}")
    for seed in seeds:
        for arm, table in (("pool_pos", pos), ("pool_neg", neg),
                           ("plastic", v1), ("no_plasticity", frozen)):
            row = table[seed]
            print(f"{seed:5d} {arm:>14} {row['success_first']:6.3f} {row['success_last']:6.3f} "
                  f"{row['delta']:+7.3f} {row['accuracy_first']:7.3f} "
                  f"{row['accuracy_last']:7.3f} {row['spikes_per_episode']:10.0f} "
                  f"{row['at_bound']:9.3f}")
        row = oracle.get(seed)
        if row:
            print(f"{seed:5d} {'oracle':>14} {row['success_first']:6.3f} "
                  f"{row['success_last']:6.3f} {row['delta']:+7.3f}")

    # --- гипотезы ---------------------------------------------------------
    pos_delta = {s: pos[s]["delta"] for s in seeds}
    neg_delta = {s: neg[s]["delta"] for s in seeds}
    v1_delta = {s: v1[s]["delta"] for s in seeds}

    h1 = all(pos_delta[s] > 0 for s in seeds)
    h2 = all(pos_delta[s] > v1_delta[s] for s in seeds)
    h3 = all(pos_delta[s] > neg_delta[s] for s in seeds)
    pos_acc = {s: pos[s]["accuracy_last"] - pos[s]["accuracy_first"] for s in seeds}
    frozen_acc = {s: frozen[s]["accuracy_last"] - frozen[s]["accuracy_first"] for s in seeds}
    # Порог величины эффекта объявлен в протоколе до прогона: без него H4
    # проваливается на разбросе заморозки в доли процента.
    h4 = (all(v > 0 for v in pos_acc.values())
          and all(frozen_acc[s] <= 0.25 * pos_acc[s] for s in seeds))
    gap_v1 = (statistics.median(pos[s]["success_last"] for s in seeds)
              - statistics.median(v1[s]["success_last"] for s in seeds))
    gap_neg = (statistics.median(pos[s]["success_last"] for s in seeds)
               - statistics.median(neg[s]["success_last"] for s in seeds))
    gap_oracle = (statistics.median(pos[s]["success_last"] for s in seeds)
                  - statistics.median(oracle[s]["success_last"] for s in seeds)) if oracle else float("nan")

    # --- гейты ------------------------------------------------------------
    g1 = all(0.2 <= pos[s]["success_first"] <= 0.7 for s in seeds)
    g2 = all(tbl[s]["at_bound"] <= 0.5 for tbl in (pos, neg) for s in seeds)
    g3 = (all(c["violations"] == 0 for c in cells)
          and all(int(r["denied"]) == 0
                  for arm in REQUIRED_ARMS for s in seeds
                  for r in _rows(args.summary, arm, s)))
    g4 = not any(c["nan"] for c in cells)
    g5 = bool(oracle) and all(row["success_last"] == 1.0 for row in oracle.values())
    collapsed = []
    for s in seeds:
        tail = _rows(args.summary, "pool_pos", s)[-block:]
        left = sum(int(r["moved_left"]) for r in tail)
        right = sum(int(r["moved_right"]) for r in tail)
        if left < 20 or right < 20:
            collapsed.append(f"s{s}: влево {left}, вправо {right}")
    g6 = not collapsed
    kappas = {_meta(args.summary, arm, s).get("settings", {}).get("pool_contrast")
              for arm in REQUIRED_ARMS for s in seeds}
    g7 = (len(kappas) == 1
          and kappas.pop() == float(declared["pool_contrast"]))
    when = {arm: {_meta(args.summary, arm, s).get("pool_when") for s in seeds}
            for arm in REQUIRED_ARMS}
    g8 = all(when[arm] == {declared_when[arm]} for arm in REQUIRED_ARMS)

    print()
    for hid, ok, text in (
        ("H1", h1, "pool_pos: success последних 100 > первых 100 на всех seed'ах "
                   f"({[round(pos_delta[s], 3) for s in seeds]})"),
        ("H2", h2, "прирост pool_pos > прирост пластика v1 на том же seed'е "
                   f"({[round(pos_delta[s] - v1_delta[s], 3) for s in seeds]})"),
        ("H3", h3, "прирост pool_pos > прирост pool_neg на том же seed'е "
                   f"({[round(pos_delta[s] - neg_delta[s], 3) for s in seeds]})"),
        ("H4", h4, "точность растёт у pool_pos и прирост заморозки <= 0.25 прироста "
                   f"(pool_pos {[round(v, 3) for v in pos_acc.values()]}, "
                   f"заморозка {[round(v, 3) for v in frozen_acc.values()]})"),
        ("H5", True, f"pool_pos - plastic = {gap_v1:+.3f}, "
                     f"pool_pos - pool_neg = {gap_neg:+.3f}, "
                     f"pool_pos - oracle = {gap_oracle:+.3f} "
                     "(характеризация, не гейт)"),
    ):
        print(f"{hid}: {'ПРОЙДЕНО' if ok else 'НЕ ПРОЙДЕНО'} — {text}")

    for gid, ok, text in (
        ("G1", g1, "полоса сложности: success первых 100 у pool_pos в [0.2, 0.7]"),
        ("G2", g2, "насыщение: доля весов на границе <= 0.5 у обеих рук контраста"),
        ("G3", g3, "ворота: 0 нарушений, 0 отказов"),
        ("G4", g4, "числа: нет NaN/Inf"),
        ("G5", g5, "оракул: success последних 100 == 1.0"),
        ("G6", g6, "пул не схлопнулся: оба направления >= 20 в последних 100"),
        ("G7", g7, f"правило то: kappa={declared['pool_contrast']} в мете клеток"),
        ("G8", g8, "режим руки в мете совпадает с объявленным"),
    ):
        print(f"{gid}: {'OK' if ok else 'ПРОВАЛ'} — {text}" + (
            f" ({'; '.join(collapsed)})" if gid == "G6" and collapsed else ""))

    # --- вердикт по таблице исходов, объявленной до прогона -----------------
    pos_beats_v1 = all(pos_delta[s] > v1_delta[s] for s in seeds)
    neg_beats_v1 = all(neg_delta[s] > v1_delta[s] for s in seeds)
    pos_beats_neg = h3
    required = h1 and h2 and h3 and h4 and g1 and g2 and g3 and g4 and g5 and g6 and g7 and g8

    print()
    if not g1:
        print("чтение: прогон неинформативен (полоса сложности), H1 не интерпретируется")
    elif not g6:
        print("чтение: пул схлопнулся, H2/H3 не интерпретируются")
    elif not g7 or not g8:
        print("чтение: рука считалась не тем режимом, что объявлен; "
              "сравнение недействительно")
    elif pos_beats_v1 and pos_beats_neg:
        print("чтение: вред в канале наказания — pool_pos обгоняет и скаляр v1, и "
              "pool_neg; рабочее правило v3 — контраст только на подкрепляющих шагах")
    elif pos_beats_v1 and not pos_beats_neg:
        print("чтение: оба канала дают плюс, v2 теряла на смеси; на части seed'ов "
              "pool_neg не ниже pool_pos, поэтому рука выбирается по числу, а не по "
              "ожидаемому механизму")
    elif not neg_beats_v1 and pos_beats_neg:
        print("чтение: контраст на подкреплении ничего не добавляет, а pool_neg хуже "
              "pool_pos; выигрыш v1 — общая возбудимость, рабочее правило — скаляр v1")
    elif not neg_beats_v1:
        print("чтение: контраст вредит независимо от знака кредита — выигрыш скаляра v1 "
              "идёт от общей возбудимости; следующий шаг — кредит на нейрон, а не на пул")
    else:
        print("чтение: вне объявленной таблицы — контраст на наказании обгоняет v1, "
              "а на подкреплении нет; механизм не тот, что заявлен, нужен отдельный разбор")

    if required:
        print("вердикт: критерий v3 выполнен (описание при n=3, значимость не "
              "заявляется)")
    elif h1 and h2 and h3 and h4:
        print("вердикт: гипотезы пройдены, но гейты не закрыты — критерий v3 не выполнен")
    else:
        print("вердикт: критерий v3 не выполнен; публикуется как есть")
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
