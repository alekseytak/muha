#!/usr/bin/env python3
"""Вердикт P4.6 (время доставки кредита) по объявленным правилам.

Прогон состоит из трёх grid'ов — отдельных запусков раннера с разной задержкой
`credit_delay_steps` и отдельными каталогами вывода, потому что имя клетки
задержку не содержит. Инструмент принимает summary.json каждого grid'а:

    python scripts/report_p4_6_credit_delay.py \
        var/p4_6_d0/summary.json var/p4_6_d20/summary.json var/p4_6_d80/summary.json

Grid определяется по `args.credit_delay` запуска и сверяется с манифестом:
каталог вывода, набор рук и набор seed'ов обязаны совпасть с объявленной записью
`grids`. Пороги, полоса предсказанной амплитуды, имена рук и текст чтения берутся
из манифеста, а не из этого файла; при расхождении protocol_id или при запуске
без `--confirm` вердикт не выносится.

Инструмент написан до подтверждающего прогона и проверен на синтетических
фикстурах — иначе вердикт подгонялся бы под уже увиденные числа.

Код возврата: 0 — все required-гипотезы и гейты пройдены, 1 — нет.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(REPO, "fly_connectome_agent", "manifests",
                                "p4_6_credit_delay.json")


def _by_seed(cells: list[dict], arm: str) -> dict[int, dict]:
    return {c["seed"]: c for c in cells if c["arm"] == arm}


def _rows(dir_path: str, arm: str, seed: int) -> list[dict]:
    path = os.path.join(dir_path, f"{arm}-s{seed}.tsv")
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _meta(dir_path: str, arm: str, seed: int) -> dict:
    path = os.path.join(dir_path, f"{arm}-s{seed}.meta.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _block_mean_dw(dir_path: str, arm: str, seed: int, block: int) -> float:
    rows = _rows(dir_path, arm, seed)[-block:]
    return statistics.mean(float(r["mean_abs_dw"]) for r in rows) if rows else float("nan")


def _block_verbs(dir_path: str, arm: str, seed: int, block: int) -> tuple[int, int]:
    rows = _rows(dir_path, arm, seed)[-block:]
    return (sum(int(r["moved_left"]) for r in rows),
            sum(int(r["moved_right"]) for r in rows))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv[1:])

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    declared = manifest["operating_point"]["values"]
    treat = manifest["treatment_arm"]
    ctrl = manifest["amplitude_control_arm"]
    ref = manifest["reference_arm"]
    primary = int(manifest["primary_delay_steps"])
    secondary = int(manifest["secondary_delay_steps"])
    grids = {int(g["credit_delay_steps"]): g for g in manifest["grids"]}
    declared_mode = {a["arm_id"]: a.get("delay_mode", "none")
                     for a in manifest["arms"]}
    tau = float(declared["tau_eligibility"])

    # --- сборка grid'ов ---------------------------------------------------
    loaded: dict[int, dict] = {}
    for path in args.summaries:
        summary = json.load(open(path, encoding="utf-8"))
        protocol = summary.get("protocol") or {}
        if protocol.get("protocol_id") != manifest["protocol_id"]:
            print(f"summary не от этого протокола: {path} -> "
                  f"{protocol.get('protocol_id')!r} != {manifest['protocol_id']!r}")
            return 1
        if not protocol.get("verified"):
            print(f"{path}: запуск без --confirm, вердикт не выносится")
            return 1
        delay = int(summary["args"]["credit_delay"])
        if delay in loaded:
            print(f"два summary объявлены как задержка {delay}: grid'ы смешаны")
            return 1
        grid = grids.get(delay)
        if grid is None:
            print(f"задержке {delay} не соответствует ни один grid манифеста "
                  f"({sorted(grids)})")
            return 1
        out = str(summary["args"]["out"])
        if out != str(grid["out"]):
            print(f"grid {grid['grid_id']}: каталог {out!r} != объявленный "
                  f"{grid['out']!r}")
            return 1
        have = sorted({c["arm"] for c in summary["cells"]})
        want = sorted(grid["arms"])
        if have != want:
            print(f"grid {grid['grid_id']}: руки {have} != объявленные {want}")
            return 1
        loaded[delay] = {"summary": summary, "dir": os.path.dirname(os.path.abspath(path)),
                         "grid": grid, "sha256": protocol["sha256"]}

    missing = sorted(set(grids) - set(loaded))
    if missing:
        print(f"нет summary для задержек {missing}: grid'а неполные, вердикт не выносится")
        return 1

    seed_sets = {d: sorted(_by_seed(g["summary"]["cells"], g["grid"]["arms"][0]))
                 for d, g in loaded.items()}
    if len({tuple(s) for s in seed_sets.values()}) != 1:
        print(f"seed'ы различаются между grid'ами: {seed_sets}")
        return 1
    seeds = seed_sets[primary]
    declared_seeds = manifest["seed_sets"]["confirmatory"]["values"]
    if sorted(seeds) != sorted(declared_seeds):
        print(f"seed'ы прогона {seeds} != объявленные подтвердительные {declared_seeds}")
        return 1

    block = int(declared["episodes"]) // 5
    print(f"протокол {manifest['protocol_id']}  sha256={loaded[primary]['sha256'][:16]}")
    print(f"блоки: первые и последние {block} эпизодов; seed'ы {seeds}; "
          f"основной уровень D={primary}, лестница D={secondary}\n")

    def table(delay: int) -> dict[str, dict[int, dict]]:
        return {arm: _by_seed(loaded[delay]["summary"]["cells"], arm)
                for arm in loaded[delay]["grid"]["arms"]}

    t0, t20, t80 = table(0), table(secondary), table(primary)
    for arm in (ref, "no_plasticity", "oracle"):
        if arm not in t0:
            print(f"в grid'е d0 нет руки {arm}")
            return 1
    print(f"{'сид':>5} {'D':>4} {'рука':>14} {'s(1)':>6} {'s(2)':>6} {'Δs':>7} "
          f"{'acc(1)':>7} {'acc(2)':>7} {'спайки/эп':>10} {'у границы':>9} {'доставок':>9}")
    for delay, tbl in ((0, t0), (secondary, t20), (primary, t80)):
        for arm in tbl:
            for seed in seeds:
                if seed not in tbl[arm]:
                    print(f"нет клетки {arm}-s{seed} при D={delay}")
                    return 1
                row = tbl[arm][seed]
                print(f"{seed:5d} {delay:4d} {arm:>14} {row['success_first']:6.3f} "
                      f"{row['success_last']:6.3f} {row['delta']:+7.3f} "
                      f"{row['accuracy_first']:7.3f} {row['accuracy_last']:7.3f} "
                      f"{row['spikes_per_episode']:10.0f} {row['at_bound']:9.3f} "
                      f"{row.get('deliveries', 0):9d}")

    dirs = {d: g["dir"] for d, g in loaded.items()}

    # --- величины ---------------------------------------------------------
    def delta(tbl: dict, arm: str) -> dict[int, float]:
        return {s: tbl[arm][s]["delta"] for s in seeds} if arm in tbl else {}

    d_ref = delta(t0, ref)
    d_live80, d_frz80 = delta(t80, treat), delta(t80, ctrl)
    d_live20, d_frz20 = delta(t20, treat), delta(t20, ctrl)

    def ratio(delay: int, arm: str) -> dict[int, float]:
        return {s: _block_mean_dw(dirs[delay], arm, s, block)
                / _block_mean_dw(dirs[0], ref, s, block) for s in seeds}

    c_frz80, c_live80 = ratio(primary, ctrl), ratio(primary, treat)
    c_frz20 = ratio(secondary, ctrl)
    acc_live80 = {s: t80[treat][s]["accuracy_last"] - t80[treat][s]["accuracy_first"]
                  for s in seeds}
    acc_ref = {s: t0[ref][s]["accuracy_last"] - t0[ref][s]["accuracy_first"]
               for s in seeds}
    med = lambda tbl, arm: statistics.median(tbl[arm][s]["success_last"] for s in seeds)

    h4_rule = next(h for h in manifest["hypotheses"] if h["id"] == "H4")
    band = float(h4_rule["relative_band"])
    expect = math.exp(-primary / tau)

    h1 = all(v > 0 for v in d_ref.values())
    a = all(d_live80[s] < d_ref[s] for s in seeds)          # H2
    b = all(d_live80[s] < d_frz80[s] for s in seeds)        # H3
    h2, h3 = a, b
    h4 = all(abs(c_frz80[s] / expect - 1.0) <= band for s in seeds)

    # --- гейты ------------------------------------------------------------
    watched = [(0, t0, ref), (secondary, t20, treat), (primary, t80, treat)]
    g1 = all(0.2 <= tbl[arm][s]["success_first"] <= 0.7
             for _, tbl, arm in watched for s in seeds)
    g2 = all(tbl[arm][s]["at_bound"] <= 0.5
             for _, tbl, arm in ((0, t0, ref), (primary, t80, treat),
                                 (primary, t80, ctrl)) for s in seeds)
    all_cells = [c for g in loaded.values() for c in g["summary"]["cells"]]
    g3 = (all(c["violations"] == 0 for c in all_cells)
          and all(int(r["denied"]) == 0
                  for delay, tbl in ((0, t0), (secondary, t20), (primary, t80))
                  for arm in tbl for s in seeds for r in _rows(dirs[delay], arm, s)))
    g4 = not any(c["nan"] for c in all_cells)
    g5 = all(t0["oracle"][s]["success_last"] == 1.0 for s in seeds)
    collapsed = []
    for s in seeds:
        left, right = _block_verbs(dirs[primary], treat, s, block)
        if left < 20 or right < 20:
            collapsed.append(f"s{s}: влево {left}, вправо {right}")
    g6 = not collapsed
    rules = {(_meta(dirs[d], arm, s).get("settings", {}).get("pool_contrast"),
              _meta(dirs[d], arm, s).get("settings", {}).get("neuron_credit"),
              _meta(dirs[d], arm, s).get("settings", {}).get("eta"),
              _meta(dirs[d], arm, s).get("settings", {}).get("tau_eligibility"))
             for d, tbl in ((0, t0), (secondary, t20), (primary, t80))
             for arm in tbl for s in seeds}
    g7 = rules == {(float(declared["pool_contrast"]), float(declared["neuron_credit"]),
                    float(declared["eta"]), float(declared["tau_eligibility"]))}
    modes_bad, delay_bad = [], []
    for d, tbl in ((0, t0), (secondary, t20), (primary, t80)):
        for arm in tbl:
            for s in seeds:
                meta = _meta(dirs[d], arm, s)
                if meta.get("delay_mode") != declared_mode.get(arm, "none"):
                    modes_bad.append(f"{arm}-s{s}@D{d}")
                if int(meta.get("settings", {}).get("credit_delay_steps", -1)) != d:
                    delay_bad.append(f"{arm}-s{s}@D{d}")
    g8 = not modes_bad and not delay_bad
    mech_bad = []
    for d, tbl in ((0, t0), (secondary, t20), (primary, t80)):
        for arm in tbl:
            for s in seeds:
                row = tbl[arm][s]
                dec, deliv, tail = (row.get("decisions", 0), row.get("deliveries", 0),
                                    row.get("pending_at_end", 0))
                want = dec - tail
                if deliv != want or (d > 0 and (deliv <= 0 or tail > 1)) \
                        or (d == 0 and deliv != dec):
                    mech_bad.append(f"{arm}-s{s}@D{d}: доставок {deliv}, "
                                    f"решений {dec}, хвост {tail}")
    g9 = not mech_bad
    g10 = (sorted(loaded) == sorted(grids)
           and all(str(loaded[d]["summary"]["args"]["out"]) == str(grids[d]["out"])
                   for d in loaded))

    print()
    for hid, ok, text in (
        ("H1", h1, f"plastic (D=0): success последних {block} > первых {block} на всех "
                   f"seed'ах ({[round(d_ref[s], 3) for s in seeds]})"),
        ("H2", h2, f"прирост delay_live при D={primary} ниже прироста plastic на всех "
                    f"seed'ах (разность "
                    f"{[round(d_ref[s] - d_live80[s], 3) for s in seeds]})"),
        ("H3", h3, f"прирост delay_live при D={primary} ниже прироста delay_frozen при "
                    f"том же D (разность "
                    f"{[round(d_frz80[s] - d_live80[s], 3) for s in seeds]})"),
        ("H4", h4, f"амплитуда delay_frozen при D={primary} в полосе +-{band} от "
                   f"exp(-{primary}/{tau:g}) = {expect:.4f} "
                   f"(C frozen {[round(c_frz80[s], 3) for s in seeds]}, "
                   f"C live {[round(c_live80[s], 3) for s in seeds]})"),
        ("H5", True, f"лестница: D={secondary} live "
                     f"{[round(d_live20[s], 3) for s in seeds]} / frozen "
                     f"{[round(d_frz20[s], 3) for s in seeds]}, D={primary} live "
                     f"{[round(d_live80[s], 3) for s in seeds]} / frozen "
                     f"{[round(d_frz80[s], 3) for s in seeds]}; C frozen при "
                     f"D={secondary} {[round(c_frz20[s], 3) for s in seeds]} "
                     f"(предсказание {math.exp(-secondary / tau):.4f}); "
                     f"медианы s(2): live {med(t80, treat):.3f}, "
                     f"frozen {med(t80, ctrl):.3f}, ref {med(t0, ref):.3f}, "
                     f"oracle {med(t0, 'oracle'):.3f}; acc live "
                     f"{[round(acc_live80[s], 3) for s in seeds]} против ref "
                     f"{[round(acc_ref[s], 3) for s in seeds]} (характеризация)"),
    ):
        label = "ИЗМЕРЕНО" if hid == "H5" else ("ПРОЙДЕНО" if ok else "НЕ ПРОЙДЕНО")
        print(f"{hid}: {label} — {text}")

    for gid, ok, text in (
        ("G1", g1, f"полоса сложности: success первых {block} у руки сравнения "
                   f"в [0.2, 0.7]"),
        ("G2", g2, f"насыщение: доля весов на границе <= 0.5 у plastic и обеих "
                   f"delay-рук при D={primary}"),
        ("G3", g3, "ворота: 0 нарушений, 0 отказов"),
        ("G4", g4, "числа: нет NaN/Inf"),
        ("G5", g5, "оракул: success последних 100 == 1.0"),
        ("G6", g6, f"пул не схлопнулся: оба направления >= 20 в последних {block} "
                   f"у delay_live при D={primary}"),
        ("G7", g7, f"правило то: kappa={declared['pool_contrast']}, "
                   f"lambda={declared['neuron_credit']}, eta={declared['eta']}, "
                   f"tau={declared['tau_eligibility']} в метах клеток"),
        ("G8", g8, f"режим руки и задержка в мете совпадают с объявленными "
                   f"(чужих режимов: {modes_bad or 'нет'}, чужих задержек: "
                   f"{delay_bad or 'нет'})"),
        ("G9", g9, "механика исполнялась: доставки сверяются с решениями и хвостом "
                   f"(расхождения: {mech_bad or 'нет'})"),
        ("G10", g10, f"целостность grid'ов: задержки {sorted(loaded)} из "
                     f"{sorted(grids)}, каталоги объявлены, protocol_id и --confirm "
                     f"на месте"),
    ):
        print(f"{gid}: {'OK' if ok else 'ПРОВАЛ'} — {text}" + (
            f" ({'; '.join(collapsed)})" if gid == "G6" and collapsed else ""))

    # --- вердикт по таблице исходов, объявленной до прогона -----------------
    required = (all([h1, h2, h3, h4])
                and all([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]))
    print()
    if not g1:
        print("чтение: прогон неинформативен (полоса сложности), H1-H3 не интерпретируются")
    elif not g6:
        print("чтение: пул схлопнулся, H2/H3 не интерпретируются")
    elif not h4:
        print(f"чтение: амплитуда задержки не сошлась с предсказанием "
              f"exp(-{primary}/{tau:g}) — механизм сделал не то, что объявлено, "
              "ни A, ни B не читаются как время доставки")
    elif not (g7 and g8 and g9 and g10):
        broken = [gid for gid, ok in (("G7", g7), ("G8", g8), ("G9", g9), ("G10", g10))
                  if not ok]
        print(f"чтение [{', '.join(broken)}]: клетка считалась не тем правилом, "
              "режимом, не той задержкой или не с тем счётом доставок, что "
              "объявлены; сравнение недействительно")
    else:
        row = next(r for r in manifest["outcome_table"] if r["A"] == a and r["B"] == b)
        print(f"чтение (A={a}, B={b}): {row['reading']}")

    if required:
        print(f"вердикт: критерий v6 выполнен (описание при n=3, значимость не "
              "заявляется)")
    elif h1 and h2 and h3 and h4:
        print("вердикт: гипотезы пройдены, но гейты не закрыты — критерий v6 не выполнен")
    else:
        print("вердикт: критерий v6 не выполнен; публикуется как есть")
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
