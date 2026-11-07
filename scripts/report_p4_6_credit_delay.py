#!/usr/bin/env python3
"""Вердикт P4.6 (время доставки кредита) по объявленным правилам.

Прогон состоит из нескольких grid'ов — отдельных запусков раннера с разной
задержкой `credit_delay_steps` и отдельными каталогами вывода, потому что имя
клетки задержку не содержит. Инструмент принимает summary.json каждого grid'а:

    python scripts/report_p4_6_credit_delay.py \
        var/p4_6_d0/summary.json var/p4_6_d20/summary.json var/p4_6_d80/summary.json

и умеет два режима. Подтверждающий (по умолчанию) сверяется с `grids`, требует
`--confirm` у каждого запуска и выносит вердикт по таблице исходов. Пилотный
(`--pilot`) берёт `pilot_grids` и `seed_sets.pilot`, допускает запуск без
`--confirm` на сожжённом seed'е, но требует совпадения sha256 манифеста, и
вердикт не выносит: у него одна работа — решить, можно ли идти на свежие seed'ы.

Grid определяется по `args.credit_delay` запуска и сверяется с манифестом:
каталог вывода, набор рук и набор seed'ов обязаны совпасть с объявленной записью.
Пороги, допуск механической проверки, имена рук и текст чтения берутся из
манифеста, а не из этого файла; при расхождении protocol_id или при запуске без
`--confirm` вердикт не выносится.

Форма механической проверки H4 читается из манифеста: `scope: within_cell`
(P4.6 v2) считает её по доставленной и заработанной массе одной клетки, без
этого поля (v1) остаётся межрукое отношение mean|dw| с полосой.

Инструмент написан до подтверждающего прогона и проверен на синтетических
фикстурах — иначе вердикт подгонялся бы под уже увиденные числа.

Код возврата: 0 — все required-гипотезы и гейты (или критерии пилота) пройдены,
1 — нет.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
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


def _within_ratio(row: dict) -> float | None:
    """Отношение доставленной массы кредита к его массе на момент действия.

    Обе величины считаются внутри одной клетки по одним и тем же событиям
    доставки (P4.6 v2), поэтому отношение — константа механизма, а не свойство
    траектории. None возвращает, если знаменателя нет: клетка без пластичности
    не может ничего ни заработать, ни доставить, и это не провал гейта.
    """
    at = float(row.get("credit_at_action") or 0.0)
    if at <= 0.0:
        return None
    return float(row.get("credit_delivered") or 0.0) / at


def _guard_thresholds(manifest: dict) -> dict:
    """Пороги гейтов берутся из текста правил манифеста, а не из этого файла.

    Проверка, порог которой живёт в том же коде, что и сама проверка, не может
    расходиться с протоколом только потому, что протокол переписали. Если в
    тексте правила числа нет, остаётся объявленное умолчание — оно совпадает с
    значениями v1 и v2, так что подмена молча не произойдёт.
    """
    rules = {g["id"]: str(g.get("rule", "")) for g in manifest["guards"]}

    def nums(gid: str, pattern: str, defaults: tuple[float, ...]) -> tuple:
        found = re.findall(pattern, rules.get(gid, ""))
        return tuple(float(v) for v in found) if found else defaults

    band = nums("G1", r"\[([0-9.]+), *([0-9.]+)\]", (0.2, 0.7))
    bound = nums("G2", r"<= *([0-9.]+)", (0.5,))
    verbs = nums("G6", r">= *([0-9.]+)", (20,))
    return {"band_lo": band[0], "band_hi": band[1], "bound": bound[0],
            "verbs": verbs[0], "declared": set(rules)}


def _load_grids(manifest: dict, paths: list[str], digest: str,
                *, pilot: bool) -> tuple[dict[int, dict], list[str]]:
    """Загрузка summary и сверка с объявленными grid'ами.

    Возвращает (loaded, список проблем) — пустой список проблем означает, что
    каталоги, руки, seed'ы и протокол совпали с манифестом. В пилотном режиме
    сетка берётся из `pilot_grids`, а требование `--confirm` заменяется
    требованием совпасть sha256 манифеста: пилот идёт на сожжённом seed'е и
    подтвердительным набором быть не может, но от версии протокола он зависеть
    обязан.
    """
    key = "pilot_grids" if pilot else "grids"
    if not manifest.get(key):
        return {}, [f"в манифесте нет {key}: этот режим не объявлен"]
    grids = {int(g["credit_delay_steps"]): g for g in manifest[key]}
    problems: list[str] = []
    loaded: dict[int, dict] = {}
    for path in paths:
        summary = json.load(open(path, encoding="utf-8"))
        protocol = summary.get("protocol") or {}
        if protocol.get("protocol_id") != manifest["protocol_id"]:
            problems.append(f"summary не от этого протокола: {path} -> "
                            f"{protocol.get('protocol_id')!r} != "
                            f"{manifest['protocol_id']!r}")
            continue
        if pilot:
            if protocol.get("sha256") != digest:
                problems.append(f"{path}: пилот сделан по другой версии манифеста "
                                f"({str(protocol.get('sha256'))[:16]} != "
                                f"{digest[:16]})")
        elif not protocol.get("verified"):
            problems.append(f"{path}: запуск без --confirm, вердикт не выносится")
        delay = int(summary["args"]["credit_delay"])
        if delay in loaded:
            problems.append(f"два summary объявлены как задержка {delay}: "
                            "grid'ы смешаны")
            continue
        grid = grids.get(delay)
        if grid is None:
            problems.append(f"задержке {delay} не соответствует ни один grid "
                            f"манифеста ({sorted(grids)})")
            continue
        out = str(summary["args"]["out"])
        if out != str(grid["out"]):
            problems.append(f"grid {grid['grid_id']}: каталог {out!r} != "
                            f"объявленный {grid['out']!r}")
        have = sorted({c["arm"] for c in summary["cells"]})
        want = sorted(grid["arms"])
        if have != want:
            problems.append(f"grid {grid['grid_id']}: руки {have} != "
                            f"объявленные {want}")
        loaded[delay] = {"summary": summary,
                        "dir": os.path.dirname(os.path.abspath(path)),
                        "grid": grid, "sha256": protocol.get("sha256")}

    missing = sorted(set(grids) - set(loaded))
    if missing:
        problems.append(f"нет summary для задержек {missing}: grid'а неполные, "
                        "вердикт не выносится")
    if problems:
        return loaded, problems

    seed_sets = {d: sorted(_by_seed(g["summary"]["cells"], g["grid"]["arms"][0]))
                 for d, g in loaded.items()}
    if len({tuple(s) for s in seed_sets.values()}) != 1:
        return loaded, [f"seed'ы различаются между grid'ами: {seed_sets}"]
    set_name = "pilot" if pilot else "confirmatory"
    label = "пилотные" if pilot else "подтвердительные"
    got_seeds = seed_sets[next(iter(sorted(loaded)))]
    declared_seeds = manifest["seed_sets"][set_name]["values"]
    if sorted(got_seeds) != sorted(declared_seeds):
        return loaded, [f"seed'ы прогона {got_seeds} != объявленные {label} "
                       f"{declared_seeds}"]
    return loaded, []


def pilot_main(manifest: dict, paths: list[str], digest: str) -> int:
    """Критерии выполнимости пилота на сожжённом seed'е.

    От подтверждающего режима отличается тремя вещами: сетка берётся из
    `pilot_grids`, запуск без `--confirm` допустим (но sha256 манифеста обязан
    совпасть — иначе пилот проверяет не тот протокол), и вердикт по таблице
    исходов не выносится: пилот ничего не подтверждает, он решает, можно ли
    вообще идти на свежие seed'ы. Пороги не переписываются здесь: они те же
    строки правил манифеста, что читаются в подтверждающем режиме, плюс точное
    внутриклеточное отношение H4.
    """
    declared = manifest["operating_point"]["values"]
    treat = manifest["treatment_arm"]
    ctrl = manifest["amplitude_control_arm"]
    ref = manifest["reference_arm"]
    primary = int(manifest["primary_delay_steps"])
    tau = float(declared["tau_eligibility"])
    h4_rule = next(h for h in manifest["hypotheses"] if h["id"] == "H4")
    th = _guard_thresholds(manifest)
    tol = float(h4_rule.get("tolerance_rel", 0.0))
    gate_arm = str(h4_rule.get("per_arm", ctrl))

    loaded, problems = _load_grids(manifest, paths, digest, pilot=True)
    if problems:
        for problem in problems:
            print(problem)
        return 1

    seeds = sorted(_by_seed(next(iter(loaded.values()))["summary"]["cells"],
                            next(iter(loaded.values()))["grid"]["arms"][0]))
    block = int(declared["episodes"]) // 5
    dirs = {d: g["dir"] for d, g in loaded.items()}
    tbls = {d: {arm: _by_seed(g["summary"]["cells"], arm)
                for arm in g["grid"]["arms"]} for d, g in loaded.items()}
    print(f"пилот {manifest['protocol_id']}  sha256={digest[:16]}  "
          f"seed'ы {seeds}  блоки {block} эпизодов")
    for delay in sorted(tbls):
        for arm in sorted(tbls[delay]):
            for s in seeds:
                row = tbls[delay][arm][s]
                print(f"  D={delay:>3} {arm:>13} s{s}: s(1)={row['success_first']:.3f} "
                      f"s(2)={row['success_last']:.3f} delta={row['delta']:+.3f} "
                      f"acc {row['accuracy_first']:.3f}->{row['accuracy_last']:.3f} "
                      f"доставок {row.get('deliveries', 0)} "
                      f"масса {row.get('credit_at_action', 0.0):.6g}/"
                      f"{row.get('credit_delivered', 0.0):.6g}")

    def ratio(delay: int, arm: str) -> dict[int, float | None]:
        return {s: _within_ratio(tbls[delay][arm][s]) for s in seeds}

    checks: list[tuple[str, bool, str]] = []
    band_cells = [(ref, 0)] + [(treat, d) for d in tbls if d > 0]
    bad = [f"{arm}-s{s}@D{d}" for arm, d in band_cells for s in seeds
           if not (th["band_lo"] <= tbls[d][arm][s]["success_first"] <= th["band_hi"])]
    checks.append(("G1 полоса сложности", not bad,
                   f"[{th['band_lo']}, {th['band_hi']}] (чужих: {bad or 'нет'})"))

    bound_cells = [(ref, 0), (treat, primary), (ctrl, primary)]
    bad = [f"{arm}-s{s}@D{d}" for arm, d in bound_cells for s in seeds
           if float(tbls[d][arm][s]["at_bound"]) > th["bound"]]
    checks.append(("G2 насыщение", not bad,
                   f"<= {th['bound']} (чужих: {bad or 'нет'})"))

    bad = []
    for s in seeds:
        left, right = _block_verbs(dirs[primary], treat, s, block)
        if left < th["verbs"] or right < th["verbs"]:
            bad.append(f"s{s}: влево {left}, вправо {right}")
    checks.append(("G6 пул не схлопнулся", not bad, f">= {th['verbs']:.0f} каждого "
                                                   f"глагола в последних {block} "
                                                   f"({'; '.join(bad) or 'ровно'})"))

    if h4_rule.get("scope") == "within_cell":
        late = [(d, ratio(d, gate_arm)) for d in sorted(tbls) if d > 0]
        bad = [f"{gate_arm}-s{s}@D{d}: {r if r is None else round(r, 9)} != "
               f"{math.exp(-d / tau):.9f}"
               for d, rr in late for s, r in rr.items()
               if r is None or abs(r / math.exp(-d / tau) - 1.0) > tol]
        zero = ratio(0, ref)
        bad += [f"{ref}-s{s}@D0: {r} != 1" for s, r in zero.items()
                if r is None or abs(r - 1.0) > tol]
        checks.append(("H4 механика внутри клетки", not bad,
                       f"внутриклеточное отношение = exp(-D/{tau:g}) с допуском {tol:g}; "
                       f"у frozen D={primary}: "
                       f"{[None if v is None else round(v, 9) for v in ratio(primary, gate_arm).values()]} "
                       f"(ожидание {math.exp(-primary / tau):.9f}), у {ref} D=0: "
                       f"{[None if v is None else round(v, 9) for v in zero.values()]} "
                       f"(расхождения: {bad or 'нет'})"))

        r_live, r_frz = ratio(primary, treat), ratio(primary, ctrl)
        bad = [f"s{s}: live {r_live[s]} <= frozen {r_frz[s]}" for s in seeds
               if r_live[s] is None or r_frz[s] is None or r_live[s] <= r_frz[s]]
        checks.append(("адресность наблюдается", not bad,
                       f"у live при D={primary} отношение выше, чем у frozen: "
                       f"{[None if v is None else round(v, 4) for v in r_live.values()]} "
                       f"против "
                       f"{[None if v is None else round(v, 4) for v in r_frz.values()]} "
                       f"(иначе у B нет предмета)"))
    else:
        checks.append(("H4 механика", False,
                       "в манифесте нет внутриклеточного гейта: пилотный режим для "
                       "такой версии не объявлен"))

    failed = []
    for name, ok, text in checks:
        print(f"ПИЛОТ {name}: {'ОК' if ok else 'НЕТ'} — {text}")
        if not ok:
            failed.append(name)
    if failed:
        print(f"пилот не прошёл: {', '.join(failed)} — подтверждающий на свежие "
              "seed'ы не выходит")
        return 1
    print("пилот пройден: критерии выполнимости закрыты, подтверждающий может идти")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--pilot", action="store_true",
                    help="режим пилота: объявленные критерии выполнимости на "
                         "сожжённом seed'е, вердикт по таблице исходов не "
                         "выносится")
    args = ap.parse_args(argv[1:])

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    digest = hashlib.sha256(open(args.manifest, "rb").read()).hexdigest()
    if args.pilot:
        return pilot_main(manifest, args.summaries, digest)
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
    loaded, problems = _load_grids(manifest, args.summaries, digest, pilot=False)
    if problems:
        for problem in problems:
            print(problem)
        return 1
    seeds = sorted(_by_seed(loaded[primary]["summary"]["cells"],
                            grids[primary]["arms"][0]))

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
    th = _guard_thresholds(manifest)
    band = float(h4_rule.get("relative_band", 0.0))
    expect = math.exp(-primary / tau)
    within = h4_rule.get("scope") == "within_cell"
    tol = float(h4_rule.get("tolerance_rel", 0.0))
    gate_arm = str(h4_rule.get("per_arm", ctrl))

    def ratio_within(delay: int, arm: str) -> dict[int, float | None]:
        tbl = table(delay)
        return {s: _within_ratio(tbl[arm][s]) for s in seeds}

    h1 = all(v > 0 for v in d_ref.values())
    a = all(d_live80[s] < d_ref[s] for s in seeds)          # H2
    b = all(d_live80[s] < d_frz80[s] for s in seeds)        # H3
    h2, h3 = a, b
    if within:
        r_frz20 = ratio_within(secondary, gate_arm)
        r_frz80 = ratio_within(primary, gate_arm)
        r_live20 = ratio_within(secondary, treat)
        r_live80 = ratio_within(primary, treat)
        r_d0 = ratio_within(0, ref)
        d0_equal = all(r is not None and abs(r - 1.0) <= tol for r in r_d0.values())
        h4 = d0_equal and all(
            r is not None and abs(r / math.exp(-delay / tau) - 1.0) <= tol
            for delay, rr in ((secondary, r_frz20), (primary, r_frz80))
            for r in rr.values())
    else:
        h4 = all(abs(c_frz80[s] / expect - 1.0) <= band for s in seeds)

    # --- гейты ------------------------------------------------------------
    watched = [(0, t0, ref), (secondary, t20, treat), (primary, t80, treat)]
    g1 = all(th["band_lo"] <= tbl[arm][s]["success_first"] <= th["band_hi"]
             for _, tbl, arm in watched for s in seeds)
    g2 = all(tbl[arm][s]["at_bound"] <= th["bound"]
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
        if left < th["verbs"] or right < th["verbs"]:
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
    mass_bad = []
    for d, tbl in ((0, t0), (secondary, t20), (primary, t80)):
        for arm in (ref, treat, ctrl):
            if arm not in tbl:
                continue
            for s in seeds:
                at = float(tbl[arm][s].get("credit_at_action") or 0.0)
                got = float(tbl[arm][s].get("credit_delivered") or 0.0)
                tag = f"{arm}-s{s}@D{d}"
                if at <= 0.0:
                    mass_bad.append(f"{tag}: массы на момент действия нет")
                elif d == 0 and abs(got - at) > tol:
                    mass_bad.append(f"{tag}: при D=0 счётчики не равны "
                                    f"({got:.6g} != {at:.6g})")
                elif d > 0 and arm == gate_arm and got / at > 1.0 + 1e-9:
                    mass_bad.append(f"{tag}: замороженный контроль доставил больше, "
                                    f"чем заработал ({got / at:.6g} > 1)")
    g11 = not mass_bad

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
        ("H4", h4, (
            f"внутриклеточный гейт: доставленная масса к массе на момент действия у "
            f"{gate_arm} = exp(-D/{tau:g}) с допуском {tol:g} — "
            f"D={secondary}: {[None if r_frz20[s] is None else round(r_frz20[s], 9) for s in seeds]} "
            f"(ожидание {math.exp(-secondary / tau):.4f}), "
            f"D={primary}: {[None if r_frz80[s] is None else round(r_frz80[s], 9) for s in seeds]} "
            f"(ожидание {expect:.4f}); {ref} при D=0: "
            f"{[None if r_d0[s] is None else round(r_d0[s], 9) for s in seeds]}; "
            f"у живых рук то же отношение свободное: "
            f"{[None if r_live80[s] is None else round(r_live80[s], 4) for s in seeds]}"
            if within else
            f"амплитуда delay_frozen при D={primary} в полосе +-{band} от "
            f"exp(-{primary}/{tau:g}) = {expect:.4f} "
            f"(C frozen {[round(c_frz80[s], 3) for s in seeds]}, "
            f"C live {[round(c_live80[s], 3) for s in seeds]})")),
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

    gate_rows = [
        ("G1", g1, f"полоса сложности: success первых {block} у руки сравнения "
                   f"в [{th['band_lo']}, {th['band_hi']}]"),
        ("G2", g2, f"насыщение: доля весов на границе <= {th['bound']} у plastic и обеих "
                   f"delay-рук при D={primary}"),
        ("G3", g3, "ворота: 0 нарушений, 0 отказов"),
        ("G4", g4, "числа: нет NaN/Inf"),
        ("G5", g5, "оракул: success последних 100 == 1.0"),
        ("G6", g6, f"пул не схлопнулся: оба направления >= {th['verbs']:.0f} в последних "
                   f"{block} у delay_live при D={primary}"),
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
    ]
    if "G11" in th["declared"]:
        gate_rows.append(("G11", g11, "счётчики массы кредита есть и сведены: "
                                       "plastic и обе delay-руки имеют заработанную "
                                       f"массу, при D=0 счётчики равны (допуск {tol:g}), "
                                       "замороженный контроль не доставляет больше, чем "
                                       f"заработал (расхождения: {mass_bad or 'нет'})"))
    for gid, ok, text in gate_rows:
        print(f"{gid}: {'OK' if ok else 'ПРОВАЛ'} — {text}" + (
            f" ({'; '.join(collapsed)})" if gid == "G6" and collapsed else ""))

    # --- вердикт по таблице исходов, объявленной до прогона -----------------
    active_gates = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10] + ([g11] if "G11"
                                                                 in th["declared"] else [])
    required = all([h1, h2, h3, h4]) and all(active_gates)
    print()
    if not g1:
        print("чтение: прогон неинформативен (полоса сложности), H1-H3 не интерпретируются")
    elif not g6:
        print("чтение: пул схлопнулся, H2/H3 не интерпретируются")
    elif not h4:
        if within:
            print(f"чтение: внутриклеточное отношение доставленного кредита к "
                  f"заработанному на момент действия не сошлось с exp(-D/{tau:g}) — "
                  "механизм сделал не то, что объявлено, ни A, ни B не читаются как "
                  "время доставки")
        else:
            print(f"чтение: амплитуда задержки не сошлась с предсказанием "
                  f"exp(-{primary}/{tau:g}) — механизм сделал не то, что объявлено, "
                  "ни A, ни B не читаются как время доставки")
    elif not (g7 and g8 and g9 and g10 and (g11 if "G11" in th["declared"] else True)):
        broken = [gid for gid, ok in (("G7", g7), ("G8", g8), ("G9", g9), ("G10", g10),
                                      ("G11", g11))
                  if not ok and gid in th["declared"]]
        print(f"чтение [{', '.join(broken)}]: клетка считалась не тем правилом, "
              "режимом, не той задержкой или не с тем счётом доставок, что "
              "объявлены; сравнение недействительно")
    else:
        row = next(r for r in manifest["outcome_table"] if r["A"] == a and r["B"] == b)
        print(f"чтение (A={a}, B={b}): {row['reading']}")

    criterion = str(manifest.get("criterion_label", "v6"))
    if required:
        print(f"вердикт: критерий {criterion} выполнен (описание при n=3, значимость не "
              "заявляется)")
    elif h1 and h2 and h3 and h4:
        print(f"вердикт: гипотезы пройдены, но гейты не закрыты — критерий {criterion} "
              "не выполнен")
    else:
        print(f"вердикт: критерий {criterion} не выполнен; публикуется как есть")
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
