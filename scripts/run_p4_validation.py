#!/usr/bin/env python
"""P4 / P4.1 — научная валидация MVP на симметричной toy-сети.

Матрица: 7 условий × 2 зеркальные задачи × N seeds.

Условия делятся на два класса, и это принципиально для трактовки Δw.

Пластичные (обучение есть, разный старт):
  plastic             R-STDP + награда, симметричный старт — главный режим
  weight_shuffled     то же обучение, но старт — случайная перестановка весов
  direction_shuffled  то же обучение, но старт — анти-рефлекс (wrong prior)

Замороженные (Δw ≡ 0 по конструкции) — вот они и есть baseline:
  no_plasticity             симметричная проводка, награда течёт, весов нет
  m_zero                    R-STDP, но M ≡ 0 (профиль no_modulator) — тайминг без награды
  weight_shuffled_frozen    «хватает ли просто случайной асимметрии»
  direction_shuffled_frozen анти-рефлекс, который никогда не переучивается

Контроль по Δw имеют право давать только замороженные условия: пластичный
shuffled-старт тоже вырабатывает специфичность, и называть его «контролем
обучения» значит мерить не то.

Задачи right/left — точные зеркала: одна длина, одно расстояние до цели,
стартовые позиции симметричны. Веса симметричны, задержки равны,
нарушитель симметрии один — шум (sigma), детерминированный по seed.

Третья задача — `mixed`: те же два зеркала чередуются внутри одного обучения
(чётность эпизода, правило одинаково для всех arms и всех seed). Она нужна
именно для критерия «лучше всех baseline»: на одиночной задаче случайно
асимметричная заморозка может выигрывать просто потому, что ей повезло с
направлением. На mixed повезть с обоими направлениями замороженной проводке
нельзя, поэтому там сравнивается устойчивость к обоим зеркалам.

Критерий P4.1 (главный): после обучения
  A (цель справа):  Δw(S_R→C_R) > Δw(S_L→C_L)
  B (цель слева):   Δw(S_L→C_L) > Δw(S_R→C_R)
плюс специфичность внутри стимулируемого сенсора (прямое − контралатеральное),
и ни одно замороженное условие её не даёт.

Запуск:  .venv/bin/python scripts/run_p4_validation.py [--seeds 20] [--episodes 20]
Быстрый прогон:  --quick   (4 seed × 6 эпизодов)
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fly_connectome_agent.src.engineering.harness import symmetric_toy as toy
from fly_connectome_agent.src.engineering.harness.symmetric_toy import (
    ARMS,
    BASELINE_ARMS,
    LEARNING_ARMS,
    MIXED,
    RunResult,
    ToySettings,
    run_arm,
)

ARM_NAMES = list(ARMS)
from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog

try:
    from scipy.stats import binomtest

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

# Позиции рёбер в dw-векторе выбранного plastic-набора.
DEFAULT_SETTINGS = ToySettings()
LABELS = toy.labels_for(DEFAULT_SETTINGS.plastic_slots)
POS = toy.edge_positions(DEFAULT_SETTINGS.plastic_slots)
DIRECT = {"left": POS["direct_L"], "right": POS["direct_R"]}
CONTRA = {"left": POS["contra_L"], "right": POS["contra_R"]}


def collect(arms: list[str], tasks: list[str], seeds: list[int], episodes: int, settings: ToySettings,
            prov: ProvenanceLog | None, verbose: bool) -> list[dict]:
    rows: list[dict] = []
    total = len(arms) * len(seeds) * len(tasks)
    done = 0
    t0 = time.time()
    for arm in arms:
        for task in tasks:
            for seed in seeds:
                r = run_arm(arm, task, seed, episodes=episodes, settings=settings,
                            prov=prov, verbose=verbose)
                rows.append(flatten(r))
                done += 1
                if done % 20 == 0:
                    print(f"  ... {done}/{total} ячеек, {time.time() - t0:.0f}с", flush=True)
    print(f"прогон завершён: {done} ячеек за {time.time() - t0:.0f}с", flush=True)
    return rows


def flatten(r: RunResult) -> dict:
    row = {
        "arm": r.arm, "task": r.task, "seed": r.seed,
        "learner": r.learner_kind, "init_regime": r.init_regime,
        "reward_coupled": int(r.reward_coupled),
        "episodes": len(r.episodes),
        "success_rate": r.success_rate,
        "steps_to_target": r.steps_to_target,
        "cumulative_reward": r.cumulative_reward,
        "moved_left": r.moved_left,
        "moved_right": r.moved_right,
        "choice_rate_right": r.choice_rate_right,
        "mean_abs_dw": r.mean_abs_dw,
        "fraction_weights_at_bound": r.fraction_at_bound,
        "spikes_cmd_left": r.spikes_per_neuron[0],
        "spikes_cmd_right": r.spikes_per_neuron[1],
        "governance_violations": r.governance_violations,
        "success_first_half": float(np.mean([e.reached for e in r.episodes[: max(1, len(r.episodes) // 2)]])),
        "success_second_half": float(np.mean([e.reached for e in r.episodes[-max(1, len(r.episodes) // 2):]])),
    }
    # На mixed зеркала идут через один эпизод, поэтому важна разбивка по
    # половинам потока: замороженная асимметрия даёт провал в одной из них.
    for side in ("right", "left"):
        hits = [e.reached for e in r.episodes if e.task == side]
        row[f"success_mirror[{side}]"] = float(np.mean(hits)) if hits else float("nan")
    halves = [row[f"success_mirror[{side}]"] for side in ("right", "left")]
    row["success_mirror_min"] = float(np.min(halves)) if not any(h != h for h in halves) else float("nan")
    for label, value in zip(LABELS, r.dw_per_edge):
        row[f"dw[{label}]"] = float(value)
    return row


def sub(rows: list[dict], arm: str, task: str) -> list[dict]:
    return [r for r in rows if r["arm"] == arm and r["task"] == task]


def mean_of(rows: list[dict], key: str) -> float:
    # r.get: replay of a CSV saved before a metric existed should print a gap,
    # not crash the verdict.
    vals = [v for v in (r.get(key) for r in rows) if isinstance(v, (int, float)) and v == v]
    return float(np.mean(vals)) if vals else float("nan")


def paired_sign_test(a: list[float], b: list[float], tol: float = 1e-12) -> tuple[int, int, float]:
    """Сколько раз a>b против b>a на одних и тех же seed; p — биномиальный."""
    wins = sum(1 for x, y in zip(a, b) if x > y + tol)
    losses = sum(1 for x, y in zip(a, b) if y > x + tol)
    n = wins + losses
    if n == 0:
        return 0, 0, 1.0
    if _HAVE_SCIPY:
        p = float(binomtest(wins, n, 0.5).pvalue)
    else:  # грубая нормальная аппроксимация, если scipy нет
        p = float(min(1.0, 2 * (1 - abs(wins - n / 2) / max(1.0, n / 2))))
    return wins, losses, p


def gate_decision(wins: int, losses: int, p: float) -> tuple[bool, bool, str]:
    """(большинство seed, статистически значимо, человекочитаемый ярлык).

    Большинство seed'ов без значимости — это подброшенная монета: 11 из 20 при
    p=0.82 ничем не отличаются от равных шансов, поэтому гейт требует оба.
    """
    better = wins > losses
    significant = (p < 0.05) if _HAVE_SCIPY else better
    if better and significant:
        return True, True, "ОК"
    if better:
        return True, False, "НЕ ДОКАЗАНО (большинство есть, значимости нет)"
    return False, False, "НЕ ЛУЧШЕ"


def fmt(x: float, prec: int = 3) -> str:
    return "nan" if x != x else f"{x:.{prec}f}"


def print_matrix(rows: list[dict], seeds: list[int], tasks: list[str]) -> None:
    print("\n=== Матрица условий (среднее по {} seeds) ===".format(len(seeds)))
    header = (f"{'arm':20s} {'task':6s} {'succ':>6s} {'succ↑':>6s} {'шаги':>6s} "
              f"{'награда':>9s} {'выбор→':>7s} {'Δw(прям)':>9s} {'Δw(контр)':>9s} "
              f"{'|dw|':>7s} {'bound':>6s} {'спайкиЛ/П':>11s} {'зерК/зЛ':>10s}")
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        for task in tasks:
            rws = sub(rows, arm, task)
            if not rws:
                continue
            dl = mean_of(rws, f"dw[{LABELS[DIRECT[task]]}]" if task in DIRECT else "success_rate")
            dc = mean_of(rws, f"dw[{LABELS[CONTRA[task]]}]" if task in CONTRA else "success_rate")
            if task not in DIRECT:  # у mixed нет единственного «прямого» ребра
                dl = dc = float("nan")
            spikes = (mean_of(rws, "spikes_cmd_left"), mean_of(rws, "spikes_cmd_right"))
            mirrors = (mean_of(rws, "success_mirror[right]"), mean_of(rws, "success_mirror[left]"))
            print(
                f"{arm:20s} {task:6s} {fmt(mean_of(rws, 'success_rate'), 2):>6s} "
                f"{fmt(mean_of(rws, 'success_second_half') - mean_of(rws, 'success_first_half'), 2):>6s} "
                f"{fmt(mean_of(rws, 'steps_to_target'), 1):>6s} "
                f"{fmt(mean_of(rws, 'cumulative_reward'), 1):>9s} "
                f"{fmt(mean_of(rws, 'choice_rate_right'), 2):>7s} "
                f"{fmt(dl):>9s} {fmt(dc):>9s} "
                f"{fmt(mean_of(rws, 'mean_abs_dw'), 4):>7s} "
                f"{fmt(mean_of(rws, 'fraction_weights_at_bound'), 2):>6s} "
                f"{spikes[0]:5.0f}/{spikes[1]:4.0f} "
                f"{fmt(mirrors[0], 2):>5s}/{fmt(mirrors[1], 2):<4s}"
            )


def p41_verdict(rows: list[dict], seeds: list[int]) -> bool:
    print("\n=== P4.1: зеркальный тест обучения направлению ===")
    ok = True
    for task, other in (("right", "left"), ("left", "right")):
        edge_task = LABELS[DIRECT[task]]      # прямое ребро этой задачи
        edge_mirror = LABELS[DIRECT[other]]   # зеркальное прямое ребро
        by_seed = {r["seed"]: r for r in sub(rows, "plastic", task)}
        a = [by_seed[s][f"dw[{edge_task}]"] for s in seeds if s in by_seed]
        b = [by_seed[s][f"dw[{edge_mirror}]"] for s in seeds if s in by_seed]
        wins, losses, p = paired_sign_test(a, b)
        sel = [by_seed[s][f"dw[{edge_task}]"] - by_seed[s][f"dw[{LABELS[CONTRA[task]]}]"]
               for s in seeds if s in by_seed]
        sel_wins, sel_losses, sel_p = paired_sign_test(sel, [0.0] * len(sel))
        mirror_pass, mirror_sig, mirror_label = gate_decision(wins, losses, p)
        sel_pass, sel_sig, sel_label = gate_decision(sel_wins, sel_losses, sel_p)
        passed = mirror_pass and mirror_sig and sel_pass and sel_sig
        ok = ok and passed
        print(
            f"  задача {task:5s}: Δw({edge_task}) > Δw({edge_mirror}) на {wins}/{wins + losses} seed "
            f"(p={fmt(p, 4)}, {mirror_label}); специфичность (прямое − контралатеральное) > 0 на "
            f"{sel_wins}/{sel_wins + sel_losses} seed (p={fmt(sel_p, 4)}, {sel_label}), "
            f"средняя специфичность {fmt(float(np.mean(sel))):s}"
        )
        toward = [by_seed[s]["choice_rate_right"] if task == "right" else 1.0 - by_seed[s]["choice_rate_right"]
                  for s in seeds if s in by_seed]
        print(f"    (поведение) доля движений в сторону цели: {fmt(float(np.mean(toward)), 2)}")

    # Зеркальная асимметрия — доказательство обучения только если замороженные
    # control-условия её не дают. Пластичные shuffled-условия НЕ являются
    # контролем по Δw: они тоже учатся и тоже вырабатывают специфичность.
    print("  контроль (замороженные условия не должны давать ни Δw, ни специфичности):")
    for arm in BASELINE_ARMS:
        for task in ("right", "left"):
            edge_t, edge_c = LABELS[DIRECT[task]], LABELS[CONTRA[task]]
            by_seed = {r["seed"]: r for r in sub(rows, arm, task)}
            sel = [by_seed[s][f"dw[{edge_t}]"] - by_seed[s][f"dw[{edge_c}]"] for s in seeds if s in by_seed]
            if not sel:
                continue
            w, l, p = paired_sign_test(sel, [0.0] * len(sel))
            moved = float(np.mean([abs(r["mean_abs_dw"]) for r in by_seed.values()])) if by_seed else 0.0
            systemic = (w > l and abs(float(np.mean(sel))) > 0.25) or moved > 1e-9
            ok = ok and not systemic
            print(
                f"    {arm:27s} {task:5s}: sel>0 на {w}/{w + l} seed, среднее {fmt(float(np.mean(sel))):s}, "
                f"|Δw|={fmt(moved, 4)} — {'НАРУШЕНО (плохо)' if systemic else 'ok'}"
            )
    print("  справочно: специфичность у пластичных shuffled-условий (тоже учатся):")
    for arm in LEARNING_ARMS:
        if arm == "plastic":
            continue
        for task in ("right", "left"):
            edge_t, edge_c = LABELS[DIRECT[task]], LABELS[CONTRA[task]]
            by_seed = {r["seed"]: r for r in sub(rows, arm, task)}
            sel = [by_seed[s][f"dw[{edge_t}]"] - by_seed[s][f"dw[{edge_c}]"] for s in seeds if s in by_seed]
            if sel:
                print(f"    {arm:20s} {task:5s}: sel {fmt(float(np.mean(sel)), 3)} на {len(sel)} seed")
    print("  вердикт P4.1:", "ВЫПОЛНЕН" if ok else "НЕ ВЫПОЛНЕН")
    return ok


def baseline_verdict(rows: list[dict], seeds: list[int], tasks: list[str]) -> bool:
    single = [t for t in tasks if t in ("right", "left")]
    if not single:
        print("\n(критерий по одиночным задачам пропущен: в этом прогоне их нет)")
        return True
    print("\n=== Критерий успеха: plastic против каждого замороженного baseline ===")
    ok = True
    for metric in ("success_rate", "cumulative_reward"):
        for arm in BASELINE_ARMS:
            for task in single:
                other = {r["seed"]: r for r in sub(rows, arm, task)}
                p_rows = sub(rows, "plastic", task)
                pmap = {r["seed"]: r for r in p_rows}
                shared = [s for s in seeds if s in pmap and s in other]
                a = [pmap[s][metric] for s in shared]
                b = [other[s][metric] for s in shared]
                wins, losses, p = paired_sign_test(a, b)
                better, significant, label = gate_decision(wins, losses, p)
                note = "" if _HAVE_SCIPY else " (p приближённый)"
                if metric == "success_rate":
                    ok = ok and better and significant
                print(
                    f"  {metric:16s} {task:5s} vs {arm:27s}: лучше на {wins}/{wins + losses} seed, "
                    f"p={fmt(p, 4)}{note} — {label}"
                )
    print("  вердикт по baseline (success_rate):", "ВЫПОЛНЕН" if ok else "НЕ ВЫПОЛНЕН")

    # Отдельно, без гейта: обучение со случайного асимметричного старта.
    print("\n  справочно (не гейт): plastic против пластичного shuffled-старта")
    for task in ("right", "left"):
        p_rows = {r["seed"]: r for r in sub(rows, "plastic", task)}
        o_rows = {r["seed"]: r for r in sub(rows, "weight_shuffled", task)}
        shared = [s for s in seeds if s in p_rows and s in o_rows]
        a = [p_rows[s]["success_rate"] for s in shared]
        b = [o_rows[s]["success_rate"] for s in shared]
        wins, losses, p = paired_sign_test(a, b)
        print(f"    {task:5s}: plastic {fmt(float(np.mean(a)), 2)} vs weight_shuffled "
              f"{fmt(float(np.mean(b)), 2)}, лучше на {wins}/{wins + losses} seed (p={fmt(p, 4)})")
    return ok


def mixed_verdict(rows: list[dict], seeds: list[int]) -> bool:
    """Критерий «лучше всех baseline» там, где он вообще имеет смысл.

    На mixed обе зеркальные задачи идут в одном обучении, поэтому сравнивается
    худшая из половин: проводка, которой повезло с направлением, обязана провалить
    половину потока, а обучение обязано подтянуть обе.
    """
    print("\n=== mixed: оба зеркала в одном обучении (метрика — худшая половина) ===")
    p = {r["seed"]: r for r in sub(rows, "plastic", MIXED)}
    if not p:
        print("  нет данных (задача mixed не прогонялась)")
        return True
    print(f"  {'arm':27s} {'succ':>6s} {'прав':>6s} {'лево':>6s} {'мин':>6s} {'разрыв':>7s}  сравнение с plastic")
    ok = True
    for arm in BASELINE_ARMS:
        other = {r["seed"]: r for r in sub(rows, arm, MIXED)}
        shared = [s for s in seeds if s in p and s in other]
        if not shared:
            continue
        pa = [p[s]["success_mirror_min"] for s in shared]
        pb = [other[s]["success_mirror_min"] for s in shared]
        wins, losses, pv = paired_sign_test(pa, pb)
        better, significant, label = gate_decision(wins, losses, pv)
        ok = ok and better and significant
        mean_p = float(np.mean(pa))
        mean_b = float(np.mean(pb))
        caveat = ""
        if better and mean_p < mean_b:
            caveat = "  (среднее ниже, но большинство seed'ов за plastic: у baseline распределение двумодальное)"
        print(
            f"  {arm:27s} {fmt(mean_of([other[s] for s in shared], 'success_rate'), 2):>6s} "
            f"{fmt(mean_of([other[s] for s in shared], 'success_mirror[right]'), 2):>6s} "
            f"{fmt(mean_of([other[s] for s in shared], 'success_mirror[left]'), 2):>6s} "
            f"{fmt(mean_b, 2):>6s} "
            f"{abs(mean_of([other[s] for s in shared], 'success_mirror[right]') - mean_of([other[s] for s in shared], 'success_mirror[left]')):>7.2f}"
            f"  plastic {fmt(mean_p, 2)} vs {fmt(mean_b, 2)}: лучше на {wins}/{wins + losses} seed, "
            f"p={fmt(pv, 4)} — {label}{caveat}"
        )
    pa_all = [p[s] for s in seeds if s in p]
    print(
        f"  {'plastic':27s} {fmt(mean_of(pa_all, 'success_rate'), 2):>6s} "
        f"{fmt(mean_of(pa_all, 'success_mirror[right]'), 2):>6s} "
        f"{fmt(mean_of(pa_all, 'success_mirror[left]'), 2):>6s} "
        f"{fmt(mean_of(pa_all, 'success_mirror_min'), 2):>6s} "
        f"{abs(mean_of(pa_all, 'success_mirror[right]') - mean_of(pa_all, 'success_mirror[left]')):>7.2f}"
    )
    print("  вердикт по mixed (min-half success):", "ВЫПОЛНЕН" if ok else "НЕ ВЫПОЛНЕН")
    return ok


def bias_probe(seeds: list[int], episodes: int, settings: ToySettings) -> None:
    """Проверка, что детектор правосторонней предвзятости вообще работает.

    `right_bias` — намеренно сконструированный рефлекс (сильный S_R→C_R) без
    пластичности. Если он даёт асимметрию право/лево, а `symmetric` — нет,
    значит матрица отличает «научился» от «так разведено».
    """
    print("\n=== Probe: виден ли встроенный right-bias без обучения ===")
    for regime in ("symmetric", "right_bias", "anti"):
        line = []
        for task in ("right", "left"):
            succs = []
            for seed in seeds:
                agent = toy.ToyAgent(learner_kind="none", init_regime=regime, seed=seed, settings=settings)
                eps = [agent.run_episode(task, e) for e in range(episodes)]
                succs.append(float(np.mean([e.reached for e in eps])))
            line.append((task, float(np.mean(succs))))
        asmm = line[0][1] - line[1][1]
        mixed_succ = []
        for seed in seeds:
            agent = toy.ToyAgent(learner_kind="none", init_regime=regime, seed=seed, settings=settings)
            eps = [agent.run_episode(MIXED, e) for e in range(episodes)]
            mixed_succ.append(float(np.mean([e.reached for e in eps])))
        mix = float(np.mean(mixed_succ))
        print(
            f"  init={regime:10s} succ(right)={line[0][1]:.2f} succ(left)={line[1][1]:.2f} Δ={asmm:+.2f} "
            f"succ(mixed)={mix:.2f} (ожидание для замороженной проводки ≈ среднее {(line[0][1] + line[1][1]) / 2:.2f})"
        )
    print("  ожидание: у symmetric Δ≈0 (направления нет «в железе»), у right_bias Δ>0 без пластичности")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n detailed rows → {path} ({len(rows)} строк)")


TEXT_FIELDS = ("arm", "task", "seed", "learner", "init_regime")


def read_csv(path: Path) -> list[dict]:
    """Reload a saved matrix so a verdict can be re-scored without re-simulating.

    Нужен, когда меняется сам гейт (например, потребовать значимости): числа
    остаются те же, меняются только правила их трактовки.
    """
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            row["seed"] = int(raw["seed"])
            for key, value in raw.items():
                if key in TEXT_FIELDS:
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = float("nan")
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4/P4.1 validation on the symmetric toy network")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=DEFAULT_SETTINGS.episodes)
    parser.add_argument("--tasks", default="right,left",
                        help="список задач через запятую: right,left,mixed")
    parser.add_argument("--arms", default=",".join(ARMS),
                        help="подмножество условий; удобно для шардинга прогона")
    parser.add_argument("--quick", action="store_true", help="4 seed, 6 эпизодов — дымовой прогон")
    parser.add_argument("--probe", action="store_true", help="дополнительно прогнать bias-probe")
    parser.add_argument("--eta", type=float, default=None, help="переопределить рабочую точку")
    parser.add_argument("--step-cost", type=float, default=None)
    parser.add_argument("--tau-eligibility", type=float, default=None)
    parser.add_argument("--command-inh", type=float, default=None)
    parser.add_argument("--replay", type=Path, default=None,
                        help="не симулировать, а заново оценить сохранённый CSV")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "var" / "p4_results.csv")
    parser.add_argument("--provenance", type=Path, default=REPO_ROOT / "var" / "p4_provenance.jsonl")
    parser.add_argument("--no-provenance", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    overrides = {"eta": args.eta, "step_cost": args.step_cost,
                 "tau_eligibility": args.tau_eligibility, "command_inh_weight": args.command_inh}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    settings = dataclasses.replace(DEFAULT_SETTINGS, **overrides) if overrides else DEFAULT_SETTINGS
    n_seeds = 4 if args.quick else args.seeds
    episodes = 6 if args.quick else args.episodes
    seeds = list(range(n_seeds))
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    known = {"right", "left", MIXED}
    unknown = [t for t in tasks if t not in known]
    if unknown:
        parser.error(f"unknown task(s): {unknown}; expected some of {sorted(known)}")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad_arms = [a for a in arms if a not in ARM_NAMES]
    if bad_arms:
        parser.error(f"unknown arm(s): {bad_arms}; expected some of {ARM_NAMES}")

    print(f"МУХА · P4/P4.1 · arms={list(arms)}")
    print(f"seeds={n_seeds} episodes={episodes} tasks={tasks} "
          f"distance={settings.cell_distance} sigma={settings.sigma} eta={settings.eta} "
          f"window={settings.window_steps} sym_weight={settings.sym_weight}")
    print(f"рабочая точка: step_cost={settings.step_cost} tau_eligibility={settings.tau_eligibility} "
          f"command_competition={settings.command_inh_weight} plastic_slots={settings.plastic_slots}")

    prov = None
    if args.replay is not None:
        rows = read_csv(args.replay)
        print(f"replay: {len(rows)} строк из {args.replay} (симуляция не запускалась)")
        tasks = sorted({r["task"] for r in rows})
        seeds = sorted({r["seed"] for r in rows})
    else:
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        if args.provenance.exists():
            args.provenance.unlink()
        prov = ProvenanceLog(path=str(args.provenance))
        rows = collect(arms, tasks, seeds, episodes, settings, prov, args.verbose)
    print_matrix(rows, seeds, tasks)
    single = [t for t in tasks if t in ("right", "left")]
    if single:
        mirror_ok = p41_verdict(rows, seeds)
    else:
        print("\nP4.1 зеркальный тест пропущен: в прогоне нет одиночных задач right/left")
        mirror_ok = True
    single_ok = baseline_verdict(rows, seeds, tasks)
    mixed_ok = mixed_verdict(rows, seeds) if MIXED in tasks else True
    baselines_ok = single_ok and mixed_ok

    violations = int(sum(r["governance_violations"] for r in rows))
    print(f"\n governance violations за весь прогон: {violations}")
    if prov is not None:
        print(f" provenance: {prov.count} записей (по одной на эпизод), цепочка цела: {prov.verify_chain()}")

    write_csv(rows, args.out) if args.replay is None else None
    if args.probe and args.replay is None:
        bias_probe(seeds[:4], min(episodes, 8), settings)

    print("\n=== Вердикт ===")
    print(" P4.1 зеркальный тест:", "ВЫПОЛНЕН" if mirror_ok else "НЕ ВЫПОЛНЕН")
    if single:
        print(" plastic > baseline по success_rate (одиночные задачи):",
              "ВЫПОЛНЕН" if single_ok else "НЕ ВЫПОЛНЕН")
    if MIXED in tasks:
        print(" plastic > baseline по min-half success (mixed):",
              "ВЫПОЛНЕН" if mixed_ok else "НЕ ВЫПОЛНЕН")
    print(" Оговорки по claims: это simulation_result на 5-нейронной toy-сети с шумом;")
    print("   никакого биологического правдоподобия и связи с реальным коннектмом здесь не проверяется.")
    return 0 if (mirror_ok and baselines_ok and violations == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
