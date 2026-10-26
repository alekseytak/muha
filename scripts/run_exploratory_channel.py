#!/usr/bin/env python
"""EXPLORATORY-замер: сдвигает ли включённый канал (4/1/6) метрики на mixed?

СТАТУС ЯВНО НЕ-CONFIRMATORY. Это НЕ заявка «обучение превзошло baseline» и НЕ
претензия на научный результат. Это зонд: двигается ли вообще метрика от канала,
в какую сторону и примерно какой величины — чтобы решить, стоит ли заводить под
это отдельную pre-registration со своим operating point и своими seed.

Что под сравнением (arm=plastic, task=mixed):
  принцип 4  verb_feedback        : OFF 0.0  vs ON 1.0
  принцип 1  self_obs_gain        : OFF 0.0  vs ON 0.5
  принцип 6  internal_observer_only: OFF F    vs ON T   ← КОНТРОЛЬ: в episode-петле
             награда идёт source='internal', страж её пропускает → разницы быть не
             должно; если она появилась — канал perturbed loop (баг).

Две меры, обе парные (binop minimal budget решают обе, но по-разному):
  success_mirror_min — первичная (бинарная, coarse): на почти-нулевом успехе любое
      возмущение траектории может НЕ переключить reached → законный null;
  cumulative_reward  — непрерывная, чувствительная: ловит сдвиг, который min-half
      не показал бы. Null и здесь = честный «канал молчит», а не «мы не видели».

Почему paired и seed'ы такие: ON/OFF одного seed отличаются ТОЛЬКО каналом (шум
env/net сидуется seed'ом) → paired sign test. Seed'ы вне замороженного 0..179,
чтобы не притронуться к сожжённым и не создать ложного «подтверждения».

Журнал на каждую ячейку: принцип 1 обязан ЧИТАТЬ собственный provenance-журнал —
без него канал молчит по построению. Отдельный файл на ячейку → no cross-seed leak.

Self-check (падение = скрипт врёт): OFF-ячейка обязана совпасть с run_arm('plastic',
'mixed', seed) по min-half И кумулятивной награде — иначе exploratory мерит не ту руку.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fly_connectome_agent.src.engineering.harness import symmetric_toy as toy
from fly_connectome_agent.src.engineering.harness.symmetric_toy import (
    MIXED,
    ToySettings,
)
from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog
from scripts.run_p4_validation import paired_sign_test, fmt  # reuse paired test verbatim

ARM = "plastic"
SPEC = toy.ARM_SPECS[ARM]
_CELL = itertools.count()
_TMP = tempfile.mkdtemp(prefix="muha_expl_")


def run_cell(seed: int, episodes: int, settings: ToySettings, *,
             verb_feedback: float = 0.0, self_obs_gain: float = 0.0,
             internal_observer_only: bool = False) -> dict:
    """Прогон plastic-руки на mixed с журналом на эту ячейку (нужен принципу 1)."""
    agent = toy.ToyAgent(
        learner_kind=SPEC["learner"], init_regime=SPEC["init"], seed=seed,
        reward_coupled=SPEC["reward_coupled"], settings=settings,
        verb_feedback=verb_feedback, self_obs_gain=self_obs_gain,
        internal_observer_only=internal_observer_only,
    )
    jpath = os.path.join(_TMP, f"cell{next(_CELL)}_{seed}.jsonl")
    prov = ProvenanceLog(path=jpath)
    results = [agent.run_episode(MIXED, ep, prov=prov) for ep in range(episodes)]
    for p in (jpath, jpath + ".head.json"):
        if os.path.exists(p):
            os.remove(p)
    right = [r.reached for r in results if r.task == "right"]
    left = [r.reached for r in results if r.task == "left"]
    mr = float(np.mean(right)) if right else float("nan")
    ml = float(np.mean(left)) if left else float("nan")
    halves = [mr, ml]
    min_half = float(np.min(halves)) if not any(h != h for h in halves) else float("nan")
    return {"right": mr, "left": ml, "min_half": min_half,
            "cum_reward": float(sum(r.reward for r in results)),
            "governance_violations": agent.prov_stats["governance_violations"]}


def selfcheck_off_matches_frozen_arm(seeds, episodes, settings) -> None:
    """OFF-ячейка обязана побитово совпасть с замороженной run_arm для тех же seed."""
    for seed in seeds[:3]:
        cell = run_cell(seed, episodes, settings)
        ref = toy.run_arm(ARM, MIXED, seed, episodes=episodes, settings=settings)
        rh = [e.reached for e in ref.episodes if e.task == "right"]
        lh = [e.reached for e in ref.episodes if e.task == "left"]
        ref_min = float(np.min([np.mean(rh), np.mean(lh)]))
        assert cell["min_half"] == ref_min, (
            f"self-check FAIL seed={seed}: min_half={cell['min_half']} != run_arm={ref_min}")
        assert abs(cell["cum_reward"] - ref.cumulative_reward) < 1e-9, (
            f"self-check FAIL seed={seed}: reward={cell['cum_reward']} != "
            f"run_arm={ref.cumulative_reward} (скрипт мерит не ту руку)")
    print("  self-check OK: OFF-путь побитово == замороженный run_arm('plastic','mixed')")


def _paired(label: str, key: str, off, on, seeds) -> None:
    a = [c[key] for c in on]
    b = [c[key] for c in off]
    wins, losses, p = paired_sign_test(a, b)
    delta = float(np.mean(a)) - float(np.mean(b))
    if wins == 0 and losses == 0:  # каждый seed побитово off==on → канал молчит
        verdict = "НЕТ СДВИГА: на каждом seed off==on (канал молчит в этом бюджете)"
    elif wins == losses:
        verdict = "НЕТ ЧИСТОГО СДВИГА: seed разошлись в обе стороны (wins==losses)"
    elif wins == len(seeds) or losses == len(seeds):
        verdict = "ОДНОРОДНЫЙ сдвиг на всех seed (всё ещё exploratory, не подтверждение)"
    elif p < 0.05:
        verdict = "направленный перевес + формальная значимость → нужен pre-reg, не verdict"
    else:
        verdict = "НЕОДНОЗНАЧНО: перевес без значимости (мало seed / шум)"
    print(f"    {label:18s} OFF={fmt(float(np.mean(b)), 3)} ON={fmt(float(np.mean(a)), 3)} "
          f"Δ={fmt(delta, 3)} | ON лучше {wins}/{wins + losses} (p={fmt(p, 3)}) → {verdict}")


def compare(name: str, seeds, episodes, settings, on_kwargs) -> list[dict]:
    off = [run_cell(s, episodes, settings) for s in seeds]
    on = [run_cell(s, episodes, settings, **on_kwargs) for s in seeds]
    viol = int(sum(c["governance_violations"] for c in on))
    print(f"\n  {name}  (governance_violations при ON: {viol})")
    _paired("success_mirror_min", "min_half", off, on, seeds)
    _paired("cumulative_reward", "cum_reward", off, on, seeds)
    return [{"probe": name, "seed": s,
             "off_min_half": o["min_half"], "on_min_half": n["min_half"],
             "off_cum_reward": o["cum_reward"], "on_cum_reward": n["cum_reward"],
             "on_gov_viol": n["governance_violations"]}
            for s, o, n in zip(seeds, off, on)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EXPLORATORY channel probe (non-confirmatory)")
    ap.add_argument("--seeds", type=int, default=8, help="СКОЛЬКО свежих seed (не значение)")
    ap.add_argument("--seed-start", type=int, default=600, help="первый seed вне 0..179")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "var" / "exploratory_channel.csv")
    args = ap.parse_args(argv)

    if args.seed_start <= 179:
        print(f"ВНИМАНИЕ: seed_start={args.seed_start} попадает в сожжённое 0..179 "
              "(подтвердительное пространство); сдвинь на --seed-start 200+")
        return 2

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    settings = ToySettings()  # замороженная рабочая точка P4; канал — только параметр агента

    print("=" * 80)
    print("МУХA · EXPLORATORY-замер канала · arm=plastic · task=mixed")
    print("  СТАТУС: НЕ confirmatory. Seed'ы %d..%d вне замороженного 0..179." % (seeds[0], seeds[-1]))
    print("  Каналы в проде ВЫКЛЮЧены по дефолту — этот прогон ничего не меняет в коде.")
    print("  episodes=%d sigma=%s alpha=%s eta=%s" % (args.episodes, settings.sigma, settings.alpha, settings.eta))
    print("=" * 80)

    t0 = time.time()
    selfcheck_off_matches_frozen_arm(seeds, args.episodes, settings)

    rows: list[dict] = []
    rows += compare("принцип 4 · verb_feedback 0.0 → 1.0", seeds, args.episodes, settings,
                    {"verb_feedback": 1.0})
    rows += compare("принцип 1 · self_obs_gain 0.0 → 0.5", seeds, args.episodes, settings,
                    {"self_obs_gain": 0.5})
    # Контроль принципа 6 структурно инертен в петле — дешёвый прогон достаточен.
    rows += compare("принцип 6 · internal_observer_only (КОНТРОЛЬ, ожидаем 0)",
                    seeds[:3], min(args.episodes, 6), settings, {"internal_observer_only": True})

    print("\n  оговорка: whiff test при малом бюджете; null/шум ожидаемы и НЕ опровергают")
    print("  канал — они лишь не дают основания его 'подтверждать'. Претензия на результат")
    print("  = отдельная pre-registration (свой operating_point, бюджет seed, заранее гейт).")
    print(f"  заняло {time.time() - t0:.0f}с")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  rows → {args.out} ({len(rows)} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
