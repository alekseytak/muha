"""Прогон контура ЭТАП 4 на замороженном канале FlyWire.

Клетка матрицы = (рука, сид). Каждая клетка независима и детерминирована,
поэтому её можно считать в отдельном процессе; сводка собирается из
per-episode TSV, а не из памяти родителя.

Руки:
    plastic        R-STDP с модулятором (награда среды), один скаляр на все рёбра
    pool           то же, но проигравший пул получает (1-kappa)*M (контраст по пулам)
    pool_pos       то же, но контраст только на подкрепляющих шагах (M > 0)
    pool_neg       то же, но контраст только на наказывающих шагах (M < 0)
    neuron         третья компонента своя на нейрон, по рангу массы eligibility
    neuron_shuffled то же, но факторы переставлены между нейронами (нулевая рука)
    neuron_renorm  то же с факторами, перенормированными до кредита v1
    neuron_renorm_shuffled  renorm-нуль: перестановка плюс перенормировка
    no_plasticity  та же сеть и проводка, Δw ≡ 0
    no_modulator   eligibility копится, M ≡ 0
    oracle         политика по сенсорике без сети — верхняя граница среды

Печатается ровно то, что заявлено в протоколе: success_rate первого и
последнего блока, спайковый бюджет, доля весов на границе, нарушения ворот.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import pathlib
import sys
from decimal import Decimal, InvalidOperation

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fly_connectome_agent.src.engineering.harness.snn_corridor import (  # noqa: E402
    ARMS,
    NEURON_ARMS,
    POOL_ARMS,
    Channel,
    CorridorAgent,
    LoopSettings,
    RunSummary,
)
from fly_connectome_agent.src.science.graph.load_connectome import load_connectome  # noqa: E402
from fly_connectome_agent.src.science.snn.weights_policy import graph_with_signed_weights  # noqa: E402

_G: dict = {}


def load_channel(data_dir: str, graph) -> Channel:
    manifest = json.loads(open(os.path.join(data_dir, "connectome_manifest.json"),
                               encoding="utf-8").read())
    spec = manifest.get("io_channel")
    if not spec:
        raise SystemExit("в манифесте нет io_channel: нужен ингест --selection role")

    flywire: dict[int, int] = {}
    with open(os.path.join(data_dir, "nodes.csv"), encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            flywire[int(row["neuron_id"])] = int(row["flywire_id"])

    side: dict[int, str] = {}
    annotations = os.path.join(REPO, "fly_connectome_agent", "data",
                               "Supplemental_file1_neuron_annotations.tsv")
    with open(annotations, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            raw = (row.get("root_id") or "").strip()
            if not raw:
                continue
            try:
                rid = int(Decimal(raw))
            except InvalidOperation:
                continue
            label = (row.get("side") or "").strip().lower()
            if label in ("left", "right"):
                side[rid] = label

    def split(ids: list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        left, right = [], []
        for local in ids:
            label = side.get(flywire.get(int(local), -1))
            if label == "left":
                left.append(int(local))
            elif label == "right":
                right.append(int(local))
        return tuple(left), tuple(right)

    ins_l, ins_r = split(spec["input_ids"])
    outs_l, outs_r = split(spec["output_ids"])
    known = {node.node_id for node in graph.nodes}
    lost = [i for i in (*ins_l, *ins_r, *outs_l, *outs_r) if i not in known]
    if lost:
        raise SystemExit(f"узлы канала отсутствуют в графе: {lost[:5]}")
    return Channel(inputs_left=ins_l, inputs_right=ins_r,
                   outputs_left=outs_l, outputs_right=outs_r)


def _init(data_dir: str, gain: float) -> None:
    graph = graph_with_signed_weights(load_connectome(
        os.path.join(data_dir, "nodes.csv"), os.path.join(data_dir, "edges.csv"),
        manifest_path=os.path.join(data_dir, "connectome_manifest.json")))
    _G.update(graph=graph, channel=load_channel(data_dir, graph))
    if gain <= 0:
        raise SystemExit("gain обязан быть положительным")


def _cell(job: dict) -> dict:
    settings = LoopSettings(**job["settings"])
    agent = CorridorAgent(graph=_G["graph"], channel=_G["channel"],
                          settings=settings, arm=job["arm"], seed=job["seed"])
    summary: RunSummary = agent.run(episodes=job["episodes"])
    return {"arm": job["arm"], "seed": job["seed"],
            "episodes": [e.__dict__ for e in summary.episodes],
            "violations": summary.governance_violations,
            "nan": summary.nan_seen}


def _write_cell(out_dir: str, cell: dict, meta: dict) -> None:
    path = os.path.join(out_dir, f"{cell['arm']}-s{cell['seed']}.tsv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["episode", "reached", "steps", "reward", "spikes", "moved_left",
                    "moved_right", "stayed", "bumps", "denied", "mean_abs_dw",
                    "fraction_at_bound", "graded_decisions", "graded_correct"])
        for e in cell["episodes"]:
            w.writerow([e["episode"], int(e["reached"]), e["steps"], round(e["reward"], 4),
                        e["spikes"], e["moved_left"], e["moved_right"], e["stayed"],
                        e["bumps"], e["denied"], round(e["mean_abs_dw"], 8),
                        round(e["fraction_at_bound"], 4), e["graded_decisions"],
                        e["graded_correct"]])
    with open(os.path.join(out_dir, f"{cell['arm']}-s{cell['seed']}.meta.json"),
              "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2, sort_keys=True)


def _cell_done(out_dir: str, arm: str, seed: int, episodes: int, meta: dict) -> bool:
    """Готовность клетки: строк ровно столько, сколько эпизодов, и те же числа."""
    tsv = os.path.join(out_dir, f"{arm}-s{seed}.tsv")
    side = os.path.join(out_dir, f"{arm}-s{seed}.meta.json")
    if not (os.path.exists(tsv) and os.path.exists(side)):
        return False
    with open(side, encoding="utf-8") as fh:
        stored = json.load(fh)
    if any(stored.get(k) != v for k, v in json.loads(json.dumps(meta, sort_keys=True)).items()):
        return False
    with open(tsv, encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh, delimiter="\t")) == episodes


def _read_cell(out_dir: str, arm: str, seed: int) -> dict:
    tsv = os.path.join(out_dir, f"{arm}-s{seed}.tsv")
    rows = []
    with open(tsv, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rows.append({
                "episode": int(row["episode"]), "reached": bool(int(row["reached"])),
                "steps": int(row["steps"]), "reward": float(row["reward"]),
                "spikes": int(row["spikes"]),
                "moved_left": int(row["moved_left"]),
                "moved_right": int(row["moved_right"]),
                "stayed": int(row["stayed"]), "bumps": int(row["bumps"]),
                "denied": int(row["denied"]),
                "mean_abs_dw": float(row["mean_abs_dw"]),
                "fraction_at_bound": float(row["fraction_at_bound"]),
                "graded_decisions": int(row["graded_decisions"]),
                "graded_correct": int(row["graded_correct"]),
            })
    with open(os.path.join(out_dir, f"{arm}-s{seed}.meta.json"), encoding="utf-8") as fh:
        side = json.load(fh)
    return {"arm": arm, "seed": seed, "episodes": rows,
            "violations": side.get("governance_violations", 0),
            "nan": side.get("nan_seen", False)}


def verify_protocol(path: str, arms: list[str], seeds: list[int],
                    settings: dict) -> dict:
    """Сверка запуска с замороженным манифестом. Любое расхождение — отказ."""
    raw = open(path, "rb").read()
    manifest = json.loads(raw.decode("utf-8"))
    digest = hashlib.sha256(raw).hexdigest()
    declared = manifest["operating_point"]["values"]
    problems: list[str] = []

    pairs = [
        ("episodes", settings["episodes"]),
        ("window_steps", settings["window_steps"]),
        ("i_drive", settings["i_drive"]),
        ("gain", settings["gain"]),
        ("sigma", settings["sigma"]),
        ("eta", settings["eta"]),
        ("pool_contrast", settings["pool_contrast"]),
        ("neuron_credit", settings["neuron_credit"]),
        ("coherence_gain", settings["coherence_gain"]),
        ("max_decisions", settings["max_episode_steps"]),
        ("width", settings["width"]),
        ("height", settings["height"]),
    ]
    for key, got in pairs:
        want = declared[key]
        if isinstance(want, (int, float)) and not isinstance(want, bool):
            if not math.isclose(float(got), float(want), rel_tol=1e-12, abs_tol=0.0):
                problems.append(f"{key}: запуск {got!r} != протокол {want!r}")
        elif got != want:
            problems.append(f"{key}: запуск {got!r} != протокол {want!r}")
    for key in ("start", "target"):
        if tuple(settings[key]) != tuple(declared[key]):
            problems.append(f"{key}: запуск {settings[key]!r} != протокол {declared[key]!r}")

    declared_arms = [a["arm_id"] for a in manifest["arms"]]
    if sorted(arms) != sorted(declared_arms):
        problems.append(f"руки: запуск {sorted(arms)} != протокол {sorted(declared_arms)}")
    declared_seeds = manifest["seed_sets"]["confirmatory"]["values"]
    if sorted(seeds) != sorted(declared_seeds):
        problems.append(f"seed'ы: запуск {sorted(seeds)} != протокол {sorted(declared_seeds)}")

    # Числа, которые берутся из умолчаний LoopSettings, тоже сверяются: иначе
    # правка умолчания в коде молча меняла бы объявленную рабочую точку.
    probe = LoopSettings(**{**settings, "seed": seeds[0]})
    for key in ("tau_eligibility", "policy_penalty", "m_max", "heading",
                "shaping_scale"):
        got = getattr(probe, key)
        want = declared[key]
        if not math.isclose(float(got), float(want), rel_tol=1e-12, abs_tol=0.0):
            problems.append(f"{key}: код {got!r} != протокол {want!r}")
    if not math.isclose(probe.coherence_gain, declared["epsilon"], rel_tol=0, abs_tol=1e-15):
        problems.append(f"epsilon: код {probe.coherence_gain!r} != протокол {declared['epsilon']!r}")
    if not math.isclose(probe.policy_penalty, declared["delta"], rel_tol=0, abs_tol=1e-15):
        problems.append(f"delta: код {probe.policy_penalty!r} != протокол {declared['delta']!r}")

    if problems:
        raise SystemExit("протокол не совпадает с запуском:\n  " + "\n  ".join(problems))
    return {"path": str(path), "protocol_id": manifest["protocol_id"],
            "sha256": digest}


def _block_rate(rows: list[dict], lo: int, hi: int) -> float:
    block = rows[lo:hi]
    return sum(1 for r in block if r["reached"]) / len(block) if block else float("nan")


def _block_accuracy(rows: list[dict], lo: int, hi: int) -> float:
    block = rows[lo:hi]
    total = sum(r["graded_decisions"] for r in block)
    if total == 0:
        return float("nan")
    return sum(r["graded_correct"] for r in block) / total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="var/flywire-role")
    ap.add_argument("--out", default="var/p4_loop")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="7,13,21")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--window-steps", type=int, default=20)
    ap.add_argument("--drive", type=float, default=20.0)
    ap.add_argument("--gain", type=float, default=6.0)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--kappa", type=float, default=0.0,
                    help="контраст проигравшего пула в руке pool (0 = правило v1)")
    ap.add_argument("--neuron-credit", type=float, default=0.0,
                    help="размах пер-нейронного фактора λ в руках neuron* (0 = v1)")
    ap.add_argument("--coherence", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=40, help="решений на эпизод")
    ap.add_argument("--size", type=int, default=7)
    ap.add_argument("--start", default="1,1")
    ap.add_argument("--target", default="5,5")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--protocol", default="fly_connectome_agent/manifests/p4_3_spike_verb.json")
    ap.add_argument("--confirm", action="store_true",
                    help="требовать полного совпадения с замороженным протоколом")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"неизвестные руки: {unknown}; допустимо {ARMS}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.out, exist_ok=True)

    settings = dict(episodes=args.episodes, window_steps=args.window_steps,
                    i_drive=args.drive, gain=args.gain, sigma=args.sigma,
                    eta=args.eta, pool_contrast=args.kappa,
                    neuron_credit=args.neuron_credit,
                    coherence_gain=args.coherence,
                    max_episode_steps=args.steps, width=args.size,
                    height=args.size, start=tuple(int(v) for v in args.start.split(",")),
                    target=tuple(int(v) for v in args.target.split(",")))
    protocol = None
    if args.confirm:
        if not args.protocol:
            raise SystemExit("--confirm требует --protocol")
        protocol = verify_protocol(args.protocol, arms, seeds, settings)
        protocol["verified"] = True
    elif args.protocol and os.path.exists(args.protocol):
        raw = open(args.protocol, "rb").read()
        protocol = {"path": args.protocol, "verified": False,
                    "sha256": hashlib.sha256(raw).hexdigest()}
    jobs = [{"arm": arm, "seed": seed, "episodes": args.episodes,
             "settings": {**settings, "seed": seed}} for arm in arms for seed in seeds]

    def cell_meta(job: dict) -> dict:
        return {"arm": job["arm"], "seed": job["seed"],
                "settings": {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in job["settings"].items()},
                "pool_when": POOL_ARMS.get(job["arm"], "always"),
                "neuron_credit": NEURON_ARMS.get(job["arm"], "none"),
                "protocol_sha256": protocol["sha256"] if protocol else None}

    pending = [j for j in jobs
               if not _cell_done(args.out, j["arm"], j["seed"], args.episodes, cell_meta(j))]
    print(f"клеток {len(jobs)}: {len(arms)} рук x {len(seeds)} сидов, "
          f"{args.episodes} эпизодов; готово {len(jobs) - len(pending)}, "
          f"считать {len(pending)}", flush=True)
    if protocol:
        print(f"протокол {protocol['protocol_id'] if 'protocol_id' in protocol else '—'} "
              f"sha256={protocol['sha256'][:16]}", flush=True)

    if pending:
        with mp.Pool(args.workers, initializer=_init,
                     initargs=(args.data, args.gain)) as pool:
            for cell in pool.imap_unordered(_cell, pending):
                job = next(j for j in pending
                           if j["arm"] == cell["arm"] and j["seed"] == cell["seed"])
                meta = {**cell_meta(job), "governance_violations": cell["violations"],
                        "nan_seen": cell["nan"]}
                _write_cell(args.out, cell, meta)
                print(f"  готово {cell['arm']}-s{cell['seed']}", flush=True)

    cells = [_read_cell(args.out, j["arm"], j["seed"]) for j in jobs]
    rows = []
    for cell in cells:
        eps = cell["episodes"]
        n = len(eps)
        block = max(1, n // 5)
        first = _block_rate(eps, 0, block)
        last = _block_rate(eps, n - block, n)
        rows.append({
            "arm": cell["arm"], "seed": cell["seed"], "episodes": n,
            "success_first": first, "success_last": last, "delta": last - first,
            "accuracy_first": _block_accuracy(eps, 0, block),
            "accuracy_last": _block_accuracy(eps, n - block, n),
            "spikes_per_episode": sum(e["spikes"] for e in eps) / n,
            "reward_mean": sum(e["reward"] for e in eps) / n,
            "steps_mean": sum(e["steps"] for e in eps) / n,
            "mean_abs_dw": sum(e["mean_abs_dw"] for e in eps) / n,
            "at_bound": eps[-1]["fraction_at_bound"],
            "violations": cell["violations"], "nan": cell["nan"],
        })
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"settings": settings, "args": vars(args), "protocol": protocol,
                   "cells": rows}, fh, ensure_ascii=False, indent=2)

    print(f"{'рука':>14} {'сид':>4} {'перв.блок':>10} {'посл.блок':>10} {'Δ':>7} "
          f"{'точн.1':>7} {'точн.2':>7} {'спайки/эп':>10} {'шаги':>6} {'у границы':>9}")
    for r in rows:
        print(f"{r['arm']:>14} {r['seed']:4d} {r['success_first']:10.3f} "
              f"{r['success_last']:10.3f} {r['delta']:+7.3f} "
              f"{r['accuracy_first']:7.3f} {r['accuracy_last']:7.3f} "
              f"{r['spikes_per_episode']:10.0f} {r['steps_mean']:6.1f} "
              f"{r['at_bound']:9.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
