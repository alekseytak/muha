#!/usr/bin/env python3
"""P4.2b: verify shard coverage — exact partition, no overlaps, no gaps.

Called by merge_p4_shards.py, but can also be used standalone for audit:
    python scripts/check_p4_shard_coverage.py --parent-manifest <path> \
        --shard-manifests <path1> <path2> [...]

Exit codes:
    0 = coverage exact
    2 = REFUSED (overlap, gap, or missing dimension)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import p4_2_protocol as proto  # noqa: E402

REFUSE = 2


def _partition(problems: list[str], parts: list[list], whole: set, dim: str, shards) -> None:
    """Exact partition of `whole` by the per-shard `parts`: no overlap, no gap,
    nothing outside the declared set. Used for the sliced axis only."""
    seen: list = []
    for s, part in zip(shards, parts):
        for item in part:
            if item in seen:
                problems.append(f"{dim} {item!r} в двух shards (overlap)")
            seen.append(item)
    missing = whole - set(seen)
    if missing:
        problems.append(f"{dim} отсутствуют во всех shards: {sorted(missing)[:5]}")
    extra = set(seen) - whole
    if extra:
        problems.append(f"shards содержат {dim} вне протокола: {sorted(extra)[:5]}")


def check_coverage(shards: list[dict], parent: dict) -> list[str]:
    """Return list of coverage problems; empty means all good.

    Режим задаётся shard_axis и обязан быть одинаковым у всех shard'ов: смешивать
    arms-axis и seeds-axis в одном merge запрещено — тогда «полное покрытие»
    нельзя свести к одному точному разбиению ни по одной оси.
    """
    problems: list[str] = []

    declared_arms = [arm["arm_id"] for arm in parent["arms"]]
    all_seeds = proto.confirmatory_seeds(parent)
    # Shard-манифесты хранят протокольные имена streams ("left_target"…), а не
    # code-маппинг ("left"…), поэтому сверка идёт по объявленному в манифесте
    # списку, иначе валидный shard вечно «не совпадает» с декодом.
    declared_streams = list(parent["task_streams"])

    # 0. Единая ось
    axes = {s.get("shard_axis") for s in shards}
    if len(axes) != 1:
        problems.append(f"смешанные оси в одном merge запрещены: {sorted(axes)}")
        return problems
    axis = axes.pop()

    if axis == "arms":
        # разбиваем arms; каждый shard держит ВСЕ seeds и ВСЕ streams
        _partition(problems, [s["arm_subset"] for s in shards], set(declared_arms), "arms", shards)
        for s in shards:
            if set(s["seed_subset"]) != set(all_seeds):
                gap = set(all_seeds) - set(s["seed_subset"])
                extra = set(s["seed_subset"]) - set(all_seeds)
                if gap:
                    problems.append(f"shard {s['shard_id']}: seeds missing {sorted(gap)[:5]}…")
                if extra:
                    problems.append(f"shard {s['shard_id']}: seeds not in protocol {sorted(extra)[:5]}…")
    elif axis == "seeds":
        # разбиваем seeds; каждый shard держит ВСЕ arms и ВСЕ streams
        _partition(problems, [s["seed_subset"] for s in shards], set(all_seeds), "seeds", shards)
        for s in shards:
            if set(s["arm_subset"]) != set(declared_arms):
                missing = set(declared_arms) - set(s["arm_subset"])
                extra = set(s["arm_subset"]) - set(declared_arms)
                if missing:
                    problems.append(f"shard {s['shard_id']}: arms missing {sorted(missing)[:5]}…")
                if extra:
                    problems.append(f"shard {s['shard_id']}: arms not in protocol {sorted(extra)[:5]}…")
    else:
        problems.append(f"неизвестная shard_axis {axis!r}")

    # Streams: во обоих режимах каждый shard обязан покрыть ВСЕ streams.
    for s in shards:
        if set(s["task_streams"]) != set(declared_streams):
            problems.append(
                f"shard {s['shard_id']}: task_streams {s['task_streams']} != "
                f"declared {declared_streams}")

    # Episodes consistency
    for s in shards:
        if s["episodes_per_seed"] != parent["episodes_per_seed"]:
            problems.append(
                f"shard {s['shard_id']}: episodes_per_seed={s['episodes_per_seed']} "
                f"!= {parent['episodes_per_seed']}")

    # Shard cell counts consistency
    for s in shards:
        expected_cells = len(s["arm_subset"]) * len(s["seed_subset"]) * len(s["task_streams"])
        actual = s.get("cells_in_shard")
        if actual is not None and actual != expected_cells:
            problems.append(
                f"shard {s['shard_id']}: cells_in_shard={actual} != "
                f"arms×seeds×streams={expected_cells}")

    # Total cells must match full experiment
    total_expected = len(declared_arms) * len(all_seeds) * len(declared_streams)
    total_from_shards = sum(
        len(s["arm_subset"]) * len(s["seed_subset"]) * len(s["task_streams"])
        for s in shards)
    if total_from_shards != total_expected:
        problems.append(
            f"merge cells: shards дают {total_from_shards}, протокол ждёт {total_expected}")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent-manifest", default=str(proto.DEFAULT_MANIFEST))
    ap.add_argument("--shard-manifests", nargs="+", required=True)
    a = ap.parse_args()

    parent = proto.load(pathlib.Path(a.parent_manifest))
    shards = []
    for p in a.shard_manifests:
        sp = pathlib.Path(p)
        if not sp.exists():
            print(f"COVERAGE REFUSED: shard manifest не найден: {p}", file=sys.stderr)
            return REFUSE
        shards.append(json.loads(sp.read_text(encoding="utf-8")))

    problems = check_coverage(shards, parent)
    if problems:
        print("COVERAGE REFUSED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return REFUSE

    print(f"coverage exact: {len(shards)} shards, "
          f"{len(parent['arms'])} arms × {len(proto.confirmatory_seeds(parent))} seeds × "
          f"{len(proto.task_streams(parent))} streams = full partition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
