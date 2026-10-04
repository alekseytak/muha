#!/usr/bin/env python3
"""P4.2b/P4.2c: run one shard of a frozen P4.2 experiment.

Infrastructure-only: wraps the existing runner logic with shard identity, a
runtime-budget planner, and shard-manifest generation. Does NOT change
scientific parameters, arms, metrics, episode counts, or provenance semantics.

Two shard axes are supported (exactly one axis per merge — never mixed):

    --axis arms   slice the arm set; every shard keeps ALL seeds and ALL streams.
    --axis seeds  slice the confirmatory seed set; every shard keeps ALL arms and
                  ALL streams. This is P4.2c: it exists because an arm-only
                  partition cannot fit a full seed set under the runtime budget
                  (3 arms × 60 seeds × 3 streams ≈ 7 560 s ≫ 1 200 s).

Usage:
    python scripts/run_p4_shard.py --parent-manifest <path> \
        --shard-id <id> --axis {arms,seeds} \
        (--arms a,b,c | --seeds 180-189) --out-dir <path> [--dry-run]

Exit codes:
    0 = shard completed (or --dry-run plan printed)
    2 = REFUSED (shard invalid, budget exceeded, identity/burned mismatch)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import p4_2_protocol as proto  # noqa: E402

REFUSE = 2
MAX_SHARD_TARGET_SECONDS = 1200  # 20 minutes

# Measured cost per cell. P4.2.v2 abort logged 120/1080 cells at ~1436 s
# (var/aborted/p4_2_v2_*/abort_metadata.json) => ~12.0 s/cell while the journal
# was still small — which is exactly the regime a shard runs in, because a shard
# starts its own fresh provenance and never reaches the O(n)-append tail that
# killed v1/v2. We keep this conservative and let the reviewer override with
# --seconds-per-cell when a fresher measurement exists; it is never tuned down
# silently to make an over-budget plan pass.
MEASURED_SECONDS_PER_CELL = 12.0

# Seeds 0–179 are all spent before v3: pilot/exploratory runs used 0–59, P4.2.v1
# burned 60–119, P4.2.v2 burned 120–179. A seed-axis shard must never touch this
# range — reusing a burned seed is a new experiment dressed as a shard.
BURNED_SEED_MAX = 179

# Back-compatible alias (earlier P4.2b tooling referenced this name).
SECONDS_PER_CELL_ESTIMATE = MEASURED_SECONDS_PER_CELL


class ShardRefused(Exception):
    pass


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def git_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                       cwd=str(REPO))
    return r.stdout.strip()


def parse_shard_arms(raw: str, declared_arms: list[str]) -> list[str]:
    chosen = [a.strip() for a in raw.split(",") if a.strip()]
    if not chosen:
        raise ShardRefused("arm_subset пуст")
    unknown = [a for a in chosen if a not in declared_arms]
    if unknown:
        raise ShardRefused(f"неизвестные arms {unknown}; протокол знает {declared_arms}")
    if len(set(chosen)) != len(chosen):
        raise ShardRefused("arm_subset содержит дубликаты")
    return chosen


def parse_seed_subset(raw: str, allowed_seeds: list[int]) -> list[int]:
    """Parse a --seeds expression ('180-189' / '180,181,182') into a seed subset.

    Two guards, and neither may be softened by the caller:
      #20 burned: any seed in 0–179 is refused outright — those seeds are spent.
      #19 no silent change: a seed outside the frozen confirmatory set is refused
          rather than dropped or expanded. The shard runs EXACTLY the requested
          subset; adding or subtracting a seed here would be a new experiment.
    """
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ShardRefused("seed_subset пуст")
    if len(set(seeds)) != len(seeds):
        raise ShardRefused("seed_subset содержит дубликаты")

    burned = sorted({s for s in seeds if 0 <= s <= BURNED_SEED_MAX})
    if burned:
        raise ShardRefused(
            f"seeds {burned[:5]}… пересекают burned-диапазон 0–{BURNED_SEED_MAX} "
            "(pilot/v1/v2) — сожжённый seed в shard не входит")

    extra = sorted(set(seeds) - set(allowed_seeds))
    if extra:
        raise ShardRefused(
            f"seeds {extra[:8]} вне замороженного confirmatory-множества; добавить seed "
            "в объявленный прогон — это новый эксперимент, а не шардинг")
    return seeds


def _seed_range_str(seeds: list[int]) -> str:
    """Compact a sorted seed list into the 'lo-hi,lo-hi' form the runner parses."""
    seeds = sorted(set(seeds))
    if not seeds:
        return ""
    parts: list[str] = []
    start = prev = seeds[0]
    for s in seeds[1:]:
        if s == prev + 1:
            prev = s
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = s
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(parts)


def estimate_shard_seconds(n_arms: int, n_seeds: int, n_streams: int,
                           seconds_per_cell: float = MEASURED_SECONDS_PER_CELL) -> float:
    cells = n_arms * n_seeds * n_streams
    return cells * seconds_per_cell


def plan_shard(n_arms: int, n_seeds: int, n_streams: int,
               seconds_per_cell: float = MEASURED_SECONDS_PER_CELL,
               budget: int = MAX_SHARD_TARGET_SECONDS) -> dict:
    """Runtime planner used by --dry-run and by the planner tests.

    Returns the exact cell count and the honest budget verdict. It never edits
    the seed/arm subsets to force a fit: if the plan is over budget, the caller
    must change the split (fewer seeds, or combine axes), not the planner.
    """
    cells = n_arms * n_seeds * n_streams
    est_seconds = cells * seconds_per_cell
    return {
        "arms": n_arms,
        "seeds": n_seeds,
        "streams": n_streams,
        "cells": cells,
        "seconds_per_cell": seconds_per_cell,
        "estimated_seconds": est_seconds,
        "budget_seconds": budget,
        "within_budget": est_seconds <= budget,
    }


def build_shard_manifest(parent: dict, shard_id: str, arm_subset: list[str],
                         seed_subset: list[int], streams: list[str],
                         csv_path: pathlib.Path, prov_path: pathlib.Path,
                         head_path: pathlib.Path,
                         started: str, finished: str,
                         cells: int, wall: float, axis: str = "arms") -> dict:
    return {
        "shard_schema_version": "1.0.0",
        "protocol_id": parent["protocol_id"],
        "manifest_digest": parent["_digest"],
        "git_sha": git_sha(),
        "shard_id": shard_id,
        "shard_axis": axis,
        "seed_subset": sorted(seed_subset),
        "arm_subset": arm_subset,
        "task_streams": streams,
        "episodes_per_seed": parent["episodes_per_seed"],
        "partial": True,
        "non_confirmatory": False,
        "csv_sha256": sha256_file(csv_path),
        "provenance_jsonl_sha256": sha256_file(prov_path),
        "provenance_head_sha256": sha256_file(head_path),
        "environment_fingerprint": {
            "python": platform.python_version(),
            "numpy": _numpy_version(),
        },
        "started_at_utc": started,
        "finished_at_utc": finished,
        "cells_in_shard": cells,
        "wall_clock_seconds": round(wall, 1),
    }


def _numpy_version() -> str:
    try:
        import numpy
        return numpy.__version__
    except ImportError:
        return "absent"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent-manifest", default=str(proto.DEFAULT_MANIFEST))
    ap.add_argument("--shard-id", required=True)
    ap.add_argument("--axis", choices=("arms", "seeds"), default="arms",
                    help="arms: режем набор arms ( держим все seeds/streams ); "
                         "seeds: режем confirmatory seeds ( держим ВСЕ arms/streams )")
    ap.add_argument("--arms", help="comma-separated arm subset (нужен для --axis arms)")
    ap.add_argument("--seeds", help="seed subset, напр. 180-189 (нужен для --axis seeds)")
    ap.add_argument("--seconds-per-cell", type=float, default=MEASURED_SECONDS_PER_CELL,
                    help="измеренная цена клетки; по умолчанию ~12 c/cell из журнала v2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and estimate only, no execution")
    a = ap.parse_args()

    try:
        parent = proto.load(pathlib.Path(a.parent_manifest))
    except proto.ProtocolError as exc:
        print(f"SHARD REFUSED: parent manifest invalid: {exc}", file=sys.stderr)
        return REFUSE

    # Only partition authorized protocols.
    allowed, flag = proto.run_authorization(parent["protocol_id"])
    if not allowed:
        print(f"SHARD REFUSED: protocol {parent['protocol_id']} run={flag!r}, "
              "shards inherit the parent authorization.", file=sys.stderr)
        return REFUSE

    declared_arms = [arm["arm_id"] for arm in parent["arms"]]
    all_seeds = proto.confirmatory_seeds(parent)
    # Shard-манифест описывает experiment протокольными именами, а не code-маппингом:
    # это то же самое множество, что объявлено в манифесте и сверяется на merge.
    streams = list(parent["task_streams"])

    # Resolve the axis: which dimension is sliced, which is kept whole.
    try:
        if a.axis == "arms":
            if not a.arms:
                raise ShardRefused("--axis arms требует --arms <subset>")
            if a.seeds:
                raise ShardRefused("--axis arms не принимает --seeds: shard держит ВСЕ seeds")
            arm_subset = parse_shard_arms(a.arms, declared_arms)
            seed_subset = list(all_seeds)
        else:  # seeds
            if not a.seeds:
                raise ShardRefused("--axis seeds требует --seeds <subset>")
            if a.arms:
                raise ShardRefused("--axis seeds не принимает --arms: shard держит ВСЕ arms")
            arm_subset = list(declared_arms)
            seed_subset = parse_seed_subset(a.seeds, all_seeds)
    except ShardRefused as exc:
        print(f"SHARD REFUSED: {exc}", file=sys.stderr)
        return REFUSE

    plan = plan_shard(len(arm_subset), len(seed_subset), len(streams),
                      seconds_per_cell=a.seconds_per_cell)
    cells = plan["cells"]
    est_seconds = plan["estimated_seconds"]
    print(f"shard {a.shard_id}: axis={a.axis}, {cells} cells "
          f"({plan['arms']} arms × {plan['seeds']} seeds × {plan['streams']} streams), "
          f"est {est_seconds:.0f}s @ {a.seconds_per_cell:.1f}s/cell "
          f"(target max {MAX_SHARD_TARGET_SECONDS}s)")

    if not plan["within_budget"]:
        if a.axis == "arms":
            print("P4.2b shard plan requires a new approved seed-axis sharding protocol.",
                  file=sys.stderr)
        else:
            print("P4.2c seed-axis shard exceeds the runtime budget: reduce seeds per "
                  "shard (or combine axes). Do NOT silently change the seed subset.",
                  file=sys.stderr)
        print(f"Estimated {est_seconds:.0f}s > {MAX_SHARD_TARGET_SECONDS}s budget; "
              "the runner will not run an over-budget shard.", file=sys.stderr)
        return REFUSE

    if a.dry_run:
        print("DRY RUN: shard plan valid, not executed.")
        return 0

    # Execute via the existing runner, restricting only the sliced axis.
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{a.shard_id}.csv"
    prov_path = out_dir / f"{a.shard_id}.prov.jsonl"
    head_path = out_dir / f"{a.shard_id}.prov.jsonl.head.json"

    cmd = [
        sys.executable, str(REPO / "scripts/run_p4_2_oracle.py"),
        "--manifest", str(a.parent_manifest),
        "--out", str(csv_path),
        "--provenance", str(prov_path),
    ]
    if a.axis == "arms":
        cmd += ["--arms", ",".join(arm_subset)]
    else:
        cmd += ["--seeds", _seed_range_str(seed_subset)]

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    finished = datetime.now(timezone.utc).isoformat()

    if res.returncode != 0:
        print(f"SHARD EXEC FAILED rc={res.returncode}", file=sys.stderr)
        print(res.stderr[-1000:], file=sys.stderr)
        return res.returncode

    if not csv_path.exists():
        print("SHARD REFUSED: runner did not produce CSV", file=sys.stderr)
        return REFUSE

    manifest = build_shard_manifest(
        parent, a.shard_id, arm_subset, seed_subset, streams,
        csv_path, prov_path, head_path, started, finished, cells, a.axis)

    shard_manifest_path = out_dir / f"{a.shard_id}.shard_manifest.json"
    shard_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    print(f"shard manifest → {shard_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
