#!/usr/bin/env python3
"""P4.2b: run one shard (arm-group) of a frozen P4.2 experiment.

Infrastructure-only: wraps the existing runner logic with shard identity,
budget estimation, and shard-manifest generation. Does NOT change
scientific parameters, arms, metrics, seeds, or provenance semantics.

Usage:
    python scripts/run_p4_shard.py --parent-manifest <path> \
        --shard-id <id> --arms <arm1,arm2,...> \
        --out-dir <path> [--dry-run]

Exit codes:
    0 = shard completed (or --dry-run plan printed)
    2 = REFUSED (shard invalid, budget exceeded, identity mismatch)
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
SECONDS_PER_CELL_ESTIMATE = 14   # conservative based on measured ~12s/cell


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


def estimate_shard_seconds(n_arms: int, n_seeds: int, n_streams: int) -> float:
    cells = n_arms * n_seeds * n_streams
    return cells * SECONDS_PER_CELL_ESTIMATE


def build_shard_manifest(parent: dict, shard_id: str, arm_subset: list[str],
                         seed_subset: list[int], streams: list[str],
                         csv_path: pathlib.Path, prov_path: pathlib.Path,
                         head_path: pathlib.Path,
                         started: str, finished: str,
                         cells: int, wall: float) -> dict:
    return {
        "shard_schema_version": "1.0.0",
        "protocol_id": parent["protocol_id"],
        "manifest_digest": parent["_digest"],
        "git_sha": git_sha(),
        "shard_id": shard_id,
        "shard_axis": "arms",
        "seed_subset": seed_subset,
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
    ap.add_argument("--arms", required=True, help="comma-separated arm subset")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and estimate only, no execution")
    a = ap.parse_args()

    try:
        parent = proto.load(pathlib.Path(a.parent_manifest))
    except proto.ProtocolError as exc:
        print(f"SHARD REFUSED: parent manifest invalid: {exc}", file=sys.stderr)
        return REFUSE

    # Only partition authorized protocols
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

    try:
        arm_subset = parse_shard_arms(a.arms, declared_arms)
    except ShardRefused as exc:
        print(f"SHARD REFUSED: {exc}", file=sys.stderr)
        return REFUSE

    # Budget estimate
    est_seconds = estimate_shard_seconds(len(arm_subset), len(all_seeds), len(streams))
    cells = len(arm_subset) * len(all_seeds) * len(streams)
    print(f"shard {a.shard_id}: {cells} cells, est {est_seconds:.0f}s "
          f"(target max {MAX_SHARD_TARGET_SECONDS}s)")

    if est_seconds > MAX_SHARD_TARGET_SECONDS:
        print("P4.2b shard plan requires a new approved seed-axis sharding protocol.",
              file=sys.stderr)
        print(f"Estimated {est_seconds:.0f}s > {MAX_SHARD_TARGET_SECONDS}s budget; "
              "arm-only partition cannot meet the target. Do NOT silently split seeds.",
              file=sys.stderr)
        return REFUSE

    if a.dry_run:
        print("DRY RUN: shard plan valid, not executed.")
        return 0

    # Execute via the existing runner, passing arm restriction
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{a.shard_id}.csv"
    prov_path = out_dir / f"{a.shard_id}.prov.jsonl"
    head_path = out_dir / f"{a.shard_id}.prov.jsonl.head.json"

    cmd = [
        sys.executable, str(REPO / "scripts/run_p4_2_oracle.py"),
        "--manifest", str(a.parent_manifest),
        "--arms", ",".join(arm_subset),
        "--out", str(csv_path),
        "--provenance", str(prov_path),
    ]
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
        parent, a.shard_id, arm_subset, all_seeds, streams,
        csv_path, prov_path, head_path, started, finished, cells, wall)

    shard_manifest_path = out_dir / f"{a.shard_id}.shard_manifest.json"
    shard_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    print(f"shard manifest → {shard_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
