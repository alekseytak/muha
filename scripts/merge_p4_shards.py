#!/usr/bin/env python3
"""P4.2b: merge N shard results into one complete experiment output.

Validates shard integrity, checks coverage, produces merged CSV and
merged result manifest. The merged manifest is the ONLY artifact that
the scientific gate can accept.

Usage:
    python scripts/merge_p4_shards.py --parent-manifest <path> \
        --shard-manifests <path1.json> <path2.json> [...] \
        --out-dir <path>

Exit codes:
    0 = merge complete
    2 = REFUSED (coverage gap, overlap, hash mismatch, etc.)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import p4_2_protocol as proto  # noqa: E402
import check_p4_shard_coverage as coverage  # noqa: E402

REFUSE = 2


class MergeRefused(Exception):
    pass


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def load_shard(path: pathlib.Path) -> dict:
    if not path.exists():
        raise MergeRefused(f"shard manifest не найден: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_shard_identity(shards: list[dict], parent: dict) -> None:
    """All shards must agree on protocol, digest, git SHA, environment."""
    if len(shards) < 2:
        raise MergeRefused("merge требует минимум 2 shards")

    expected_digest = parent["_digest"]
    expected_protocol = parent["protocol_id"]

    for s in shards:
        if s["protocol_id"] != expected_protocol:
            raise MergeRefused(
                f"shard {s['shard_id']}: protocol_id {s['protocol_id']!r} != {expected_protocol!r}")
        if s["manifest_digest"] != expected_digest:
            raise MergeRefused(
                f"shard {s['shard_id']}: manifest_digest разошёлся с parent")
        if s.get("partial") is not True:
            raise MergeRefused(f"shard {s['shard_id']}: partial должен быть true")
        if s.get("non_confirmatory"):
            raise MergeRefused(f"shard {s['shard_id']}: non_confirmatory shard в merge не входит")
        if s["shard_axis"] not in ("arms", "seeds"):
            raise MergeRefused(f"shard {s['shard_id']}: shard_axis {s['shard_axis']!r} вне arms/seeds")

    # Ось обязана быть единой: смешивать arms-axis и seeds-axis в одном merge
    # нельзя — полное покрытие тогда не сводится к одному точному разбиению.
    axes = {s["shard_axis"] for s in shards}
    if len(axes) != 1:
        raise MergeRefused(
            f"смешанные оси в одном merge запрещены: {sorted(axes)} — нужен отдельный merge на каждую ось")

    ref = shards[0]
    for s in shards[1:]:
        if s["git_sha"] != ref["git_sha"]:
            raise MergeRefused(
                f"shards {ref['shard_id']} и {s['shard_id']}: git_sha различается")
        if s["environment_fingerprint"] != ref["environment_fingerprint"]:
            raise MergeRefused(
                f"shards {ref['shard_id']} и {s['shard_id']}: environment_fingerprint различается")


def validate_shard_hashes(shards: list[dict], shard_paths: list[pathlib.Path]) -> None:
    """Re-hash files on disk and compare to recorded digests."""
    for s, manifest_path in zip(shards, shard_paths):
        d = manifest_path.parent
        csv_file = d / f"{s['shard_id']}.csv"
        prov_file = d / f"{s['shard_id']}.prov.jsonl"
        head_file = d / f"{s['shard_id']}.prov.jsonl.head.json"

        for f, key, label in [(csv_file, "csv_sha256", "CSV"),
                              (prov_file, "provenance_jsonl_sha256", "provenance JSONL"),
                              (head_file, "provenance_head_sha256", "provenance head")]:
            if not f.exists():
                raise MergeRefused(f"shard {s['shard_id']}: {label} файл не найден: {f.name}")
            actual = sha256_file(f)
            if actual != s[key]:
                raise MergeRefused(
                    f"shard {s['shard_id']}: {label} hash разошёлся "
                    f"(записан {s[key][:19]}…, на диске {actual[:19]}…)")


def verify_shard_provenance(shards: list[dict], shard_paths: list[pathlib.Path]) -> None:
    """Проверить саму цепь provenance каждого shard, независимо от записанных хешей.

    validate_shard_hashes сверяет файл с манифестом (#10). Здесь — другой отказ
    (#12): журнал и witness internally противоречат друг другу (разрыв цепи,
    подрезанный хвост, подправленный payload), даже если хеши в манифесте честные.
    Используется тот же ProvenanceLog.verify_chain(), что и у раннера: один
    определитель целостности, а не второй, свой.
    """
    from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog
    for s, manifest_path in zip(shards, shard_paths):
        prov_file = manifest_path.parent / f"{s['shard_id']}.prov.jsonl"
        log = ProvenanceLog(str(prov_file))
        if not log.verify_chain():
            raise MergeRefused(
                f"shard {s['shard_id']}: provenance chain verification failed "
                f"(цепь журнала противоречит witness: {prov_file.name})")


def check_not_aborted(shards: list[dict]) -> None:
    """Не merge-ить чанки от abortивного или неавторизованного протокола (#4).

    Реестр FROZEN_PROTOCOLS — единственный источник статуса: манифест shard не
    может объявить свой прогон завершённым. Статус со слов 'aborted' и снятое
    право на запуск (run != authorized) — оба отказа; неизвестный будущий
    протокол пропускается (его связывают digest + git_sha + coverage).
    """
    for s in shards:
        entry = proto.FROZEN_PROTOCOLS.get(s["protocol_id"])
        if entry is None:
            continue
        if "aborted" in entry["status"]:
            raise MergeRefused(
                f"shard {s['shard_id']}: протокол {s['protocol_id']} со статусом "
                f"«{entry['status']}» — aborted-прогон в merge не входит")
        allowed, flag = proto.run_authorization(s["protocol_id"])
        if not allowed:
            raise MergeRefused(
                f"shard {s['shard_id']}: протокол {s['protocol_id']} не авторизован "
                f"(run={flag!r}) — merge по неавторизованному прогону запрещён")


def check_no_burned_overlap(shards: list[dict]) -> None:
    """Refuse to merge shards using seeds from aborted v1/v2 runs."""
    burned_ranges = [(60, 119), (120, 179)]  # P4.2.v1, P4.2.v2
    for s in shards:
        seeds = set(s["seed_subset"])
        for lo, hi in burned_ranges:
            overlap = seeds & set(range(lo, hi + 1))
            if overlap:
                raise MergeRefused(
                    f"shard {s['shard_id']}: seeds {sorted(overlap)[:5]}… "
                    f"пересекаются с burned-диапазоном {lo}–{hi}")


def merge_csvs(shards: list[dict], shard_paths: list[pathlib.Path]) -> list[dict]:
    """Concatenate rows from all shard CSVs. Rows must not overlap."""
    all_rows: list[dict] = []
    seen_cells: set[tuple] = set()

    for s, mp in zip(shards, shard_paths):
        csv_path = mp.parent / f"{s['shard_id']}.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get("arm_id") or row.get("arm"),
                       row.get("seed"), row.get("task") or row.get("stream"))
                if key in seen_cells:
                    raise MergeRefused(f"дубликат cell {key} в shard {s['shard_id']}")
                seen_cells.add(key)
                all_rows.append(row)
    return all_rows


def build_merged_manifest(parent: dict, shards: list[dict],
                          merged_csv_path: pathlib.Path,
                          total_cells: int,
                          shard_paths: list[pathlib.Path] | None = None) -> dict:
    all_arms = sorted(set(a for s in shards for a in s["arm_subset"]))
    all_seeds = sorted(set(seed for s in shards for seed in s["seed_subset"]))
    expected = len(all_arms) * len(all_seeds) * len(parent["task_streams"])

    # source_shards — заново проверяемое evidence: merged-gate adapter сверяет эти
    # хеши и подмножества с диском, поэтому merged-результат нельзя подделать,
    # пересобрав только сам манифест. shard_manifest_path — указатель, по которому
    # гейт доходит до исходных CSV/provenance/head; без него «source shard hash»
    # было бы числом, которое не с чем сверить.
    source_shards = [{
        "shard_id": s["shard_id"],
        "shard_axis": s["shard_axis"],
        "manifest_digest": s["manifest_digest"],
        "git_sha": s["git_sha"],
        "arm_subset": sorted(s["arm_subset"]),
        "seed_subset": sorted(s["seed_subset"]),
        "task_streams": sorted(s["task_streams"]),
        "csv_sha256": s["csv_sha256"],
        "provenance_jsonl_sha256": s["provenance_jsonl_sha256"],
        "provenance_head_sha256": s["provenance_head_sha256"],
        "shard_manifest_path": (str(shard_paths[i]) if shard_paths
                                 else f"{s['shard_id']}.shard_manifest.json"),
    } for i, s in enumerate(shards)]

    return {
        "merge_schema_version": "1.0.0",
        "protocol_id": parent["protocol_id"],
        "manifest_digest": parent["_digest"],
        "git_sha": shards[0]["git_sha"],
        "merge_status": "COMPLETE",
        "merge_axis": shards[0]["shard_axis"],
        "partial": False,
        "coverage_verified": True,
        "shard_count": len(shards),
        "shard_ids": [s["shard_id"] for s in shards],
        "source_shards": source_shards,
        "total_cells": total_cells,
        "expected_cells": expected,
        "arms": all_arms,
        "seeds": all_seeds,
        "task_streams": parent["task_streams"],
        "episodes_per_seed": parent["episodes_per_seed"],
        "merged_csv_path": str(merged_csv_path),
        "merged_csv_sha256": sha256_file(merged_csv_path),
        "merged_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_fingerprint": shards[0]["environment_fingerprint"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent-manifest", default=str(proto.DEFAULT_MANIFEST))
    ap.add_argument("--shard-manifests", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    try:
        parent = proto.load(pathlib.Path(a.parent_manifest))
    except proto.ProtocolError as exc:
        print(f"MERGE REFUSED: {exc}", file=sys.stderr)
        return REFUSE

    shard_paths = [pathlib.Path(p) for p in a.shard_manifests]
    try:
        shards = [load_shard(p) for p in shard_paths]
        validate_shard_identity(shards, parent)
        validate_shard_hashes(shards, shard_paths)
        verify_shard_provenance(shards, shard_paths)
        check_not_aborted(shards)
        check_no_burned_overlap(shards)

        # Coverage check
        problems = coverage.check_coverage(shards, parent)
        if problems:
            for p in problems:
                print(f"  COVERAGE: {p}", file=sys.stderr)
            raise MergeRefused("coverage check failed")

        out_dir = pathlib.Path(a.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        merged_csv = out_dir / "p4_2_merged.csv"
        rows = merge_csvs(shards, shard_paths)

        # Write merged CSV
        if rows:
            import csv as csvmod
            with open(merged_csv, "w", newline="", encoding="utf-8") as f:
                writer = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        manifest = build_merged_manifest(parent, shards, merged_csv, len(rows), shard_paths)
        manifest_path = out_dir / "p4_2_merged_result_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")

        # Merged sidecar — единственный артефакт, который scientific gate готов
        # принять как «полный прогон»: partial=false + ссылки на merged-манифест и
        # merged-CSV. Строится кодом гейта, чтобы sidecar и гейт сверяли одно и то же.
        import check_p4_2_oracle_gate as gate
        sidecar = gate.build_merged_sidecar(manifest_path)
        sidecar_path = gate.sidecar_path_for(merged_csv)
        sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")

        print(f"merged: {len(rows)} cells → {merged_csv}")
        print(f"merged manifest → {manifest_path}")
        print(f"merged sidecar → {sidecar_path} (partial=false, merge_status=COMPLETE)")
        return 0

    except MergeRefused as exc:
        print(f"MERGE REFUSED: {exc}", file=sys.stderr)
        return REFUSE


if __name__ == "__main__":
    raise SystemExit(main())
