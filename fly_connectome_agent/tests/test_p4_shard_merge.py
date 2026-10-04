"""P4.2b merge and coverage tests.

Proves that:
- two valid arm-shards merge correctly;
- each specific malformation causes refusal;
- individual shard CSV cannot pass scientific gate.

Tests work with synthetic fixtures (tmp CSV + shard manifests) and do NOT
run any simulation.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import pathlib
import sys

import pytest

REPO = Path_file = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import merge_p4_shards as merge_mod  # noqa: E402
import check_p4_shard_coverage as cov_mod  # noqa: E402
import p4_2_protocol as proto  # noqa: E402


ARMS = ["rstdp_mixed", "no_plasticity", "m_zero",
        "weight_shuffled_frozen", "direction_shuffled_frozen", "oracle_reflex"]
SEEDS = list(range(180, 240))
STREAMS = ["left_target", "right_target", "mixed"]
DIGEST = "sha256:" + "f" * 64
GIT_SHA = "a" * 40
ENV_FP = {"python": "3.13.0", "numpy": "2.0.0"}


def _make_shard(shard_id: str, arm_subset: list[str], seeds: list[int] | None = None,
                streams: list[str] | None = None, **overrides) -> dict:
    s = seeds or SEEDS
    st = streams or STREAMS
    return {
        "shard_schema_version": "1.0.0",
        "protocol_id": "p4.2.oracle-baseline.v3",
        "manifest_digest": DIGEST,
        "git_sha": GIT_SHA,
        "shard_id": shard_id,
        "shard_axis": "arms",
        "seed_subset": s,
        "arm_subset": arm_subset,
        "task_streams": st,
        "episodes_per_seed": 80,
        "partial": True,
        "non_confirmatory": False,
        "csv_sha256": "sha256:" + "b" * 64,
        "provenance_jsonl_sha256": "sha256:" + "c" * 64,
        "provenance_head_sha256": "sha256:" + "d" * 64,
        "environment_fingerprint": ENV_FP,
        "started_at_utc": "2026-10-04T12:00:00Z",
        "finished_at_utc": "2026-10-04T12:15:00Z",
        "cells_in_shard": len(arm_subset) * len(s) * len(st),
        **overrides,
    }


@pytest.fixture()
def parent_manifest() -> dict:
    """Minimal parent manifest for coverage checks."""
    return {
        "protocol_id": "p4.2.oracle-baseline.v3",
        "_digest": DIGEST,
        "arms": [{"arm_id": a} for a in ARMS],
        "task_streams": STREAMS,
        "episodes_per_seed": 80,
        "seed_sets": {"confirmatory": {"range": [180, 239], "count": 60}},
    }


def _patch_parent_seeds(monkeypatch, parent):
    """confirmatory_seeds() reads from manifest; patch for synthetic parent."""
    monkeypatch.setattr(proto, "confirmatory_seeds", lambda m: SEEDS)


# --- Coverage: merge success (5 tests) ----------------------------------------

def test_two_valid_arm_shards_cover_all_arms(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:])
    problems = cov_mod.check_coverage([s1, s2], parent_manifest)
    assert problems == [], problems


def test_merged_cells_equal_full_experiment(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:])
    total = sum(len(s["arm_subset"]) * len(s["seed_subset"]) * len(s["task_streams"])
                for s in [s1, s2])
    assert total == 6 * 60 * 3  # 1080


def test_merge_csvs_concatenate_without_overlap(tmp_path):
    """Merge produces byte-for-byte shard rows."""
    rows_a = [{"arm_id": "rstdp_mixed", "seed": "180", "task": "mixed", "val": "1"}]
    rows_b = [{"arm_id": "oracle_reflex", "seed": "180", "task": "mixed", "val": "2"}]
    csv_a = tmp_path / "shard-a.csv"
    csv_b = tmp_path / "shard-b.csv"
    for path, rows in [(csv_a, rows_a), (csv_b, rows_b)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["arm_id", "seed", "task", "val"])
            w.writeheader()
            w.writerows(rows)

    shards = [
        _make_shard("shard-a", ARMS[:3], csv_sha256=_file_hash(csv_a)),
        _make_shard("shard-b", ARMS[3:], csv_sha256=_file_hash(csv_b)),
    ]
    shard_paths = [tmp_path / "shard-a.shard_manifest.json",
                   tmp_path / "shard-b.shard_manifest.json"]
    for sp, s in zip(shard_paths, shards):
        sp.write_text(json.dumps(s))

    merged = merge_mod.merge_csvs(shards, shard_paths)
    assert len(merged) == 2
    assert merged[0]["arm_id"] == "rstdp_mixed"
    assert merged[1]["arm_id"] == "oracle_reflex"


def test_merged_manifest_has_correct_constants(parent_manifest, monkeypatch, tmp_path):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    merged_csv = tmp_path / "merged.csv"
    merged_csv.write_text("arm_id,seed,task\n")
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:])
    m = merge_mod.build_merged_manifest(parent_manifest, [s1, s2], merged_csv, 1080)
    assert m["partial"] is False
    assert m["coverage_verified"] is True
    assert m["merge_status"] == "COMPLETE"
    assert m["shard_count"] == 2
    assert m["total_cells"] == 1080
    assert m["expected_cells"] == 1080


def test_burned_seed_overlap_is_refused(parent_manifest, monkeypatch):
    """Shard using P4.2.v1/v2 seeds must be refused by merge."""
    _patch_parent_seeds(monkeypatch, parent_manifest)
    burned_seeds = list(range(120, 180))  # v2 seeds
    s1 = _make_shard("shard-x", ARMS[:3], seeds=burned_seeds)
    s2 = _make_shard("shard-y", ARMS[3:], seeds=burned_seeds)
    with pytest.raises(merge_mod.MergeRefused, match="burned"):
        merge_mod.check_no_burned_overlap([s1, s2])


# --- Coverage: merge rejection (14 tests) -------------------------------------

def test_overlapping_arm_subsets_fail(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    s1 = _make_shard("shard-a", ARMS[:4])  # overlaps on arm index 3
    s2 = _make_shard("shard-b", ARMS[3:])
    problems = cov_mod.check_coverage([s1, s2], parent_manifest)
    assert any("overlap" in p for p in problems), problems


def test_missing_arm_fails(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:5])  # oracle_reflex missing
    problems = cov_mod.check_coverage([s1, s2], parent_manifest)
    assert any("отсутствуют" in p for p in problems), problems


def test_different_manifest_digest_fails():
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:])
    s2["manifest_digest"] = "sha256:" + "e" * 64
    parent = {"protocol_id": "p4.2.oracle-baseline.v3", "_digest": s1["manifest_digest"],
              "arms": [{"arm_id": a} for a in ARMS], "task_streams": STREAMS,
              "episodes_per_seed": 80,
              "seed_sets": {"confirmatory": {"range": [180, 239], "count": 60}}}
    with pytest.raises(merge_mod.MergeRefused, match="manifest_digest"):
        merge_mod.validate_shard_identity([s1, s2], parent)


def test_different_git_sha_fails():
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:])
    s2["git_sha"] = "b" * 40
    parent = {"protocol_id": "p4.2.oracle-baseline.v3", "_digest": DIGEST,
              "arms": [{"arm_id": a} for a in ARMS]}
    with pytest.raises(merge_mod.MergeRefused, match="git_sha"):
        merge_mod.validate_shard_identity([s1, s2], parent)


def test_different_environment_fingerprint_fails():
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:])
    s2["environment_fingerprint"] = {"python": "3.12.0", "numpy": "2.0.0"}
    parent = {"protocol_id": "p4.2.oracle-baseline.v3", "_digest": DIGEST,
              "arms": [{"arm_id": a} for a in ARMS]}
    with pytest.raises(merge_mod.MergeRefused, match="environment_fingerprint"):
        merge_mod.validate_shard_identity([s1, s2], parent)


def test_duplicate_csv_cell_fails(tmp_path):
    csv_file = tmp_path / "shard-a.csv"
    rows = [{"arm_id": "rstdp_mixed", "seed": "180", "task": "mixed"}]
    with open(csv_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm_id", "seed", "task"])
        w.writeheader()
        w.writerows(rows + rows)  # duplicate

    s1 = _make_shard("shard-a", ARMS[:3])
    shard_path = tmp_path / "shard-a.shard_manifest.json"
    shard_path.write_text(json.dumps(s1))
    with pytest.raises(merge_mod.MergeRefused, match="дубликат"):
        merge_mod.merge_csvs([s1], [shard_path])


def test_missing_seed_in_shard_fails(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    partial_seeds = SEEDS[:50]  # only 50 of 60
    s1 = _make_shard("shard-a", ARMS[:3], seeds=partial_seeds)
    s2 = _make_shard("shard-b", ARMS[3:], seeds=partial_seeds)
    problems = cov_mod.check_coverage([s1, s2], parent_manifest)
    assert any("seeds missing" in p for p in problems), problems


def test_missing_stream_fails(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    s1 = _make_shard("shard-a", ARMS[:3], streams=["left_target", "right_target"])
    s2 = _make_shard("shard-b", ARMS[3:])
    problems = cov_mod.check_coverage([s1, s2], parent_manifest)
    assert any("task_streams" in p for p in problems), problems


def test_bad_csv_sha_fails(tmp_path):
    s1 = _make_shard("shard-a", ARMS[:3], csv_sha256="sha256:" + "0" * 64)
    csv_file = tmp_path / "shard-a.csv"
    csv_file.write_text("arm_id,seed,task\n")
    head_file = tmp_path / "shard-a.prov.jsonl.head.json"
    head_file.write_text("{}")
    prov_file = tmp_path / "shard-a.prov.jsonl"
    prov_file.write_text("{}\n")
    sp = tmp_path / "shard-a.shard_manifest.json"
    sp.write_text(json.dumps(s1))
    with pytest.raises(merge_mod.MergeRefused, match="CSV hash"):
        merge_mod.validate_shard_hashes([s1], [sp])


def test_bad_provenance_jsonl_sha_fails(tmp_path):
    s1 = _make_shard("shard-a", ARMS[:3], provenance_jsonl_sha256="sha256:" + "0" * 64)
    csv_file = tmp_path / "shard-a.csv"
    csv_file.write_text("arm_id,seed,task\n")
    s1["csv_sha256"] = _file_hash(csv_file)
    prov_file = tmp_path / "shard-a.prov.jsonl"
    prov_file.write_text("{}\n")
    head_file = tmp_path / "shard-a.prov.jsonl.head.json"
    head_file.write_text("{}")
    s1["provenance_head_sha256"] = _file_hash(head_file)
    sp = tmp_path / "shard-a.shard_manifest.json"
    sp.write_text(json.dumps(s1))
    with pytest.raises(merge_mod.MergeRefused, match="provenance JSONL hash"):
        merge_mod.validate_shard_hashes([s1], [sp])


def test_bad_head_witness_sha_fails(tmp_path):
    s1 = _make_shard("shard-a", ARMS[:3], provenance_head_sha256="sha256:" + "0" * 64)
    csv_file = tmp_path / "shard-a.csv"
    csv_file.write_text("arm_id,seed,task\n")
    s1["csv_sha256"] = _file_hash(csv_file)
    prov_file = tmp_path / "shard-a.prov.jsonl"
    prov_file.write_text("{}\n")
    s1["provenance_jsonl_sha256"] = _file_hash(prov_file)
    head_file = tmp_path / "shard-a.prov.jsonl.head.json"
    head_file.write_text("{}")
    sp = tmp_path / "shard-a.shard_manifest.json"
    sp.write_text(json.dumps(s1))
    with pytest.raises(merge_mod.MergeRefused, match="provenance head hash"):
        merge_mod.validate_shard_hashes([s1], [sp])


def test_non_confirmatory_shard_identity_fails():
    s1 = _make_shard("shard-a", ARMS[:3], non_confirmatory=True)
    s2 = _make_shard("shard-b", ARMS[3:])
    parent = {"protocol_id": "p4.2.oracle-baseline.v3", "_digest": DIGEST,
              "arms": [{"arm_id": a} for a in ARMS]}
    with pytest.raises(merge_mod.MergeRefused, match="non_confirmatory"):
        merge_mod.validate_shard_identity([s1, s2], parent)


def test_failed_provenance_verification_fails(tmp_path):
    """(#12) Цепь provenance обязана сходиться сама по себе, даже если хеши
    в манифесте честные: манифест ничего не обещал о сломанной цепи, но журнал
    противоречит собственному witness.

    Сначала собираем валидный журнал писателем, потом портим payload одной строки,
    не пересчитывая entry_hash — verify_chain обязан вернуть False, merge — отказаться.
    """
    from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog

    prov = tmp_path / "shard-a.prov.jsonl"
    log = ProvenanceLog(str(prov))
    log.append({"kind": "start"})
    log.append({"kind": "cell"})

    s1 = _make_shard("shard-a", ARMS[:3])
    sp = tmp_path / "shard-a.shard_manifest.json"
    sp.write_text(json.dumps(s1))
    # валидная цепь не вызывает отказа
    assert merge_mod.verify_shard_provenance([s1], [sp]) is None

    # ломаем цепь: payload первой записи меняется, её хеши остаются старыми
    lines = prov.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["payload"] = {"kind": "tampered"}
    lines[0] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    prov.write_text("\n".join(lines) + "\n")
    with pytest.raises(merge_mod.MergeRefused, match="chain verification failed"):
        merge_mod.verify_shard_provenance([s1], [sp])


def test_aborted_protocol_result_fails():
    """(#13/4) Shard от закрытого aborted-протокола (P4.2.v1) не merge-ится:
    статус живёт в реестре кода, а не в манифесте shard."""
    s1 = _make_shard("shard-a", ARMS[:3], protocol_id="p4.2.oracle-baseline.v1")
    s2 = _make_shard("shard-b", ARMS[3:])
    with pytest.raises(merge_mod.MergeRefused, match="aborted"):
        merge_mod.check_not_aborted([s1, s2])


def test_individual_shard_cannot_pass_scientific_gate(tmp_path):
    """(#14) Научный гейт отвергает sidecar отдельного shard (partial=true) и
    принимает только merged результат. Проверка идёт настоящим кодом гейта
    check_p4_2_oracle_gate.check_sidecars, а не репликой."""
    import check_p4_2_oracle_gate as gate

    digest = proto.FROZEN_PROTOCOL_DIGEST
    csv_file = tmp_path / "shard-a.csv"
    csv_file.write_text("arm,seed,stream,success_rate\n")
    sidecar = gate.sidecar_path_for(csv_file)
    sidecar.write_text(json.dumps({
        "manifest_digest": digest, "partial": True, "non_confirmatory": False}))
    # отдельный shard: partial=true -> гейт отказывает
    with pytest.raises(gate.Refused, match="partial"):
        gate.check_sidecars([str(csv_file)], [str(sidecar)], digest)
    # тот же CSV после merge (partial=false) проходит эту проверку sidecar'ов
    sidecar.write_text(json.dumps({
        "manifest_digest": digest, "partial": False, "non_confirmatory": False}))
    sides = gate.check_sidecars([str(csv_file)], [str(sidecar)], digest)
    assert len(sides) == 1


# --- Seed-axis coverage / merge (P4.2c) ---------------------------------------

def _seed_shard(shard_id: str, seeds: list[int], **ov) -> dict:
    """Seed-axis shard: держит ВСЕ arms и ВСЕ streams, режет лишь seeds."""
    return _make_shard(shard_id, ARMS, seeds=seeds, shard_axis="seeds", **ov)


def _six_seed_shards() -> list[dict]:
    return [_seed_shard(f"shard-s{i}", list(range(180 + i * 10, 190 + i * 10)))
            for i in range(6)]


def _seed_csv_rows(arms, seeds, streams):
    return [{"arm_id": a, "seed": str(sd), "task": st}
            for a in arms for sd in seeds for st in streams]


def _write_shard(tmp_path, shard: dict):
    """Materialise a shard dir (CSV + prov + head + manifest) with REAL hashes,
    so hash-dependent guards are genuinely exercised."""
    sid = shard["shard_id"]
    d = tmp_path / sid
    d.mkdir(parents=True, exist_ok=True)
    csvf = d / f"{sid}.csv"
    with open(csvf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm_id", "seed", "task"])
        w.writeheader()
        w.writerows(_seed_csv_rows(shard["arm_subset"], shard["seed_subset"], shard["task_streams"]))
    provf = d / f"{sid}.prov.jsonl"
    provf.write_text('{"kind":"start"}\n')
    headf = d / f"{sid}.prov.jsonl.head.json"
    headf.write_text('{"last_hash":"x"}')
    shard["csv_sha256"] = _file_hash(csvf)
    shard["provenance_jsonl_sha256"] = _file_hash(provf)
    shard["provenance_head_sha256"] = _file_hash(headf)
    sp = d / f"{sid}.shard_manifest.json"
    sp.write_text(json.dumps(shard))
    return shard, sp


# (1) six 10-seed shards merge successfully; (2) 1080 exact cells.
def test_six_seed_shards_cover_experiment(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    shards = _six_seed_shards()
    problems = cov_mod.check_coverage(shards, parent_manifest)
    assert problems == [], problems
    total = sum(len(s["arm_subset"]) * len(s["seed_subset"]) * len(s["task_streams"])
                for s in shards)
    assert total == 6 * 60 * 3 == 1080


# (3) no duplicate cells across the merged CSV.
def test_six_seed_shards_merge_without_duplicates(tmp_path):
    shards = _six_seed_shards()
    written = [_write_shard(tmp_path, s) for s in shards]
    shard_paths = [sp for _, sp in written]
    merged = merge_mod.merge_csvs([s for s, _ in written], shard_paths)
    assert len(merged) == 1080
    keys = {(r["arm_id"], r["seed"], r["task"]) for r in merged}
    assert len(keys) == 1080  # unique cells, no dup


# (4) merged manifest carries the seed-axis constants.
def test_merged_manifest_seed_axis(tmp_path):
    parent = {"protocol_id": "p4.2.oracle-baseline.v3", "_digest": DIGEST,
              "arms": [{"arm_id": a} for a in ARMS], "task_streams": STREAMS,
              "episodes_per_seed": 80}
    shards = _six_seed_shards()
    written = [_write_shard(tmp_path, s) for s in shards]
    shard_paths = [sp for _, sp in written]
    merged_csv = tmp_path / "merged.csv"
    merged_csv.write_text("arm_id,seed,task\n")
    m = merge_mod.build_merged_manifest(parent, [s for s, _ in written], merged_csv, 1080,
                                        shard_paths)
    assert m["partial"] is False
    assert m["coverage_verified"] is True
    assert m["merge_status"] == "COMPLETE"
    assert m["merge_axis"] == "seeds"
    assert m["shard_count"] == 6
    assert len(m["source_shards"]) == 6


# (5) overlap seed 190 across two shards fails.
def test_seed_overlap_fails(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    s1 = _seed_shard("shard-a", list(range(180, 191)))   # includes 190
    s2 = _seed_shard("shard-b", list(range(190, 200)))   # also 190
    problems = cov_mod.check_coverage([s1, s2], parent_manifest)
    assert any("overlap" in p for p in problems), problems


# (6) missing seed 239 fails.
def test_seed_gap_fails(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    shards = _six_seed_shards()
    shards[-1]["seed_subset"] = list(range(230, 239))     # drops 239
    shards[-1]["cells_in_shard"] = 6 * 9 * 3
    problems = cov_mod.check_coverage(shards, parent_manifest)
    assert any("отсутствуют" in p for p in problems), problems


# (7) one seed-shard missing the oracle arm fails.
def test_seed_shard_missing_arm_fails(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    shards = _six_seed_shards()
    shards[0]["arm_subset"] = ARMS[:-1]                    # oracle_reflex dropped
    shards[0]["cells_in_shard"] = 5 * 10 * 3
    problems = cov_mod.check_coverage(shards, parent_manifest)
    assert any("arms missing" in p for p in problems), problems


# (8) one seed-shard missing the mixed stream fails.
def test_seed_shard_missing_stream_fails(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    shards = _six_seed_shards()
    shards[1] = _seed_shard("shard-s1", list(range(190, 200)),
                            streams=["left_target", "right_target"])
    problems = cov_mod.check_coverage(shards, parent_manifest)
    assert any("task_streams" in p for p in problems), problems


# (9) mixed arms-axis and seeds-axis in one merge fails (coverage + identity).
def test_mixed_axis_shards_fail_coverage(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    arms_shard = _make_shard("shard-arms", ARMS[:3])       # axis arms
    seed_shard = _seed_shard("shard-seed", list(range(180, 240)))
    problems = cov_mod.check_coverage([arms_shard, seed_shard], parent_manifest)
    assert any("смешанные оси" in p for p in problems), problems


def test_mixed_axis_shards_fail_identity():
    arms_shard = _make_shard("shard-arms", ARMS[:3])
    seed_shard = _seed_shard("shard-seed", list(range(180, 240)))
    parent = {"protocol_id": "p4.2.oracle-baseline.v3", "_digest": DIGEST,
              "arms": [{"arm_id": a} for a in ARMS]}
    with pytest.raises(merge_mod.MergeRefused, match="смешанные оси"):
        merge_mod.validate_shard_identity([arms_shard, seed_shard], parent)


# (10) one shard with 79 episodes fails.
def test_seed_shard_wrong_episodes_fails(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    shards = _six_seed_shards()
    shards[2] = _seed_shard("shard-s2", list(range(200, 210)), episodes_per_seed=79)
    problems = cov_mod.check_coverage(shards, parent_manifest)
    assert any("episodes_per_seed" in p for p in problems), problems


# (11) one shard from another Git SHA fails.
def test_seed_shard_other_git_sha_fails():
    s1 = _seed_shard("shard-a", list(range(180, 190)))
    s2 = _seed_shard("shard-b", list(range(190, 200)))
    s2["git_sha"] = "b" * 40
    parent = {"protocol_id": "p4.2.oracle-baseline.v3", "_digest": DIGEST,
              "arms": [{"arm_id": a} for a in ARMS]}
    with pytest.raises(merge_mod.MergeRefused, match="git_sha"):
        merge_mod.validate_shard_identity([s1, s2], parent)


# (12) one shard with a foreign digest fails.
def test_seed_shard_foreign_digest_fails():
    s1 = _seed_shard("shard-a", list(range(180, 190)))
    s2 = _seed_shard("shard-b", list(range(190, 200)))
    s2["manifest_digest"] = "sha256:" + "9" * 64
    parent = {"protocol_id": "p4.2.oracle-baseline.v3", "_digest": DIGEST,
              "arms": [{"arm_id": a} for a in ARMS]}
    with pytest.raises(merge_mod.MergeRefused, match="manifest_digest"):
        merge_mod.validate_shard_identity([s1, s2], parent)


# --- Runtime planner (P4.2c, run_p4_shard) ------------------------------------

import run_p4_shard as shard_mod  # noqa: E402


def test_planner_ten_seed_shard_is_180_cells():
    # (17) 10 seeds × 6 arms × 3 streams produces 180 cells.
    p = shard_mod.plan_shard(6, 10, 3)
    assert p["cells"] == 180


def test_planner_refuses_over_budget():
    # (18) refuses if estimated duration > 1200s (180 cells @ ~12s/cell = 2160s).
    p = shard_mod.plan_shard(6, 10, 3)
    assert p["within_budget"] is False
    assert p["estimated_seconds"] > shard_mod.MAX_SHARD_TARGET_SECONDS


def test_planner_accepts_within_budget_shard():
    # the compliant seed-axis split (5 seeds) fits; planner does not force it.
    p = shard_mod.plan_shard(6, 5, 3)
    assert p["cells"] == 90 and p["within_budget"] is True


def test_planner_does_not_silently_change_seed_subset():
    # (19) the parsed subset is EXACTLY what was asked; out-of-protocol refuses.
    got = shard_mod.parse_seed_subset("180-189", list(range(180, 240)))
    assert got == list(range(180, 190))
    with pytest.raises(shard_mod.ShardRefused):
        shard_mod.parse_seed_subset("180-189,250", list(range(180, 240)))


def test_planner_refuses_burned_seeds():
    # (20) refuses any seed overlapping burned range 0-179.
    with pytest.raises(shard_mod.ShardRefused, match="burned"):
        shard_mod.parse_seed_subset("150-159", list(range(180, 240)))


# --- Helper ---

def _file_hash(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()}"
