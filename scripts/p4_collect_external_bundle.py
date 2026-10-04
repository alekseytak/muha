#!/usr/bin/env python
"""P4.2d — сборка SHA-256 манифеста для bundle ОДНОГО внешнего прогона P4.2.

Отличие от `collect_p4_2_bundle.py`: тот собирает bundle по списку ролей из
замороженного манифеста (9 артефактов). Внешний (без IDE-timeout) прогон обязан
оставить два дополнительных доказательства хода исполнения — `start_metadata`
(чем и откуда запущен процесс) и `run_log` (stdout/stderr самого прогона), — без
них нельзя отличить «прогон упал на seed 40» от «прогон вообще не запускался».
Нового манифеста v3 здесь нет и быть не может (он вне scope), поэтому контракт
ролей живёт в коде этого скрипта — ровно как `REQUIRED_BUNDLE_ROLES` живёт в коде
протокола, а не в самодекларации файла.

Что гарантирует collector:
  - каждый обязательный артефакт присутствует на диске (нет CSV / sidecar /
    head-witness — отказ, а не «недописанный bundle»);
  - provenance JSONL и его head-witness — пара: журнал без свидетеля обрезан;
  - каждый обязательный файл хешится sha256; bundle COMPLETE только когда
    захешены ВСЕ обязательные артефакты.

Никакой науки: скрипт ничего не считает по метрикам, не запускает runner/gate и
не создаёт v3, seed'ы или запуск. Он упаковывает уже существующие артефакты.

Коды возврата: 2 = bundle неполон (обязательный артефакт отсутствует), 0 = собран.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import collect_p4_2_bundle as bundle_mod  # noqa: E402 (переиспользуем sha256_of)
from fly_connectome_agent.src.engineering.logging.provenance_log import HEAD_SUFFIX  # noqa: E402

REFUSE = 2
COLLECT_RECORD_VERSION = "1.0.0"
COLLECT_KIND = "p4_2d_external_bundle_hashes"

# Базовые имена, принятые в runner/gate (см. p4_2_oracle_confirmatory_v2.json и
# run_p4_2_oracle.py): sidecar и witness выводятся из них, а не задаются отдельно,
# чтобы суффикс-контракт писателя не разъехался со списком ролей.
CSV_NAME = "run.csv"
PROV_NAME = "provenance.jsonl"

SIDECAR_NAME = CSV_NAME + ".meta.json"        # run.csv -> run.csv.meta.json
WITNESS_NAME = PROV_NAME + HEAD_SUFFIX        # provenance.jsonl -> …/provenance.jsonl.head.json

# Роль -> имя файла. file_hashes_manifest — сам манифест хешей, он исключён из
# собственного списка (не может содержать собственный sha256, см. self_excluded).
REQUIRED_ARTIFACTS: dict[str, str] = {
    "run_csv": CSV_NAME,
    "csv_sidecar": SIDECAR_NAME,
    "provenance_jsonl": PROV_NAME,
    "provenance_head_witness": WITNESS_NAME,
    "start_metadata": "run.start.json",
    "environment_fingerprint": "environment.json",
    "run_log": "run.log",
    "gate_stdout": "gate.stdout.txt",
    "gate_verdict_json": "gate.verdict.json",
    "per_seed_outcomes": "per_seed_outcomes.csv",
    "file_hashes_manifest": "bundle_sha256.json",
}

# Хешуем всё, кроме самого списка хешей.
HASHED_ROLES = tuple(r for r in REQUIRED_ARTIFACTS if r != "file_hashes_manifest")


class Refused(Exception):
    """Bundle не собирается: обязательного артефакта нет или witness-пара нарушена."""


def enumerate_bundle(bundle_dir: pathlib.Path) -> tuple[list[dict], list[str]]:
    """(захашированные записи, список отсутствующих обязательных ролей)."""
    base = pathlib.Path(bundle_dir)
    entries, missing = [], []
    for role in HASHED_ROLES:
        path = base / REQUIRED_ARTIFACTS[role]
        if not path.exists():
            missing.append(role)
            continue
        entries.append({
            "role": role,
            "filename": path.name,
            "path": str(path),
            "sha256": bundle_mod.sha256_of(path),
            "bytes": path.stat().st_size,
        })
    return entries, missing


def is_complete(record: dict) -> bool:
    """COMPLETE только если захешен каждый обязательный артефакт с непустым sha256."""
    hashed = {e["role"] for e in record.get("files", []) if e.get("sha256")}
    return set(HASHED_ROLES) <= hashed


def build_bundle_record(bundle_dir: pathlib.Path) -> dict:
    base = pathlib.Path(bundle_dir)
    if not base.is_dir():
        raise Refused(f"каталог bundle {base} не найден — собирать нечего")

    entries, missing = enumerate_bundle(base)
    if missing:
        by_name = {r: REQUIRED_ARTIFACTS[r] for r in missing}
        raise Refused("внешний bundle неполон: нет обязательных артефактов "
                      + ", ".join(f"{r} ({by_name[r]})" for r in missing)
                      + ". Прогон без них не подтверждён: missing != partial, это незавершённый запуск.")

    # witness-контракт: журнал без свидетеля неотличим от обрезанного (дублируем
    # явно — даже если кто-то положил JSONL, но «потерял» head-witness вне списка).
    by_role = {e["role"]: e for e in entries}
    if "provenance_jsonl" in by_role and "provenance_head_witness" not in by_role:
        raise Refused(f"provenance-журнал есть, а его witness {WITNESS_NAME} — нет: "
                      "список хешей по одному файлу ничего не доказывает")

    pairs = "\n".join(f"{e['filename']}:{e['sha256']}" for e in sorted(entries, key=lambda e: e["filename"]))
    record = {
        "record_version": COLLECT_RECORD_VERSION,
        "kind": COLLECT_KIND,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "directory": str(base),
        "hashing": "sha256",
        "required_roles": list(HASHED_ROLES),
        "files": sorted(entries, key=lambda e: e["role"]),
        "bundle_digest": "sha256:" + hashlib.sha256(pairs.encode("utf-8")).hexdigest(),
        "self_excluded": REQUIRED_ARTIFACTS["file_hashes_manifest"],
        "complete": False,
        "missing": [],
    }
    record["complete"] = is_complete(record)
    if not record["complete"]:
        # Defensive: перечисление уже отказало на пропуске, но COMPLETE обязан
        # следовать из хешей, а не из «дошли до конца функции».
        hashed = {e["role"] for e in record["files"] if e.get("sha256")}
        record["missing"] = sorted(set(HASHED_ROLES) - hashed)
        raise Refused(f"bundle не может быть помечен COMPLETE: без хеша {record['missing']}")
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-dir", required=True, help="каталог с артефактами внешнего прогона")
    ap.add_argument("--out", default=None,
                    help="куда записать bundle_sha256.json (по умолчанию в --bundle-dir)")
    a = ap.parse_args()

    base = pathlib.Path(a.bundle_dir)
    try:
        record = build_bundle_record(base)
    except Refused as exc:
        print(f"BUNDLE COLLECTION REFUSED: {exc}", file=sys.stderr)
        return REFUSE

    out = pathlib.Path(a.out) if a.out else base / REQUIRED_ARTIFACTS["file_hashes_manifest"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"external bundle → {out} ({len(record['files'])} артефактов захешено, "
          f"complete={record['complete']}, digest {record['bundle_digest'][:19]}…)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
