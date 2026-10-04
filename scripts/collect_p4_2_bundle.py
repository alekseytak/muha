#!/usr/bin/env python
"""P4.2.v2 — сборка и проверка result_bundle.

Зачем отдельный скрипт, если runner и gate уже пишут свои файлы: bundle — это не
папка с артефактами, а договорённость «что именно лежит внутри и чем хешится».
Манифест объявляет девять ролей, runner пишет три из них, gate — три, две
(environment fingerprint и список хешей) не принадлежат ни одному прогоновому
скрипту. Если собрать их по месту, девять файлов окажутся девятью разными
представлениями об одном прогоне, и расхождение между ними никто не заметит.

Что здесь проверяется:
  check-empty — ни один объявленный путь не занят. Это то, ради чего v1 нельзя
                «допровадить»: свежий provenance-путь проверяется до запуска, а не
                после того, как к цепи добавят чужие события;
  collect     — environment.json (машина, на которой считали) и bundle_hashes.json
                (sha256 каждого артефакта + bundle_digest по паре имя:хеш);
  verify      — пересобрать список хешей и сравнить с записанным: артефакт,
                изменённый после вердикта, ловится здесь, а не в пересказе README.

Коды возврата, как в остальном P4.2: 2 = проверка не состоялась (манифест не
проходит замороженный протокол, объявленный артефакт отсутствует, witness потерян),
1 = состоялась и не прошла (путь занят, хеш не совпал), 0 = чисто.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import socket
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import p4_2_protocol as proto  # noqa: E402

REFUSE = 2
DIRTY = "не известно"
BUNDLE_RECORD_VERSION = "1.0.0"


class Refused(Exception):
    """Bundle не собирается: заявленного артефакта нет или контракт не выполнен."""


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def declared_paths(manifest: dict) -> dict[str, pathlib.Path]:
    """role -> абсолютный путь, как объявлено в замороженном манифесте."""
    bundle = manifest.get("result_bundle")
    if not bundle:
        raise Refused(
            f"в {manifest['protocol_id']} нет result_bundle: собирать нечего. "
            "Bundle-контракт появился в v2, для более ранних протоколов его не было.")
    base = REPO / bundle["directory"]
    return {a["role"]: base / a["filename"] for a in bundle["artifacts"]}


def by_role(manifest: dict) -> dict[str, dict]:
    bundle = manifest.get("result_bundle") or {}
    return {a["role"]: a for a in bundle.get("artifacts", [])}


def bundle_directory(manifest: dict) -> pathlib.Path:
    return (REPO / manifest["result_bundle"]["directory"]).resolve()


def git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001 — нет git не отменяет fingerprint машины
            return ""

    full = run("rev-parse", "HEAD")
    return {"rev": full, "rev_short": run("rev-parse", "--short", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": "yes" if run("status", "--porcelain") else "no",
            "dirty_paths": run("status", "--porcelain").splitlines()}


def _mem_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGESIZE")
        return int(pages) * int(size)
    except (ValueError, KeyError, OSError):
        return None


def environment_fingerprint(manifest: dict, digest: str) -> dict:
    """На чём и из чего считали. Без этого вердикт неотпроизводим даже в теории.

    Список версий ограничен тем, что реально участвует в вычислении: python,
    numpy, scipy (опционально — у sign-теста есть точный путь без неё). Железо
    указывается справочно: результаты детерминированы по seed'ам, но время прогона
    и shared-memory поведение от машины зависят.
    """
    import numpy as np

    try:
        import scipy
        scipy_version = getattr(scipy, "__version__", "present, version unreadable")
    except ImportError:
        scipy_version = None
    mem = _mem_bytes()
    return {
        "record_version": BUNDLE_RECORD_VERSION,
        "kind": "p4_2_result_environment",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_id": manifest["protocol_id"],
        "manifest_digest": digest,
        "frozen_protocol_digest": proto.FROZEN_PROTOCOL_DIGEST,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "libraries": {"numpy": np.__version__, "scipy": scipy_version},
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "macos": platform.mac_ver()[0] or None,
        },
        "hardware": {
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "physical_memory_bytes": mem,
            "hostname": socket.gethostname(),
        },
        "git": git_state(),
        "what_this_does_not_prove": (
            "что прогон воспроизводим по этому описанию: здесь зафиксировано состояние "
            "машины на момент сборки bundle, а не гарантия идентичности чисел на другой."),
    }


def check_empty(manifest: dict) -> dict:
    """Каждая объявленная дорожка свободна, и каталог bundle ничего не содержит.

    Проверка идёт по всему каталогу, а не только по девяти именам: leftover-файл
    с прошлой попытки (shard, .tmp писателя, чужой CSV) не попадёт ни в какой
    объявленный список, но попадёт в будущий bundle_hashes.json при --collect.
    """
    base = bundle_directory(manifest)
    paths = declared_paths(manifest)
    occupied = []
    for role, path in sorted(paths.items(), key=lambda kv: kv[0]):
        if path.exists():
            occupied.append({"role": role, "path": str(path.relative_to(REPO)),
                             "bytes": path.stat().st_size})
    stray = []
    if base.exists():
        declared_set = {p.resolve() for p in paths.values()}
        for entry in sorted(base.rglob("*")):
            if entry.is_file() and entry.resolve() not in declared_set:
                stray.append(str(entry.relative_to(REPO)))
    return {"directory": str(base.relative_to(REPO)), "directory_exists": base.exists(),
            "declared": len(paths), "occupied": occupied, "stray_files": stray,
            "clean": not occupied and not stray}


def hashes_manifest(manifest: dict, digest: str) -> dict:
    """Список хешей объявленных артефактов. Сам себя не хешит — и говорит об этом.

    bundle_hashes.json физически не может содержать собственный sha256 (он меняется
    при записи), поэтому роль file_hashes_manifest идёт отдельным полем
    self_excluded, а вместо «хеша файла» здесь bundle_digest: sha256 по
    отсортированным парам «имя:хеш». Это то, что можно сравнить с копией списка,
    не попадая в рекурсию.
    """
    paths = declared_paths(manifest)
    meta = by_role(manifest)
    self_name = manifest["result_bundle"]["hashes_manifest"]
    entries, missing = [], []
    for role, path in sorted(paths.items()):
        if path.name == self_name:
            continue                      # сам себя не хешит, см. self_excluded
        if not path.exists():
            missing.append({"role": role, "path": str(path.relative_to(REPO))})
            continue
        entries.append({"role": role, "path": str(path.relative_to(REPO)),
                        "filename": path.name, "sha256": sha256_of(path),
                        "bytes": path.stat().st_size, "producer": meta[role]["producer"]})
    if missing:
        raise Refused(
            "в bundle нет обязательных артефактов: "
            + ", ".join(f"{m['role']} ({m['path']})" for m in missing)
            + ". required=true в манифесте нельзя снять молча: отсутствие файла — это "
              "не часть результата, а незавершённый прогон.")
    # witness-контракт писателя: журнал без sidecar-свидетеля неотличим от обрезанного.
    prov = next((e for e in entries if e["role"] == "provenance_jsonl"), None)
    witness = next((e for e in entries if e["role"] == "provenance_head_witness"), None)
    if prov and not witness:
        raise Refused(f"provenance-журнал {prov['path']} объявлен, а его witness "
                      f"{prov['filename']}.head.json в bundle не попал: список хешей "
                      "по одному файлу ничего не доказывает")
    # Пары для bundle_digest упорядочены по имени файла: тот же порядок повторного
    # счёта в verify(). Сортировка по роли выглядела бы невинно, но две сортировки
    # дают два разных bundle_digest над одним набором файлов, и сверка начала бы
    # «расхождение» там, где расхождения нет.
    pairs = "\n".join(f"{e['filename']}:{e['sha256']}"
                      for e in sorted(entries, key=lambda e: e["filename"]))
    return {
        "record_version": BUNDLE_RECORD_VERSION,
        "kind": "p4_2_bundle_hashes",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_id": manifest["protocol_id"],
        "manifest_digest": digest,
        "directory": manifest["result_bundle"]["directory"],
        "hashing": "sha256",
        "files": entries,
        "bundle_digest": "sha256:" + hashlib.sha256(pairs.encode("utf-8")).hexdigest(),
        "self_excluded": self_name,
        "why_self_excluded": ("список не может содержать собственный хеш: он меняется в момент "
                             "записи. bundle_digest считается по парам имя:хеш и сравнивается "
                             "между копиями этого файла."),
        "not_covered": ["stdout самого collect_p4_2_bundle.py (он пишет артефакты, а не результат)"],
    }


def verify(manifest: dict, digest: str) -> tuple[bool, list[str]]:
    """Пересобрать хеши с диска и сравнить с записанным bundle_hashes.json.

    Здесь важно, что пересчёт идёт по фактическим файлам, а не по записанным
    значениям: сверка «списка с самим собой» была бы тавтологией и ловила бы только
    правку JSON, но не подмену CSV после вердикта.
    """
    path = bundle_directory(manifest) / manifest["result_bundle"]["hashes_manifest"]
    if not path.exists():
        raise Refused(f"{path} нет на диске: верифицировать нечего — сначала --collect")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if recorded.get("manifest_digest") != digest:
        problems.append(f"записанный manifest_digest {recorded.get('manifest_digest')} против "
                        f"текущего {digest}")
    if recorded.get("protocol_id") != manifest["protocol_id"]:
        problems.append(f"в списке протокол {recorded.get('protocol_id')}, а манифест "
                        f"{manifest['protocol_id']}")
    declared_names = {p.name for p in declared_paths(manifest).values()
                      if p.name != manifest["result_bundle"]["hashes_manifest"]}
    listed = {e["filename"] for e in recorded.get("files", [])}
    for name in sorted(declared_names - listed):
        problems.append(f"{name} объявлен в манифесте, но вычеркнут из списка хешей")
    for name in sorted(listed - declared_names):
        problems.append(f"{name} есть в списке хешей, но не объявлен в манифесте")
    recomputed: dict[str, str] = {}
    for entry in recorded.get("files", []):
        fpath = REPO / entry["path"]
        if not fpath.exists():
            problems.append(f"{entry['path']} объявлен в списке, но удалён с диска")
            continue
        now = sha256_of(fpath)
        recomputed[fpath.name] = now
        if now != entry["sha256"]:
            problems.append(f"{entry['path']} изменился после сборки bundle: было "
                            f"{entry['sha256']}, стало {now}")
        if int(entry.get("bytes", -1)) != fpath.stat().st_size:
            problems.append(f"{entry['path']}: размер {fpath.stat().st_size} против записанного "
                            f"{entry.get('bytes')}")
    if listed <= declared_names and len(recomputed) == len(listed):
        pairs = "\n".join(f"{name}:{recomputed[name]}" for name in sorted(recomputed))
        want = "sha256:" + hashlib.sha256(pairs.encode("utf-8")).hexdigest()
        if recorded.get("bundle_digest") != want:
            problems.append(f"bundle_digest {recorded.get('bundle_digest')} не сходится с "
                            f"пересчётом по файлам на диске: {want}")
    return not problems, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(proto.DEFAULT_MANIFEST))
    ap.add_argument("--mode", choices=("check-empty", "collect", "verify"), default="check-empty")
    a = ap.parse_args()

    try:
        manifest = proto.load(pathlib.Path(a.manifest))
    except proto.ProtocolError as exc:
        print(f"BUNDLE НЕ СОБИРАЕТСЯ: манифест не проходит замороженный протокол: {exc}",
              file=sys.stderr)
        return REFUSE
    digest = proto.digest_of_file(pathlib.Path(a.manifest))
    try:
        proto.require_frozen(digest, where="collect", protocol_id=manifest["protocol_id"])
    except proto.ProtocolError as exc:
        print(f"BUNDLE НЕ СОБИРАЕТСЯ: {exc}", file=sys.stderr)
        return REFUSE

    print(proto.describe(manifest))
    try:
        if a.mode == "check-empty":
            report = check_empty(manifest)
            print(f"\nbundle-каталог: {report['directory']} "
                  f"(существует: {report['directory_exists']}), объявлено артефактов: {report['declared']}")
            for item in report["occupied"]:
                print(f"  ЗАНЯТО  {item['role']:24s} {item['path']} ({item['bytes']} байт)")
            for stray in report["stray_files"]:
                print(f"  ЛИШНЕЕ  {stray}")
            if report["clean"]:
                print("  все объявленные пути свободны, каталог пуст — прогон ещё не начинался")
                return 0
            print(f"  чистых путей: {report['declared'] - len(report['occupied'])}/{report['declared']}")
            return 1
        if a.mode == "verify":
            ok, problems = verify(manifest, digest)
            for p in problems:
                print("  РАСХОЖДЕНИЕ  " + p)
            print("bundle_hashes.json сходится с файлами на диске" if ok
                  else "bundle изменён после сборки — хеши не совпадают")
            return 0 if ok else 1
        base = bundle_directory(manifest)
        env_path = declared_paths(manifest)["environment_fingerprint"]
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env = environment_fingerprint(manifest, digest)
        env_path.write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nenvironment → {env_path.relative_to(REPO)} "
              f"(git {env['git']['rev_short'] or DIRTY}, dirty={env['git']['dirty']}, "
              f"numpy {env['libraries']['numpy']})")
        record = hashes_manifest(manifest, digest)
        out = base / manifest["result_bundle"]["hashes_manifest"]
        out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"bundle_hashes → {out.relative_to(REPO)} ({len(record['files'])} файлов, "
              f"digest {record['bundle_digest'][:19]}…)")
        print("  note: пересобрать список можно только после финальной правки артефактов — "
              "сам bundle_hashes.json в него не входит")
        return 0
    except Refused as exc:
        print(f"BUNDLE НЕ СОБИРАЕТСЯ: {exc}", file=sys.stderr)
        return REFUSE


if __name__ == "__main__":
    raise SystemExit(main())
