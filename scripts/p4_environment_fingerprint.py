#!/usr/bin/env python
"""P4.2d — environment fingerprint для одного внешнего (без IDE-timeout) прогона P4.2.

Зачем отдельный скрипт, если `collect_p4_2_bundle.py` уже пишет `environment.json`:
тот fingerprint снимает машину в момент сборки bundle и содержит сырой `hostname`.
Внешний прогон (терминал/SSH/Colab) требует другого контракта: (а) явно записанное
`execution_mode` — чем именно был запущен прогон, (б) обезличенный хост вместо
сырого имени машины (внешняя среда расшаривается вместе с bundle), (в) полный
набор обязательных полей, который preflight обязан проверить до запуска, а не после
вердикта. Это надстройка над `collect_p4_2_bundle`, а не его замена: формат `environment.json`
в замороженном манифесте мы не правим (он вне scope P4.2d).

Никакой научной работы здесь нет: модуль только считывает состояние окружения.
Никакого прогона, манифеста v3, seed'ов или запуска Colab он не создаёт.

Обязательные поля (preflight сверяет по этому же кортежу):
    git_sha, protocol_id, manifest_digest, python, numpy, scipy,
    os, cpu_arch, host, execution_mode, captured_utc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import socket
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import p4_2_protocol as proto  # noqa: E402

FP_RECORD_VERSION = "1.0.0"
FP_KIND = "p4_2d_external_execution_environment"

# Единственный список того, что делает fingerprint достаточным для воспроизведения
# внешнего прогона. Preflight и тесты сверяются с этим кортежем, а не с текстом.
REQUIRED_FIELDS = (
    "schema_version",
    "kind",
    "git_sha",
    "protocol_id",
    "manifest_digest",
    "python",
    "numpy",
    "scipy",
    "os",
    "cpu_arch",
    "host",
    "execution_mode",
    "captured_utc",
)

# Допустимые режимы исполнения. Внешний прогон обязан назвать, чем он запущен:
# mixing шардов между разными средами запрещён, поэтому execution_mode — часть
# отпечатка, а не свободный комментарий.
EXECUTION_MODES = (
    "external-full-run",       # локальный терминал / SSH, среда без IDE-timeout
    "colab-isolated-run",      # Colab: одна изолированная среда, ровно один прогон
)


def _library_versions() -> dict[str, str | None]:
    import numpy as np

    try:
        import scipy
        scipy_version = getattr(scipy, "__version__", "present, version unreadable")
    except ImportError:
        scipy_version = None
    return {"numpy": np.__version__, "scipy": scipy_version}


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 — отсутствие git не отменяет поле, оно станет пустым
        return ""


def anonymize_host(hostname: str) -> dict[str, str]:
    """Обезличивание хоста: класс машины + необратимый (несолёный) digest имени.

    Сырое `socket.gethostname()` в расшариваемом bundle — это утечка инфраструктуры;
    внешняя среда (Colab) вообще даёт эфемерные имена. Основной признак — machine_class
    (system/arch); `hostname_sha256` остаётся только как стабильный id «тот ли это хост»
    при сравнении двух записей, без восстановления имени прямым чтением.
    """
    digest = "sha256:" + hashlib.sha256(hostname.encode("utf-8")).hexdigest()
    return {
        "machine_class": f"{platform.system()}/{platform.machine()}",
        "hostname_sha256": digest,
    }


def collect_fingerprint(manifest: dict, digest: str, *,
                        execution_mode: str = "external-full-run",
                        git_sha: str | None = None,
                        hostname: str | None = None) -> dict:
    """Собрать отпечаток окружения внешнего прогона.

    Значения по умолчанию берутся из живого окружения; тесты и preflight могут
    подставить явные `git_sha` / `hostname`, чтобы проверить поле `host` и
    сверку SHA без зависимости от того, на какой машине запущен pytest.
    """
    libs = _library_versions()
    host = anonymize_host(socket.gethostname() if hostname is None else hostname)
    fp = {
        "schema_version": FP_RECORD_VERSION,
        "kind": FP_KIND,
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": (git_sha if git_sha is not None else _git_sha()),
        "protocol_id": manifest["protocol_id"],
        "manifest_digest": digest,
        "frozen_protocol_digest": proto.FROZEN_PROTOCOL_DIGEST,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "numpy": libs["numpy"],
        "scipy": libs["scipy"],
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "cpu_arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "host": host,
        "execution_mode": execution_mode,
    }
    return fp


def missing_fields(fp: dict) -> list[str]:
    """Каких обязательных полей нет (или они пустые).

    Пустая строка `git_sha` считается отсутствием: fingerprint с неизвестным
    коммитом не годится для preflight — нечем подтвердить «exact SHA checkout».
    """
    missing = []
    for name in REQUIRED_FIELDS:
        value = fp.get(name)
        if value is None or value == "" or value == {}:
            missing.append(name)
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(proto.DEFAULT_MANIFEST))
    ap.add_argument("--execution-mode", choices=EXECUTION_MODES, default="external-full-run",
                    help="чем запущен внешний прогон; mixing-среды запрещены")
    ap.add_argument("--out", default=None, help="записать fingerprint в файл (по умолчанию stdout)")
    a = ap.parse_args()

    try:
        manifest = proto.load(pathlib.Path(a.manifest))
    except proto.ProtocolError as exc:
        print(f"FINGERPRINT REFUSED: манифест не проходит замороженный протокол: {exc}",
              file=sys.stderr)
        return 2
    digest = proto.digest_of_file(pathlib.Path(a.manifest))

    fp = collect_fingerprint(manifest, digest, execution_mode=a.execution_mode)
    gaps = missing_fields(fp)
    if gaps:
        print(f"FINGERPRINT REFUSED: нет обязательных полей {gaps}", file=sys.stderr)
        return 2

    payload = json.dumps(fp, ensure_ascii=False, indent=2) + "\n"
    if a.out:
        out = pathlib.Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"environment fingerprint → {out} "
              f"(git {str(fp['git_sha'])[:12] or '…'}, mode={fp['execution_mode']}, "
              f"host={fp['host']['machine_class']})", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
