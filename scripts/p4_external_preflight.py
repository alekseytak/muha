#!/usr/bin/env python
"""P4.2d — preflight для одного внешнего (без IDE-timeout) full-прогона P4.2.

Смысл: убедиться, что среда годится для ЕДИНСТВЕННОГО подтверждённого прогона,
ДО того как будут сожжены seed'ы. Preflight ничего не считает и ничего не пишет
в выходной bundle — он только отказывает, если что-то не так. Отказ здесь дешёв;
та же проблема, обнаруженная после 60 seed'ов, стоит закрытого протокола.

Проверяется (каждый пункт обязан уметь покраснеть):
  exact git SHA   — HEAD обязан равняться ожидающему закреплённому SHA;
  clean tree      — `git status --porcelain` пуст: прогон по модифицированному коду
                    не воспроизводится и его fingerprint врёт;
  pinned deps     — внешний lock (`--requirements …==…`): без закрепления среда
                    «та же» только по словам;
  tests           — pytest / check.sh зелёные до запуска;
  frozen digest   — digest манифеста обязан совпасть с якорем в коде;
  fingerprint     — полный набор обязательных полей окружения;
  empty outputs   — объявленные пути bundle свободны, каталог без leftover-файлов;
  fresh journal   — provenance JSONL и его head-witness отсутствуют.

Коды возврата как в остальном P4.2: 2 = preflight не пройден (запуск запрещён),
0 = все проверки зелёные. Preflight не запускает науку: он читает состояние.

Ядро `evaluate_preflight(inputs)` — чистая функция над словарём снятых фактов,
поэтому тесты проверяют каждый отказ без зависимости от реальной машины и реального
git: собирают pass-базу и портят ровно одно поле.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import p4_2_protocol as proto  # noqa: E402
import p4_environment_fingerprint as fpmod  # noqa: E402
import collect_p4_2_bundle as bundle_mod  # noqa: E402

REFUSE = 2

# Ровно те проверки, что обязан пройти внешний прогон. Порядок — цена отказа:
# дешёвые статические проверки раньше, дорогие (тесты) позже.
CHECK_NAMES = (
    "git_sha",
    "clean_tree",
    "requirements",
    "tests",
    "frozen_digest",
    "fingerprint",
    "empty_outputs",
    "fresh_journal",
)


class PreflightError(Exception):
    """Preflight не смог вообще сняться (нет манифеста, нет git) — это не «провал»."""


def _req_name(raw: str) -> str:
    """Имя пакета из строки requirements: снять extras `ray[default]` и комментарий."""
    raw = raw.split("#", 1)[0].strip()
    return re.split(r"[\[\=<>!~]", raw, 1)[0].strip()


def check_requirements(text: str) -> list[str]:
    """Вернуть список проблем закрепления. Пустой список = lock валиден.

    Требование жёсткое: каждая значимая строка обязана быть `name==version`.
    Диапазоны (`>=`, `~=`, `<`) не закрепляют среду — две установки одного
    «requirements.txt» дадут разные числа, а воспроизводимость внешнего прогона
    ровно в этом и состоит.
    """
    problems: list[str] = []
    seen = 0
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if not s or s.startswith("-"):          # пустые строки, флаги (-r, -e, --index-url)
            continue
        seen += 1
        if "==" not in s:
            problems.append(f"{s!r}: не закреплено (нужен ==), внешний прогон идёт по lock-файлу")
            continue
        name, _, pinned = s.partition("==")
        name = _req_name(name)
        pinned = pinned.strip()
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{name}: не установлен (заявлено {pinned})")
            continue
        if installed != pinned:
            problems.append(f"{name}: установлено {installed}, в lock {pinned}")
    if seen == 0:
        problems.append("lock-файл пуст: нечего сверять — preflight отказывает вместо «ок»")
    return problems


def evaluate_preflight(inputs: dict) -> tuple[bool, list[str]]:
    """Чистая сверка снятых фактов. Возвращает (ok, список отказов).

    Каждый отказ — строка с именем проверки в начале, чтобы и человек, и тест
    видели, какой именно гейт непустил прогон.
    """
    failures: list[str] = []

    if inputs.get("git_sha_current") != inputs.get("git_sha_expected"):
        failures.append(f"git_sha: HEAD {inputs.get('git_sha_current')!r} != ожидаемый "
                        f"{inputs.get('git_sha_expected')!r} — прогон обязан идти на точном коммите")

    porcelain = inputs.get("working_tree_porcelain") or []
    if porcelain:
        failures.append(f"clean_tree: рабочее дерево не чистое ({len(porcelain)} изменений: "
                        f"{porcelain[:3]}); fingerprint по модифицированному коду не подтверждает запуск")

    req = inputs.get("requirements_problems") or []
    for p in req:
        failures.append(f"requirements: {p}")

    if not inputs.get("tests_passed"):
        failures.append("tests: pytest/check.sh не зелёные — нечего запускать поверх красного дерева")

    if inputs.get("manifest_digest") != inputs.get("frozen_protocol_digest"):
        failures.append(f"frozen_digest: digest манифеста {inputs.get('manifest_digest')} не "
                        f"совпал с замороженным {inputs.get('frozen_protocol_digest')}")

    gaps = fpmod.missing_fields(inputs.get("fingerprint") or {})
    if gaps:
        failures.append(f"fingerprint: нет обязательных полей {gaps}")

    occupied = inputs.get("occupied_paths") or []
    stray = inputs.get("stray_files") or []
    if occupied or stray:
        failures.append(f"empty_outputs: выход не пуст — занято {occupied[:3]}, "
                        f"лишние файлы {stray[:3]}; bundle обязан начинаться с чистого каталога")

    if inputs.get("provenance_jsonl_exists") or inputs.get("provenance_head_exists"):
        failures.append("fresh_journal: provenance JSONL/-head-witness уже существует — это не "
                        "свежий прогон; продолжение журнала и resume по тем же seed'ам запрещены")

    return (not failures), failures


# ── сбор реальных фактов для CLI (не нужен тестам, которые зовут evaluate напрямую) ──

def _git_porcelain() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                         capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def _git_head() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def gather_inputs(manifest_path: pathlib.Path, expected_git_sha: str,
                  requirements_path: pathlib.Path | None,
                  execution_mode: str, run_tests: bool) -> dict:
    manifest = proto.load(manifest_path)
    digest = proto.digest_of_file(manifest_path)
    # Замороженный якорь по protocol_id: require_frozen бросает ProtocolError на расхождении,
    # но preflight обязан ПОКАЗАТЬ расхождение как отказ, а не упасть — сравниваем мягко.
    frozen_digest = proto.frozen_entry(manifest["protocol_id"])["digest"]

    fingerprint = fpmod.collect_fingerprint(manifest, digest, execution_mode=execution_mode)

    empty = bundle_mod.check_empty(manifest)
    declared = bundle_mod.declared_paths(manifest)
    from fly_connectome_agent.src.engineering.logging.provenance_log import HEAD_SUFFIX
    prov_jsonl = declared.get("provenance_jsonl")
    prov_head = declared.get("provenance_head_witness")

    req_problems = None
    if requirements_path is not None:
        req_problems = check_requirements(pathlib.Path(requirements_path).read_text(encoding="utf-8"))
    else:
        req_problems = ["requirements не переданы (--requirements lock==ver): внешний прогон "
                        "обязан идти по закреплённому lock-файлу"]

    tests_passed = True
    if run_tests:
        rc = subprocess.run(["./scripts/check.sh"], cwd=str(REPO)).returncode
        tests_passed = rc == 0

    return {
        "git_sha_current": _git_head(),
        "git_sha_expected": expected_git_sha.strip(),
        "working_tree_porcelain": _git_porcelain(),
        "requirements_problems": req_problems,
        "tests_passed": tests_passed,
        "manifest_digest": digest,
        "frozen_protocol_digest": frozen_digest,
        "protocol_id": manifest["protocol_id"],
        "fingerprint": fingerprint,
        "occupied_paths": [o["role"] for o in empty["occupied"]],
        "stray_files": empty["stray_files"],
        "provenance_jsonl_exists": bool(prov_jsonl and prov_jsonl.exists()),
        "provenance_head_exists": bool(prov_head and prov_head.exists()),
        "_head_suffix": HEAD_SUFFIX,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(proto.DEFAULT_MANIFEST))
    ap.add_argument("--expected-git-sha", required=True,
                    help="точный коммит, на котором разрешён внешний прогон")
    ap.add_argument("--requirements", default=None,
                    help="путь к закреплённому lock-файлу (name==version на строку)")
    ap.add_argument("--execution-mode", choices=fpmod.EXECUTION_MODES, default="external-full-run")
    ap.add_argument("--run-tests", action="store_true",
                    help="запустить ./scripts/check.sh как часть preflight (по умолчанию — пропустить)")
    a = ap.parse_args()

    try:
        inputs = gather_inputs(
            pathlib.Path(a.manifest), a.expected_git_sha,
            pathlib.Path(a.requirements) if a.requirements else None,
            a.execution_mode, a.run_tests,
        )
    except proto.ProtocolError as exc:
        print(f"PREFLIGHT ERROR: манифест не проходит замороженный протокол: {exc}", file=sys.stderr)
        return REFUSE

    ok, failures = evaluate_preflight(inputs)
    print(f"preflight: protocol {inputs['protocol_id']}, execution_mode {a.execution_mode}")
    if ok:
        print("  все проверки зелёные — внешний прогон в этой среде допустим")
        print("  note: preflight ничего не запускал и не писал bundle; запуск — отдельный акт.")
        return 0
    for f in failures:
        print("  ОТКАЗ  " + f, file=sys.stderr)
    print(f"PREFLIGHT REFUSED: {len(failures)} причин — прогон запрещён до устранения.", file=sys.stderr)
    return REFUSE


if __name__ == "__main__":
    raise SystemExit(main())
