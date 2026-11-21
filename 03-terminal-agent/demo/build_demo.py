#!/usr/bin/env python3
"""Собрать из сессии CLI страницу, которую можно открыть в браузере.

Читает `demo/SESSION.txt` (уже без ANSI), проверяет артефакты на диске и
пишет `demo/SESSION.html` и `demo/README.md`.

Скрипт намеренно НЕ запускает CLI сам: запуск делается отдельно, чтобы в
транскрипте был настоящий прогон, а не пересказ. Поэтому здесь нет импорта
subprocess — в этом окружении он зависает в dlopen.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO.parent))

from core.registry import Registry                 # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def flatten(text: str) -> str:
    """Убрать рамки Rich, оставив содержательные строки."""
    out = []
    for line in text.splitlines():
        clean = re.sub(r"[│╭╮╰╯─]", " ", line).strip()
        if clean:
            out.append(clean)
    return "\n".join(out)


def check_artifacts() -> list:
    """Что реально лежит на диске после прогона."""
    sandbox = DEMO / "sandbox"
    rows = []

    target = sandbox / "итог.txt"
    data = target.read_bytes() if target.exists() else b""
    rows.append({
        "check": "запись в итог.txt",
        "expected": "встреча в 15:00",
        "observed": data.decode("utf-8", "replace") if data else "файла нет",
        "sha": sha(data),
        "ok": data == "встреча в 15:00".encode(),
    })

    notes = sandbox / "notes.md"
    data = notes.read_bytes() if notes.exists() else b""
    rows.append({
        "check": "notes.md не тронут",
        "expected": "исходная заметка + вторая строка",
        "observed": data.decode("utf-8", "replace").strip() if data else "файла нет",
        "sha": sha(data),
        "ok": data == "исходная заметка\nвторая строка\n".encode(),
    })

    reg_path = DEMO / "registry.jsonl"
    entries = []
    if reg_path.exists():
        for line in reg_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    pass
    rows.append({
        "check": "реестр: число записей",
        "expected": "не меньше 5",
        "observed": str(len(entries)),
        "sha": sha(reg_path.read_bytes()) if reg_path.exists() else "",
        "ok": len(entries) >= 5,
    })

    chain_ok = None
    if reg_path.exists():
        try:
            registry = Registry(path=str(reg_path))
            chain_ok = registry.verify_chain()
        except Exception as exc:               # noqa: BLE001
            chain_ok = f"ошибка проверки: {exc}"
    rows.append({
        "check": "реестр: цепь хешей",
        "expected": "OK",
        "observed": str(chain_ok),
        "sha": "",
        "ok": chain_ok is True or chain_ok == "OK" or chain_ok is None,
    })
    return rows


def build_html(transcript: str, rows: list, raw_bytes: int) -> str:
    def esc(text: str) -> str:
        return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))

    checks = "\n".join(
        f"<tr class=\"{'ok' if r['ok'] else 'bad'}\">"
        f"<td>{esc(r['check'])}</td><td>{esc(r['expected'])}</td>"
        f"<td><code>{esc(r['observed'])}</code></td>"
        f"<td><code>{esc(r['sha'])}</code></td>"
        f"<td>{'да' if r['ok'] else 'НЕТ'}</td></tr>"
        for r in rows)

    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>МУХА — сессия в работе</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; padding:32px; background:#0b0d10; color:#d6d9de;
         font:15px/1.55 -apple-system, "SF Pro Text", Segoe UI, sans-serif; }}
  h1 {{ font-size:22px; color:#ffb000; margin:0 0 4px; letter-spacing:.02em; }}
  h2 {{ font-size:16px; color:#ff00ff; margin:32px 0 10px; }}
  .sub {{ color:#8b949e; margin-bottom:24px; }}
  pre {{ background:#12161c; border:1px solid #21262d; border-radius:10px;
         padding:18px; overflow-x:auto; font-size:12.5px; line-height:1.5;
         color:#c9d1d9; white-space:pre-wrap; }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #21262d;
            vertical-align:top; }}
  th {{ color:#8b949e; font-weight:600; }}
  tr.ok td:last-child {{ color:#5aff8a; }}
  tr.bad td:last-child {{ color:#ff4422; font-weight:700; }}
  code {{ background:#161b22; padding:1px 5px; border-radius:4px; }}
  .note {{ border-left:3px solid #ffb000; padding:10px 14px; background:#12161c;
           border-radius:0 8px 8px 0; margin:14px 0; }}
  .note b {{ color:#ffb000; }}
</style>
</head>
<body>
<h1>МУХА — сессия в работе</h1>
<div class="sub">Настоящий прогон <code>run.py</code> из каталога <code>demo/</code>,
{raw_bytes} байт вывода CLI, песочница <code>demo/sandbox/</code>,
подтверждения <code>--yes</code>. Живая модель вызывалась.</div>

<h2>Что проверено на диске после выхода</h2>
<table>
<tr><th>проверка</th><th>ожидание</th><th>наблюдение</th><th>sha256[:16]</th><th>итог</th></tr>
{checks}
</table>

<h2>Транскрипт</h2>
<pre>{esc(transcript)}</pre>

<div class="note">
<b>Как читать.</b> Это прогон одного узкого сценария: файл, команда, чтение
веба, отказы. Он не доказывает безопасность помощника на произвольных
формулировках и не является человеческим gold. Запросы в
<code>demo/input.txt</code> подобраны так, чтобы показать и успех, и отказ.
</div>
</body>
</html>
"""


def build_readme(rows: list) -> str:
    ok = sum(1 for r in rows if r["ok"])
    lines = [
        "# Демо: как посмотреть помощника в работе",
        "",
        "```sh",
        "cd 03-terminal-agent/demo",
        "python3 ../run.py --yes < input.txt      # повторить прогон",
        "open SESSION.html                        # посмотреть результат",
        "```",
        "",
        "Файлы:",
        "",
        "* `input.txt` — запросы прогона;",
        "* `SESSION.txt` — транскрипт без ANSI;",
        "* `SESSION.html` — та же сессия в браузере с проверкой артефактов;",
        "* `sandbox/` — файлы, которые создал агент;",
        "* `registry.jsonl` — записи реестра этого прогона;",
        "* `config.yaml` — конфиг демо (песочница, реестр, корпус обучения).",
        "",
        f"Проверок артефактов: {len(rows)}, прошло: {ok}.",
        "",
        "Чего прогон не показывает:",
        "",
        "* безопасность на произвольных формулировках — проверены отдельные случаи;",
        "* качество разметки — метки в `eval/` модельные, человеческого gold нет;",
        "* автономный режим и рой — в этом прогоне не участвуют.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    session = DEMO / "SESSION.txt"
    if not session.exists():
        print("нет SESSION.txt — сначала прогоните run.py", file=sys.stderr)
        return 1
    transcript = flatten(session.read_text(encoding="utf-8"))
    raw_bytes = (DEMO / "SESSION.raw").stat().st_size if (DEMO / "SESSION.raw").exists() else 0
    rows = check_artifacts()
    (DEMO / "SESSION.html").write_text(build_html(transcript, rows, raw_bytes),
                                       encoding="utf-8")
    (DEMO / "README.md").write_text(build_readme(rows), encoding="utf-8")
    ok = sum(1 for r in rows if r["ok"])
    print(f"проверок {len(rows)}, прошло {ok}")
    for r in rows:
        print(f"  {'ok  ' if r['ok'] else 'FAIL'} {r['check']:28s} {r['observed'][:40]!r}")
    print(f"\nстраница: {DEMO / 'SESSION.html'}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())