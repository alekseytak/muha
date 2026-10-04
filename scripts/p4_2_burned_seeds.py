#!/usr/bin/env python3
"""Сжигатель seed'ов: превращает «эти seed'ы уже тронуты» в записанный в git факт.

Зачем это отдельным скриптом: требование «v2 обязан быть disjoint от всех seed'ов,
которые есть в прерванном v1-журнале» проверяется один раз на одной машине. Если
результат остаётся в памяти разговора (или в gitignored `var/`), на новой машине
его нельзя ни перепроверить, ни опровергнуть — он превращается в обещание. Здесь
вывод пишется JSON-файлом в `fly_connectome_agent/manifests/`, и манифест v2
ссылается на него как на evidence: расхождение ловится кросс-проверкой протокола.

Что считается сожжённым: seed'ы, у которых в журнале есть хотя бы одно событие
эпизода (`kind=p4_episode`). Диапазон из заголовочной записи `p4_2_protocol` —
это *объявление*, а не факт прогона: оно показывает, куда собирались дойти, и
пишется в вывод отдельно, чтобы их нельзя было перепутать.

Запуск:
    .venv/bin/python scripts/p4_2_burned_seeds.py \
        --provenance var/aborted/p4_2_v1_infrastructure_abort/partial_provenance_final.jsonl \
        --out fly_connectome_agent/manifests/p4_2_v1_burned_seeds.json
Код возврата: 0 — запись собрана; 2 — журнал не читается целиком (обрыванный хвост,
битый JSON), то есть утверждать несечение по нему нельзя.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
REFUSE = 2

EPISODE_KINDS = ("p4_episode",)
HEADER_KIND = "p4_2_protocol"


class Refused(Exception):
    """Журнал непригоден как доказательство: вывод про него нельзя записать."""


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def scan_provenance(path: pathlib.Path) -> dict:
    """Один проход по JSONL: seed'ы эпизодов, заголовок прогона, целостность строк.

    Обрывок в хвосте — не повод падать молча: он зачитывается отдельно, и тогда
    запись помечается `evidence_complete=false`, потому что потерянная строка могла
    содержать seed, которого больше нигде нет.
    """
    if not path.exists():
        raise Refused(f"журнал не найден: {path}")

    episodes: collections.Counter[int] = collections.Counter()
    kinds: collections.Counter[str] = collections.Counter()
    headers: list[dict] = []
    lines_total = 0
    broken: list[int] = []

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            lines_total += 1
            try:
                payload = json.loads(line)["payload"]
            except (json.JSONDecodeError, KeyError, TypeError):
                broken.append(lineno)
                continue
            kind = payload.get("kind", "?")
            kinds[kind] += 1
            if kind in EPISODE_KINDS:
                seed = payload.get("seed")
                if not isinstance(seed, int):
                    raise Refused(f"{path.name}: строка {lineno}: эпизод без целого seed")
                episodes[seed] += 1
            elif kind == HEADER_KIND:
                headers.append({
                    "protocol_id": payload.get("protocol_id"),
                    "manifest_digest": payload.get("manifest_digest"),
                    "declared_seeds": payload.get("seeds"),
                    "n_seeds": payload.get("n_seeds"),
                    "non_confirmatory": payload.get("non_confirmatory"),
                })

    if not episodes:
        raise Refused(f"{path.name}: ни одного события эпизода — журнал ничего не утверждает "
                      "о тронутых seed'ах")

    seeds = sorted(episodes)
    counts = [episodes[s] for s in seeds]
    resolved = path.resolve()
    try:
        shown_path = str(resolved.relative_to(REPO))
    except ValueError:      # журнал вне репозитория — пишем абсолютный путь, не падая
        shown_path = str(resolved)
    return {
        "source": {
            "path": shown_path,
            "sha256": sha256_of(path),
            "bytes": path.stat().st_size,
            "lines_total": lines_total,
            "lines_unparseable": len(broken),
            "unparseable_line_numbers": broken[:20],
            "event_kinds": dict(kinds),
        },
        "seeds_touched": {
            "count": len(seeds),
            "range": [seeds[0], seeds[-1]],
            "contiguous": seeds == list(range(seeds[0], seeds[-1] + 1)),
            "values": seeds,
            "episodes_per_seed_min": min(counts),
            "episodes_per_seed_max": max(counts),
            "episode_events_total": sum(counts),
        },
        "run_headers": headers,
    }


def build_record(scans: list[dict], *, protocol_id: str, status: str, note: str) -> dict:
    """Объединение нескольких журналов в одно множество сожжённых seed'ов."""
    all_seeds = sorted({s for scan in scans for s in scan["seeds_touched"]["values"]})
    if not all_seeds:
        raise Refused("объединённое множество пусто — нечего записывать как сожжённое")
    appearances: collections.Counter[int] = collections.Counter()
    for scan in scans:
        appearances.update(set(scan["seeds_touched"]["values"]))
    partial = [s for s, n in appearances.items() if n < len(scans)]
    return {
        "record_version": "1.0.0",
        "kind": "burned_seeds",
        "protocol_id": protocol_id,
        "status": status,
        "note": note,
        "burned_seed_set": {
            "count": len(all_seeds),
            "range": [all_seeds[0], all_seeds[-1]],
            "contiguous": all_seeds == list(range(all_seeds[0], all_seeds[-1] + 1)),
            "values": all_seeds,
        },
        "coverage": {
            "journals_scanned": len(scans),
            "seeds_present_in_all_journals": len(all_seeds) - len(partial),
            "seeds_not_in_every_journal": sorted(partial)[:20],
            "evidence_complete": all(s["source"]["lines_unparseable"] == 0 for s in scans),
        },
        "journals": scans,
        "what_this_does_not_prove": (
            "то, что эти seed'ы дали полный прогон: они тронуты частично. Здесь "
            "зафиксировано только несечение — какое множество нельзя использовать заново "
            "как confirmatory."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provenance", nargs="+", required=True,
                    help="provenance JSONL (можно несколько журналов одного прогона)")
    ap.add_argument("--protocol-id", required=True, help="какому протоколу принадлежит журнал")
    ap.add_argument("--status", default="aborted",
                    help="статус прогона, который сжёг seed'ы")
    ap.add_argument("--note", default="", help="почему множество нельзя повторять")
    ap.add_argument("--out", help="куда положить запись; без него — только печать в stdout")
    a = ap.parse_args()

    try:
        scans = [scan_provenance(pathlib.Path(p)) for p in a.provenance]
        record = build_record(scans, protocol_id=a.protocol_id, status=a.status,
                              note=a.note or "эти seed'ы тронуты прогоном; как confirmatory "
                                            "повторять нельзя")
    except Refused as exc:
        print(f"ЗАПИСЬ НЕ СОБРАНА: {exc}", file=sys.stderr)
        return REFUSE

    text = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if a.out:
        out = pathlib.Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"запись о сожжённых seed'ах → {out}")
    burned = record["burned_seed_set"]
    print(f"  protocol {record['protocol_id']} ({record['status']}): "
          f"{burned['count']} seed'ов, {burned['range'][0]}..{burned['range'][1]}, "
          f"непрерывно={burned['contiguous']}")
    for scan in record["journals"]:
        src = scan["source"]
        print(f"  журнал {src['path']}: строк {src['lines_total']}, "
              f"битых строк {src['lines_unparseable']}, "
              f"эпизодов {scan['seeds_touched']['episode_events_total']}, "
              f"sha256 {src['sha256'][:23]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
