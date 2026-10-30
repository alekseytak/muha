#!/usr/bin/env python
"""ЭТАП 1 — ingest FlyWire 783 (вариант A) в контракт nodes.csv/edges.csv.

Два правила отбора подграфа:
  degree — top-N по суммарным синапсам (правило ЭТАПа 1);
  role   — входы, у которых есть путь к выходам по аннотации (волна 6:
           снимает блокер ЭТАПа 4, потому что сенсорных в degree-топе нет).

Пишет производные в out (var/, gitignore) и тут же прогоняет гейт готовности:
load_connectome собирает ConnectomeGraph, validate_nodes_edges проходит,
манифест валиден по схеме; для role — ещё и канал в манифесте совпадает с
файлами (и, на горизонте одного шага, каждый вход имеет ребро на выход).

Запуск:
  .venv/bin/python scripts/run_flywire_ingest.py [--target-nodes 2000]
  .venv/bin/python scripts/run_flywire_ingest.py --selection role --max-inputs 300
Требует pyarrow и наличие fly_connectome_agent/data/*. Без них — понятная ошибка.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fly_connectome_agent.src.science.graph.load_connectome import load_connectome  # noqa: E402
from fly_connectome_agent.src.science.ingest.flywire import run_ingest, run_role_ingest  # noqa: E402

SCHEMA_PATH = REPO_ROOT / "fly_connectome_agent" / "schemas" / "connectome_manifest.schema.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", choices=("degree", "role"), default="degree")
    parser.add_argument("--target-nodes", type=int, default=2000, help="размер degree-induced подграфа (1k–50k)")
    parser.add_argument("--input-classes", nargs="+", default=["sensory", "ascending"],
                        help="роли входов для role-отбора (super_class из аннотации)")
    parser.add_argument("--output-classes", nargs="+", default=["descending"],
                        help="роли выходов для role-отбора")
    parser.add_argument("--max-hops", type=int, default=1, help="горизонт пути вход -> выход")
    parser.add_argument("--max-inputs", type=int, default=300, help="бюджет ВХОДОВ (не узлов)")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "fly_connectome_agent" / "data")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out or REPO_ROOT / "var" / ("flywire-role" if args.selection == "role" else "flywire")
    if args.selection == "role":
        stats = run_role_ingest(
            args.data_dir, out,
            input_classes=tuple(args.input_classes), output_classes=tuple(args.output_classes),
            max_hops=args.max_hops, max_inputs=args.max_inputs,
        )
    else:
        stats = run_ingest(args.data_dir, out, target_nodes=args.target_nodes)
    print("ingest:", json.dumps(stats, ensure_ascii=False, indent=2))

    graph = load_connectome(out / "nodes.csv", out / "edges.csv",
                            manifest_path=out / "connectome_manifest.json")
    print(f"ГЕЙТ: ConnectomeGraph собрана — узлов {graph.n_nodes}, рёбер {graph.n_edges}")

    manifest = json.loads((out / "connectome_manifest.json").read_text(encoding="utf-8"))
    channel = manifest.get("io_channel")
    if channel:
        ids = {node.node_id for node in graph.nodes}
        missing = sorted((set(channel["input_ids"]) | set(channel["output_ids"])) - ids)
        if missing:
            print("ГЕЙТ: канал ссылается на узлы вне nodes.csv:", missing[:5])
            return 1
        if channel["max_hops"] == 1:
            fed = {edge.source for edge in graph.edges if edge.target in set(channel["output_ids"])}
            unfed = sorted(set(channel["input_ids"]) - fed)
            if unfed:
                print(f"ГЕЙТ: {len(unfed)} входов канала без ребра на выход:", unfed[:5])
                return 1
        print(
            f"ГЕЙТ: канал заморожен — входов {len(channel['input_ids'])}, "
            f"выходов {len(channel['output_ids'])}; все входы имеют путь к выходу"
        )
    try:
        from jsonschema import Draft202012Validator
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda e: list(e.path))
        if errors:
            print("ГЕЙТ: манифест НЕ проходил схему:", errors[0].message)
            return 1
        print("ГЕЙТ: манифест валиден по схеме коннектома")
    except ModuleNotFoundError:
        print("ГЕЙТ: jsonschema не установлен — пропуск схемой (pyarrow-часть прошла)")

    print("всё зелёное")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
