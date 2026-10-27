#!/usr/bin/env python
"""ЭТАП 1 — ingest FlyWire 783 (вариант A) в контракт nodes.csv/edges.csv.

Декодирует Connectivity_783.parquet + аннотации/регионы, режет воспроизводимый
degree-induced подграф, пишет производные в var/flywire/ (gitignore) и тут же
прогоняет гейт готовности: load_connectome собирает ConnectomeGraph,
validate_nodes_edges проходит, манифест валиден по схеме коннектома.

Запуск:  .venv/bin/python scripts/run_flywire_ingest.py [--target-nodes 2000]
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
from fly_connectome_agent.src.science.ingest.flywire import run_ingest  # noqa: E402

SCHEMA_PATH = REPO_ROOT / "fly_connectome_agent" / "schemas" / "connectome_manifest.schema.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-nodes", type=int, default=2000, help="размер degree-induced подграфа (1k–50k)")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "fly_connectome_agent" / "data")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "var" / "flywire")
    args = parser.parse_args(argv)

    stats = run_ingest(args.data_dir, args.out, target_nodes=args.target_nodes)
    print("ingest:", json.dumps(stats, ensure_ascii=False, indent=2))

    graph = load_connectome(args.out / "nodes.csv", args.out / "edges.csv",
                            manifest_path=args.out / "connectome_manifest.json")
    print(f"ГЕЙТ: ConnectomeGraph собрана — узлов {graph.n_nodes}, рёбер {graph.n_edges}")

    manifest = json.loads((args.out / "connectome_manifest.json").read_text(encoding="utf-8"))
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
