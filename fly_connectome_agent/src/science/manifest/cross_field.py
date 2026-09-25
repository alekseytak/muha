"""Межполевые правила манифеста эксперимента.

JSON Schema умеет проверять поля по отдельности, но не умеет сравнивать два
числа между собой: правило «quantiles.low < quantiles.high» в ней выразить
нечем. Поэтому структурные правила живут в схеме
(`schemas/experiment_manifest.schema.json`), а межполевые — здесь, и тест
`tests/test_manifest_validation.py` проверяет именно этот путь.
"""
from __future__ import annotations

from typing import Any

__all__ = ["cross_field_errors", "validate_cross_field"]


def cross_field_errors(manifest: dict[str, Any]) -> list[str]:
    """Возвращает список нарушений межполевых правил (пустой список — чисто).

    Правила:
    - `weight_mapping.quantiles.low < quantiles.high` — иначе квантильный
      диапазон вывернут и маппинг весов смысла не имеет;
    - `weight_mapping.w_exc_max > 0` и `w_inh_max > 0` — пределы весов
      обязаны быть положительными.
    """
    problems: list[str] = []

    weight_mapping = manifest.get("weight_mapping") or {}
    quantiles = weight_mapping.get("quantiles") or {}
    low = quantiles.get("low")
    high = quantiles.get("high")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        if not low < high:
            problems.append(
                f"quantiles.low ({low}) must be < quantiles.high ({high})"
            )

    for field in ("w_exc_max", "w_inh_max"):
        value = weight_mapping.get(field)
        if isinstance(value, (int, float)) and not value > 0:
            problems.append(f"weight_mapping.{field} must be > 0, got {value}")

    return problems


def validate_cross_field(manifest: dict[str, Any]) -> None:
    """Бросает ValueError, если межполевые правила нарушены."""
    problems = cross_field_errors(manifest)
    if problems:
        raise ValueError("; ".join(problems))
