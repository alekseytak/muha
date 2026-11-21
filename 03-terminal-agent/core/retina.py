"""Локальное зрение: картинка превращается в вектор признаков, а не в текст.

Зачем это вместо модели. `deepseek-v4-flash-vision-exp` работает и стоит
копейки, но он **пересказывает** картинку словами. Мухе нужен не пересказ, а
восприятие: числа, по которым можно решать. Здесь картинка превращается в
короткий вектор — контраст, края, раскладка, доля «чернил», полосы текста —
и ни одного сетевого вызова.

Что это даёт практически: муха может отличить скриншот терминала от списка
файлов, плотный экран от пустого, ровные строки от свободной вёрстки. Она не
«понимает» картинку — она её измеряет, и это честнее, чем называть пересказ
большой модели зрением.

Ограничение, которое надо помнить: вектор признаков не различает смысл.
«Ошибка» и «обычный вывод» с одинаковой плотностью текста дадут близкие
векторы. Поэтому тип наблюдения (ошибка/команда/список) по-прежнему
определяется по словам, если они есть, — но уже по распознанному локально.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:                                    # Pillow и numpy стоят, но модуль
    from PIL import Image               # должен импортироваться и без них.
    import numpy as np
    HAVE_IMAGING = True
except Exception:                       # noqa: BLE001
    HAVE_IMAGING = False


FEATURES = (
    "width", "height", "aspect",
    "mean", "contrast", "dark_share", "bright_share",
    "edges", "ink", "row_bands", "col_bands",
    "monochrome",
)


@dataclass
class Retina:
    """Вектор восприятия: числа, а не слова."""

    values: Dict[str, float] = field(default_factory=dict)
    image: str = ""
    source: str = "локально"

    def as_list(self) -> List[float]:
        return [round(self.values.get(name, 0.0), 4) for name in FEATURES]

    def summary(self) -> str:
        v = self.values
        return (f"{int(v.get('width', 0))}x{int(v.get('height', 0))}, "
                f"контраст {v.get('contrast', 0):.2f}, "
                f"чернил {v.get('ink', 0):.2f}, "
                f"краёв {v.get('edges', 0):.3f}, "
                f"строк {int(v.get('row_bands', 0))}, "
                f"монохром {v.get('monochrome', 0):.0f}")


def look(path: str, max_side: int = 320) -> Retina:
    """Измерить картинку локально. Сети нет, денег нет."""
    if not HAVE_IMAGING:
        raise RuntimeError("нужны Pillow и numpy для локального зрения")
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"картинка не найдена: {path}")

    with Image.open(image_path) as raw:
        image = raw.convert("L")
        width, height = image.size
        if max(width, height) > max_side:
            scale = max_side / max(width, height)
            image = image.resize((max(1, int(width * scale)),
                                  max(1, int(height * scale))))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        colour = raw.convert("RGB")
        colour_pixels = np.asarray(colour.resize(
            (max(1, image.width // 4), max(1, image.height // 4))),
            dtype=np.float32)

    mean = float(pixels.mean())
    contrast = float(pixels.std())
    dark = float((pixels < 0.25).mean())
    bright = float((pixels > 0.75).mean())

    # Края: разница между соседними пикселями. Дёшево и без свёрток.
    vertical = np.abs(np.diff(pixels, axis=1)).mean() if pixels.shape[1] > 1 else 0.0
    horizontal = np.abs(np.diff(pixels, axis=0)).mean() if pixels.shape[0] > 1 else 0.0
    edges = float((vertical + horizontal) / 2)

    # «Чернила»: доля пикселей, отличающихся от фона заметно.
    background = float(np.median(pixels))
    ink = float((np.abs(pixels - background) > 0.25).mean())

    # Полосы: строки и столбцы с ненулевой «чернильной» нагрузкой.
    row_load = (np.abs(pixels - background) > 0.25).mean(axis=1)
    col_load = (np.abs(pixels - background) > 0.25).mean(axis=0)
    row_bands = _count_bands(row_load, 0.02)
    col_bands = _count_bands(col_load, 0.02)

    # Монохром: у серого скриншота каналы почти совпадают.
    if colour_pixels.size:
        spread = float(np.abs(colour_pixels.max(axis=2)
                              - colour_pixels.min(axis=2)).mean()) / 255.0
        monochrome = 1.0 if spread < 0.04 else 0.0
    else:
        monochrome = 1.0

    values = {
        "width": float(width), "height": float(height),
        "aspect": float(width) / float(height or 1),
        "mean": mean, "contrast": contrast,
        "dark_share": dark, "bright_share": bright,
        "edges": edges, "ink": ink,
        "row_bands": float(row_bands), "col_bands": float(col_bands),
        "monochrome": monochrome,
    }
    return Retina(values=values, image=str(image_path))


def _count_bands(load, threshold: float) -> int:
    """Сколько раз нагрузка пересекает порог вверх. Полоса = одна строка текста."""
    bands = 0
    inside = False
    for value in load:
        active = value > threshold
        if active and not inside:
            bands += 1
        inside = active
    return bands


def describe(retina: Retina) -> str:
    """Словесное описание вектора — для человека, не для решения."""
    v = retina.values
    parts = [retina.summary()]
    if v.get("monochrome", 0) >= 1:
        parts.append("похоже на снимок экрана: цвета почти серые")
    if v.get("row_bands", 0) >= 4 and v.get("edges", 0) > 0.02:
        parts.append("ровные строки — вероятно текст")
    if v.get("ink", 0) < 0.02:
        parts.append("почти пусто")
    if v.get("contrast", 0) > 0.25 and v.get("ink", 0) > 0.15:
        parts.append("плотный экран")
    return "; ".join(parts)


def distance(left: Retina, right: Retina) -> float:
    """Расстояние между двумя восприятиями, по нормированным признакам.

    Нужно, чтобы муха могла сравнить: «этот экран похож на тот, где была
    ошибка» — без всякой модели.
    """
    scales = {
        "width": 1000.0, "height": 1000.0, "aspect": 3.0,
        "mean": 1.0, "contrast": 1.0, "dark_share": 1.0,
        "bright_share": 1.0, "edges": 0.2, "ink": 1.0,
        "row_bands": 40.0, "col_bands": 80.0, "monochrome": 1.0,
    }
    total = 0.0
    for name in FEATURES:
        scale = scales.get(name, 1.0)
        diff = (left.values.get(name, 0.0)
                - right.values.get(name, 0.0)) / scale
        total += diff * diff
    return total ** 0.5