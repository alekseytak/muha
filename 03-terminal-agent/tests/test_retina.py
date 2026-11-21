"""Проверки локального зрения: вектор признаков без сети.

Главное, что здесь проверяется: признаки РАЗЛИЧАЮТ картинки. Вектор, который
одинаков для пустого экрана и для плотного текста, бесполезен ровно так же,
как гейт, постоянный по построению.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.retina import (                       # noqa: E402
    FEATURES, HAVE_IMAGING, describe, distance, look,
)

if HAVE_IMAGING:
    from PIL import Image, ImageDraw


def make_blank(path: str, size=(400, 200), colour=(255, 255, 255)) -> str:
    Image.new("RGB", size, colour).save(path)
    return path


def make_lines(path: str, lines: int = 6, size=(400, 240)) -> str:
    image = Image.new("RGB", size, (12, 14, 18))
    draw = ImageDraw.Draw(image)
    step = size[1] // (lines + 1)
    for index in range(lines):
        y = step * (index + 1)
        draw.rectangle([20, y, size[0] - 20, y + 8], fill=(230, 230, 230))
    image.save(path)
    return path


def make_colour(path: str, size=(200, 200)) -> str:
    image = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            image.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
    image.save(path)
    return path


@unittest.skipUnless(HAVE_IMAGING, "нужны Pillow и numpy")
class RetinaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def path(self, name: str) -> str:
        return os.path.join(self.tmp, name)

    def test_vector_has_every_feature(self):
        retina = look(make_blank(self.path("blank.png")))
        self.assertEqual(sorted(retina.values), sorted(FEATURES))
        self.assertEqual(len(retina.as_list()), len(FEATURES))
        self.assertTrue(all(isinstance(value, float)
                            for value in retina.as_list()))

    def test_blank_screen_has_no_ink_and_no_bands(self):
        retina = look(make_blank(self.path("blank.png")))
        self.assertLess(retina.values["ink"], 0.01)
        self.assertEqual(int(retina.values["row_bands"]), 0)
        self.assertLess(retina.values["edges"], 0.01)

    def test_lines_are_counted_as_bands(self):
        retina = look(make_lines(self.path("lines.png"), lines=6))
        self.assertGreaterEqual(int(retina.values["row_bands"]), 5)
        self.assertLessEqual(int(retina.values["row_bands"]), 8)
        self.assertGreater(retina.values["ink"], 0.01)

    def test_distance_separates_blank_from_text(self):
        blank = look(make_blank(self.path("blank.png")))
        lines = look(make_lines(self.path("lines.png")))
        self.assertGreater(distance(blank, lines), 0.05,
                           "пустой экран и текст должны различаться")
        self.assertAlmostEqual(distance(lines, lines), 0.0, places=6)

    def test_monochrome_flag_tells_screenshot_from_colour_photo(self):
        grey = look(make_blank(self.path("grey.png"), colour=(20, 20, 20)))
        colour = look(make_colour(self.path("colour.png")))
        self.assertEqual(grey.values["monochrome"], 1.0)
        self.assertEqual(colour.values["monochrome"], 0.0)

    def test_missing_file_is_reported_not_silently_zero(self):
        with self.assertRaises(FileNotFoundError):
            look(self.path("нет-такого.png"))

    def test_description_is_human_readable(self):
        text = describe(look(make_lines(self.path("lines.png"))))
        self.assertIn("x", text)
        self.assertIn("строк", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)