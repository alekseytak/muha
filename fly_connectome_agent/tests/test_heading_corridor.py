"""Тесты 2D-коридора: кинематика, цель, стены, сенсорика."""
import numpy as np
import pytest

from fly_connectome_agent.src.engineering.environments.heading_corridor import (
    HeadingConfig,
    HeadingCorridor,
)


def make(**overrides) -> HeadingCorridor:
    return HeadingCorridor(HeadingConfig(**overrides))


def test_povorot_i_shag_vpered():
    env = make(start=(1, 1), target=(5, 5))
    obs, info = env.reset()
    assert info["position"] == (1, 1) and info["heading"] == 0

    _, reward, terminated, _, info = env.step("move_left")
    assert info["position"] == (1, 2)          # повернул на север и шагнул
    assert info["heading"] == 1
    assert reward == pytest.approx(1.0)        # расстояние 8 -> 7
    assert not terminated

    _, _, _, _, info = env.step("move_right")
    assert info["position"] == (2, 2)          # повернул на восток и шагнул
    assert info["heading"] == 0


def test_stay_idet_po_kursu_ne_ostanavlivaet():
    env = make(start=(1, 1), target=(5, 1))
    env.reset()
    _, reward, _, _, info = env.step("stay")
    assert info["position"] == (2, 1) and info["heading"] == 0
    assert reward == pytest.approx(1.0)


def test_stena_ne_propuskaet_i_shtrafuet():
    env = make(start=(1, 0), target=(5, 0), heading=3)  # курс на юг, стена рядом
    env.reset()
    _, reward, _, _, info = env.step("stay")
    assert info["position"] == (1, 0)
    assert info["bumped"] is True
    assert reward == pytest.approx(-1.0)


def test_dostizhenie_celi_zavershaet_epizod():
    env = make(start=(0, 0), target=(2, 0), width=3, height=3)
    env.reset()
    total = 0.0
    for _ in range(2):
        _, reward, terminated, _, info = env.step("stay")
        total += reward
    assert info["position"] == (2, 0)
    assert terminated is True
    assert total == pytest.approx(1.0 + 1.0 + 10.0)

    _, reward, terminated, truncated, info = env.step("stay")
    assert reward == 0.0 and terminated and truncated
    assert info["reason"] == "already_terminated"


def test_shaping_otricatelen_pri_shage_ot_celi():
    env = make(start=(1, 1), target=(5, 1), heading=1)  # курс на север, цель на востоке
    env.reset()
    _, reward, _, _, info = env.step("stay")
    assert info["position"] == (1, 2)     # расстояние 4 -> 5
    assert reward == pytest.approx(-1.0)


def test_obstacle_blokiruet_dvizhenie():
    env = make(start=(1, 1), target=(5, 1), obstacles=frozenset({(2, 1)}))
    env.reset()
    _, reward, _, _, info = env.step("stay")
    assert info["position"] == (1, 1)
    assert info["bumped"] is True and reward == pytest.approx(-1.0)


@pytest.mark.parametrize("heading,expected", [
    (0, "ahead"),    # восток, цель восточнее
    (1, "right"),    # север, цель восточнее -> по правому борту
    (2, "behind"),   # запад, цель за кормой
    (3, "left"),     # юг, цель восточнее -> по левому борту
])
def test_bearing_po_kursu(heading, expected):
    env = make(start=(1, 1), target=(3, 1), heading=heading)
    env.reset()
    assert env.bearing() == expected


def test_bearing_za_kormoy_ne_pritvoryaetsya_pryamo():
    env = make(start=(3, 1), target=(1, 1), heading=0)  # смотрим на восток, цель на западе
    env.reset()
    assert env.bearing() == "behind"


def test_truncation_po_max_steps():
    env = make(start=(1, 1), target=(5, 5), max_steps=3)
    env.reset()
    truncated = False
    for _ in range(3):
        _, _, _, truncated, _ = env.step("stay")
    assert truncated is True and env.terminated is False


def test_reset_vozvrashchaet_sostoyanie():
    env = make(start=(1, 1), target=(5, 5))
    env.reset()
    for action in ("move_left", "stay", "move_right"):
        env.step(action)
    obs, info = env.reset()
    assert info["position"] == (1, 1)
    assert info["heading"] == 0
    assert np.array_equal(obs, np.array([1, 1, 0]))


def test_dvizhenie_deystvitelno_2d():
    """Маршрут меняет обе координаты: одной осью его не объяснить."""
    env = make(start=(1, 1), target=(3, 3), width=5, height=5)
    env.reset()
    for action in ("move_left", "stay", "move_right", "stay"):
        _, _, terminated, _, info = env.step(action)
    assert info["position"] == (3, 3)
    assert terminated is True


def test_reaktivnaya_politika_dostigaet_celi():
    """Верхняя граница: сенсорика достаточна, задача решаема без обучения.

    Политика «поворачивай в сторону цели, иначе прямо» — не часть контура, а
    доказательство, что успех вообще достижим на этой геометрии.
    """
    env = make(start=(1, 1), target=(5, 5))
    env.reset()
    for _ in range(40):
        bearing = env.bearing()
        action = {"left": "move_left", "right": "move_right",
                  "ahead": "stay", "behind": "move_left"}[bearing]
        _, _, terminated, _, info = env.step(action)
        if terminated:
            break
    assert terminated is True
    assert info["position"] == (5, 5)


@pytest.mark.parametrize("kwargs", [
    {"start": (1, 1), "target": (1, 1)},
    {"start": (1, 1), "target": (9, 9)},
    {"start": (1, 1), "target": (5, 5), "obstacles": frozenset({(1, 1)})},
    {"start": (1, 1), "target": (5, 5), "heading": 4},
    {"start": (1, 1), "target": (5, 5), "max_steps": 0},
])
def test_konfiguraciya_proveryaetsya(kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


def test_neizvestnoe_deystvie_otvergaetsya():
    env = make()
    env.reset()
    with pytest.raises(ValueError):
        env.step("jump")
