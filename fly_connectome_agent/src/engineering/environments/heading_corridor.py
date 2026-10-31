"""2D-коридор с целью и танковой кинематикой (ЭТАП 4).

Зачем именно танк. Глагольный декодер выдаёт два направленных действия плюс
«прямо». Если трактовать их как абсолютные шаги по осям, среда снова
становится 1D — просто с другими именами, и «2D-полигон» из инструкции был бы
декорацией. Поэтому:

    move_left  -> поворот на 90° против часовой, затем шаг вперёд
    move_right -> поворот на 90° по часовой, затем шаг вперёд
    stay       -> шаг вперёд без поворота

Такая последовательность решений действительно разворачивается в 2D: маршрут
зависит от того, ЧТО и КОГДА повернулось, а не от одной координаты.

Кинематика детерминирована: ни шума, ни скрытого состояния, кроме позиции и
курса. Награда — потенциальная shaping по манхэттенскому расстоянию (шаг к
цели положителен, шаг от цели отрицателен) плюс штраф за удар в стену. Shaping
не меняет оптимальную политику: это разность потенциалов, а не подсказка.

Сенсорика `bearing()` — восприятие, а не команда: среда сообщает, каким
бортом стоит цель, но не говорит, какое действие правильное. Один и тот же
сигнал получают все руки контура, включая замороженные контроли.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 0 = восток, 1 = север, 2 = запад, 3 = юг (против часовой).
DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass(frozen=True)
class HeadingConfig:
    width: int = 7
    height: int = 7
    start: tuple[int, int] = (1, 1)
    heading: int = 0
    target: tuple[int, int] = (5, 5)
    obstacles: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    max_steps: int = 40
    target_reward: float = 10.0
    wall_penalty: float = -1.0
    step_cost: float = 0.0
    shaping_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("коридор должен быть хотя бы 2x2")
        if not 0 <= self.heading < 4:
            raise ValueError(f"heading обязан быть 0..3, получен {self.heading!r}")
        if self.max_steps <= 0:
            raise ValueError("max_steps обязан быть положительным")
        if self.shaping_scale < 0:
            raise ValueError("shaping_scale не может быть отрицательным")


class HeadingCorridor:
    """Сетка с целью; агент рулит, а не телепортируется по осям."""

    ACTION_SPACE = ("move_left", "move_right", "stay")

    def __init__(self, config: HeadingConfig | None = None):
        self.config = config or HeadingConfig()
        self._validate_config()
        self.x, self.y = self.config.start
        self.heading = self.config.heading
        self.terminated = False
        self._steps_this_episode = 0

    def _validate_config(self) -> None:
        c = self.config
        for name, cell in (("start", c.start), ("target", c.target)):
            if not self._inside(cell):
                raise ValueError(f"{name} вне сетки: {cell!r}")
        if c.start == c.target:
            raise ValueError("старт и цель совпадают — задачи нет")
        blocked = {c.start, c.target} & set(c.obstacles)
        if blocked:
            raise ValueError(f"препятствия перекрывают старт/цель: {sorted(blocked)!r}")

    def _inside(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def _free(self, cell: tuple[int, int]) -> bool:
        return self._inside(cell) and cell not in self.config.obstacles

    def distance(self) -> int:
        return abs(self.config.target[0] - self.x) + abs(self.config.target[1] - self.y)

    def bearing(self) -> str:
        """Где цель относительно курса: left / right / ahead / behind.

        Считается по знаку векторного произведения курса и направления на цель.
        Случай «ровно позади» возвращается отдельно, а не притворяется «прямо»:
        иначе агент шёл бы от цели, считая, что она впереди.
        """
        tx, ty = self.config.target
        vx, vy = tx - self.x, ty - self.y
        if vx == 0 and vy == 0:
            return "ahead"
        hx, hy = DIRECTIONS[self.heading]
        cross = hx * vy - hy * vx
        if cross > 0:
            return "left"
        if cross < 0:
            return "right"
        return "ahead" if hx * vx + hy * vy > 0 else "behind"

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Среда детерминирована; seed принимается ради общего интерфейса."""
        self.x, self.y = self.config.start
        self.heading = self.config.heading
        self.terminated = False
        self._steps_this_episode = 0
        return self._obs(), self._info(prev=(self.x, self.y), reward=0.0, bumped=False)

    def step(self, action: str) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if action not in self.ACTION_SPACE:
            raise ValueError(f"Invalid action: {action!r}. Valid: {self.ACTION_SPACE}")
        if self.terminated:
            return self._obs(), 0.0, True, True, self._info(
                prev=(self.x, self.y), reward=0.0, bumped=False, reason="already_terminated")

        c = self.config
        prev = (self.x, self.y)
        prev_distance = self.distance()

        if action == "move_left":
            self.heading = (self.heading + 1) % 4
        elif action == "move_right":
            self.heading = (self.heading - 1) % 4

        dx, dy = DIRECTIONS[self.heading]
        candidate = (self.x + dx, self.y + dy)
        bumped = not self._free(candidate)
        reward = 0.0
        if bumped:
            reward += c.wall_penalty
        else:
            self.x, self.y = candidate
            reward += c.shaping_scale * (prev_distance - self.distance())
        reward += c.step_cost

        self._steps_this_episode += 1
        if (self.x, self.y) == c.target:
            reward += c.target_reward
            self.terminated = True
        truncated = self._steps_this_episode >= c.max_steps and not self.terminated
        return self._obs(), reward, self.terminated, truncated, self._info(
            prev=prev, reward=reward, bumped=bumped)

    def _obs(self) -> np.ndarray:
        return np.array([self.x, self.y, self.heading], dtype=np.int64)

    def _info(self, *, prev: tuple[int, int], reward: float, bumped: bool,
              reason: str = "") -> dict[str, Any]:
        info = {
            "position": (self.x, self.y),
            "prev_position": prev,
            "heading": self.heading,
            "distance": self.distance(),
            "bearing": self.bearing(),
            "bumped": bumped,
            "reward": reward,
            "step": self._steps_this_episode,
        }
        if reason:
            info["reason"] = reason
        return info
