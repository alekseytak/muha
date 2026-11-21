"""ZaLoop: контроллер лупов для MUHA.

Перенос спецификации `~/DSH-Research-Lab/library/ergo/zaloop.md`. Основание
документа — `~/документы /🔄Os глаголы и лупы .md` (разделы 1.2, 2.1, 4, 5)
и тормоза из `core/autonomous.py`.

Контроллер **читает** журнал шагов и **возвращает** уровень автономии. Он не
исполняет действия, не меняет цель и не трогает песочницу.

Пять ступеней, внутри одного лупа только вверх:

| ступень | условие | что разрешено |
|---|---|---|
| 0 normal | ни один порог не превышен | всё |
| 1 nudge | один повтор подряд | всё + подсказка |
| 2 reduce | два повтора подряд или доля повторов выше порога | меньше шагов, окно истории 3 |
| 3 shadow | три повтора, цикл периода 1-4 или доля отказов выше порога | только предложения |
| 4 stop | цикл после shadow, две ошибки подряд или бюджет кончился | остановка, задача в PENDING |

Сигналом считается отпечаток действия `verb|domain|object`, а не текст
команды: `cat notes/a.txt` и `sed -n 1p notes/a.txt` — одно действие.

Пороги без `sha256` не принимаются: контроллер отказывается работать.
Верхняя граница рассеяния **не реализована как гейт** — в документе она
объявлена, но не измерена (доля новизны 1.00 во всех 47 ячейках), а гейт,
постоянный по построению, ничего не измеряет.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

NORMAL, NUDGE, REDUCE, SHADOW, STOP = (
    "normal", "nudge", "reduce", "shadow", "stop")
STAGES = (NORMAL, NUDGE, REDUCE, SHADOW, STOP)
RANK = {name: index for index, name in enumerate(STAGES)}

PENDING = "PENDING"


class ZaLoopError(RuntimeError):
    """Пороги непригодны: нет хеша, битый файл, чужая модель."""


@dataclass
class Thresholds:
    """Пороги. Значений по умолчанию нет: они выводятся из пилота."""

    repeat_ratio_reduce: float
    repeat_ratio_shadow: float
    denied_rate_shadow: float
    max_steps: int
    model: str = ""
    calibrated_on: str = ""
    hash: str = ""

    @classmethod
    def load(cls, path: str) -> "Thresholds":
        """Прочитать пороги из YAML-подобного файла и проверить хеш.

        Файл разбирается вручную: PyYAML на этой машине падает в dlopen,
        а тащить зависимость ради шести чисел незачем.
        """
        text = Path(path).read_text(encoding="utf-8")
        values: Dict[str, str] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            values[key.strip()] = value.strip().strip("'\"")
        claimed = values.get("hash", "")
        if not claimed:
            raise ZaLoopError(
                "пороги без hash не принимаются: сначала пилот, потом прогон")
        body = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("hash:"))
        actual = "sha256:" + hashlib.sha256(
            body.encode("utf-8")).hexdigest()
        if claimed != actual:
            raise ZaLoopError(
                f"hash не совпадает: в файле {claimed[:20]}, "
                f"посчитано {actual[:20]}")
        try:
            return cls(
                repeat_ratio_reduce=float(values["repeatRatio_reduce"]),
                repeat_ratio_shadow=float(values["repeatRatio_shadow"]),
                denied_rate_shadow=float(values["deniedRate_shadow"]),
                max_steps=int(values["max_steps"]),
                model=values.get("model", ""),
                calibrated_on=values.get("calibrated_on", ""),
                hash=claimed,
            )
        except KeyError as missing:
            raise ZaLoopError(f"в порогах нет поля {missing}") from None


@dataclass
class Signals:
    """Сигналы, посчитанные по журналу шагов."""

    steps: int = 0
    repeat_ratio: float = 0.0
    consecutive_repeats: int = 0
    cyclic: bool = False
    period: int = 0
    denied_rate: float = 0.0
    escalation_rate: float = 0.0
    failures_in_a_row: int = 0
    budget_left: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "steps": self.steps,
            "repeatRatio": round(self.repeat_ratio, 3),
            "consecutiveRepeats": self.consecutive_repeats,
            "cycle": {"cyclic": self.cyclic, "period": self.period},
            "deniedRate": round(self.denied_rate, 3),
            "escalationRate": round(self.escalation_rate, 3),
            "failuresInARow": self.failures_in_a_row,
            "budgetLeft": self.budget_left,
        }


@dataclass
class HumanReviewTicket:
    """Тикет на ступени stop. Причина — наблюдение, а не оценка."""

    task_id: str
    stop_reason: str
    stage: str
    steps: List[Dict[str, object]] = field(default_factory=list)
    signals: Dict[str, object] = field(default_factory=dict)
    repeat_ratio: float = 0.0
    cycle: Dict[str, object] = field(default_factory=dict)
    resume_from: str = PENDING


@dataclass
class Decision:
    """Что разрешено на следующем шаге."""

    stage: str
    reason: str
    allow_execution: bool = True
    max_steps: Optional[int] = None
    history_window: Optional[int] = None
    nudge: str = ""
    ticket: Optional[HumanReviewTicket] = None
    signals: Optional[Signals] = None
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "stage": self.stage,
            "reason": self.reason,
            "allowExecution": self.allow_execution,
            "maxSteps": self.max_steps,
            "historyWindow": self.history_window,
            "nudge": self.nudge,
            "warnings": list(self.warnings),
            "ticket": (None if self.ticket is None else {
                "taskId": self.ticket.task_id,
                "stopReason": self.ticket.stop_reason,
                "stage": self.ticket.stage,
                "repeatRatio": round(self.ticket.repeat_ratio, 3),
                "cycle": self.ticket.cycle,
                "resumeFrom": self.ticket.resume_from,
            }),
            "signals": (None if self.signals is None
                        else self.signals.as_dict()),
        }


def object_cluster(obj: str) -> str:
    """Кластер объекта: каталог плюс расширение.

    Нужен, чтобы отличить «ту же рамку» от «новой»: `notes/a.txt` и
    `notes/b.txt` — один кластер, `sys:uname` — другой.
    """
    text = (obj or "").strip()
    if ":" in text.split("/")[0]:
        scheme, _, rest = text.partition(":")
        return f"{scheme}:"
    path = Path(text)
    suffix = path.suffix.lower()
    return f"{path.parent}/{suffix}" if suffix else str(path.parent)


def detect_cycle(signatures: Sequence[str], max_period: int = 4) -> int:
    """Найти период повторяющегося хвоста. 0 — цикла нет.

    Проверяются периоды от 1 до `max_period`: `A A A` — период 1,
    `A B A B` — период 2. Смотрим только хвост, чтобы не ловить случайные
    совпадения в начале длинной истории.
    """
    for period in range(1, max_period + 1):
        if len(signatures) < 2 * period:
            continue
        tail = list(signatures[-2 * period:])
        if tail[:period] == tail[period:]:
            return period
    return 0


class ZaLoop:
    """Контроллер. Один экземпляр на задачу."""

    def __init__(self, thresholds: Thresholds,
                 expected_model: str = "") -> None:
        self.thresholds = thresholds
        self.stage = NORMAL
        self._frame = ""
        # Начало текущего лупа: после смены рамки сигналы считаются заново,
        # иначе доля повторов тянет за собой историю прошлой рамки.
        self._frame_start = 0
        self.warnings: List[str] = []
        if (expected_model and thresholds.model
                and thresholds.model != expected_model):
            self.warnings.append(
                f"пороги откалиброваны на модели {thresholds.model}, "
                f"а проверка идёт на {expected_model}")

    # ── сигналы ────────────────────────────────────────────────────────────
    def measure(self, steps: Sequence, budget_left: int = 0) -> Signals:
        signatures = [step.signature() for step in steps]
        total = len(signatures)
        if total == 0:
            return Signals(budget_left=budget_left)
        unique = len(set(signatures))
        repeats = 0
        if total > 1 and signatures[-1] == signatures[-2]:
            repeats = 1
            for index in range(total - 2, 0, -1):
                if signatures[index] == signatures[index - 1]:
                    repeats += 1
                else:
                    break
        denied = sum(1 for step in steps if not step.success)
        escalated = sum(
            1 for step in steps
            if getattr(step, "verb", "") in ("escalate", "veto"))
        failures = 0
        for step in reversed(list(steps)):
            if step.success:
                break
            failures += 1
        period = detect_cycle(signatures)
        return Signals(
            steps=total,
            repeat_ratio=1.0 - unique / total,
            consecutive_repeats=repeats,
            cyclic=period > 0,
            period=period,
            denied_rate=denied / total,
            escalation_rate=escalated / total,
            failures_in_a_row=failures,
            budget_left=budget_left,
        )

    # ── ступени ────────────────────────────────────────────────────────────
    def _raise(self, stage: str) -> str:
        """Поднять ступень, но не опустить: внутри лупа только вверх."""
        if RANK[stage] > RANK[self.stage]:
            self.stage = stage
        return self.stage

    def _reset_if_frame_changed(self, steps: Sequence) -> bool:
        if not steps:
            return False
        last = steps[-1]
        frame = f"{last.verb}|{last.domain}|{object_cluster(last.object)}"
        if not self._frame:
            self._frame = frame
            return False
        if frame != self._frame:
            previous = self._frame
            self._frame = frame
            same_cluster = previous.rsplit("|", 1)[-1] == frame.rsplit("|", 1)[-1]
            if same_cluster:
                self.stage = NORMAL
                self._frame_start = len(steps) - 1
                return True
        return False

    def supervise(self, steps: Sequence, budget_left: int = 0,
                  task_id: str = "") -> Decision:
        """Решить, что разрешено на следующем шаге."""
        steps = list(steps)
        reset = self._reset_if_frame_changed(steps)
        # Ступень, стоявшая ДО этого шага: нужна, чтобы отличить «только что
        # вошли в shadow» от «цикл продолжается в shadow». Иначе цикл периода 1
        # валил задачу в stop в тот же момент, когда её надо было лишь
        # притормозить.
        previous_stage = self.stage
        signals = self.measure(steps[self._frame_start:], budget_left)
        limits = self.thresholds

        # stop: две ошибки подряд или кончился бюджет.
        if signals.failures_in_a_row >= 2:
            self._raise(STOP)
            reason = (f"две ошибки подряд: последние шаги "
                      f"{signals.failures_in_a_row} без успеха")
            return self._stop(steps, signals, task_id, reason)
        if budget_left <= 0 and steps:
            self._raise(STOP)
            return self._stop(steps, signals, task_id, "бюджет исчерпан")

        # shadow: три повтора, цикл периода 1-4, много отказов.
        if signals.consecutive_repeats >= 3:
            self._raise(SHADOW)
        elif signals.cyclic and self.stage != NORMAL:
            # Цикл после уже поднятой ступени — это петля, а не разведка.
            self._raise(SHADOW)
        elif signals.cyclic and signals.period >= 2:
            self._raise(SHADOW)
        elif signals.denied_rate > limits.denied_rate_shadow:
            self._raise(SHADOW)

        if self.stage == SHADOW:
            # Цикл, найденный после shadow, ведёт в stop — но только если
            # shadow уже стоял на входе, а не выставлен этим же шагом.
            if (previous_stage == SHADOW and signals.cyclic
                    and (signals.consecutive_repeats >= 3
                         or signals.period >= 2)):
                self._raise(STOP)
                return self._stop(
                    steps, signals, task_id,
                    f"цикл периода {signals.period} после теневого режима")
            return Decision(
                stage=SHADOW,
                reason=(f"повторов подряд {signals.consecutive_repeats}, "
                        f"цикл {signals.period or 'нет'}, отказов "
                        f"{signals.denied_rate:.0%}"),
                allow_execution=False, max_steps=None, history_window=3,
                signals=signals, warnings=list(self.warnings))

        # reduce: два повтора или высокая доля повторов.
        #
        # Доля повторов считается только на истории от четырёх шагов: на двух
        # шагах один повтор даёт 0.5, что выше любого разумного порога, и
        # ступень reduce срабатывала бы там, где спецификация требует nudge.
        if signals.consecutive_repeats >= 2:
            self._raise(REDUCE)
        elif (signals.steps >= 4
              and signals.repeat_ratio > limits.repeat_ratio_reduce):
            self._raise(REDUCE)

        if self.stage == REDUCE:
            return Decision(
                stage=REDUCE,
                reason=(f"повторов подряд {signals.consecutive_repeats}, "
                        f"доля повторов {signals.repeat_ratio:.0%}"),
                max_steps=max(1, limits.max_steps // 2), history_window=3,
                signals=signals, warnings=list(self.warnings))

        # nudge: один повтор — подсказка, а не сдача.
        if signals.consecutive_repeats == 1:
            self._raise(NUDGE)
            last = steps[-1]
            return Decision(
                stage=NUDGE,
                reason=f"действие повторено: {last.verb} {last.object[:40]}",
                nudge=("ВАЖНО: это действие уже выполнено, его результат выше. "
                       "Не повторяй его — ответь итогом или выбери ДРУГОЕ "
                       "действие."),
                signals=signals, warnings=list(self.warnings))

        reason = "рамка сменилась" if reset else "ни один порог не превышен"
        return Decision(stage=NORMAL, reason=reason, signals=signals,
                        max_steps=limits.max_steps,
                        warnings=list(self.warnings))

    def _stop(self, steps: Sequence, signals: Signals, task_id: str,
              reason: str) -> Decision:
        ticket = HumanReviewTicket(
            task_id=task_id,
            stop_reason=reason,
            stage=self.stage,
            steps=[{"index": getattr(step, "index", index + 1),
                    "signature": step.signature(),
                    "success": bool(step.success)}
                   for index, step in enumerate(steps)],
            signals=signals.as_dict(),
            repeat_ratio=signals.repeat_ratio,
            cycle={"cyclic": signals.cyclic, "period": signals.period},
        )
        return Decision(stage=STOP, reason=reason, allow_execution=False,
                        max_steps=0, history_window=3, ticket=ticket,
                        signals=signals, warnings=list(self.warnings))