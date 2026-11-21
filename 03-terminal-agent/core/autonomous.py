"""Автономный цикл: агент сам берёт задачи из очереди и доводит их до конца.

Экономия заложена в конструкцию, а не в благие намерения:

* **Один вызов модели на шаг.** Никаких «сначала план, потом уточнение» —
  план и есть шаг.
* **Жёсткий бюджет вызовов.** `max_llm_calls` проверяется ДО запроса, поэтому
  превысить его нельзя даже при ошибках.
* **Компактная история.** В контекст идут глагол, объект и первые 80 символов
  результата — не полные тексты. Это главный рычаг расхода: промпт не растёт
  линейно от объёма прочитанных данных.
* **Раннее завершение.** Ответ THINK означает «задача решена словами» — цикл
  останавливается, лишнего вызова нет.
* **Детектор зацикливания.** Повтор того же действия дважды подряд —
  остановка. Без него агент честно сожжёт весь бюджет, топчась на месте.
* **Учёт токенов.** Расход берётся из ответов провайдера, а не оценивается.

Цикл не изобретает полномочий: каждое действие проходит обычный путь
режим -> гейт -> подтверждение -> инструмент -> реестр.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .tasks import DONE, FAILED, PENDING, Task

# Сколько символов результата шага попадает в историю для модели.
RESULT_SNIPPET = 120          # старые шаги: только смысл
RESULT_SNIPPET_LAST = 400     # последний шаг: данные видны, повтор не нужен
# Сколько последних шагов показывать: дальше история дороже, чем полезна.
HISTORY_WINDOW = 6


@dataclass
class StepRecord:
    """Один шаг цикла — то, что реально произошло."""

    index: int
    verb: str
    domain: str
    object: str
    success: bool
    tool: Optional[str] = None
    result: str = ""
    refused: str = ""
    llm_calls: int = 0
    # Сырой ответ модели: без него невозможно понять, почему шаг получился
    # таким — парсер мог не распознать формат и уйти в THINK.
    raw_answer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def signature(self) -> str:
        """Отпечаток действия — по нему ловим зацикливание."""
        return f"{self.verb}|{self.domain}|{self.object}"

    def history_line(self, limit: int = RESULT_SNIPPET) -> str:
        mark = "ок" if self.success else "отказ"
        body = (self.result or self.refused or "").replace("\n", " ")[:RESULT_SNIPPET]
        return f"{self.index}. {self.verb.upper()} {self.object} -> {mark}: {body}"


@dataclass
class RunReport:
    """Итог прогона по одной задаче."""

    task_id: str
    task_text: str
    status: str
    stop_reason: str
    steps: List[StepRecord] = field(default_factory=list)
    llm_calls: int = 0
    tokens: int = 0
    tokens_reported: bool = True
    duration_s: float = 0.0

    @property
    def solved(self) -> bool:
        return self.status == DONE

    @property
    def final_answer(self) -> str:
        """Итог задачи — текст последнего ответа THINK, а не причина остановки."""
        for step in reversed(self.steps):
            if step.verb == "think" and step.success and step.result:
                return step.result
        for step in reversed(self.steps):
            if step.success and step.result:
                return step.result
        return self.stop_reason

    def summary(self) -> str:
        return (
            f"задача {self.task_id}: {self.status} ({self.stop_reason}), "
            f"шагов {len(self.steps)}, вызовов модели {self.llm_calls}, "
            f"токенов {self.tokens if self.tokens_reported else '?'}"
        )


class AutonomousRunner:
    """Прогоняет задачи из очереди через обычный путь агента."""

    def __init__(self, agent: Any, queue: Any, compact_prompt: bool = False) -> None:
        self.agent = agent
        self.queue = queue
        # Компактный промпт (263 токена вместо 876) экономит 70% промпта, но
        # замер на контрольной выборке дал 60% точности разбора против 100%.
        # По умолчанию выключен: тонкий промпт ломает разбор глаголов.
        self.compact_prompt = compact_prompt
        # Пороги ZaLoop. Без файла порогов контроллер не работает: значения
        # выводятся из пилота, а не берутся с потолка.
        self.zaloop_thresholds = None
        self.zaloop_model = ""
        self.last_ticket = None

    def load_zaloop(self, path: str, model: str = "") -> None:
        """Подключить пороги ZaLoop. Бросает ZaLoopError, если файл плох."""
        from .zaloop import Thresholds

        self.zaloop_thresholds = Thresholds.load(path)
        self.zaloop_model = model or getattr(
            getattr(self.agent, "llm", None), "model", "")

    def _make_zaloop(self):
        """Контроллер на задачу. Без порогов — None, и работает старый тормоз."""
        if self.zaloop_thresholds is None:
            return None
        from .zaloop import ZaLoop

        return ZaLoop(self.zaloop_thresholds,
                      expected_model=self.zaloop_model)

    # ---------- контекст для модели ----------

    def build_prompt(self, task: Task, steps: List[StepRecord],
                     steps_left: int, calls_left: int,
                     nudge: str = "") -> str:
        """Компактный запрос: задача, короткая история, остаток бюджета."""
        lines = [f"Задача: {task.text}", f"Выполнено шагов: {len(steps)}"]
        if steps:
            lines.append("История (последние шаги):")
            lines += [record.history_line() for record in steps[-HISTORY_WINDOW:]]
        lines.append(f"Осталось шагов: {steps_left}, вызовов модели: {calls_left}.")
        if steps:
            lines.append(
                "Не повторяй действие, которое уже выполнено: если нужные данные "
                "есть в истории, ответь THINK: <итог> | {} — иначе простоишь."
            )
        lines.append(
            "Выполни следующий шаг: одно действие в формате ГЛАГОЛ: объект | {}. "
            "Если задача уже решена — ответь THINK: <итог> | {}"
        )
        if nudge:
            lines.append(nudge)
        return "\n".join(lines)

    # ---------- расход ----------

    def _usage(self) -> Dict[str, int]:
        totals = {"calls": 0, "tokens": 0, "unreported": 0}
        for client in (getattr(self.agent, "llm", None),
                       getattr(self.agent, "fallback_llm", None)):
            if client is None:
                continue
            report = client.usage_report()
            totals["calls"] += report["calls"]
            totals["tokens"] += report["total_tokens"]
            totals["unreported"] += report["calls_without_usage"]
        return totals

    # ---------- цикл ----------

    def run_task(self, task: Task, max_steps: int = 8,
                 max_llm_calls: int = 6) -> RunReport:
        """Довести одну задачу до конца в пределах бюджетов."""
        started = time.time()
        before = self._usage()
        # На прогон ставим нулевую температуру: разбор глаголов должен быть
        # воспроизводимым, а «творчество» здесь только портит формат.
        saved_temps = []
        for client in (getattr(self.agent, "llm", None),
                       getattr(self.agent, "fallback_llm", None)):
            if client is not None:
                saved_temps.append((client, client.temperature))
                client.temperature = 0.0
        steps: List[StepRecord] = []
        stop_reason = "бюджет шагов"
        status = FAILED
        budget_stop = False
        last_signature: Optional[str] = None
        repeats = 0
        failures_in_a_row = 0
        zaloop = self._make_zaloop()
        zaloop_decision = None
        shadow_steps = 0
        format_failures = 0
        nudge = ""

        while True:
            calls_used = self._usage()["calls"] - before["calls"]
            if len(steps) >= max_steps:
                stop_reason = f"достигнут предел шагов ({max_steps})"
                break
            if calls_used >= max_llm_calls:
                stop_reason = f"исчерпан бюджет вызовов модели ({max_llm_calls})"
                budget_stop = True
                break

            prompt = self.build_prompt(
                task, steps, max_steps - len(steps),
                max_llm_calls - calls_used, nudge=nudge
            )
            system_prompt = None
            if self.compact_prompt and steps:
                system_prompt = self.agent.compact_planning_prompt()
            try:
                answer = self.agent.think(prompt, system_prompt=system_prompt)
            except Exception as exc:  # noqa: BLE001 — цикл не должен падать целиком
                stop_reason = f"модель недоступна: {type(exc).__name__}: {str(exc)[:120]}"
                break

            calls_used = self._usage()["calls"] - before["calls"]
            verb_act = self.agent._parse_verb_act(answer, prompt)

            from .agent import FORMAT_FAILURE

            parse_failed = str(verb_act.parse_reason or "").startswith(FORMAT_FAILURE)
            if parse_failed:
                # Модель не выдала формат. Парсер подставил THINK с текстом
                # запроса — это не ответ, и закрывать задачу нельзя: иначе
                # в отчёте окажется «решено», а результатом будет наш же промпт.
                format_failures += 1
                steps.append(StepRecord(
                    index=len(steps) + 1, verb="think", domain="ui",
                    object="(ответ не по формату)", success=False,
                    refused=f"модель ответила не по формату: {answer[:160]}",
                    llm_calls=calls_used, raw_answer=answer[:400],
                ))
                if format_failures >= 2:
                    stop_reason = "модель дважды ответила не в формате ГЛАГОЛ"
                    break
                nudge = (
                    "ВАЖНО: предыдущий ответ не соответствовал формату. Ответь "
                    "РОВНО одной строкой вида THINK: <итог> | {} — без пояснений."
                )
                continue
            nudge = ""

            if verb_act.verb.value == "think":
                steps.append(StepRecord(
                    index=len(steps) + 1, verb="think", domain=verb_act.domain or "ui",
                    object=verb_act.object[:120], success=True,
                    result=verb_act.object[:600], llm_calls=calls_used,
                    raw_answer=answer[:400],
                ))
                status = DONE
                stop_reason = "задача решена ответом THINK"
                break

            self.agent.execute(verb_act)
            record = StepRecord(
                index=len(steps) + 1,
                verb=verb_act.verb.value,
                domain=verb_act.domain or "ui",
                object=verb_act.object[:120],
                success=bool(verb_act.success),
                tool=verb_act.tool,
                result=str(verb_act.result)[:400] if verb_act.success else "",
                refused="" if verb_act.success else str(verb_act.result)[:400],
                llm_calls=calls_used,
                raw_answer=answer[:400],
            )
            steps.append(record)

            if zaloop is not None:
                zaloop_decision = zaloop.supervise(
                    steps, budget_left=max(0, max_steps - len(steps)),
                    task_id=task.id)
                if zaloop_decision.stage == "stop":
                    stop_reason = f"ZaLoop: {zaloop_decision.reason}"
                    if zaloop_decision.ticket is not None:
                        self.last_ticket = zaloop_decision.ticket
                    status = PENDING
                    break
                if zaloop_decision.stage == "shadow":
                    # Теневой режим: исполнение запрещено, шаг не тратит
                    # бюджет инструментов. Здесь просто не идём дальше.
                    shadow_steps += 1
                    if shadow_steps >= 2:
                        stop_reason = (
                            f"ZaLoop: теневой режим не вывел из петли "
                            f"({zaloop_decision.reason})")
                        break
                if zaloop_decision.nudge:
                    nudge = zaloop_decision.nudge

            signature = record.signature()
            if signature == last_signature:
                repeats += 1
            else:
                repeats = 0
                last_signature = signature
            if repeats >= 2:
                stop_reason = (
                    f"зациклился: действие {record.verb} {record.object[:40]} "
                    f"повторено трижды"
                )
                break
            if repeats == 1:
                # Один повтор — повод объяснить модели, что данные уже собраны,
                # а не повод сдаваться: обычно она их просто не видит.
                nudge = (
                    f"ВАЖНО: действие {record.verb} {record.object[:60]} уже "
                    "выполнено, его результат выше. Не повторяй его — ответь "
                    "THINK: <итог> | {} или выбери ДРУГОЕ действие."
                )

            failures_in_a_row = 0 if record.success else failures_in_a_row + 1
            if failures_in_a_row >= 2:
                stop_reason = "две ошибки подряд — дальнейшие шаги бессмысленны"
                break

        for client, temperature in saved_temps:
            client.temperature = temperature
        after = self._usage()
        if budget_stop:
            # Задача не провалена — она не доведена. Возвращаем в очередь,
            # чтобы следующий прогон продолжил, а не начинал заново.
            status = PENDING
        report = RunReport(
            task_id=task.id,
            task_text=task.text,
            status=status,
            stop_reason=stop_reason,
            steps=steps,
            llm_calls=after["calls"] - before["calls"],
            tokens=after["tokens"] - before["tokens"],
            tokens_reported=after["unreported"] == before["unreported"],
            duration_s=round(time.time() - started, 2),
        )
        return report

    def run(self, max_tasks: int = 1, max_steps: int = 8,
            max_llm_calls: int = 6) -> List[RunReport]:
        """Пройти по очереди: не больше `max_tasks` задач и общего бюджета."""
        reports: List[RunReport] = []
        spent = 0
        while len(reports) < max_tasks:
            task = self.queue.pending()
            if task is None:
                break
            budget = max_llm_calls - spent
            if budget <= 0:
                break
            report = self.run_task(task, max_steps=max_steps, max_llm_calls=budget)
            spent += report.llm_calls
            # Счётчики обновляем ДО записи, чтобы в очереди оказался один
            # согласованный снимок, а не два подряд.
            task.steps += len(report.steps)
            task.llm_calls += report.llm_calls
            task.tokens += report.tokens
            self.queue.close(task, report.status, report.final_answer, report.stop_reason)
            reports.append(report)
        return reports