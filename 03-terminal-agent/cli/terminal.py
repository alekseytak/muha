"""Терминальный интерфейс агента МУХА (Rich + кассетный футуризм)."""
from __future__ import annotations

import os
import sys
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    from rich import box
    from rich.markup import escape
    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover — деградация без rich
    RICH_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.agent import MODES, Agent
from core.verb_act import VerbActGrammar, VerbActType
from core.gatekeeper import GateKeeper

# Rich принимает ТОЛЬКО реальные цвета/styles, поэтому символьные имена
# («amber», «warning», «success») транслируются в hex из config.yaml -> cli.colors.
DEFAULT_COLORS = {
    "primary": "#ffb000",
    "secondary": "#ff00ff",
    "success": "#5aff8a",
    "warning": "#ff4422",
    "info": "#94a3b8",
}

STYLE_ALIASES = {
    "amber": "primary",
    "primary": "primary",
    "secondary": "secondary",
    "success": "success",
    "warning": "warning",
    "danger": "warning",
    "info": "info",
    "white": None,
    "cyan": "info",
}


class TerminalUI:
    """Терминальный интерфейс агента."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.console = Console() if RICH_AVAILABLE else None
        self.palette = dict(DEFAULT_COLORS)
        self.palette.update(agent.config.get("cli", {}).get("colors", {}) or {})
        # Подтверждение опасных действий — через этот же интерфейс.
        agent.confirm_callback = self._confirm

    # ---------- отрисовка ----------

    def _style(self, name: str) -> str:
        key = STYLE_ALIASES.get(name)
        if key is None:
            return "white"
        return self.palette.get(key) or "white"

    def _print(self, text: str, style: str = "white") -> None:
        if self.console:
            # Пользовательский текст и id действий не должны трактоваться
            # как rich-markup: [381cd8b0] иначе съедается как цвет.
            self.console.print(escape(text), style=self._style(style))
        else:
            print(text)

    def _panel(self, title: str, content: str, style: str = "amber") -> None:
        if self.console:
            self.console.print(
                Panel(
                    escape(content),
                    title=escape(title),
                    border_style=self._style(style),
                    box=box.ROUNDED,
                )
            )
        else:
            print(f"--- {title} ---")
            print(content)
            print("---")

    def _llm_label(self, stats: Optional[dict] = None) -> str:
        """Честный статус: провайдер, модель и ключ — отдельно."""
        stats = stats if stats is not None else self.agent.stats()
        if not stats["llm_available"]:
            return "недоступен"
        if not stats.get("llm_model_present", False):
            return f"отвечает, но модель {stats['llm_model']} не подтверждена"
        return "готов"

    def _llm_warning(self) -> Optional[str]:
        """Текст предупреждения о LLM или None, если всё в порядке."""
        stats = self.agent.stats()
        provider_line = (
            f"провайдер: {stats['llm_provider']} ({stats['llm_api_style']}), "
            f"модель: {stats['llm_model']}"
        )
        fallback = stats.get("fallback")
        if stats["llm_available"] and stats.get("llm_model_present", False):
            return None
        if not stats["llm_available"]:
            lines = [f"LLM недоступна. {provider_line}"]
            if stats.get("llm_error"):
                lines.append(f"причина: {stats['llm_error']}")
            if fallback:
                lines.append(f"запасной провайдер: {fallback}")
            lines.append(
                "Ключ задаётся переменной окружения (см. llm.api_key_env в config.yaml) "
                "или файлом credentials_file."
            )
            return "\n".join(lines)
        try:
            models = ", ".join(self.agent.llm.list_models()[:10]) or "(список пуст)"
        except Exception:
            models = "(не удалось получить)"
        return (
            f"{provider_line}\n"
            f"модель не подтверждена; доступны: {models}\n"
            "Действия через LLM могут падать — используйте `run` для прямого вызова."
        )

    def _confirm(self, verb_act, spec) -> bool:
        """Спрашивает подтверждение перед изменяющим действие."""
        self._panel(
            "Требуется подтверждение",
            f"Инструмент: {spec.name} (риск: {spec.danger})\n"
            f"Действие: {verb_act.verb.value} {verb_act.object}\n"
            f"Параметры: {verb_act.parameters}\n"
            "Домен: " + VerbActGrammar.domain_for_object(verb_act.object),
            style="warning",
        )
        try:
            answer = input("Выполнить? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes", "д", "да", "1", "true", "approve")

    # ---------- команды ----------

    def _show_stats(self) -> None:
        stats = self.agent.stats()
        llm_status = self._llm_label(stats)
        gk = stats["gatekeeper"]
        reg = stats["registry"]
        ga = stats["golden_apples"]
        ho = stats["hormones"]
        ko = stats["kozyrev"]
        tools = stats["tools"]
        chain_status = "OK" if reg["chain_valid"] else "ОШИБКА"

        if self.console:
            table = Table(title="Статус агента МУХА", box=box.SIMPLE)
            table.add_column("Модуль", style="cyan")
            table.add_column("Значение", style="white")
            table.add_row("LLM", f"{stats['llm_provider']}/{stats['llm_model']} — {llm_status}")
            table.add_row("Роль", stats["role"])
            table.add_row("Режим", stats["mode"])
            table.add_row("Действий", str(stats["actions_total"]))
            table.add_row("Подтверждение", self.agent.confirm_policy)
            table.add_row("GateKeeper", f"ALLOW:{gk['allow']} DENY:{gk['deny']} ESC:{gk['escalate']}")
            table.add_row("Реестр", f"{reg['total_entries']} записей, цепь: {chain_status}")
            table.add_row("Инструменты", f"{tools['tools']} в реестре, {tools['calls']} вызовов")
            table.add_row("Золотые яблоки", f"{ga['canonical']} канон, {ga['pending']} в голосовании")
            table.add_row("Гормоны", f"Доф:{ho['dopamine']} Серот:{ho['serotonin']} Корт:{ho['cortisol']}")
            table.add_row("Риск-аппетит", str(ho["risk_appetite"]))
            table.add_row("Козырев", f"Парадоксов:{ko['total_events']} Эскалаций:{ko['escalated']}")
            self.console.print(table)
        else:
            print(f"LLM: {stats['llm_provider']}/{stats['llm_model']} ({llm_status})")
            print(f"Роль: {stats['role']} | Режим: {stats['mode']}")
            print(f"Действий: {stats['actions_total']}")
            print(f"Инструменты: {tools['tools']}, вызовов: {tools['calls']}")

    def _show_connectome(self, argument: str) -> None:
        raw = (argument or "stats").strip()
        params = {}
        # `connectome neighbors:0 {"limit": 3}` — отделяем хвостовой JSON.
        if raw.rstrip().endswith("}"):
            brace = raw.rfind("{")
            if brace > 0:
                try:
                    import json
                    params = json.loads(raw[brace:])
                    if not isinstance(params, dict):
                        params = {}
                    else:
                        raw = raw[:brace].strip()
                except Exception:
                    params = {}
        act = self.agent._make_act(
            VerbActType.READ, f"connectome:{raw}", params, raw, "cli:connectome"
        )
        result = self.agent.execute(act)
        self._panel(
            "Коннектом",
            str(result),
            style="success" if act.success else "warning",
        )

    def _run_direct(self, argument: str) -> None:
        """`run <глагол> <объект> [json-параметры]` — исполнение без LLM."""
        parts = argument.split(maxsplit=2)
        if not parts:
            self._print("Использование: run <глагол> <объект> [json-параметры]")
            return
        verb_raw = parts[0]
        verb = self.agent._coerce_verb(verb_raw)
        if verb is None:
            self._print(f"Неизвестный глагол: {verb_raw}")
            return
        obj = parts[1] if len(parts) > 1 else ""
        params = {}
        if len(parts) > 2:
            import json
            try:
                params = json.loads(parts[2])
                if not isinstance(params, dict):
                    params = {"value": params}
            except Exception:
                self._print(f"Параметры должны быть JSON-объектом: {parts[2]}")
                return
        act = self.agent._make_act(verb, obj, params, argument, "cli:run")
        self._panel("Действие", f"{act}\nДомен: {act.domain}\nПараметры: {act.parameters}")
        result = self.agent.execute(act)
        self._panel(
            "Результат",
            f"{result}",
            style="success" if act.success else "warning",
        )

    def run(self) -> None:
        """Запуск интерактивного цикла."""
        title = f"Муха-Агент v{self.agent.config.get('agent', {}).get('version', '1.1.0')}"
        if self.console:
            self.console.print(f"[bold {self._style('amber')}]{escape(title)}[/]")
        else:
            print(title)
        self._print("Терминальный агент на базе коннектома дрозофилы\n")

        if (warning_text := self._llm_warning()) is not None:
            self._panel("Внимание", warning_text, style="warning")
        else:
            self._print(f"LLM: {self.agent.llm.describe()}\n", style="success")

        while True:
            try:
                if self.console:
                    user_input = Prompt.ask(f"[bold {self._style('amber')}]муха>[/]")
                else:
                    user_input = input("муха> ")
            except (EOFError, KeyboardInterrupt):
                self._print("\nВыход из агента МУХА.")
                break

            if not user_input.strip():
                continue

            raw = user_input.strip()
            cmd = raw.lower()

            if cmd in ("quit", "exit", "q"):
                self._print("До свидания.")
                break

            if cmd == "stats":
                self._show_stats()
                continue

            if cmd == "help":
                self._show_help()
                continue

            if cmd == "tools":
                self._panel("Инструменты", self.agent.tools_roster())
                continue

            if cmd == "grammar":
                self._panel("Грамматика глаголов", VerbActGrammar.describe())
                continue

            if cmd == "clear":
                if self.console:
                    self.console.clear()
                else:
                    print("\n" * 50)
                continue

            if cmd.startswith("connectome"):
                self._show_connectome(raw[10:].strip())
                continue

            if cmd.startswith("run "):
                self._run_direct(raw[4:].strip())
                continue

            if cmd.startswith("confirm "):
                policy = cmd[8:].strip()
                if policy in ("ask", "allow", "deny"):
                    self.agent.confirm_policy = policy
                    self._print(f"Политика подтверждения: {policy}")
                else:
                    self._print("Использование: confirm <ask|allow|deny>")
                continue

            if cmd.startswith("role "):
                new_role = cmd[5:].strip()
                if new_role not in GateKeeper.ROLE_HIERARCHY:
                    self._print(
                        f"Неизвестная роль: {new_role}. Доступные: "
                        + ", ".join(GateKeeper.ROLE_HIERARCHY)
                    )
                else:
                    self.agent.role = new_role
                    self._print(f"Роль изменена на: {new_role}")
                continue

            if cmd == "task":
                stats = self.agent.task_queue.stats()
                self._print(
                    f"Задачи: {stats['total']} всего | ожидают {stats['pending']}, "
                    f"готово {stats['done']}, провалено {stats['failed']}"
                )
                for task in self.agent.task_queue.all()[-10:]:
                    self._print(
                        f"  [{task.id}] {task.status:8s} {task.text[:60]}"
                        + (f" :: {task.stop_reason[:50]}" if task.stop_reason else "")
                    )
                continue

            if cmd.startswith("task add "):
                text = raw.split("task add ", 1)[1].strip()
                try:
                    task = self.agent.task_queue.add(text)
                except ValueError as exc:
                    self._print(str(exc))
                else:
                    self._print(f"Задача добавлена: [{task.id}] {task.text}")
                continue

            if cmd == "auto" or cmd.startswith("auto "):
                limits = self.agent.autonomous_limits
                calls = limits["max_llm_calls"]
                if cmd.startswith("auto "):
                    try:
                        calls = int(cmd[5:].strip())
                    except ValueError:
                        self._print("Использование: auto [<бюджет вызовов модели>]")
                        continue
                if self.agent.mode != "autonomous":
                    self._print(
                        f"Сейчас режим {self.agent.mode}. Автономный прогон имеет смысл "
                        f"в режиме autonomous: выполните mode autonomous"
                    )
                if self.agent.task_queue.pending() is None:
                    self._print("Очередь пуста: добавьте задачу командой task add <текст>")
                    continue
                self._print(
                    f"Автономный прогон: до {limits['max_tasks']} задач, "
                    f"до {limits['max_steps']} шагов каждая, бюджет {calls} вызовов модели"
                )
                reports = self.agent.run_autonomous(max_llm_calls=calls)
                for report in reports:
                    self._print(f"  {report.summary()}")
                    for step in report.steps:
                        mark = "ок" if step.success else "отказ"
                        self._print(
                            f"    {step.index}. {step.verb} {step.object[:50]} — {mark}"
                        )
                if not reports:
                    self._print("  ничего не выполнено")
                continue

            if cmd == "providers":
                self._print("Проверяю провайдеров реальным запросом…")
                for item in self.agent.probe_providers():
                    verdict = "работает" if item["works"] else "НЕ РАБОТАЕТ"
                    reason = item.get("reason", "")
                    self._print(
                        f"  {item['role']}: {item.get('provider') or '—'}"
                        f"/{item.get('model') or '—'} — {verdict}"
                        + (f" ({reason})" if reason else "")
                    )
                continue

            if cmd == "mode":
                self._print(f"Режим: {self.agent.mode} (доступны: {', '.join(MODES)})")
                continue

            if cmd.startswith("mode "):
                new_mode = cmd[5:].strip()
                try:
                    self.agent.set_mode(new_mode)
                except ValueError as exc:
                    self._print(str(exc))
                else:
                    self._print(f"Режим изменён на: {new_mode}")
                continue

            if cmd.startswith("vote "):
                parts = raw.split()
                if len(parts) >= 3:
                    apple_id = parts[1]
                    approve = parts[2].lower() in ("yes", "y", "1", "true", "approve")
                    voter = parts[3] if len(parts) >= 4 else self.agent.role
                    success = self.agent.vote_golden_apple(apple_id, approve, voter)
                    self._print(
                        f"Голос {'принят' if success else 'не принят'}: {apple_id} -> "
                        f"{'одобрено' if approve else 'отклонено'} (голосующий: {voter})"
                    )
                else:
                    self._print("Использование: vote <id_яблока> <yes|no> [голосующий]")
                continue

            if cmd.startswith("apple "):
                proposal = raw[6:].strip()
                if proposal:
                    apple_id = self.agent.propose_golden_apple(proposal)
                    self._print(f"Предложено золотое яблоко: {apple_id}")
                else:
                    self._print("Использование: apple <предложение>")
                continue

            if cmd.startswith("reflect "):
                description = raw[8:].strip()
                result = self.agent.reflect(description)
                self._panel(
                    "Рефлексия (Зеркало Козырева)",
                    f"Уровень парадокса: {result['paradox_level']}\n"
                    f"Эскалация: {'ДА' if result['escalated'] else 'нет'}\n"
                    f"Описание: {result['description']}",
                )
                continue

            # Обычное действие через LLM
            verb_act = self.agent.propose_action(raw)
            self._panel(
                "Предложение действия",
                f"{verb_act}\nДомен: {verb_act.domain}\n"
                f"Параметры: {verb_act.parameters}\n"
                f"Источник: {verb_act.parse_reason}",
            )
            result = self.agent.execute(verb_act)
            self._panel(
                "Результат",
                str(result),
                style="success" if verb_act.success else "warning",
            )

    def _show_help(self) -> None:
        help_text = f"""Доступные команды:
  quit / exit / q     — выйти из агента
  help                — эту справку
  stats               — статистика всех модулей
  tools               — список инструментов с уровнями риска
  grammar             — матрица домен x глагол
  connectome [запрос] — данные коннектома: stats | classes | find:X | neighbors:X | <id>
  run <глагол> <объект> [json] — прямое исполнение без LLM
  confirm <ask|allow|deny>     — политика подтверждения опасных действий
  role <имя>          — сменить роль ({'/'.join(GateKeeper.ROLE_HIERARCHY)})
  mode <имя>          — сменить режим ({'/'.join(MODES)})
  providers           — реальная проверка провайдеров (ключ, модели, ответ)
  task / task add <т> — очередь задач автономного режима
  auto [<бюджет>]     — автономный прогон очереди (по умолчанию бюджет из конфига)
  vote <id> <yes|no> [голосующий] — голосовать за золотое яблоко
  apple <текст>       — предложить золотое яблоко
  reflect <текст>     — рефлексия через Зеркало Козырева
  clear               — очистить экран

  Любой другой текст — передаётся LLM как запрос на действие.

Примеры прямых вызовов:
  run read file:config.yaml
  run read connectome:stats
  run read connectome:neighbors:12 {{"limit": 5}}
  run write file:/tmp/muha-note.txt {{"content": "привет"}}
  run execute sys:uname -a
  run select _ {{"options": ["a","b"], "strategy": "first"}}
"""
        self._panel("Справка", help_text)


def main() -> None:
    """Точка входа."""
    from pathlib import Path

    argv = sys.argv[1:]
    confirm_policy: Optional[str] = None
    for arg in argv:
        if arg.startswith("--confirm="):
            confirm_policy = arg.split("=", 1)[1]
        elif arg in ("--yes", "-y"):
            confirm_policy = "allow"
        elif arg == "--no-confirm":
            confirm_policy = "deny"

    config_path = "config.yaml"
    if not Path(config_path).exists():
        config_path = str(Path(__file__).parent.parent / "config.yaml")

    agent = Agent(config_path=config_path, confirm_policy=confirm_policy)
    TerminalUI(agent).run()


if __name__ == "__main__":
    main()
