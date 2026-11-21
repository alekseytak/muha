"""Основной цикл агента МУХА."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .llm_client import LLMClient, LLMMessage, LLMUnavailable
from .verb_act import VerbAct, VerbActType, VerbActGrammar
from .gatekeeper import GateKeeper, GateDecision, GateResult
from .registry import Registry
from .tasks import TaskQueue
from .autonomous import AutonomousRunner
from .hierarchy import HierarchicalCascade
from .micromodels import cue_verb
from .readiness import (DEICTIC_RE, NEGATION_IMPERSONAL_RE,
                        NEGATION_STEM_RE,
                        SEQUENCE_RE, SHELL_META_RE, UNSUPPORTED_RE,
                        action_targets,
                        canonicalize_object, check_readiness,
                        distinct_action_verbs, extract_append, intent_guard,
                        object_of_registry)
from .micromodels import MicroCascade
from modules.web import WebFetcher
from .golden_apples import GoldenApples
from .swarm import Swarm
from .swarm_agents import FlySwarm
from .hormones import Hormones
from .kozyrev_mirror import KozyrevMirror

from modules.connectome import Connectome
from modules.filesystem import FsSandbox, ToolError
from modules.system import (
    SystemSandbox,
    set_backend as set_commands_backend,
    set_enabled as set_commands_enabled,
)
from tools.registry import ToolContext, ToolSpec, ToolRegistry, build_default_registry

ConfirmCallback = Callable[[VerbAct, ToolSpec], bool]


# Режимы работы. Ключ `agent.mode` в config.yaml управляет тем, что агенту
# вообще позволено, независимо от роли и гейта.
MODES: tuple[str, ...] = ("interactive", "read-only", "autonomous")

# В read-only запрещены действия, меняющие состояние.
READ_ONLY_FORBIDDEN = ("write", "execute", "anchor", "veto")

# Так помечается разбор, при котором модель не выдала формат и парсер
# подставил THINK. Это НЕ ответ модели, и считать задачу решённой нельзя.
FORMAT_FAILURE = "llm не вернул формат"


class Agent:
    """Муха-Агент — терминальный агент с MUHA архитектурой."""

    LLM_ERROR_PREFIX = "[Ошибка LLM: "

    def __init__(
        self,
        config_path: str = "config.yaml",
        confirm_policy: Optional[str] = None,
    ) -> None:
        from .miniyaml import load_file

        self.config_path = Path(config_path).resolve()
        self.project_root = self.config_path.parent
        self.config = load_file(self.config_path) or {}

        # --- LLM и запасной провайдер ---
        llm_cfg = self.config.get("llm", {}) or {}
        self.llm = self._build_llm(llm_cfg)
        self.fallback_llm: Optional[LLMClient] = None
        fallback_cfg = llm_cfg.get("fallback") or {}
        if fallback_cfg:
            self.fallback_llm = self._build_llm(
                {**fallback_cfg, "credentials_file": llm_cfg.get("credentials_file")}
            )

        # --- MUHA модули ---
        gate_cfg = self.config.get("gatekeeper", {}) or {}
        self.gatekeeper = GateKeeper(
            current_role=gate_cfg.get("default_role", "decider"),
            roles=gate_cfg.get("roles"),
        )

        auto_cfg = self.config.get("autonomous", {}) or {}
        self.autonomous_limits = {
            "max_tasks": int(auto_cfg.get("max_tasks", 3)),
            "max_steps": int(auto_cfg.get("max_steps", 8)),
            "max_llm_calls": int(auto_cfg.get("max_llm_calls", 6)),
        }
        tasks_path = auto_cfg.get("tasks_path", "./memory/tasks.jsonl")
        self.task_queue = TaskQueue(self._resolve(tasks_path))

        # Каскад микромоделей: экономный первый шаг перед моделью. Обучается
        # на размеченных кейсах из репозитория; при малом числе примеров и на
        # незнакомых фразах обязан воздерживаться (см. core/micromodels.py).
        micro_cfg = self.config.get("micromodels", {}) or {}
        self.micromodels_enabled = bool(micro_cfg.get("enabled", True))
        self.micromodels_floor = float(micro_cfg.get("confidence_floor", 0.55))
        # Иерархия «уровень L0–L5 -> глагол» вместо плоского выбора. Покрытие
        # 100% против 92% при точности 0.987 против 0.986 (eval/hierarchical.py,
        # cluster bootstrap). Оговорка: вся прибавка получена правилами μ1a,
        # ветка уровней не сработала ни разу, поэтому это расширение политики
        # для veto/escalate, а не выигрыш обученной иерархии.
        # Плоский каскад доступен флагом hierarchical: false.
        self.micromodels_hierarchical = bool(micro_cfg.get("hierarchical", True))
        if self.micromodels_hierarchical:
            self.cascade = HierarchicalCascade(floor=self.micromodels_floor)
        else:
            self.cascade = MicroCascade(floor=self.micromodels_floor)
        self.micro_stats = {"decided": 0, "escalated": 0}
        self._train_cascade()
        self.runner = AutonomousRunner(
            self, self.task_queue,
            compact_prompt=bool(auto_cfg.get("compact_prompt", False)),
        )
        # Разрешение менять состояние без человека — отдельный флаг, а не
        # следствие gatekeeper.confirm: тот отвечает на вопрос «спрашивать ли»,
        # а не «можно ли действовать без человека».
        self.autonomous_pre_approved = bool(auto_cfg.get("pre_approved", False))

        reg_cfg = self.config.get("registry", {}) or {}
        self.registry = Registry(path=self._resolve(reg_cfg.get("path", "./memory/registry.jsonl")))

        ga_cfg = self.config.get("golden_apples", {}) or {}
        self.golden_apples = GoldenApples(
            vote_threshold=ga_cfg.get("vote_threshold", 0.66),
            min_votes=ga_cfg.get("min_votes", 3),
        )

        swarm_cfg = self.config.get("swarm", {}) or {}
        self.swarm = Swarm(min_agents=swarm_cfg.get("min_agents", 3))

        harm_cfg = self.config.get("hormones", {}) or {}
        self.hormones = Hormones(
            dopamine_base=harm_cfg.get("dopamine_base", 0.5),
            serotonin_base=harm_cfg.get("serotonin_base", 0.5),
            cortisol_base=harm_cfg.get("cortisol_base", 0.1),
            risk_appetite_base=harm_cfg.get("risk_appetite_base", 0.5),
        )

        koz_cfg = self.config.get("kozyrev", {}) or {}
        self.kozyrev = KozyrevMirror(paradox_threshold=koz_cfg.get("paradox_threshold", 0.66))

        # --- Исполнительный слой: песочницы и инструменты ---
        tools_cfg = self.config.get("tools", {}) or {}
        self.fs_sandbox = FsSandbox(
            root=self._resolve(tools_cfg.get("filesystem_root", ".")),
            max_read_bytes=int(tools_cfg.get("max_read_bytes", 65536)),
            max_write_bytes=int(tools_cfg.get("max_write_bytes", 262144)),
        )
        # Флаг и бэкенд: posix не требует C-расширения _posixsubprocess.
        set_commands_enabled(bool(tools_cfg.get("commands_enabled", True)))
        try:
            set_commands_backend(str(tools_cfg.get("commands_backend", "posix")))
        except Exception:
            set_commands_backend("posix")
        self.system_sandbox = SystemSandbox(
            allowed_commands=tuple(tools_cfg.get("allowed_commands", ())),
            denied_patterns=tuple(str(p) for p in tools_cfg.get("denied_patterns", ())),
            timeout=int(tools_cfg.get("command_timeout", 10)),
            privileged_commands=tuple(tools_cfg.get("privileged_commands", ())),
            privileged_roles=tuple(
                tools_cfg.get("privileged_roles", ("guardian", "registrar"))
            ),
            # Команды и файловые инструменты должны смотреть в ОДИН корень.
            # Раньше здесь стоял project_root, а FsSandbox — filesystem_root:
            # «прочитай file:notes.md» находил файл, а «wc -l notes.md» нет,
            # потому что процесс стартовал в другом каталоге.
            cwd=self.fs_sandbox.root,
        )

        conn_cfg = self.config.get("connectome", {}) or {}
        self.connectome = Connectome.open(self._resolve(conn_cfg.get("path", "../var/flywire-role/")))
        self.connectome_enabled = bool(conn_cfg.get("enabled", True))

        tools_cfg = self.config.get("tools", {}) or {}
        tool_ctx = ToolContext(
            fs=self.fs_sandbox,
            system=self.system_sandbox,
            connectome=self.connectome,
            project_root=self.project_root,
            owner=self,
        )
        tool_ctx.web_fetcher = WebFetcher(
            timeout=int(tools_cfg.get("web_timeout", 15)),
            max_bytes=int(tools_cfg.get("web_max_bytes", 2_000_000)),
        )
        self.web_max_chars = int(tools_cfg.get("web_max_chars", 20000))
        # Браузер: путь и таймаут берём из конфига, но не падаем, если бинаря нет.
        from modules import web as web_module
        web_module.STATE["browser_timeout"] = int(tools_cfg.get("browser_timeout", 30))
        if tools_cfg.get("browser_binary"):
            import os
            os.environ["OBSCURA_BIN"] = os.path.expanduser(
                str(tools_cfg["browser_binary"]))
        self.tools: ToolRegistry = build_default_registry(tool_ctx)

        # --- Состояние ---
        self.running = False
        self.action_history: List[VerbAct] = []
        self.role = gate_cfg.get("default_role", "decider")
        mode = str((self.config.get("agent") or {}).get("mode", "interactive")).strip()
        if mode not in MODES:
            raise ValueError(
                f"agent.mode={mode!r} неизвестен; допустимо: {', '.join(MODES)}"
            )
        self.mode = mode
        self.confirm_policy = (confirm_policy or gate_cfg.get("confirm", "ask")).lower()
        self.confirm_callback: Optional[ConfirmCallback] = None
        self.seen_prompts: List[str] = []
        # Счётчик ставит _train_cascade до этой строки — не затирать.
        self.cascade_trained_on = getattr(self, "cascade_trained_on", 0)
        # Рой создаётся после обучения каскада: мухи голосуют по нему.
        # Движки разные, иначе голосование бессмысленно.
        engines = {"полный": self.cascade}
        if getattr(self, "cascade_rules", None) is not None:
            engines["правила"] = self.cascade_rules
        if getattr(self, "cascade_stats", None) is not None:
            engines["статистика"] = self.cascade_stats
        self.fly_swarm = FlySwarm(engines)

        # --- Доступность LLM ---
        self.llm_available = False
        self.llm_model_present = False
        self.llm_error: Optional[str] = None
        self._check_llm()

    # ---------- инфраструктура ----------

    def _resolve(self, value: str) -> Path:
        """Путь из конфига: относительные считаются от каталога проекта."""
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _build_llm(self, cfg: Dict[str, Any]) -> LLMClient:
        return LLMClient(
            base_url=cfg.get("base_url", "https://api.oneprovider.dev/v1"),
            model=cfg.get("model", "deepseek-v4.1-flash"),
            provider=cfg.get("provider", "oneprovider"),
            temperature=float(cfg.get("temperature", 0.1)),
            max_tokens=int(cfg.get("max_tokens", 512)),
            timeout=int(cfg.get("timeout", 60)),
            api_key=cfg.get("api_key"),
            api_key_env=cfg.get("api_key_env"),
            credentials_file=cfg.get("credentials_file"),
            tls_verify=bool(cfg.get("tls_verify", True)),
            ca_bundle=cfg.get("ca_bundle"),
        )

    def _check_llm(self) -> None:
        """Проверяет доступность провайдера и наличие модели."""
        self.llm_available = False
        self.llm_model_present = False
        self.llm_error = None
        try:
            self.llm_available = self.llm.is_available()
        except Exception as exc:
            self.llm_error = f"{type(exc).__name__}: {exc}"
            return
        if not self.llm_available:
            if self.llm.api_style == "openai" and not self.llm.api_key:
                self.llm_error = (
                    f"нет ключа: задайте {self.llm.api_key_env} "
                    f"или api_key в {self.llm.credentials_file}"
                )
            else:
                self.llm_error = "провайдер не отвечает"
            return
        try:
            self.llm_model_present = self.llm.ensure_model_available()
        except Exception:
            self.llm_model_present = False

    # ---------- мышление ----------

    def _build_system_prompt(self) -> str:
        return f"""Ты — Муха-Агент, терминальный агент с архитектурой MUHA.
Твоя задача — превратить просьбу пользователя в ОДНО действие формата:

ГЛАГОЛ: объект | {{"параметр": "значение"}}

Глаголы и когда их выбирать:
  READ      — показать содержимое: файл, данные коннектома, записи реестра.
  SELECT    — выбрать вариант ИЛИ получить метаданные объекта (размер, тип,
              права). Отличается от READ: READ даёт содержимое, SELECT — выбор.
  WRITE     — создать или изменить файл; нужен параметр content.
  EXECUTE   — запустить разрешённую команду. Опасную команду всё равно разбирай
              как EXECUTE: отказ делает гейт, а не ты.
  LOG       — оставить пометку в журнале/реестре («запиши в журнал», «отметь»).
              Не путать с WRITE: журнал — это reg:, а не файл.
  VERSION   — узнать версию агента, состояние реестра («какая версия»).
  ANCHOR    — закрепить/канонизировать запись в реестре.
  VETO      — наложить вето на предложение роя.
  ESCALATE  — передать решение на уровень выше, если не хватает полномочий.
  RECOMMEND — совет, рекомендация, «что делать дальше».
  THINK     — ответ словами, когда действие не требуется или данных не хватает.

Домены задаются префиксом объекта:
  file:<путь> / dir:<путь>  — файлы (чтение, листинг, запись)
  sys:<команда>             — запуск разрешённой команды
  connectome:<подкоманда>   — данные коннектома FlyWire
  reg:<подкоманда>          — реестр: list, search:<текст>, <текст>
  swarm:<текст>             — рой (veto/escalate)
  без префикса              — домен ui (select/recommend/escalate)

Подкоманды коннектома: stats, classes, find:<текст>, neighbors:<id>, <id>.
Подкоманды реестра: list (последние записи), search:<текст>.

Примеры:
  READ: file:config.yaml | {{}}
  READ: connectome:stats | {{}}
  READ: connectome:neighbors:12 | {{"direction": "out", "limit": 5}}
  READ: reg:list | {{"limit": 5}}
  SELECT: file:config.yaml | {{}}
  SELECT: options | {{"options": ["a", "b"], "strategy": "first"}}
  WRITE: file:notes/out.txt | {{"content": "текст"}}
  EXECUTE: sys:uname -a | {{}}
  LOG: reg:проверка запуска | {{}}
  VERSION: reg:version | {{}}
  ANCHOR: reg:канон-1 | {{}}
  VETO: swarm:спорное предложение | {{}}
  ESCALATE: swarm:вопрос к хранителю | {{}}
  RECOMMEND: план на день | {{}}

Правила:
1. Отвечай ТОЛЬКО одной строкой указанного формата, без пояснений и markdown.
2. Если действие не требуется — ответь: THINK: <краткий ответ> | {{}}
3. Не выдумывай пути и идентификаторы: если данных нет, используй THINK.
4. Слова «тип», «размер», «сколько места» про файл — это SELECT.
5. «Журнал», «реестр», «отметь», «запиши в лог» — это LOG, а не WRITE.
6. Отвечай по-русски."""

    def set_sandbox_root(self, root) -> None:
        """Один корень для файлов И для команд.

        Две песочницы легко расходятся: файловые инструменты читают из
        `fs_sandbox.root`, а процесс стартует в `system_sandbox.cwd`. Если
        двигать только первую, «прочитай file:x» работает, а «wc -l x» — нет.
        """
        from pathlib import Path as _Path
        path = _Path(root)
        self.fs_sandbox.root = path
        self.system_sandbox.cwd = path

    def perceive(self, image: str, model: Optional[str] = None):
        """Посмотреть на картинку и вернуть наблюдение.

        Зрение — это восприятие, а не действие: здесь ничего не исполняется.
        Наблюдение превращается в текст задачи, и дальше решает та же
        грамматика глаголов, что и для набранной руками фразы.
        """
        from .vision import VISION_MODEL, perceive as _perceive

        if self.llm is None:
            raise RuntimeError("зрение требует модели: llm не настроен")
        return _perceive(self.llm, image, model or VISION_MODEL)

    def see(self, image: str):
        """Локальное зрение: вектор признаков без сети и без денег.

        Это не пересказ картинки словами, а её измерение: контраст, края,
        полосы текста, монохром. Дёшево настолько, что можно звать на каждом
        шаге, и не требует ни одной модели.
        """
        from .retina import look

        return look(image)

    def propose_from_image(self, image: str, model: Optional[str] = None):
        """Увидеть картинку и предложить действие. (наблюдение, действие)."""
        observation = self.perceive(image, model)
        phrase = observation.as_phrase()
        act = self.propose_action(phrase)
        return observation, act

    def propose_action_swarm(self, user_input: str) -> Optional[VerbAct]:
        """Разбор роем: несколько мух голосуют, омега — только при расколе.

        Отличие от `propose_action_fast` не в скорости, а в том, кто решает:
        там глагол берётся из одного каскада, здесь — из голосования мух с
        разными порогами и разной готовностью рисковать. Если рой не сошёлся,
        возвращается None: решение уходит омеге, а не выдумывается.
        """
        if not self.micromodels_enabled:
            return None
        if self._phrase_guard(user_input):
            self.micro_stats["not_ready"] = self.micro_stats.get("not_ready", 0) + 1
            return None
        decision = self.fly_swarm.vote(user_input)
        self.micro_stats["swarm_votes"] = self.micro_stats.get("swarm_votes", 0) + 1
        if decision.verb is None:
            self.micro_stats["escalated"] = self.micro_stats.get("escalated", 0) + 1
            return None
        verb = self._coerce_verb(decision.verb)
        if verb is None:
            return None
        act = self._gate_and_build(
            verb, user_input, decision.object, decision.domain,
            f"рой ({decision.agreement:.0%}, {decision.spoken} из "
            f"{len(decision.votes)} мух)",
        )
        if act is None:
            self.micro_stats["not_ready"] = self.micro_stats.get("not_ready", 0) + 1
        return act

    def _train_cascade(self) -> None:
        """Обучение микромоделей на размеченных кейсах, если они есть.

        Корпус задаётся `micromodels.training_corpus`. Раньше путь был
        зашит на `eval/cases.json` (39 фраз), а все замеры делались на
        `eval/cases_distill.json` — то есть опубликованное качество
        описывало не тот каскад, который запускается.

        Берутся ЧЁТНЫЕ семейства: нечётные остаются контрольными, поэтому
        замеры на 152 фразах остаются замерами, а не подгонкой.
        """
        micro_cfg = self.config.get("micromodels", {}) or {}
        corpus = str(micro_cfg.get("training_corpus")
                     or "./eval/cases_distill.json")
        cases_path = self._resolve(corpus)
        pairs = []
        if cases_path.exists():
            try:
                payload = json.loads(cases_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = {}
            cases = payload.get("cases") or payload or []
            labeled = [c for c in cases
                       if (c.get("prompt") or c.get("phrase"))
                       and (c.get("verbs") or c.get("expected_verbs"))]
            # Деление ПО СЕМЕЙСТВАМ, а не по индексам. При делении по
            # индексам половина фраз каждого шаблона попадала в обучение, и
            # «контроль» на 96% состоял из родственников обучающих фраз.
            families = sorted({c.get("family") for c in labeled
                               if c.get("family")})
            if families:
                train_families = {name for index, name in enumerate(families)
                                  if index % 2 == 0}
                chosen = [c for c in labeled
                          if c.get("family") in train_families]
            else:
                chosen = labeled[::2]
            for case in chosen:
                phrase = case.get("prompt") or case.get("phrase")
                verbs = case.get("verbs") or case.get("expected_verbs")
                pairs.append((phrase, verbs[0]))

        # Регрессионный набор добавляется всегда: он маленький и содержит
        # формулировки, на которых каскад уже ошибался.
        regression = self.project_root / "eval" / "cases.json"
        if regression.exists():
            try:
                payload = json.loads(regression.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = {}
            for case in (payload.get("cases") or payload or []):
                phrase = case.get("prompt") or case.get("phrase") or ""
                verbs = case.get("verbs") or case.get("expected_verbs") or []
                if phrase and verbs:
                    pairs.append((phrase, verbs[0]))
        if pairs:
            self.cascade.fit(pairs)
            self.cascade_trained_on = len(pairs)
            # Отдельные движки для роя. Это не копии одного каскада: один
            # верит только правилам-уликам, второй только статистике n-грамм.
            # Без этого рой вырождается в одну муху — проверено замером:
            # на 152 фразах все мухи голосовали одинаково.
            self.cascade_rules = MicroCascade(floor=self.micromodels_floor,
                                              use_stats=False).fit(pairs)
            self.cascade_stats = MicroCascade(floor=self.micromodels_floor,
                                              use_rules=False).fit(pairs)

    def propose_action_fast(self, user_input: str) -> Optional[VerbAct]:
        """Экономный разбор: каскад микромоделей, без вызова модели.

        Возвращает None, если каскад воздерживается — тогда решает омега
        (большая модель). Это не отказ, а маршрутизация: экономия не должна
        покупаться ошибками.
        """
        if not self.micromodels_enabled:
            return None
        decision = self.cascade.decide(user_input)
        if decision is None:
            return None
        verb = self._coerce_verb(decision.verb)
        if verb is None:
            return None

        blocked = self._phrase_guard(user_input)
        if blocked:
            self.micro_stats["not_ready"] = self.micro_stats.get("not_ready", 0) + 1
            return None
        return self._gate_and_build(
            verb, user_input, decision.object, decision.domain,
            f"микромодель ({decision.source}, уверенность {decision.confidence:.2f})",
        )

    def _phrase_guard(self, phrase: str) -> Optional[str]:
        """Проверки уровня фразы, не зависящие от выбранного глагола.

        Возвращает причину отказа или None. Вынесено отдельным методом,
        потому что эти же проверки нужны и рою: голосовать по фразе, которая
        заведомо не является приказом, бессмысленно.
        """
        # Отрицание проверяется независимо от того, кто выбрал глагол:
        # статистика тоже могла ошибиться, а «не сохраняй» — это запрет.
        if (NEGATION_STEM_RE.search(phrase)
                or NEGATION_IMPERSONAL_RE.search(phrase)):
            return "отрицание: фраза запрещает действие, а не просит его"
        # Оболочки нет: цепочки и подстановки не поддерживаются.
        if SHELL_META_RE.search(phrase):
            return "оболочка: цепочки и подстановки не поддерживаются"
        # Глаголов вне набора из одиннадцати локальный роутер не решает.
        if UNSUPPORTED_RE.search(phrase):
            return "глагол вне набора одиннадцати"
        # Местоимение без объекта — не приказ, а недостающая информация.
        if DEICTIC_RE.search(phrase) and not action_targets(phrase):
            return "местоимение без предмета: непонятно, о чём речь"
        several = distinct_action_verbs(phrase, cue_verb)
        if (len(several) > 1 or len(action_targets(phrase)) > 1
                or SEQUENCE_RE.search(phrase)):
            return "фраза требует нескольких действий"
        return None

    def _gate_and_build(self, verb, phrase: str, obj: Optional[str],
                        domain_hint: Optional[str],
                        source: str) -> Optional[VerbAct]:
        """Собрать действие и проверить, что его вообще можно исполнять.

        Общий путь для одиночного агента и для роя: граница готовности должна
        быть одна, иначе рой сможет предложить то, что одиночный путь уже
        отсекает.
        """
        verb_name = verb.value if hasattr(verb, "value") else str(verb)
        allowed, _reason = intent_guard(phrase, verb_name)
        if not allowed:
            return None

        act = self._make_act(verb, obj or phrase, {}, phrase, source)
        # Домен: если объект несёт префикс (file:/sys:/connectome:/web:…), его
        # домен определён грамматикой детерминированно — верить каскаду там
        # не нужно. Доменная метка омеги шумная, а ошибка домена меняет набор
        # прав в GateKeeper, то есть это не косметика.
        if act.domain == "ui" and domain_hint:
            act.domain = domain_hint
        # Канонизация до проверки прав: объект должен нести префикс своего
        # домена, иначе GateKeeper выведет домен из объекта заново и получит
        # другую пару «глагол/домен», чем исполнитель.
        if act.domain == "registry" and verb_name == "version":
            act.object = object_of_registry(phrase) or act.object
        act.object = canonicalize_object(act.domain, act.object)

        # Инструмент должен существовать для этой пары «глагол/домен».
        # Иначе проверка готовности молчит: «сохрани результат» получало
        # домен ui, для которого write-инструмента нет, и проходило насквозь.
        if self.tools.find(verb, act.domain) is None:
            return None

        # Готовность проверяется по ДОМЕНУ АКТА, а не по домену каскада:
        # раньше проверка видела «ui» и пропускала write без содержимого.
        ruled_verb, _ = cue_verb(phrase)
        ready, params, _why = check_readiness(
            verb_name, act.domain, phrase, act.object,
            confirmed_by_cue=(ruled_verb == verb_name))
        if not ready:
            return None
        act.parameters = params
        act.context = {**(act.context or {}), "micro": True,
                       "micro_source": source}
        return act

    def compact_planning_prompt(self) -> str:
        """Короткий промпт для шага-продолжения автономного цикла.

        Полный промпт (2630 символов) описывает все глаголы с примерами — это
        нужно на первом шаге и в интерактиве. Но он уходит в модель НА КАЖДОМ
        шаге, а в цикле шагов много: замер показал 876 токенов системного
        промпта из 1069 токенов запроса. Здесь остаётся только то, без чего
        модель не соберёт формат: список глаголов, префиксы доменов и правила.
        """
        return f"""Ты — Муха-Агент. Ответь ОДНОЙ строкой: ГЛАГОЛ: объект | {{"параметр": "значение"}}

Глаголы: {", ".join(v.value for v in VerbActType)}.
Префиксы объектов: file:<путь>, dir:<путь> — файлы; sys:<команда> — команда;
connectome:<stats|classes|find:<текст>|neighbors:<id>|<id>> — коннектом;
web:<адрес> или url:<адрес> — прочитать веб-страницу текстом (READ) или её
метаданные (SELECT);
reg:<list|search:<текст>|<текст>> — реестр; swarm:<текст> — рой; без префикса — ui.

READ — содержимое; SELECT — выбрать вариант или метаданные (размер, тип);
WRITE — файл, нужен content; LOG — пометка в журнале (reg:); VERSION — версия;
ANCHOR — закрепить; VETO — вето; ESCALATE — передать наверх; RECOMMEND — совет;
EXECUTE — команда; THINK — ответ словами, когда действие не нужно.

Если задача решена — ответь THINK: <итог> | {{}}. Не выдумывай пути. По-русски."""

    def think(self, prompt: str, context: Optional[str] = None,
              system_prompt: Optional[str] = None) -> str:
        """Размышление через LLM с откатом на запасной провайдер."""
        messages: List[LLMMessage] = []
        if context:
            messages.append(LLMMessage(role="system", content=context))
        messages.append(LLMMessage(role="user", content=prompt))
        if system_prompt is None:
            system_prompt = self._build_system_prompt()

        errors: List[str] = []
        for client in (self.llm, self.fallback_llm):
            if client is None:
                continue
            try:
                response = client.generate(messages, system_prompt=system_prompt)
                content = (response.content or "").strip()
                if not content:
                    raise LLMUnavailable("пустой ответ модели")
                self.hormones.reward(0.02)
                return content
            except LLMUnavailable as exc:
                errors.append(f"{client.provider}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 — сеть/парсинг чужих ответов
                errors.append(f"{client.provider}: {type(exc).__name__}: {exc}")
                continue

        self.hormones.stress(0.05)
        self.llm_error = " | ".join(errors) if errors else "провайдеры не сконфигурированы"
        return f"{self.LLM_ERROR_PREFIX}{self.llm_error}]"

    # ---------- разбор ответа LLM ----------

    VERB_PATTERN = re.compile(
        r"^\s*(?P<verb>[A-Za-zА-Яа-я_]+)\s*:\s*(?P<rest>.+)$", re.MULTILINE
    )
    RU_VERBS = {
        "прочитай": "read", "прочитать": "read", "читай": "read", "read": "read",
        "запиши": "write", "записать": "write", "write": "write",
        "выполни": "execute", "запусти": "execute", "execute": "execute",
        "подумай": "think", "think": "think",
        "выбери": "select", "select": "select",
        "рекомендуй": "recommend", "recommend": "recommend",
        "эскалируй": "escalate", "escalate": "escalate",
        "вето": "veto", "veto": "veto",
        "якорь": "anchor", "anchor": "anchor",
        "лог": "log", "log": "log",
        "версия": "version", "version": "version",
    }

    def _strip_fences(self, text: str) -> str:
        cleaned = text.strip()
        fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", cleaned, re.DOTALL)
        if fence:
            return fence.group(1).strip()
        return cleaned

    def _coerce_verb(self, raw: str) -> Optional[VerbActType]:
        token = (raw or "").strip().strip("*`_ ").lower()
        if token in self.RU_VERBS:
            token = self.RU_VERBS[token]
        try:
            return VerbActType(token)
        except ValueError:
            return None

    def _parse_json_verb(self, text: str) -> Optional[Dict[str, Any]]:
        """Пытается разобрать JSON-объект с полями verb/object/parameters."""
        for candidate in (text, ):
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end <= start:
                continue
            try:
                data = json.loads(candidate[start:end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "verb" in data:
                return data
        return None

    def _parse_verb_act(self, response: str, user_input: str) -> VerbAct:
        """Разбирает ответ LLM в VerbAct. Неудача -> THINK с причиной."""
        text = self._strip_fences(response or "")

        data = self._parse_json_verb(text)
        if data:
            verb = self._coerce_verb(str(data.get("verb", "")))
            if verb is not None:
                params = data.get("parameters") or {}
                if not isinstance(params, dict):
                    params = {"value": params}
                obj = str(data.get("object") or data.get("target") or user_input)
                return self._make_act(verb, obj, params, user_input, "json")

        for match in self.VERB_PATTERN.finditer(text):
            verb = self._coerce_verb(match.group("verb"))
            if verb is None:
                continue
            rest = match.group("rest").strip()
            obj, params = rest, {}
            if "|" in rest:
                obj_part, params_part = rest.split("|", 1)
                obj = obj_part.strip().strip("*` ")
                raw_params = params_part.strip()
                if raw_params:
                    try:
                        parsed = json.loads(raw_params)
                        params = parsed if isinstance(parsed, dict) else {"value": parsed}
                    except json.JSONDecodeError:
                        params = {"raw": raw_params.strip("`* ")}
            obj = obj.strip("`* ")
            return self._make_act(verb, obj, params, user_input, "text")

        return self._make_act(
            VerbActType.THINK, user_input, {}, user_input,
            f"{FORMAT_FAILURE} ГЛАГОЛ: объект | {{}}",
        )

    def _make_act(
        self,
        verb: VerbActType,
        obj: str,
        params: Dict[str, Any],
        user_input: str,
        reason: str,
    ) -> VerbAct:
        act = VerbAct.create(
            verb=verb, object=obj, parameters=params,
            context={"user_input": user_input},
        )
        act.parse_reason = reason
        act.domain = VerbActGrammar.domain_for_object(obj)
        return act

    def propose_action(self, user_input: str) -> VerbAct:
        """Предложить действие: сначала микромодель, при её отказе — LLM."""
        self.seen_prompts.append(user_input)
        fast = self.propose_action_fast(user_input)
        if fast is not None:
            self.micro_stats["decided"] += 1
            return fast
        self.micro_stats["escalated"] += 1
        response = self.think(
            "Пользователь сказал: "
            f"{user_input}\n"
            f"Преобразуй в глагол-действие. Доступные глаголы: "
            f"{', '.join(v.value for v in VerbActType)}.\n"
            "Ответь только форматом: ГЛАГОЛ: объект | {\"параметр\": \"значение\"}"
        )
        return self._parse_verb_act(response, user_input)

    # ---------- гейт и исполнение ----------

    def gate(self, verb_act: VerbAct) -> GateDecision:
        """Проверить действие через GateKeeper."""
        if verb_act.domain is None:
            verb_act.domain = VerbActGrammar.domain_for_object(verb_act.object)
        result = self.gatekeeper.check(verb_act, user_role=self.role)
        verb_act.gate_decision = result.decision.value
        verb_act.gate_reason = result.reason
        return result.decision

    def _mode_block(self, verb_act: VerbAct) -> Optional[str]:
        """Причина отказа по режиму работы, если действие в нём недопустимо.

        * read-only — менять состояние нельзя: ни записи файлов, ни команд,
          ни закреплений и вето. Журналирование разрешено: это аудит.
        * autonomous — спросить человека не у кого, поэтому всё, что требует
          подтверждения, запрещается, а не подтверждается молча.
        """
        if self.mode == "read-only" and verb_act.verb.value in READ_ONLY_FORBIDDEN:
            return (
                f"Запрещено режимом read-only: глагол {verb_act.verb.value!r} "
                f"меняет состояние (разрешены read/select/think/log/recommend)"
            )
        if self.mode == "autonomous":
            spec = self.tools.find(verb_act.verb, verb_act.domain or "ui")
            if spec is not None and getattr(spec, "requires_confirmation", False):
                if not getattr(self, "autonomous_pre_approved", False):
                    return (
                        f"Запрещено режимом autonomous: инструмент {spec.name} требует "
                        f"подтверждения человека, которого в этом режиме нет "
                        f"(включить предодобрение: autonomous.pre_approved: true)"
                    )
                # Предодобрение выдано оператором в конфиге: выполняем, но
                # помечаем действие, чтобы его нельзя было спутать с тем,
                # что человек подтвердил вживую.
                verb_act.auto_approved = True
        return None

    def set_mode(self, mode: str) -> str:
        """Переключить режим с проверкой допустимости."""
        mode = mode.strip()
        if mode not in MODES:
            raise ValueError(f"режим {mode!r} неизвестен; допустимо: {', '.join(MODES)}")
        self.mode = mode
        return mode

    def needs_confirmation(self, verb_act: VerbAct) -> Optional[ToolSpec]:
        """Возвращает spec, если действие требует подтверждения."""
        spec = self.tools.find(verb_act.verb, verb_act.domain or "ui")
        if spec and spec.requires_confirmation:
            return spec
        return None

    def _ask_confirmation(self, verb_act: VerbAct, spec: ToolSpec) -> bool:
        if self.confirm_policy == "allow":
            return True
        if self.confirm_policy == "deny":
            return False
        if self.confirm_callback is None:
            return False
        return bool(self.confirm_callback(verb_act, spec))

    def execute(self, verb_act: VerbAct) -> Any:
        """Выполнить действие: гейт -> подтверждение -> инструмент -> реестр."""
        started = time.time()
        decision = self.gate(verb_act)
        domain = verb_act.domain or "ui"
        # Режим проверяется ПЕРЕД гейтом: это ограничение оператора, а не роль.
        # Иначе в read-only роль могла бы «эскалировать» к запрещённому действию.

        def record(result: str, success: bool, extra: Optional[Dict[str, Any]] = None) -> None:
            metadata = {
                "domain": domain,
                "tool": verb_act.tool,
                "duration_ms": int((time.time() - started) * 1000),
            }
            if verb_act.gate_reason:
                metadata["gate_reason"] = verb_act.gate_reason
            metadata.update(extra or {})
            self.registry.record(
                action=f"{verb_act.verb.value} {verb_act.object}",
                actor=verb_act.actor,
                result=result,
                success=success,
                metadata=metadata,
            )

        user_input = str((verb_act.context or {}).get("user_input") or "")
        intent_ok, intent_reason = intent_guard(user_input, verb_act.verb.value)
        if verb_act.verb == VerbActType.WRITE and (
                extract_append(user_input)
                or (verb_act.parameters or {}).get("mode") == "append"):
            verb_act.success = False
            verb_act.result = "Дописывание не поддерживается: перезапись вместо append запрещена"
            record(verb_act.result, False, {"unsupported_mode": "append"})
            return verb_act.result
        if user_input and verb_act.verb != VerbActType.THINK and not intent_ok:
            verb_act.success = False
            verb_act.gate_decision = "DENY"
            verb_act.result = f"Запрещено намерением пользователя: {intent_reason}"
            record(verb_act.result, False, {"gate": "DENY", "intent_blocked": True})
            return verb_act.result

        blocked = self._mode_block(verb_act)
        if blocked:
            verb_act.success = False
            verb_act.result = blocked
            record(blocked, False, {"mode": self.mode})
            self.hormones.stress(0.02)
            return blocked

        if decision == GateDecision.DENY:
            verb_act.success = False
            verb_act.result = f"Запрещено: {verb_act.gate_reason}"
            record(verb_act.result, False, {"gate": "DENY"})
            self.hormones.stress(0.05)
            return verb_act.result

        if decision == GateDecision.ESCALATE:
            verb_act.success = False
            verb_act.result = f"Эскалация требуется: {verb_act.gate_reason}"
            record(verb_act.result, False, {"gate": "ESCALATE", "escalation": True})
            self.hormones.stress(0.03)
            return verb_act.result

        spec = self.needs_confirmation(verb_act)
        if spec is not None and not verb_act.auto_approved \
                and not self._ask_confirmation(verb_act, spec):
            verb_act.success = False
            verb_act.result = f"Отменено: требуется подтверждение для инструмента {spec.name}"
            record(verb_act.result, False, {"cancelled": True})
            self.hormones.stress(0.02)
            return verb_act.result

        try:
            result = self._do_action(verb_act)
            verb_act.success = True
            verb_act.result = str(result)
            extra = {"auto_approved": True} if verb_act.auto_approved else None
            record(verb_act.result, True, extra)
            self.hormones.reward(0.05)
            self.action_history.append(verb_act)
            return result
        except (ToolError, RuntimeError) as exc:
            verb_act.success = False
            verb_act.result = f"Ошибка выполнения: {exc}"
            record(verb_act.result, False)
            self.hormones.stress(0.1)
            return verb_act.result
        except Exception as exc:  # noqa: BLE001 — непредвиденный сбой инструмента
            verb_act.success = False
            verb_act.result = f"Внутренняя ошибка {type(exc).__name__}: {exc}"
            record(verb_act.result, False, {"unexpected": True})
            self.hormones.stress(0.15)
            return verb_act.result

    def _do_action(self, verb_act: VerbAct) -> Any:
        """Выполнение действия через реестр инструментов."""
        if verb_act.verb == VerbActType.THINK:
            params = verb_act.parameters or {}
            result = self.think(str(params.get("prompt") or verb_act.object))
            if result.startswith(self.LLM_ERROR_PREFIX):
                # Отказ LLM не должен попадать в append-only реестр как успех.
                raise RuntimeError(result)
            return result

        domain = verb_act.domain or VerbActGrammar.domain_for_object(verb_act.object)
        result = self.tools.execute(
            verb_act.verb, domain, verb_act.object, verb_act.parameters or {}
        )
        verb_act.tool = result.tool
        if not result.ok:
            raise ToolError(result.output)
        return result.output

    # ---------- прочие операции ----------

    def run_autonomous(self, max_tasks: Optional[int] = None,
                       max_steps: Optional[int] = None,
                       max_llm_calls: Optional[int] = None) -> List[Any]:
        """Прогнать очередь задач автономно, в пределах бюджетов.

        Ничего не выполняет в обход обычного пути: каждое действие идёт через
        режим, гейт, подтверждение (или предодобрение) и реестр.
        """
        limits = self.autonomous_limits
        return self.runner.run(
            max_tasks=limits["max_tasks"] if max_tasks is None else max_tasks,
            max_steps=limits["max_steps"] if max_steps is None else max_steps,
            max_llm_calls=(
                limits["max_llm_calls"] if max_llm_calls is None else max_llm_calls
            ),
        )

    def reflect(self, description: str) -> Dict[str, Any]:
        """Рефлексия через Зеркало Козырева."""
        h = self.hormones.state
        event = self.kozyrev.reflect(
            state_description=description,
            hormone_dopamine=h.dopamine,
            hormone_cortisol=h.cortisol,
            risk_appetite=h.risk_appetite,
        )
        return {
            "paradox_level": round(event.level, 4),
            "escalated": event.escalated,
            "description": event.description,
        }

    def propose_golden_apple(self, proposal: str) -> str:
        """Предложить золотое яблоко."""
        apple_id = self.golden_apples.propose(proposal=proposal, proposer=self.role)
        self.registry.record(
            action=f"golden_apple:propose {proposal[:50]}",
            actor=self.role,
            result=apple_id,
            success=True,
        )
        return apple_id

    def vote_golden_apple(
        self, apple_id: str, approve: bool, voter_id: Optional[str] = None
    ) -> bool:
        """Голосовать за золотое яблоко.

        voter_id позволяет имитировать разных участников роя: без него все
        голоса идут от текущей роли агента, и порог min_votes недостижим.
        """
        return self.golden_apples.vote(
            apple_id=apple_id, voter_id=voter_id or self.role, approve=approve
        )

    def stats(self) -> Dict[str, Any]:
        """Общая статистика агента."""
        return {
            "llm_available": self.llm_available,
            "llm_model_present": getattr(self, "llm_model_present", False),
            "llm_model": self.llm.model,
            "llm_provider": self.llm.provider,
            "llm_api_style": self.llm.api_style,
            "llm_error": self.llm_error,
            "fallback": self.fallback_llm.describe() if self.fallback_llm else None,
            "role": self.role,
            "mode": self.mode,
            "actions_total": len(self.action_history),
            "gatekeeper": self.gatekeeper.stats(),
            "registry": self.registry.stats(),
            "golden_apples": self.golden_apples.stats(),
            "swarm": self.swarm.stats(),
            "hormones": self.hormones.stats(),
            "kozyrev": self.kozyrev.stats(),
            "tools": self.tools.stats(),
            "tasks": self.task_queue.stats(),
            "autonomous_limits": self.autonomous_limits,
            "micromodels": {
                "enabled": self.micromodels_enabled,
                "trained_on": self.cascade.trained_on,
                "floor": self.micromodels_floor,
                **self.micro_stats,
            },
        }

    def probe_providers(self, max_tokens: int = 8, timeout: int = 30) -> List[Dict[str, Any]]:
        """Реально проверить провайдеров: ключ, список моделей и живой ответ.

        Флаг `available` у клиента означает лишь «ключ найден и /models ответил».
        Списания, баланс и работоспособность модели так не видны: ключ может
        быть валиден, а генерация отвечать 402 или 401. Поэтому пробуем
        настоящий короткий запрос и отличаем «настроен» от «работает».
        """
        report: List[Dict[str, Any]] = []
        for role, client in (("основной", self.llm), ("фолбэк", self.fallback_llm)):
            if client is None:
                report.append({"role": role, "provider": None, "works": False,
                               "reason": "не настроен"})
                continue
            item: Dict[str, Any] = {
                "role": role,
                "provider": client.provider,
                "model": client.model,
                "base_url": client.base_url,
                "key_present": bool(client.api_key),
                "works": False,
            }
            # Ключ НЕ считаем решающим: локальным провайдерам он не нужен, а
            # отсутствие ключа у сетевого клиент сам назовёт точной причиной.
            # Раньше проба обрывалась здесь и скрывала настоящий отказ
            # (например, падение локальной модели в llama.cpp).
            try:
                item["models_listed"] = client.list_models()[:0] or True
            except Exception as exc:  # noqa: BLE001 — диагностика чужих ответов
                item["models_listed"] = False
                item["models_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            previous = client.max_tokens
            client.max_tokens = max_tokens
            try:
                answer = client.chat("Ответь одним словом: да")
                item["works"] = True
                item["answer"] = answer[:60]
            except Exception as exc:  # noqa: BLE001
                item["reason"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            finally:
                client.max_tokens = previous
            report.append(item)
        return report

    def tools_roster(self) -> str:
        return self.tools.roster()

    def run_interactive(self) -> None:
        """Интерактивный цикл терминала."""
        from cli.terminal import TerminalUI

        TerminalUI(self).run()
