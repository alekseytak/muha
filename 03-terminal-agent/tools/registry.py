"""Регистр инструментов — соответствие «глагол + домен -> обработчик».

Глагол описывает намерение (что агент хочет сделать), инструмент — реализацию.
Домен определяется по префиксу объекта (`file:`, `sys:`, `connectome:` …),
поэтому один и тот же глагол может вести в разные инструменты.

Каждый инструмент несёт уровень опасности. `requires_confirmation`
выставляется для всего, что меняет состояние среды (запись, исполнение).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from core.verb_act import VerbActType

from modules.connectome import Connectome
from modules.filesystem import (
    FsSandbox,
    ToolError,
    list_dir,
    read_file,
    stat_file,
    write_file,
)
from modules.system import SystemSandbox, run_command
from modules.web import WebFetcher, page_info, read_web
import time

Handler = Callable[[str, Dict], str]


class ToolContext:
    """Окружение, в котором работают инструменты.

    Ссылки на реестр/рой/гормоны берутся у агента **в момент вызова**, а не
    копируются при сборке: агент может заменить реестр (например, в тестах на
    временный каталог), и инструменты обязаны видеть актуальный объект,
    а не устаревшую ссылку.
    """

    def __init__(
        self,
        fs: FsSandbox,
        system: SystemSandbox,
        connectome: Connectome,
        project_root: Path,
        owner: object = None,
        registry: object = None,
        swarm: object = None,
        hormones: object = None,
        version: str = "1.1.0",
    ) -> None:
        self.fs = fs
        self.system = system
        self.connectome = connectome
        self.project_root = project_root
        self.owner = owner
        self._registry = registry
        self._swarm = swarm
        self._hormones = hormones
        self._version = version

    def _resolve(self, name: str, explicit: object) -> object:
        if explicit is not None:
            return explicit
        return getattr(self.owner, name, None) if self.owner is not None else None

    @property
    def registry(self) -> object:
        return self._resolve("registry", self._registry)

    @registry.setter
    def registry(self, value: object) -> None:
        self._registry = value

    @property
    def swarm(self) -> object:
        return self._resolve("swarm", self._swarm)

    @swarm.setter
    def swarm(self, value: object) -> None:
        self._swarm = value

    @property
    def hormones(self) -> object:
        return self._resolve("hormones", self._hormones)

    @hormones.setter
    def hormones(self, value: object) -> None:
        self._hormones = value

    @property
    def role(self) -> Optional[str]:
        """Роль, от имени которой действует агент (нужна для привилегий)."""
        return getattr(self.owner, "role", None) if self.owner is not None else None

    @property
    def version(self) -> str:
        if self._version:
            return self._version
        config = getattr(self.owner, "config", {}) or {}
        return str((config.get("agent") or {}).get("version", "1.1.0"))


@dataclass
class ToolResult:
    """Результат работы инструмента."""

    output: str
    tool: str
    ok: bool = True


@dataclass
class ToolSpec:
    """Описание инструмента."""

    name: str
    verbs: Tuple[VerbActType, ...]
    domains: Tuple[str, ...]
    handler: Handler
    description: str
    danger: str = "low"            # low | medium | high
    requires_confirmation: bool = False

    def matches(self, verb: VerbActType, domain: str) -> bool:
        return verb in self.verbs and domain in self.domains


def _strip_prefix(obj: str, prefixes: Tuple[str, ...]) -> str:
    """Срезает известный префикс объекта: `file:./x` -> `./x`."""
    text = (obj or "").strip()
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


# ---------- обработчики домена filesystem ----------

FS_PREFIXES = ("file:", "path:", "dir:", "folder:", "fs:")
WEB_PREFIXES = ("web:", "url:")


def _fs_read(obj: str, params: Dict, ctx: ToolContext) -> str:
    raw = _strip_prefix(obj, FS_PREFIXES)
    if not raw:
        raise ToolError("не указан путь: используйте file:<путь> или dir:<путь>")
    target = ctx.fs.resolve(raw)
    if target.is_dir():
        return list_dir(raw, ctx.fs)
    return read_file(raw, ctx.fs)


def _fs_write(obj: str, params: Dict, ctx: ToolContext) -> str:
    raw = _strip_prefix(obj, FS_PREFIXES)
    if not raw:
        raise ToolError("не указан путь для записи")
    content = params.get("content")
    if content is None:
        content = params.get("text", params.get("data"))
    if content is None:
        raise ToolError(
            "не передан parameters.content — нечего записывать (пустая запись запрещена)"
        )
    return write_file(raw, str(content), ctx.fs)


def _fs_stat(obj: str, params: Dict, ctx: ToolContext) -> str:
    raw = _strip_prefix(obj, FS_PREFIXES)
    return stat_file(raw or ".", ctx.fs)


# ---------- обработчики домена system ----------


def _web_dispatch(fn, obj: str, params: Dict, ctx: ToolContext) -> str:
    """Общий вход веб-инструментов: `web:<url>` и `url:<url>` без схемы."""
    url = _strip_prefix(obj, WEB_PREFIXES)
    fetcher = getattr(ctx, "web_fetcher", None) or WebFetcher()
    return fn(url, params, fetcher)


def _web_read(obj: str, params: Dict, ctx: ToolContext) -> str:
    return _web_dispatch(read_web, obj, params, ctx)


def _web_info(obj: str, params: Dict, ctx: ToolContext) -> str:
    return _web_dispatch(page_info, obj, params, ctx)


def _sys_execute(obj: str, params: Dict, ctx: ToolContext) -> str:
    import shlex

    command = _strip_prefix(obj, ("sys:", "cmd:", "sh:", "exec:"))
    if not command:
        raise ToolError("пустая команда: используйте sys:<команда>")
    argv = params.get("argv")
    if argv is not None:
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
            raise ToolError("argv должен быть непустым списком строк")
        if shlex.split(command)[0] != argv[0]:
            raise ToolError("argv не соответствует команде объекта")
        command = shlex.join(argv)
    return run_command(command, ctx.system, role=ctx.role)


# ---------- обработчики домена connectome ----------

CONNECTOME_PREFIXES = ("connectome:", "connectom:", "neuron:", "fly:", "fw:")


def _connectome_dispatch(obj: str, params: Dict, ctx: ToolContext) -> str:
    """Подкоманды: stats | classes | find:<текст> | <id> | neighbors:<id>."""
    raw = _strip_prefix(obj, CONNECTOME_PREFIXES) or "stats"
    action, _, argument = raw.partition(":")
    action = (action or "stats").strip().lower()
    argument = argument.strip()

    if action in ("stats", "stat", "info", "статы"):
        info = ctx.connectome.stats()
        if not info.get("available"):
            raise ToolError(
                f"коннектом недоступен ({info.get('directory')}): {info.get('error')}"
            )
        classes = ", ".join(f"{k}={v}" for k, v in info["classes"].items())
        return (
            f"датасет {info['dataset_id']}\n"
            f"узлов: {info['nodes']}, связей: {info['edges']}\n"
            f"классы: {classes}"
        )
    if action in ("classes", "roles", "классы"):
        return ctx.connectome.classes_report()
    if action in ("find", "search", "найти"):
        return ctx.connectome.find(argument or params.get("query", ""))
    if action in ("neighbors", "neighbours", "соседи"):
        target = argument or params.get("node", "")
        return ctx.connectome.neighbors(
            target, direction=str(params.get("direction", "out")),
            limit=int(params.get("limit", 10)),
        )
    # иначе считаем, что аргумент — идентификатор узла
    target = argument or action
    return ctx.connectome.neuron_report(target)


# ---------- домен registry ----------


def _record(ctx: ToolContext, action: str, result: str, metadata: Dict) -> str:
    """Пишет запись в настоящий append-only реестр и возвращает хеш."""
    if ctx.registry is None:
        raise ToolError("реестр недоступен: инструмент вызван без ToolContext.registry")
    entry = ctx.registry.record(
        action=action,
        actor="tool",
        result=result,
        success=True,
        metadata=metadata,
    )
    return entry.hash


def _registry_log(obj: str, params: Dict, ctx: ToolContext) -> str:
    text = _strip_prefix(obj, ("reg:", "registry:")) or str(params.get("text", ""))
    if not text:
        raise ToolError("нечего записывать в лог: пустой объект")
    entry_hash = _record(ctx, action=f"log:{text[:60]}", result=text,
                         metadata={"tool": "registry.log", "note": text})
    total = len(getattr(ctx.registry, "entries", []))
    return f"записано в реестр: {entry_hash[:12]}… (всего записей: {total})"


def _newest_record_for(obj: str, ctx: ToolContext) -> Optional[object]:
    """Самая свежая запись реестра по этому объекту.

    Правило источника версии: если объект назван, версия — это последняя
    запись о нём по времени создания. Раньше `registry.version` игнорировал
    объект и печатал версию агента, поэтому «какая версия у файла todo.md?»
    получало ответ не про todo.md.
    """
    registry = ctx.registry
    if registry is None or not obj:
        return None
    wanted = _strip_prefix(obj, ("file:", "reg:", "registry:", "fs:"))
    if not wanted:
        return None
    try:
        found = registry.search(wanted)
    except Exception:                              # noqa: BLE001
        return None
    if not found:
        return None
    return max(found, key=lambda entry: entry.timestamp)


def _newest_file_for(obj: str, ctx: ToolContext) -> Optional[object]:
    """Самая свежая файловая запись по объекту, если реестр молчит.

    Берётся время создания (`st_birthtime`), а при его отсутствии —
    изменения. На macOS `st_birthtime` есть; на других системах честно
    откатываемся на mtime, а не выдумываем дату.
    """
    # Объект приходит и как `reg:todo.md`: глагол `version` живёт в домене
    # registry, поэтому префикс реестра надо снимать тоже, иначе файл не
    # найдётся и ответ будет «версии нет» при существующем файле.
    wanted = _strip_prefix(obj, ("file:", "fs:", "reg:", "registry:"))
    if not wanted:
        return None
    root = getattr(ctx.fs, "root", None)
    if root is None:
        return None
    root = Path(root)
    if not root.exists():
        return None
    name = Path(wanted).name
    candidates = []
    try:
        for path in root.rglob(name):
            if not path.is_file():
                continue
            stat = path.stat()
            created = getattr(stat, "st_birthtime", None) or stat.st_mtime
            candidates.append((created, stat.st_mtime, path))
    except OSError:
        return None
    if not candidates:
        return None
    best = max(candidates, key=lambda item: (item[0], item[1]))
    return best[0], best[1], best[2], len(candidates)


def _registry_version(obj: str, params: Dict, ctx: ToolContext) -> str:
    parts = [f"Муха-Агент v{ctx.version}"]

    # Источник версии названного объекта — самая свежая запись о нём.
    # «Смотри по дате создания: последний и нужен» — правило, а не догадка.
    record = _newest_record_for(obj, ctx)
    if record is not None:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.timestamp))
        parts.append(
            f"объект {obj!r}: последняя запись [{record.hash[:8]}] от {when} "
            f"({record.action}, {'ok' if record.success else 'FAIL'}) — "
            f"она и считается текущей версией"
        )
    else:
        newest = _newest_file_for(obj, ctx)
        if newest is not None:
            created, _modified, path, count = newest
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created))
            plural = ("кандидат" if count == 1 else
                      "кандидата" if count < 5 else "кандидатов")
            parts.append(
                f"объект {obj!r}: записей в реестре нет, но на диске найден "
                f"{path} от {when} (создан) — он и считается текущим, "
                f"потому что версия берётся по дате создания; "
                f"файлов-{plural} с таким именем: {count}"
            )
        elif obj:
            parts.append(
                f"объект {obj!r}: ни записей в реестре, ни файла с таким "
                f"именем не найдено — версии нет"
            )
    if ctx.registry is not None:
        stats = ctx.registry.stats()
        parts.append(
            f"реестр: {stats.get('total_entries', '?')} записей, "
            f"алгоритм {stats.get('hash_algorithm', 'sha256')}, "
            f"цепь {'валидна' if ctx.registry.verify_chain() else 'ПОВРЕЖДЕНА'}"
        )
    return "\n".join(parts)


def _registry_query(obj: str, params: Dict, ctx: ToolContext) -> str:
    """Чтение реестра: последние записи или поиск по подстроке."""
    if ctx.registry is None:
        raise ToolError("реестр недоступен: инструмент вызван без ToolContext.registry")
    raw = _strip_prefix(obj, ("reg:", "registry:"))
    limit = int(params.get("limit", 10))
    if raw.startswith("search:"):
        query = raw[len("search:"):].strip()
        found = ctx.registry.search(query)[:limit]
        if not found:
            return f"в реестре нет записей по запросу {query!r}"
        lines = [f"найдено {len(found)} записей по {query!r}:"]
        lines += [f"  [{e.hash[:8]}] {e.action} -> {str(e.result)[:60]}" for e in found]
        return "\n".join(lines)

    entries = ctx.registry.last_n(limit)
    if not entries:
        return "реестр пуст"
    stats = ctx.registry.stats()
    lines = [
        f"последние {len(entries)} записей из {stats.get('total_entries', '?')} "
        f"(цепь {'валидна' if ctx.registry.verify_chain() else 'ПОВРЕЖДЕНА'}):"
    ]
    lines += [
        f"  [{e.hash[:8]}] {e.action} -> {str(e.result)[:60]}"
        f" ({'ok' if e.success else 'FAIL'})"
        for e in entries
    ]
    return "\n".join(lines)


def _registry_anchor(obj: str, params: Dict, ctx: ToolContext) -> str:
    text = _strip_prefix(obj, ("reg:", "registry:")) or str(params.get("text", ""))
    if not text:
        raise ToolError("нечего якорить: пустой объект")
    entry_hash = _record(
        ctx,
        action=f"anchor:{text[:60]}",
        result=f"заякорено: {text}",
        metadata={"tool": "registry.anchor", "anchored": text, "canonical": True},
    )
    return f"заякорено: {text} (хеш {entry_hash[:12]}…)"


# ---------- домен swarm ----------


def _swarm_field(ctx: ToolContext):
    """Достаёт поле феромонов из роя (у Swarm оно называется `field`)."""
    swarm = ctx.swarm
    if swarm is None:
        raise ToolError("рой недоступен: инструмент вызван без ToolContext.swarm")
    for attribute in ("field", "pheromone_field"):
        field = getattr(swarm, attribute, None)
        if field is not None and hasattr(field, "deposit"):
            return field
    raise ToolError("у роя нет поля феромонов — проверьте core/swarm.py")


def _swarm_veto(obj: str, params: Dict, ctx: ToolContext) -> str:
    target = _strip_prefix(obj, ("swarm:", "pheromone:")) or obj
    if not target:
        raise ToolError("не указано, что ветировать")
    field = _swarm_field(ctx)
    # Вето — сильный отрицательный феромон в точке решения.
    pheromone_id = field.deposit(
        position=(0.0, 0.0),
        strength=float(params.get("strength", 1.0)),
        agent_id=str(params.get("agent_id", "agent:veto")),
        pheromone_type="veto",
    )
    return (
        f"вето наложено: {target} (феромон {pheromone_id}, "
        f"в поле {field.stats()['total_pheromones']} меток)"
    )


def _swarm_escalate(obj: str, params: Dict, ctx: ToolContext) -> str:
    target = _strip_prefix(obj, ("swarm:", "pheromone:")) or obj or "(без объекта)"
    field = _swarm_field(ctx)
    pheromone_id = field.deposit(
        position=(0.0, 0.0),
        strength=float(params.get("strength", 0.8)),
        agent_id=str(params.get("agent_id", "agent:escalate")),
        pheromone_type="escalate",
    )
    return (
        f"эскалировано: {target} (феромон {pheromone_id}, "
        f"в поле {field.stats()['total_pheromones']} меток)"
    )


# ---------- домен ui ----------


def _ui_select(obj: str, params: Dict, ctx: ToolContext) -> str:
    options: List[str] = params.get("options") or []
    if not options:
        raise ToolError("select требует parameters.options — список вариантов")
    strategy = str(params.get("strategy", "first")).lower()
    if strategy == "last":
        chosen = options[-1]
    elif strategy == "max" and all(isinstance(o, (int, float)) for o in options):
        chosen = max(options)
    else:
        chosen = options[0]
    return f"выбрано: {chosen} (из {len(options)} вариантов, стратегия {strategy})"


def _ui_recommend(obj: str, params: Dict, ctx: ToolContext) -> str:
    """Рекомендация строится на реальном состоянии, а не на эхе запроса."""
    subject = obj or str(params.get("text", "")) or "(без объекта)"
    parts = [f"рекомендация по «{subject}»"]
    if ctx.hormones is not None:
        appetite = float(getattr(ctx.hormones, "risk_appetite", 0.5))
        mood = "действовать" if appetite >= 0.5 else "сначала проверить"
        parts.append(f"риск-аппетит {appetite:.2f} -> {mood}")
    if ctx.registry is not None:
        recent = ctx.registry.last_n(3)
        failed = sum(1 for e in recent if not e.success)
        parts.append(
            f"из последних {len(recent)} действий провалено {failed}"
            + (" -> снизить риск" if failed else " -> можно продолжать")
        )
    if ctx.connectome is not None and ctx.connectome.available:
        parts.append("коннектом доступен")
    return "\n".join(parts)


# ---------- сборка ----------


def build_default_registry(ctx: ToolContext) -> "ToolRegistry":
    """Создаёт регистр с инструментами по умолчанию."""
    registry = ToolRegistry()

    registry.register(ToolSpec(
        name="fs.read",
        verbs=(VerbActType.READ,),
        domains=("filesystem",),
        handler=lambda o, p: _fs_read(o, p, ctx),
        description="чтение файла или листинг каталога внутри песочницы",
        danger="low",
    ))
    registry.register(ToolSpec(
        name="fs.stat",
        verbs=(VerbActType.SELECT,),
        domains=("filesystem",),
        handler=lambda o, p: _fs_stat(o, p, ctx),
        description="метаданные файла (select по объекту файловой системы)",
        danger="low",
    ))
    registry.register(ToolSpec(
        name="fs.write",
        verbs=(VerbActType.WRITE,),
        domains=("filesystem",),
        handler=lambda o, p: _fs_write(o, p, ctx),
        description="атомарная запись файла; нужен parameters.content",
        danger="medium",
        requires_confirmation=True,
    ))
    registry.register(ToolSpec(
        name="web.read",
        verbs=(VerbActType.READ,),
        domains=("web",),
        handler=lambda o, p: _web_read(o, p, ctx),
        description="загрузить страницу и вернуть её текст (web:<url>), параметр selector",
        danger="low",
    ))
    registry.register(ToolSpec(
        name="web.info",
        verbs=(VerbActType.SELECT,),
        domains=("web",),
        handler=lambda o, p: _web_info(o, p, ctx),
        description="метаданные страницы: код, тип, размер, заголовок",
        danger="low",
    ))
    registry.register(ToolSpec(
        name="sys.execute",
        verbs=(VerbActType.EXECUTE,),
        domains=("system",),
        handler=lambda o, p: _sys_execute(o, p, ctx),
        description="запуск команды из allowlist без шелла, с таймаутом",
        danger="high",
        requires_confirmation=True,
    ))
    registry.register(ToolSpec(
        name="connectome.query",
        verbs=(VerbActType.READ, VerbActType.SELECT),
        domains=("connectome",),
        handler=lambda o, p: _connectome_dispatch(o, p, ctx),
        description="запросы к коннектому FlyWire: stats/classes/find/neighbors/<id>",
        danger="low",
    ))
    registry.register(ToolSpec(
        name="registry.log",
        verbs=(VerbActType.LOG,),
        domains=("ui", "filesystem", "system", "registry", "swarm", "connectome"),
        handler=lambda o, p: _registry_log(o, p, ctx),
        description="запись пометки в журнал",
        danger="low",
    ))
    registry.register(ToolSpec(
        name="registry.version",
        verbs=(VerbActType.VERSION,),
        domains=("registry", "ui"),
        handler=lambda o, p: _registry_version(o, p, ctx),
        description="версия агента и форматов",
        danger="low",
    ))
    registry.register(ToolSpec(
        name="registry.query",
        verbs=(VerbActType.READ, VerbActType.SELECT),
        domains=("registry",),
        handler=lambda obj, params: _registry_query(obj, params, ctx),
        description="чтение реестра: reg:list (последние) или reg:search:<текст>",
        danger="low",
    ))

    registry.register(ToolSpec(
        name="registry.anchor",
        verbs=(VerbActType.ANCHOR,),
        domains=("registry",),
        handler=lambda o, p: _registry_anchor(o, p, ctx),
        description="канонизация записи",
        danger="medium",
        requires_confirmation=True,
    ))
    registry.register(ToolSpec(
        name="swarm.veto",
        verbs=(VerbActType.VETO,),
        domains=("swarm", "registry"),
        handler=lambda o, p: _swarm_veto(o, p, ctx),
        description="вето на предложение роя",
        danger="low",
    ))
    registry.register(ToolSpec(
        name="swarm.escalate",
        verbs=(VerbActType.ESCALATE,),
        domains=("swarm", "registry", "ui", "system", "filesystem", "connectome"),
        handler=lambda o, p: _swarm_escalate(o, p, ctx),
        description="эскалация решения на уровень выше",
        danger="low",
    ))
    registry.register(ToolSpec(
        name="ui.select",
        verbs=(VerbActType.SELECT,),
        domains=("ui", "swarm", "connectome"),
        handler=lambda o, p: _ui_select(o, p, ctx),
        description="выбор варианта из parameters.options",
        danger="low",
    ))
    registry.register(ToolSpec(
        name="ui.recommend",
        verbs=(VerbActType.RECOMMEND,),
        domains=("ui", "swarm"),
        handler=lambda o, p: _ui_recommend(o, p, ctx),
        description="сформулировать рекомендацию",
        danger="low",
    ))
    return registry


class ToolRegistry:
    """Хранит инструменты и исполняет их по (глагол, домен)."""

    def __init__(self) -> None:
        self._specs: List[ToolSpec] = []
        self.calls: List[Dict] = []

    def register(self, spec: ToolSpec) -> None:
        self._specs.append(spec)

    def roster_specs(self) -> List[ToolSpec]:
        """Все зарегистрированные спецификации (для проверок и отчётов)."""
        return list(self._specs)

    def find(self, verb: VerbActType, domain: str) -> Optional[ToolSpec]:
        for spec in self._specs:
            if spec.matches(verb, domain):
                return spec
        return None

    def execute(self, verb: VerbActType, domain: str, obj: str, params: Dict) -> ToolResult:
        """Находит инструмент и выполняет его. ToolError -> ToolResult(ok=False)."""
        spec = self.find(verb, domain)
        if spec is None:
            raise ToolError(f"нет инструмента для глагола {verb.value!r} в домене {domain!r}")
        try:
            output = spec.handler(obj, params or {})
        except ToolError as exc:
            return ToolResult(output=f"{spec.name}: {exc}", tool=spec.name, ok=False)
        self.calls.append({"tool": spec.name, "verb": verb.value, "domain": domain})
        return ToolResult(output=output, tool=spec.name, ok=True)

    def roster(self) -> str:
        """Текстовая таблица возможностей — для help и системного промпта."""
        lines = ["инструменты:"]
        for spec in self._specs:
            verbs = ",".join(v.value for v in spec.verbs)
            domains = ",".join(spec.domains)
            flag = " (спросит подтверждение)" if spec.requires_confirmation else ""
            lines.append(
                f"  {spec.name:18s} {verbs:10s} домены: {domains:12s} "
                f"риск: {spec.danger}{flag} — {spec.description}"
            )
        return "\n".join(lines)

    def stats(self) -> Dict:
        return {"tools": len(self._specs), "calls": len(self.calls)}
