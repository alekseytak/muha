"""Чтение веб-страниц как текста: загрузка + извлечение содержимого.

Разделение обязанностей:

* загрузка — `urllib` с нашим SSL-контекстом (`core.llm_client.build_ssl_context`),
  потому что штатный бандл macOS во framework-сборке Python неполный;
* разбор — Scrapling (`scrapling.parser.Adaptor`) из `pip install scrapling`,
  то есть только парсер, **без** fetchers и браузеров: 300 МБ Chromium агенту
  на терминале не нужны. Если Scrapling не установлен, работает встроенный
  регэксп-фолбэк и результат помечается как degraded.

Защита: только http/https, никаких localhost и приватных адресов, ограничение
размера и таймаута. Агент читает внешний мир, а не нашу же машину.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

from .filesystem import ToolError

ALLOWED_SCHEMES = ("http", "https")
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 muha-agent/1.0"
)
# Теги, содержимое которых не является текстом страницы.
DROP_TAGS = ("script", "style", "noscript", "template", "svg", "iframe", "canvas")
# Кандидаты на «основное содержимое», от лучшего к худшему.
CONTENT_SELECTORS = ("article", "main", "[role=main]", ".post-content", ".article-body", "#content")
# Блочные элементы: их текст — уже готовый абзац, инлайн-куски внутри
# склеиваются пробелом, поэтому ссылка не рвёт строку.
BLOCK_SELECTOR = "h1,h2,h3,h4,h5,h6,p,li,blockquote,pre,td,th,figcaption,dd,dt"
STATE = {"scrapling_ok": None, "scrapling_error": "", "browser_ok": None}

# Порог «страница пустая»: SPA отдаёт каркас на пару сотен символов, а после
# исполнения JS там десятки тысяч. Если текста меньше — пробуем браузер.
THIN_TEXT_CHARS = 400
# Каркас SPA: HTML большой, а текста почти нет. Короткая страница-визитка
# (example.com) под это не подпадает — браузер для неё только тратит время.
SPA_HTML_TO_TEXT_RATIO = 20
JS_HINTS = ("enable javascript", "включите javascript", "requires javascript",
            "you need to enable javascript")


def _scrapling():
    """Ленивый импорт Scrapling: без него модуль остаётся работоспособным."""
    if STATE["scrapling_ok"] is False:
        return None
    try:
        from scrapling.parser import Adaptor  # noqa: PLC0415
    except Exception as exc:  # ImportError и любые сборочные проблемы
        STATE["scrapling_ok"] = False
        STATE["scrapling_error"] = f"{type(exc).__name__}: {exc}"
        return None
    STATE["scrapling_ok"] = True
    return Adaptor


def scrapling_available() -> bool:
    _scrapling()
    return bool(STATE["scrapling_ok"])


def _host_is_private(host: str) -> bool:
    """Приватный или служебный адрес — читать запрещено (SSRF)."""
    name = (host or "").strip("[]").lower()
    if not name:
        return True
    if name in ("localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"):
        return True
    if name.endswith(".local") or name.endswith(".internal") or name.endswith(".localhost"):
        return True
    try:
        infos = socket.getaddrinfo(name, None)
    except socket.gaierror:
        # Не разрешается — пусть загрузчик скажет об этом понятнее.
        return False
    except Exception:
        return True
    for info in infos:
        raw = info[4][0]
        try:
            address = ipaddress.ip_address(raw.split("%")[0])
        except ValueError:
            return True
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            return True
    return False


def validate_url(url: str) -> str:
    """Проверяет адрес и возвращает нормализованный вид."""
    text = (url or "").strip().strip("<>\"'")
    if not text:
        raise ToolError("не указан адрес: используйте web:https://example.com/страница")
    if "://" not in text:
        text = "https://" + text
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ToolError(
            f"схема {parsed.scheme!r} запрещена: разрешены только "
            + ", ".join(ALLOWED_SCHEMES)
        )
    host = parsed.hostname or ""
    if not host:
        raise ToolError(f"в адресе нет хоста: {text}")
    if _host_is_private(host):
        raise ToolError(
            f"адрес {host!r} ведёт в локальную сеть или на служебный адрес — "
            "агент читает только внешний веб"
        )
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, "")
    )


class WebFetcher:
    """Загружает страницы с ограничениями по размеру, времени и адресам."""

    def __init__(self, timeout: int = 15, max_bytes: int = 2_000_000,
                 user_agent: str = DEFAULT_USER_AGENT) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.user_agent = user_agent

    def _context(self):
        try:
            from core.llm_client import build_ssl_context  # noqa: PLC0415
            return build_ssl_context()
        except Exception:
            return None

    def fetch(self, url: str) -> Dict[str, object]:
        target = validate_url(url)
        request = urllib.request.Request(
            target,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
                "Accept-Language": "ru,en;q=0.8",
            },
        )
        context = self._context()
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=context
            ) as response:
                raw = response.read(self.max_bytes + 1)
                status = getattr(response, "status", 200)
                content_type = response.headers.get("Content-Type", "")
                final_url = response.geturl()
        except urllib.error.HTTPError as exc:
            raise ToolError(f"страница вернула HTTP {exc.code} ({target})") from exc
        except urllib.error.URLError as exc:
            raise ToolError(f"не удалось загрузить {target}: {exc.reason}") from exc
        except socket.timeout as exc:
            raise ToolError(f"таймаут {self.timeout} с при загрузке {target}") from exc

        charset = "utf-8"
        match = re.search(r"charset=([\w-]+)", content_type or "", re.I)
        if match:
            charset = match.group(1)
        try:
            text = raw[: self.max_bytes].decode(charset, "replace")
        except LookupError:
            text = raw[: self.max_bytes].decode("utf-8", "replace")
        return {
            "url": target,
            "final_url": final_url,
            "status": status,
            "content_type": content_type,
            "bytes": len(raw),
            "truncated": len(raw) > self.max_bytes,
            "text": text,
        }


def _fallback_text(html: str) -> str:
    """Без Scrapling: вырезаем служебные блоки и теги регулярками."""
    text = html
    for tag in DROP_TAGS:
        text = re.sub(rf"<{tag}\b.*?</{tag}>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (
        text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
        .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    )
    return text


def html_to_text(html: str, max_chars: int = 20000,
                 selector: Optional[str] = None) -> Tuple[str, Dict[str, object]]:
    """Извлекает читаемый текст. Возвращает текст и метаданные разбора."""
    meta: Dict[str, object] = {"parser": "fallback", "title": "", "selector_used": ""}
    Adaptor = _scrapling()
    if Adaptor is None:
        meta["degraded"] = "scrapling не установлен: используется регэксп-фолбэк"
        return _squash(_fallback_text(html), max_chars, reflow=True), meta

    try:
        page = Adaptor(content=html)
        meta["parser"] = "scrapling"
        title = page.css("title::text").get()
        meta["title"] = (title or "").strip()[:200]

        node = None
        if selector:
            found = page.css(selector)
            node = found[0] if found else None
            if node is not None:
                meta["selector_used"] = selector
        if node is None:
            for candidate in CONTENT_SELECTORS:
                found = page.css(candidate)
                if found:
                    node = found[0]
                    meta["selector_used"] = candidate
                    break
        if node is None:
            node = page
        blocks = node.css(BLOCK_SELECTOR)
        if blocks:
            parts = []
            for block in blocks:
                chunk = block.get_all_text(separator=" ", strip=True)
                chunk = " ".join(str(chunk).split())
                if chunk:
                    parts.append(chunk)
            raw = "\n".join(parts)
            meta["blocks"] = len(parts)
        else:
            raw = node.get_all_text(separator=" ", strip=True)
    except Exception as exc:
        meta["degraded"] = f"scrapling не справился ({type(exc).__name__}): {exc}"
        return _squash(_fallback_text(html), max_chars, reflow=True), meta
    return _squash(raw, max_chars), meta


END_PUNCT = ".!?:;»\"'”’)"
LONG_LINE = 60


def _reflow(lines: List[str]) -> List[str]:
    """Склеивает обрывки в абзацы.

    Scrapling отдаёт текст по строчным узлам, поэтому ссылка или <em> внутри
    абзаца рвёт строку: «Второй абзац со» / «ссылкой» / «.». Абзац собирается
    обратно, пока строка не закончилась знаком конца или не стала длинной.
    """
    merged: List[str] = []
    for line in lines:
        if merged:
            previous = merged[-1]
            continues = (
                previous
                and not previous.endswith(tuple(END_PUNCT))
                and len(previous) < 400
            )
            if continues:
                merged[-1] = previous.rstrip() + " " + line
                continue
        merged.append(line)
    fixed = []
    for line in merged:
        for mark in END_PUNCT:
            line = line.replace(" " + mark, mark)
        fixed.append(line.strip())
    return [line for line in fixed if line]


def _squash(text: str, max_chars: int, reflow: bool = False) -> str:
    lines: List[str] = []
    for line in (text or "").splitlines():
        clean = " ".join(line.split())
        if clean:
            lines.append(clean)
    normalized = []
    for line in (lines if not reflow else _reflow(lines)):
        for mark in END_PUNCT:
            line = line.replace(" " + mark, mark)
        normalized.append(line.strip())
    joined = "\n".join(line for line in normalized if line)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + f"\n…обрезано (всего {len(joined)} символов)"
    return joined


def _render_with_browser(url: str, max_chars: int) -> Tuple[str, dict]:
    """Пробует настоящий браузер. Возвращает пустой текст, если его нет."""
    from . import browser as browser_module

    if not browser_module.is_available():
        STATE["browser_ok"] = False
        return "", {"backend": "нет", "reason": "бинарь Obscura не найден"}
    try:
        text, meta = browser_module.render(
            url, timeout=int(STATE.get("browser_timeout", 30)), fmt="text"
        )
    except browser_module.BrowserUnavailable as exc:
        STATE["browser_ok"] = False
        return "", {"backend": "obscura", "reason": str(exc)[:200]}
    STATE["browser_ok"] = True
    return _squash(text, max_chars, reflow=False), meta


def read_web(url: str, params: Dict, fetcher: Optional[WebFetcher] = None,
             max_chars: int = 20000) -> str:
    """READ: загружает страницу и возвращает её текст с шапкой.

    Если обычный HTML дал подозрительно мало текста (страница на JS), домен
    не сдаётся, а пробует браузер — и честно пишет в шапке, кто именно читал.
    """
    fetcher = fetcher or WebFetcher()
    page = fetcher.fetch(url)
    selector = str(params.get("selector") or "").strip() or None
    if selector and not re.fullmatch(r"[#.\[\]\w\s\-=:'\"]{1,120}", selector):
        raise ToolError(f"подозрительный CSS-селектор: {selector[:60]!r}")
    text, meta = html_to_text(str(page["text"]), max_chars=max_chars, selector=selector)

    content_type = str(page["content_type"] or "")
    browser_note = ""
    visible = len(text.strip())
    raw_size = len(str(page["text"]))
    looks_like_spa = (
        visible < THIN_TEXT_CHARS
        and (raw_size > visible * SPA_HTML_TO_TEXT_RATIO
             or any(hint in str(page["text"]).lower() for hint in JS_HINTS))
    )
    use_browser = bool(params.get("browser")) or (not selector and looks_like_spa)
    if use_browser and "html" in content_type.lower():
        rendered, browser_meta = _render_with_browser(str(page["url"]), max_chars)
        if rendered.strip() and len(rendered.strip()) > len(text.strip()):
            text = rendered
            meta = {**meta, "parser": "obscura (после JS)"}
            browser_note = (f"браузер: obscura, за {browser_meta.get('bytes', 0)} байт "
                            f"после исполнения JS")
        elif browser_meta.get("reason"):
            browser_note = f"браузер не помог: {browser_meta['reason']}"

    if not text.strip():
        raise ToolError(
            f"страница {page['url']} не дала текста (тип {content_type or 'неизвестен'}); "
            "возможно, содержимое грузится JavaScript, а браузер недоступен"
        )
    header = [f"URL: {page['url']}"]
    if page["final_url"] and page["final_url"] != page["url"]:
        header.append(f"перенаправление: {page['final_url']}")
    header.append(f"HTTP: {page['status']} | тип: {content_type or 'неизвестен'} | "
                  f"размер: {page['bytes']} байт")
    if meta.get("title"):
        header.append(f"заголовок: {meta['title']}")
    header.append(f"разбор: {meta['parser']}" + (
        f" (селектор {meta['selector_used']})" if meta.get("selector_used") else ""
    ))
    if browser_note:
        header.append(browser_note)
    if meta.get("degraded"):
        header.append(f"ВНИМАНИЕ: {meta['degraded']}")
    return "\n".join(header) + "\n\n" + text


def page_info(url: str, fetcher: Optional[WebFetcher] = None) -> str:
    """SELECT: метаданные страницы без выгрузки всего текста."""
    fetcher = fetcher or WebFetcher()
    page = fetcher.fetch(url)
    text, meta = html_to_text(str(page["text"]), max_chars=400)
    lines = [
        f"URL: {page['url']}",
        f"HTTP: {page['status']}",
        f"тип: {page['content_type'] or 'неизвестен'}",
        f"размер: {page['bytes']} байт" + (" (обрезано)" if page["truncated"] else ""),
        f"заголовок: {meta.get('title') or '—'}",
        f"разбор: {meta['parser']}",
        f"начало текста: {text[:200]}",
    ]
    return "\n".join(lines)