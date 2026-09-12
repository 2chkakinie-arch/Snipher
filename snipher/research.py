"""軽量なウェブ調査層。

Snipher は知識ベースだけで現在の出来事を答えない。まずローカルの
``ResearchPolicy`` が「今の情報が要るか」を数マイクロ秒で判定し、必要な
ときだけ Edge の検索結果 HTML と対象ページを取得する。このモジュールは
外部 SDK/ブラウザ/requests に依存せず、標準ライブラリだけで動く。

設計上の要点:

* Edge のブラウザに近い User-Agent で Bing HTML を取得する
* HTML は ``HTMLParser`` で本文・title・リンクを抽出し、script/style を捨てる
* URL、サイズ、時間、リダイレクトを制限する（SSRF と巨大ページ対策）
* 検索失敗は例外を外へ漏らさず、取得できた材料だけを返す
* 結果は短い TTL のメモリキャッシュに置き、同じ質問を瞬時に再利用する

ネットワークは既定で「必要な質問」にしか発生しない。テストでは fetcher を
差し替えれば、ネットワークなしで全文を検証できる。
"""

from __future__ import annotations

import base64
import html
import ipaddress
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import socket
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

_EDGE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
)
_DEFAULT_SEARCH_URL = "https://www.bing.com/search?q={query}&count={count}&setlang=ja-JP"
_FALLBACK_SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"

# HTML の検索結果や記事に混じるノイズ。本文はさらに空白を正規化する。
_NOISE_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "canvas", "nav", "footer"})


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _clean_text(value: str, limit: int | None = None) -> str:
    value = html.unescape(str(value or ""))
    value = re.sub(r"[\t\r\n\f\v ]+", " ", value)
    value = re.sub(r"\s+([。、！？!?：:，,）】』])", r"\1", value)
    value = re.sub(r"([（【『])\s+", r"\1", value)
    value = value.strip()
    return value[:limit] if limit else value


def _private_or_reserved(host: str) -> bool:
    """ホスト名が localhost/予約 IP なら True。

    DNS 解決は fetch の直前に行う。解決できないテスト用ホストはここでは
    弾かず、実際の urlopen が失敗した結果として安全に返す。
    """
    h = (host or "").strip("[]").lower().rstrip(".")
    if not h or h in {"localhost", "localhost.localdomain"} or h.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(h)
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast)
    except ValueError:
        return False


def validate_url(url: str, *, allow_private: bool = False) -> tuple[bool, str]:
    """外部 HTTP URL として安全に扱えるかを返す。"""
    try:
        p = urlparse(str(url or "").strip())
    except ValueError:
        return False, "invalid_url"
    if p.scheme not in {"http", "https"} or not p.hostname:
        return False, "http_or_https_required"
    if p.username or p.password:
        return False, "credentials_in_url_not_allowed"
    if not allow_private and _private_or_reserved(p.hostname):
        return False, "private_host_blocked"
    return True, "ok"


class _PageParser(HTMLParser):
    """記事ページを壊れにくく読む最小 HTML パーサー。"""

    def __init__(self, *, max_links: int = 32):
        super().__init__(convert_charrefs=True)
        self.title: list[str] = []
        self.meta: dict[str, str] = {}
        self.text: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._skip = 0
        self._title_depth = 0
        self._link_href = ""
        self._link_text: list[str] = []
        self._max_links = max_links

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr = {str(k).lower(): str(v or "") for k, v in attrs}
        if tag in _NOISE_TAGS:
            self._skip += 1
            return
        if tag == "title":
            self._title_depth += 1
        if tag == "meta":
            key = (attr.get("name") or attr.get("property") or "").lower()
            content = attr.get("content", "")
            if key and content and key in {"description", "og:description", "twitter:description"}:
                self.meta[key] = content
        if tag == "a" and len(self.links) < self._max_links:
            href = attr.get("href", "").strip()
            if href:
                self._link_href = href
                self._link_text = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _NOISE_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
        if tag == "a" and self._link_href:
            label = _clean_text("".join(self._link_text), 300)
            self.links.append((self._link_href, label))
            self._link_href = ""
            self._link_text = []

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        data = _clean_text(data)
        if not data:
            return
        if self._title_depth:
            self.title.append(data)
        self.text.append(data)
        if self._link_href:
            self._link_text.append(data)


@dataclass(slots=True)
class FetchResult:
    url: str
    status: int = 0
    title: str = ""
    text: str = ""
    links: list[dict] = field(default_factory=list)
    content_type: str = ""
    elapsed_ms: float = 0.0
    cached: bool = False
    error: str | None = None
    # 末尾に置いて旧 positional 初期化 (url,status,title,text,links,...) を壊さない。
    description: str = ""

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "status": self.status,
            "title": self.title,
            "text": self.text,
            "description": self.description,
            "links": self.links,
            "content_type": self.content_type,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "cached": self.cached,
            "error": self.error,
        }


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    source: str = "bing"
    rank: int = 0

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "rank": self.rank,
        }


@dataclass(slots=True)
class ResearchResult:
    query: str
    needed: bool
    reason: str
    results: list[SearchResult] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    fetched: list[FetchResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    cached: bool = False
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "needed": self.needed,
            "reason": self.reason,
            "results": [r.as_dict() for r in self.results],
            "sources": self.sources,
            "fetched": [f.as_dict() for f in self.fetched],
            "elapsed_ms": round(self.elapsed_ms, 2),
            "cached": self.cached,
            "error": self.error,
        }


class ResearchPolicy:
    """ウェブが必要な質問だけを高速に選別する。"""

    # 現在性が回答の正しさに直結する語。形態素解析を必要としない。
    CURRENT_MARKS = (
        "最新", "今日", "きょう", "現在", "今の", "いまの", "リアルタイム", "速報", "ニュース",
        "天気", "気温", "降水", "株価", "為替", "価格", "値段", "在庫", "発売", "アップデート",
        "更新", "いつの情報", "最近の", "今週", "今月", "今年", "2026", "2025",
    )
    EXPLICIT_MARKS = ("検索", "調べて", "調べる", "検索して", "ウェブ", "ネットで", "URL", "出典", "ソース", "引用")
    # 計算・コードは外部検索よりローカルの厳密な実行/生成を優先する。
    LOCAL_MARKS = ("計算", "解いて", "方程式", "コード", "プログラム", "実装", "書いて", "デバッグ")

    def decide(self, query: str, explicit: bool | None = None) -> tuple[bool, str]:
        q = str(query or "").strip()
        if explicit is False:
            return False, "disabled"
        if explicit is True:
            return True, "explicit"
        if not q:
            return False, "empty"
        if any(m in q for m in self.LOCAL_MARKS) and not any(m in q for m in self.CURRENT_MARKS):
            # 「最新のPython」は現在性があるのでウェブへ回す。
            return False, "local_task"
        if re.search(r"https?://|www\.", q, re.I):
            return True, "url_or_source"
        if any(m in q for m in self.EXPLICIT_MARKS):
            return True, "explicit_keyword"
        if any(m in q for m in self.CURRENT_MARKS) or re.search(r"(?<!\d)20\d{2}(?!\d)", q):
            return True, "current_information"
        return False, "stable_knowledge"


class _SearchParser(HTMLParser):
    """Bing/DuckDuckGo の検索 HTML からリンクと短い説明を拾う。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items: list[SearchResult] = []
        self._href = ""
        self._text: list[str] = []
        self._in_p = 0
        self._snippet: list[str] = []
        self._class_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = {str(k).lower(): str(v or "") for k, v in attrs}
        cls = attrs_d.get("class", "").lower()
        self._class_stack.append(cls)
        if tag.lower() == "a":
            href = attrs_d.get("href", "")
            if href and ("bing.com/ck/a" in href or href.startswith("http")):
                self._href, self._text = href, []
        if tag.lower() in {"p", "div"} and any("snippet" in c or "b_algo" in c for c in self._class_stack):
            self._in_p += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a" and self._href:
            label = _clean_text("".join(self._text), 240)
            # ナビゲーションリンクを除外し、見出しらしいリンクを先に保持する。
            if label and len(label) >= 2:
                self.items.append(SearchResult(title=label, url=self._href))
            self._href, self._text = "", []
        if tag in {"p", "div"} and self._in_p:
            self._in_p -= 1
        if self._class_stack:
            self._class_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)
        if self._in_p:
            self._snippet.append(data)

    def finalize(self, provider: str, limit: int) -> list[SearchResult]:
        snippets = [_clean_text("".join(self._snippet), 600)] if self._snippet else []
        out: list[SearchResult] = []
        seen: set[str] = set()
        for item in self.items:
            url = _unwrap_search_url(item.url)
            if not url or url in seen or not url.startswith(("http://", "https://")):
                continue
            seen.add(url)
            item.url = url
            item.source = provider
            item.rank = len(out) + 1
            if snippets:
                item.snippet = snippets[min(len(out), len(snippets) - 1)]
            out.append(item)
            if len(out) >= limit:
                break
        return out


def _unwrap_search_url(url: str) -> str:
    """Bing/DuckDuckGo のリダイレクトリンクを実 URL に戻す。"""
    try:
        p = urlparse(html.unescape(url))
        qs = parse_qs(p.query)
        for key in ("url", "u", "uddg"):
            if qs.get(key):
                candidate = unquote(qs[key][0])
                if candidate.startswith(("http://", "https://")):
                    return candidate
                # Bing の新しい /ck/a リンクは u=a1 + base64url の形式がある。
                if key == "u" and candidate.startswith("a1"):
                    try:
                        encoded = candidate[2:]
                        padded = encoded + "=" * (-len(encoded) % 4)
                        decoded = base64.urlsafe_b64decode(padded).decode("utf-8", "ignore")
                        if decoded.startswith(("http://", "https://")):
                            return decoded
                    except (ValueError, UnicodeError):
                        pass
        return html.unescape(url)
    except Exception:
        return url


class HtmlFetcher:
    """制限付き HTML fetcher。``fetch`` はテスト時に opener を差し替えられる。"""

    def __init__(self, *, timeout: float | None = None, max_bytes: int | None = None,
                 max_text: int | None = None, opener: Callable | None = None,
                 allow_private: bool | None = None):
        # ネットワークが無い環境でもチャットを長く止めない。必要なら環境変数で延長可能。
        self.timeout = timeout if timeout is not None else _env_float("SNIPHER_WEB_TIMEOUT", 1.5)
        self.max_bytes = max_bytes if max_bytes is not None else _env_int("SNIPHER_WEB_MAX_BYTES", 1_500_000)
        self.max_text = max_text if max_text is not None else _env_int("SNIPHER_WEB_MAX_TEXT", 12_000)
        self.opener = opener or urlopen
        self.allow_private = bool(allow_private) if allow_private is not None else os.environ.get("SNIPHER_WEB_ALLOW_PRIVATE") == "1"

    def fetch(self, url: str, *, base_url: str | None = None) -> FetchResult:
        if base_url:
            url = urljoin(base_url, url)
        ok, reason = validate_url(url, allow_private=self.allow_private)
        if not ok:
            return FetchResult(url=str(url), error=reason)
        if not self.allow_private:
            # ドメイン名が内部 IP に解決される DNS rebinding/SSRF も止める。
            host = urlparse(str(url)).hostname or ""
            try:
                addresses = {item[4][0] for item in socket.getaddrinfo(host, None)}
            except (OSError, socket.gaierror):
                addresses = set()  # 接続可否は opener に任せる（テスト用ホストも許可）
            if any(_private_or_reserved(addr) for addr in addresses):
                return FetchResult(url=str(url), error="private_host_blocked")
        started = time.perf_counter()
        req = Request(str(url), headers={
            "User-Agent": _EDGE_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
            "Accept-Language": "ja,en;q=0.8",
            "Cache-Control": "no-cache",
        })
        try:
            with self.opener(req, timeout=self.timeout) as response:
                final_url = str(response.geturl()) if callable(getattr(response, "geturl", None)) else str(url)
                final_ok, final_reason = validate_url(final_url, allow_private=self.allow_private)
                if not final_ok:
                    return FetchResult(url=final_url, error=final_reason,
                                       elapsed_ms=(time.perf_counter() - started) * 1000)
                status = int(getattr(response, "status", getattr(response, "code", 200)) or 200)
                headers = getattr(response, "headers", {})
                ctype = str(headers.get("Content-Type", "") if hasattr(headers, "get") else "")
                # Content-Length が危険なら本文を読む前に止める。
                try:
                    length = int(headers.get("Content-Length", "0") or 0)
                except (ValueError, TypeError):
                    length = 0
                if length > self.max_bytes:
                    return FetchResult(url=str(url), status=status, content_type=ctype,
                                       error="response_too_large",
                                       elapsed_ms=(time.perf_counter() - started) * 1000)
                body = response.read(self.max_bytes + 1)
                if len(body) > self.max_bytes:
                    return FetchResult(url=str(url), status=status, content_type=ctype,
                                       error="response_too_large",
                                       elapsed_ms=(time.perf_counter() - started) * 1000)
            charset = "utf-8"
            m = re.search(r"charset\s*=\s*['\"]?([\w-]+)", ctype, re.I)
            if m:
                charset = m.group(1)
            text = body.decode(charset, "replace")
            parser = _PageParser()
            parser.feed(text)
            parser.close()
            links = []
            for href, label in parser.links:
                absolute = urljoin(str(url), href)
                if absolute.startswith(("http://", "https://")):
                    links.append({"url": absolute, "text": label})
            page_text = _clean_text(" ".join(parser.text), self.max_text)
            return FetchResult(
                url=str(url), status=status, title=_clean_text(" ".join(parser.title), 300),
                text=page_text,
                description=_clean_text(parser.meta.get("description") or parser.meta.get("og:description", ""), 600),
                links=links, content_type=ctype,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except HTTPError as exc:
            return FetchResult(url=str(url), status=int(exc.code), error=f"http_{exc.code}",
                               elapsed_ms=(time.perf_counter() - started) * 1000)
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            return FetchResult(url=str(url), error=f"fetch_error:{type(exc).__name__}",
                               elapsed_ms=(time.perf_counter() - started) * 1000)
        except Exception as exc:  # pragma: no cover - defensive boundary for third-party openers
            log.debug("HTML取得に失敗: %s", exc, exc_info=True)
            return FetchResult(url=str(url), error=f"fetch_error:{type(exc).__name__}",
                               elapsed_ms=(time.perf_counter() - started) * 1000)

    # html-fetch と同じ役割を名前でも公開する（内部 API / 外部利用向け）。
    html_fetch = fetch


class EdgeSearchProvider:
    """Edge/Bing HTML 検索を行う軽量プロバイダ。"""

    name = "edge-bing-html"

    def __init__(self, fetcher: HtmlFetcher | None = None, *, endpoint: str | None = None):
        self.fetcher = fetcher or HtmlFetcher()
        self.endpoint = endpoint or os.environ.get("SNIPHER_EDGE_SEARCH_URL", _DEFAULT_SEARCH_URL)

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        url = self.endpoint.format(query=quote_plus(str(query)), count=max(1, min(10, limit)))
        page = self.fetcher.fetch(url)
        if page.error or not page.text and not page.title:
            return []
        # _PageParser の text だけでは class 情報が失われるので、リンクを記事候補として
        # ラベル順に使う。検索 HTML の anchor は fetcher.links に残る。
        parser = _SearchParser()
        # fetcher は安全性と decode を担当済みだが、テスト可能性のため page.text だけでなく
        # links も利用する。Bing の parser は raw HTML opener を差し替えた時に下記を通る。
        skip_labels = {"画像", "動画", "ニュース", "地図", "ショッピング", "検索", "サインイン", "images", "videos"}
        for link in page.links:
            title = _clean_text(link.get("text", ""), 240)
            url = _unwrap_search_url(str(link.get("url", "")))
            host = (urlparse(url).hostname or "").lower()
            if title and title.casefold() not in {x.casefold() for x in skip_labels} \
                    and host not in {"bing.com", "www.bing.com", "microsoft.com", "www.microsoft.com"}:
                parser.items.append(SearchResult(title=title, url=url))
        out = parser.finalize(self.name, limit)
        if not out:
            # FetchResult のリンクがナビゲーションだけの場合でも title/text を材料に返す。
            return []
        for i, item in enumerate(out):
            item.rank = i + 1
            item.source = self.name
            item.snippet = page.description or page.text[:360]
        return out


class DuckDuckGoProvider(EdgeSearchProvider):
    """Bing が遮断されたときの HTML-only フォールバック。"""

    name = "duckduckgo-html"

    def __init__(self, fetcher: HtmlFetcher | None = None):
        super().__init__(fetcher, endpoint=_FALLBACK_SEARCH_URL)


class ResearchEngine:
    """検索 → 上位 HTML fetch → 引用可能な材料、を束ねる内部ツール。"""

    def __init__(self, *, fetcher: HtmlFetcher | None = None,
                 providers: Iterable | None = None, policy: ResearchPolicy | None = None,
                 cache_ttl: float | None = None, cache_size: int | None = None):
        self.fetcher = fetcher or HtmlFetcher()
        self.policy = policy or ResearchPolicy()
        self.providers = list(providers) if providers is not None else [EdgeSearchProvider(self.fetcher), DuckDuckGoProvider(self.fetcher)]
        self.cache_ttl = cache_ttl if cache_ttl is not None else _env_float("SNIPHER_WEB_CACHE_TTL", 90.0)
        self.cache_size = cache_size if cache_size is not None else _env_int("SNIPHER_WEB_CACHE_SIZE", 64)
        self._cache: OrderedDict[str, tuple[float, ResearchResult]] = OrderedDict()
        self._lock = threading.RLock()

    def should_research(self, query: str, explicit: bool | None = None) -> tuple[bool, str]:
        return self.policy.decide(query, explicit)

    def fetch(self, url: str) -> FetchResult:
        return self.fetcher.fetch(url)

    # html-fetch という呼び名を内部/外部どちらにも提供する。
    html_fetch = fetch

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        seen: set[str] = set()
        output: list[SearchResult] = []
        for provider in self.providers:
            try:
                got = provider.search(query, limit=limit)
            except Exception as exc:  # pragma: no cover - provider boundary
                log.debug("検索プロバイダ %s が失敗: %s", provider, exc)
                continue
            for item in got:
                url = str(getattr(item, "url", "") or "")
                if url in seen:
                    continue
                seen.add(url)
                item.rank = len(output) + 1
                output.append(item)
                if len(output) >= limit:
                    return output
        return output

    def research(self, query: str, *, explicit: bool | None = None,
                 limit: int = 5, fetch_pages: int = 2) -> ResearchResult:
        started = time.perf_counter()
        q = _clean_text(query, 500)
        needed, reason = self.should_research(q, explicit)
        if not needed:
            return ResearchResult(query=q, needed=False, reason=reason,
                                  elapsed_ms=(time.perf_counter() - started) * 1000)
        key = q.casefold()
        with self._lock:
            cached = self._cache.get(key)
            if cached and time.time() - cached[0] < self.cache_ttl:
                result = cached[1]
                return ResearchResult(query=result.query, needed=True, reason=result.reason,
                                      results=result.results, sources=result.sources,
                                      fetched=result.fetched, elapsed_ms=(time.perf_counter() - started) * 1000,
                                      cached=True, error=result.error)
            if cached:
                self._cache.pop(key, None)
        # URL が直接送られた場合は検索エンジンを経由せず html-fetch する。
        direct = q if re.match(r"^https?://", q, re.I) else ""
        if direct:
            page = self.fetch(direct)
            fetched = [page] if not page.error and page.status < 400 else []
            sources = []
            if fetched:
                page = fetched[0]
                sources.append({"title": page.title or page.url, "url": page.url,
                                "snippet": page.description or page.text[:360],
                                "content": page.text[:4000], "provider": "html-fetch", "rank": 1})
            result = ResearchResult(query=q, needed=True, reason=reason, results=[],
                                    sources=sources, fetched=fetched,
                                    elapsed_ms=(time.perf_counter() - started) * 1000,
                                    error=None if fetched else (page.error if page else "fetch_error"))
            with self._lock:
                self._cache[key] = (time.time(), result)
                self._cache.move_to_end(key)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            return result

        results = self.search(q, limit=max(1, min(10, limit)))
        fetched: list[FetchResult] = []
        # 検索スニペットを即時の材料とし、上位ページは少数だけ本文を取る。
        # ネットワーク失敗でも search result は残るため、速度と実用性を両立する。
        targets = results[: max(0, min(3, fetch_pages))]
        if len(targets) <= 1:
            pages = [self.fetch(item.url) for item in targets]
        else:
            # 本文取得は独立しているので並列化し、検索結果の待ち時間を増やさない。
            with ThreadPoolExecutor(max_workers=len(targets), thread_name_prefix="snipher-fetch") as pool:
                futures = [pool.submit(self.fetch, item.url) for item in targets]
                pages = [f.result() for f in futures]
        for page in pages:
            if not page.error and page.status < 400:
                fetched.append(page)
        sources: list[dict] = []
        for item in results:
            sources.append({"title": item.title, "url": item.url, "snippet": item.snippet,
                            "provider": item.source, "rank": item.rank})
        for page in fetched:
            # 同じ URL の検索結果に本文を合流し、引用に使える文字量を確保する。
            for src in sources:
                if src["url"] == page.url:
                    src["title"] = page.title or src["title"]
                    src["content"] = page.text[:4000]
                    break
        result = ResearchResult(query=q, needed=True, reason=reason, results=results,
                                sources=sources, fetched=fetched,
                                elapsed_ms=(time.perf_counter() - started) * 1000,
                                error=None if results or fetched else "no_search_results")
        with self._lock:
            self._cache[key] = (time.time(), result)
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result

    def status(self) -> dict:
        with self._lock:
            return {"providers": [getattr(p, "name", type(p).__name__) for p in self.providers],
                    "cache_entries": len(self._cache), "cache_ttl": self.cache_ttl,
                    "timeout": self.fetcher.timeout, "max_bytes": self.fetcher.max_bytes}


# 公開名を短くした別名（利用者が `html_fetch` を探しても迷わないようにする）。
EdgeScraper = EdgeSearchProvider
HtmlFetch = HtmlFetcher


def html_fetch(url: str, **kwargs) -> FetchResult:
    """一回だけ安全に HTML を取得する便利関数。"""
    return HtmlFetcher(**kwargs).fetch(url)


def edge_search(query: str, *, limit: int = 5, **kwargs) -> list[SearchResult]:
    """Edge/Bing HTML 検索を一回だけ行う便利関数。"""
    return EdgeSearchProvider(**kwargs).search(query, limit=limit)


__all__ = [
    "EdgeSearchProvider", "EdgeScraper", "DuckDuckGoProvider", "FetchResult", "SearchResult",
    "ResearchEngine", "ResearchPolicy", "ResearchResult", "HtmlFetcher", "HtmlFetch",
    "html_fetch", "edge_search", "validate_url",
]
