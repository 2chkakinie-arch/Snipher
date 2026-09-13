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
import json
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
    raw_html: str = ""            # 検索結果 HTML を構造で読むための原文（上限あり）

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "status": self.status,
            "raw_chars": len(self.raw_html),
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
                 allow_private: bool | None = None, max_raw_bytes: int | None = None):
        # ネットワークが無い環境でもチャットを長く止めない。必要なら環境変数で延長可能。
        self.timeout = timeout if timeout is not None else _env_float("SNIPHER_WEB_TIMEOUT", 1.5)
        self.max_bytes = max_bytes if max_bytes is not None else _env_int("SNIPHER_WEB_MAX_BYTES", 1_500_000)
        self.max_text = max_text if max_text is not None else _env_int("SNIPHER_WEB_MAX_TEXT", 12_000)
        self.opener = opener or urlopen
        self.allow_private = bool(allow_private) if allow_private is not None else os.environ.get("SNIPHER_WEB_ALLOW_PRIVATE") == "1"
        # 検索結果ページを *構造で* 解析するための原文バッファ（既定 360 KB）。
        self.max_raw_bytes = max_raw_bytes if max_raw_bytes is not None else _env_int(
            "SNIPHER_WEB_MAX_RAW", 360_000)

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
            raw = text[: self.max_raw_bytes] if self.max_raw_bytes else ""
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
                links=links, content_type=ctype, raw_html=raw,
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


_RESULT_CLS = ("b_algo", "result__body", "e algo", "algo", "result results_links",
               "titledLink", "result", "b_results")
_AD_CLS = ("b_ad", "b_ps", "adsbygoogle", "sponsored", "b_topAl", "b_adt", "b_ans",
           "b_vTPay", "ppc", "promo", "organic-ad", "b_slidebar")
_NAV_HOSTS = {"bing.com", "www.bing.com", "microsoft.com", "www.microsoft.com",
              "go.microsoft.com", "duckduckgo.com", "duck.co", "yahoo.co.jp"}


class SerpParser(HTMLParser):
    """検索結果 HTML を「1 件 = 1 ブロック」として読む。

    v2 まではページ全体のリンクと説明を拾っていたため、Bing 自身の案内文
    （リワード・サインイン等）が *回答の本文* として混ざりました。ここからは
    li/div のクラスで結果ブロックを決め、ブロック内の h2/h3 アンカーと
    p の説明文だけに対応づけます。広告・計算回答ブロックは捨てます。
    """

    def __init__(self, *, limit: int = 10):
        super().__init__(convert_charrefs=True)
        self.items: list[SearchResult] = []
        self._limit = limit
        self._stack: list[dict] = []

    @staticmethod
    def _cls(attrs: list[tuple[str, str | None]]) -> str:
        d = {str(k).lower(): str(v or "") for k, v in attrs}
        return (d.get("class", "") + " " + d.get("id", "")).lower()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        cls = self._cls(attrs)
        if tag in {"li", "div", "ol", "section", "article"} and any(c in cls for c in _RESULT_CLS):
            block = {"tag": tag, "ad": any(a in cls for a in _AD_CLS), "title": "",
                     "href": "", "tbuf": None, "pbuf": [], "snippet": [],
                     "in_title": False, "skip": 0, "nested": 0}
            if self._stack:
                # 内側に結果ブロックを持つ外枠（ol.b_results 等）は、自分では出力しない。
                # さもないと広告ブロックの語が外枠の題目として漏れます（v2 のバグ）。
                self._stack[-1]["nested"] = int(self._stack[-1].get("nested", 0)) + 1
            self._stack.append(block)
        if not self._stack:
            return
        top = self._stack[-1]
        d = {str(k).lower(): str(v or "") for k, v in attrs}
        if tag in {"h1", "h2", "h3"} and not top["title"]:
            top["in_title"] = True
        if tag == "a" and top["in_title"] and not top["href"]:
            top["href"] = d.get("href", "")
            top["tbuf"] = []
        if tag == "p":
            top["pbuf"].append([])
        if tag in {"script", "style", "noscript"}:
            top["skip"] += 1

    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        top = self._stack[-1]
        if tag in {"h1", "h2", "h3"}:
            top["in_title"] = False
        if tag == "a" and top["tbuf"] is not None:
            top["title"] = _clean_text("".join(top["tbuf"]), 260)
            top["tbuf"] = None
        if tag == "p" and top["pbuf"]:
            buf = top["pbuf"].pop()
            text = _clean_text("".join(buf), 600)
            if text and len(text) >= 12:
                top["snippet"].append(text)
        if tag in {"script", "style", "noscript"}:
            top["skip"] = max(0, top["skip"] - 1)
        if tag == top["tag"]:
            self._pop_block()

    def handle_data(self, data: str) -> None:
        if not self._stack:
            return
        top = self._stack[-1]
        if top["skip"]:
            return
        if top["tbuf"] is not None:
            top["tbuf"].append(data)
        if top["pbuf"]:
            top["pbuf"][-1].append(data)

    def _pop_block(self) -> None:
        top = self._stack.pop()
        if top["ad"] or int(top.get("nested", 0) or 0) > 0:
            return
        title = top.get("title", "")
        href = html.unescape(top.get("href", "") or "")
        if not title or not href:
            return
        url = _unwrap_search_url(href)
        if not url.startswith(("http://", "https://")):
            return
        host = (urlparse(url).hostname or "").lower()
        if host in _NAV_HOSTS:
            return
        snippet = _clean_text(" ".join(top["snippet"])[:600], 600)
        self.items.append(SearchResult(title=title[:240], url=url, snippet=snippet,
                                       rank=len(self.items) + 1))
        if len(self.items) >= self._limit:
            self._stack.clear()

    def finalize(self, provider: str) -> list[SearchResult]:
        for i, item in enumerate(self.items, 1):
            item.source = provider
            item.rank = i
        return self.items



class EdgeSearchProvider:
    """Edge/Bing の検索 HTML を取り、*結果ブロック単位*で読む軽量プロバイダ。"""

    name = "edge-bing-html"

    def __init__(self, fetcher: HtmlFetcher | None = None, *, endpoint: str | None = None):
        self.fetcher = fetcher or HtmlFetcher()
        self.endpoint = endpoint or os.environ.get("SNIPHER_EDGE_SEARCH_URL", _DEFAULT_SEARCH_URL)

    def _parse(self, page: FetchResult, limit: int) -> list[SearchResult]:
        out: list[SearchResult] = []
        raw = getattr(page, "raw_html", "") or ""
        if raw:
            parser = SerpParser(limit=limit)
            try:
                parser.feed(raw)
                parser.close()
                out = parser.finalize(self.name)
            except Exception:  # noqa: BLE001
                log.debug("SERP の構造解析に失敗", exc_info=True)
                out = []
        if out:
            return out[:limit]
        # 縮退: アンカーの羅列から見出し候補だけ拾う。スニペットは空のままにし、
        # ページ全体の文字列を「根拠」として流し込むことは絶対にしない。
        skip = {"画像", "動画", "ニュース", "地図", "ショッピング", "検索", "サインイン",
                "images", "videos", "maps", "more", "menu"}
        for link in page.links[: limit * 8]:
            title = _clean_text(str(link.get("text", "")), 240)
            url = _unwrap_search_url(str(link.get("url", "")))
            host = (urlparse(url).hostname or "").lower()
            if not title or title.casefold() in {x.casefold() for x in skip}:
                continue
            if host in _NAV_HOSTS or not url.startswith(("http://", "https://")):
                continue
            if len(title) < 8 or not re.search(r"[ぁ-んァ-ヶ一-龯A-Za-z]{3,}", title):
                continue
            if any(url.startswith(x) for x in ("#", "javascript:")):
                continue
            out.append(SearchResult(title=title, url=url, snippet="", source=self.name,
                                    rank=len(out) + 1))
            if len(out) >= limit:
                break
        return out

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        url = self.endpoint.format(query=quote_plus(str(query)), count=max(1, min(10, limit)))
        page = self.fetcher.fetch(url)
        if page.error and not (getattr(page, "raw_html", "") or ""):
            return []
        return self._parse(page, max(1, min(10, limit)))


class DuckDuckGoProvider(EdgeSearchProvider):
    """Bing が遮断されたときの HTML-only フォールバック。"""

    name = "duckduckgo-html"

    def __init__(self, fetcher: HtmlFetcher | None = None):
        super().__init__(fetcher, endpoint=_FALLBACK_SEARCH_URL)


class WikipediaApiProvider:
    """日本語 Wikipedia の opensearch/summary API（HTML スクレイピングより速く正確）。"""

    name = "wikipedia-api"
    API = "https://ja.wikipedia.org/w/api.php"

    def __init__(self, fetcher: HtmlFetcher | None = None):
        self.fetcher = fetcher or HtmlFetcher()

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        url = (f"{self.API}?action=query&list=search&srsearch={quote_plus(str(query))}"
               f"&srlimit={max(1, min(10, limit))}&format=json&srprop=snippet|words&utf8=1")
        page = self.fetcher.fetch(url)
        if page.error or not page.text:
            return []
        try:
            data = json.loads(page.text)
        except Exception:  # noqa: BLE001
            return []
        out: list[SearchResult] = []
        for hit in (data.get("query", {}) or {}).get("search", []) or []:
            title = _clean_text(str(hit.get("title", "")), 200)
            snippet = re.sub(r"<[^>]+>", "", str(hit.get("snippet", "")))
            snippet = _clean_text(snippet, 480)
            if not title:
                continue
            out.append(SearchResult(title=title,
                                    url="https://ja.wikipedia.org/wiki/" + quote_plus(title),
                                    snippet=snippet, source=self.name, rank=len(out) + 1))
        return out[:limit]



class ResearchEngine:
    """検索 → 上位 HTML fetch → 引用可能な材料、を束ねる内部ツール。"""

    def __init__(self, *, fetcher: HtmlFetcher | None = None,
                 providers: Iterable | None = None, policy: ResearchPolicy | None = None,
                 cache_ttl: float | None = None, cache_size: int | None = None):
        self.fetcher = fetcher or HtmlFetcher()
        self.policy = policy or ResearchPolicy()
        if providers is not None:
            self.providers = list(providers)
        else:
            self.providers = [EdgeSearchProvider(self.fetcher), DuckDuckGoProvider(self.fetcher)]
            if os.environ.get("SNIPHER_WEB_WIKIPEDIA", "1") != "0":
                from .research import WikipediaApiProvider

                self.providers.append(WikipediaApiProvider(self.fetcher))
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
