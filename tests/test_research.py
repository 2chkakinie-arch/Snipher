"""Edge HTML 検索と html-fetch のネットワークなしテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher.research import (  # noqa: E402
    EdgeSearchProvider,
    FetchResult,
    HtmlFetcher,
    ResearchEngine,
    ResearchPolicy,
    validate_url,
)


_HTML = """<!doctype html>
<html><head><title>検索結果</title><meta name="description" content="概要です"></head>
<body><script>ignored()</script>
<li class="b_algo"><h2><a href="https://example.org/article">記事タイトル</a></h2>
<p>検索結果の概要です。</p></li></body></html>"""


class _Response:
    status = 200
    headers = {"Content-Type": "text/html; charset=utf-8", "Content-Length": "512"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, _limit=-1):
        return _HTML.encode("utf-8")


def _opener(request, timeout=0):
    assert "Edg/" in request.headers.get("User-agent", "")
    assert timeout > 0
    return _Response()


def test_html_fetch_extracts_title_text_and_links_without_browser():
    page = HtmlFetcher(opener=_opener).fetch("https://example.org")
    assert page.status == 200
    assert page.title == "検索結果"
    assert "ignored" not in page.text
    assert "検索結果の概要です" in page.text
    assert page.links[0]["url"] == "https://example.org/article"
    assert page.description == "概要です"


def test_edge_search_scrapes_result_links_and_is_injectable():
    provider = EdgeSearchProvider(
        HtmlFetcher(opener=_opener),
        endpoint="https://www.bing.com/search?q={query}&count={count}",
    )
    results = provider.search("Snipher", limit=3)
    assert len(results) == 1
    assert results[0].title == "記事タイトル"
    assert results[0].url == "https://example.org/article"
    assert results[0].source == "edge-bing-html"


def test_policy_only_selects_web_when_current_or_explicit():
    policy = ResearchPolicy()
    assert policy.decide("2+2を計算") == (False, "local_task")
    assert policy.decide("Node.jsのコードを書いて") == (False, "local_task")
    assert policy.decide("今日のニュース") == (True, "current_information")
    assert policy.decide("Pythonの公式ドキュメントを検索して") == (True, "explicit_keyword")
    assert policy.decide("猫とは何ですか") == (False, "stable_knowledge")
    assert policy.decide("https://example.org", explicit=False) == (False, "disabled")


def test_research_engine_returns_sources_and_reuses_cache():
    provider = EdgeSearchProvider(HtmlFetcher(opener=_opener), endpoint="https://bing.example/{query}")
    engine = ResearchEngine(fetcher=HtmlFetcher(opener=_opener), providers=[provider], cache_ttl=60)
    first = engine.research("最新のSnipherニュース", limit=2, fetch_pages=0)
    second = engine.research("最新のSnipherニュース", limit=2, fetch_pages=0)
    assert first.needed is True
    assert first.sources[0]["url"] == "https://example.org/article"
    assert second.cached is True


def test_direct_url_uses_html_fetch_not_search():
    engine = ResearchEngine(fetcher=HtmlFetcher(opener=_opener), providers=[])
    result = engine.research("https://example.org/article", fetch_pages=0)
    assert result.sources[0]["provider"] == "html-fetch"
    assert result.sources[0]["title"] == "検索結果"


def test_private_hosts_are_blocked_before_network():
    assert validate_url("http://127.0.0.1:8000")[0] is False
    assert validate_url("file:///etc/passwd")[0] is False
    assert validate_url("https://example.org")[0] is True
