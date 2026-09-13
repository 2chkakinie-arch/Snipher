"""内部化されたネット裏取り（snipher.ground.web + research）の実測テスト。

検索エンジン HTML を *ページ全体の文字列* として扱わず結果ブロック単位で読むこと、
広告・サイト側の案内文（リワード / サインイン / 表現の自由 …）を回答に混ぜないこと、
html-fetch が既定でローカル・リンクローカルアドレスを弾くことを、
ローカルフィクスチャ（tests/web_fixtures.py）のサーバで実測する。
"""

from __future__ import annotations

import pytest

from snipher.ground.web import (
    BOILERPLATE,
    WebGrounding,
    is_boilerplate,
    query_terms,
    score_sentence,
    split_sentences,
)
from snipher.research import SerpParser, validate_url
from web_fixtures import SERP, engine, grounding, server  # noqa: F401  (fixtures)


# --------------------------------------------------------------------------- #
# SERP を「ブロック単位」で読む
# --------------------------------------------------------------------------- #
def test_serp_parser_reads_result_blocks_only() -> None:
    p = SerpParser(limit=10)
    p.feed(SERP.format(base="http://example.com"))
    p.close()
    got = p.finalize("test")
    assert [x.title for x in got] == ["GLM5.3 とは - 例示事典", "GLM5.3 に関する別のページ"]
    # 広告・検索エンジン自身の計算回答ブロックは候補にしない
    assert not any("リワード" in (x.title + x.snippet) for x in got)
    assert not any("表現の自由" in (x.title + x.snippet) for x in got)
    # ページ下部の chrome（Cookie 案内）がスニペットに混ざらない
    assert all("Cookie" not in x.snippet for x in got)


def test_snippets_are_per_result_not_page_level() -> None:
    p = SerpParser(limit=10)
    p.feed(SERP.format(base="http://example.com"))
    p.close()
    first = p.finalize("test")[0]
    assert "2026 年 4 月" in first.snippet
    assert len(first.snippet) < 200


# --------------------------------------------------------------------------- #
# 証拠文の品質
# --------------------------------------------------------------------------- #
def test_boilerplate_blacklist_blocks_search_engine_chrome() -> None:
    for bad in ["本日はサービスを停止しています。表現の自由について。",
                "リワードでポイントがもらえます。サインインしてください。",
                "This site uses cookies. Privacy policy."]:
        assert is_boilerplate(bad), bad
    assert not is_boilerplate("GLM5.3 は 2026 年 4 月に公開された会話用の基盤モデルです。")
    assert any("リワード" in x for x in BOILERPLATE)


def test_query_terms_strips_question_chrome() -> None:
    terms = query_terms("GLM5.3とは何ですか？")
    assert "glm5.3" in [t.lower() for t in terms]
    assert not any("とは" in t or "ですか" in t for t in terms)


def test_score_sentence_prefers_on_topic_sentences() -> None:
    terms = ["glm5.3", "glm"]
    good = "GLM5.3 は会話用の基盤モデルです。"
    junk = "このページでは検索の仕方をご案内しています。"
    assert score_sentence(good, terms) > score_sentence(junk, terms)


def test_split_sentences_bounds() -> None:
    text = "短文です。" + ("とても長い" * 40) + "で終わります。"
    out = split_sentences(text)
    assert out and all(12 <= len(s) <= 230 for s in out)


# --------------------------------------------------------------------------- #
# html-fetch
# --------------------------------------------------------------------------- #
def test_private_hosts_are_blocked_by_default() -> None:
    assert validate_url("http://127.0.0.1:80/x")[0] is False
    assert validate_url("http://169.254.169.254/latest/meta-data/")[0] is False
    assert validate_url("file:///etc/passwd")[0] is False
    assert validate_url("http://127.0.0.1:80/x", allow_private=True)[0] is True


def test_fetch_returns_text_and_raw_html(engine) -> None:
    got = engine.fetch(engine.base_url + "/glm")
    assert got.error is None
    assert "GLM5.3" in got.text
    assert "<article>" in (got.raw_html or "")          # 構造解析に使う原文が残っている


def test_search_uses_the_fixture_serp(engine) -> None:
    got = engine.search("GLM5.3", limit=5)
    assert got
    assert any("/glm" in x.url for x in got)
    assert not any("/ad" in x.url or "/ans" in x.url for x in got)


def test_grounding_gathers_evidence_from_search_then_fetch(grounding) -> None:
    got = grounding.gather("GLM5.3とは", explicit=True)
    assert got.ok, got.as_dict()
    texts = " ".join(e.text for e in got.evidence)
    assert "GLM5.3" in texts
    for banned in ("リワード", "サインイン", "表現の自由", "Cookie", "アクセスが集中"):
        assert banned not in texts
    assert any("2026" in e.text for e in got.evidence)   # 本文から抜いた文が並ぶ
    assert got.sources and all(s.get("url") for s in got.sources)
    assert not any("リワード" in str(s.get("title", "")) for s in got.sources)
    assert got.elapsed_ms > 0


def test_grounding_reports_failure_without_inventing(server) -> None:
    from snipher.research import EdgeSearchProvider, HtmlFetcher, ResearchEngine

    broken = ResearchEngine(
        fetcher=HtmlFetcher(timeout=0.4, allow_private=True),
        providers=[EdgeSearchProvider(HtmlFetcher(timeout=0.4, allow_private=True),
                                      endpoint="http://127.0.0.1:1/search?q={query}&count={count}")],
    )
    wg = WebGrounding(broken, enabled=True)
    got = wg.gather("まったく未知の語彙xyz", explicit=True)
    assert not got.ok
    assert got.error                                      # 失敗をごまかさない


def test_disabled_grounding_is_inert() -> None:
    wg = WebGrounding(None, enabled=False)
    got = wg.gather("GLM5.3とは", explicit=True)
    assert got.error == "web_disabled" and not got.evidence
