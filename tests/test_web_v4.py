"""v4 のネット裏取り: 未知語でも検索 → html-fetch → 応答までを *実経路* で測る。

ネットワークに出ない CI でも通るよう、すべて tests/web_fixtures.py のローカル
mini web（Bing と同じ class 構造の SERP + 記事ページ）に対して検証します。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from web_fixtures import SERP, engine, grounding, server  # noqa: F401  (fixtures)

from snipher.ground.web import WebGrounding
from snipher.instruction.run import run as run_instruction
from snipher.research import ResearchEngine

ANSWER_PROMPT = (
    "以下の質問に、です・ます調で150文字程度で答えてください。\n\n"
    "質問：GLM5.3 とは何ですか？"
)


# --------------------------------------------------------------------------- #
# 既定値（v3 は 1.5 秒で切れていて、未知語のときに答えが薄かった）
# --------------------------------------------------------------------------- #
def test_fetch_timeout_default_is_generous() -> None:
    eng = ResearchEngine()
    assert eng.fetcher.timeout >= 4.0, eng.fetcher.timeout


def test_cache_keeps_failures_shorter_than_success() -> None:
    eng = ResearchEngine()
    assert eng.cache_ttl_fail < eng.cache_ttl


# --------------------------------------------------------------------------- #
# 裏取りそのもの
# --------------------------------------------------------------------------- #
def test_unknown_term_is_answered_from_the_web(grounding) -> None:  # noqa: F811
    """手元に無い語でも、検索と本文から証拠文が取れる。"""
    out = grounding.gather("GLM5.3 とは何ですか？", explicit=True)
    assert out.ok, out.as_dict()
    text = " ".join(e.text for e in out.evidence)
    assert "GLM5.3" in text
    assert "2026" in text or "基盤モデル" in text
    # chrome・広告は証拠に入らない
    for banned in ("リワード", "サインイン", "アカウントを選択", "Cookie"):
        assert banned not in text


def test_instruction_answer_cites_the_fetched_page(grounding) -> None:  # noqa: F811
    res = run_instruction(ANSWER_PROMPT, web=grounding)
    assert res is not None and res.ok
    assert "GLM5.3" in res.text
    assert res.meta.get("sources"), res.meta
    for banned in ("できません", "分かりません", "検索に出られない", "手元に無い"):
        assert banned not in res.text, banned


def test_composer_uses_the_same_web_window(grounding) -> None:  # noqa: F811
    """composer は同じ裏取り窓を使い、検索が通らないときも辞書引きに逃げない。"""
    from snipher.composer import Composer
    from snipher.knowledge import KnowledgeBase

    c = Composer(kb=KnowledgeBase.shared())
    c._web = grounding
    r = c.compose("GLM5.3 とは何ですか？", web=True)
    assert r.text
    assert not any(x in r.text for x in ("拍", "品詞", "索引", "語彙バンク", "できません")), r.text


# --------------------------------------------------------------------------- #
# 通らないネットワークで会話を止めない・毎回待たない
# --------------------------------------------------------------------------- #
def test_repeated_network_failures_put_the_grounding_on_cooldown() -> None:
    class DeadEngine:
        status_calls = 0

        def research(self, q, *, explicit=False, limit=3, fetch_pages=1):
            raise OSError("network unreachable")

        def status(self) -> dict:
            return {"providers": []}

    g = WebGrounding(DeadEngine(), backoff=60.0)
    for _ in range(3):
        out = g.gather("からっぽの語 について")
        assert not out.ok
    assert g.available() is False          # 失敗が続いたら少し休む
    g2 = WebGrounding(DeadEngine(), backoff=0.0)
    for _ in range(3):
        g2.gather("からっぽの語 について")
    assert g2.available() is True          # backoff=0 は休まない


def test_off_switch_is_respected() -> None:
    g = WebGrounding(ResearchEngine(), enabled=False)
    assert g.available() is False
    out = g.gather("GLM5.3 とは")
    assert out.error == "web_disabled"
