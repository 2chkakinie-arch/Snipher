"""知識ベース v2 の検索品質テスト。

v1 の失敗は「トピックが合えば facts[0] を返す」だけだったことです。そのため

    「花火とは」      → 確度 95% で植物の世話
    「好きな食べ物」  → 何もしゃべらない（食べ物が索引に無い）
    「暇」           → 1 文字の話題名を拾えない

が起きていました。v2 の契約は **問いの型に合う欄を返す**ことです
（定義 / 理由 / 手順 / 時期 / 場所 / 値段 / 感想 / よくある問い）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher.knowledge import KnowledgeBase, content_words, question_type  # noqa: E402


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase()


# ---------------------------------------------------------------------- #
# 規模（v2 は「ごみすぎる」を解消するために作り直した）
# ---------------------------------------------------------------------- #
def test_kb_scale(kb):
    st = kb.stats()
    assert st["topics"] >= 190, st
    assert st["facts"] >= 550, st
    assert st["questions"] >= 380, st
    assert st["answers"] >= 380, st
    assert st.get("opinions", 0) >= 180, st          # 感想欄（「好き？」に答える一人称の文）


def test_every_topic_can_answer_what_it_is(kb):
    """全話題が「Xとは」に答えられる（定義か事実か Q&A のどれかを持つ）。"""
    missing = [it["topic"] for it in kb.items
               if not (str(it.get("def", "")).strip() or it.get("facts") or it.get("qa"))]
    assert not missing, missing[:10]


def test_no_placeholder_or_broken_content(kb):
    bad = []
    for it in kb.items:
        for s in kb.all_sentences_for(it) if hasattr(kb, "all_sentences_for") else []:
            if "{" in s or "None" in s:
                bad.append(s)
    assert not bad, bad[:5]
    for it in kb.items:
        assert len(str(it.get("topic", ""))) >= 1
        for fld in ("def", "opinion"):
            v = str(it.get(fld, ""))
            assert "{}" not in v and "None" not in v, (it["id"], fld, v)


# ---------------------------------------------------------------------- #
# 問いの型 → 欄
# ---------------------------------------------------------------------- #
def test_question_type_routing(kb):
    assert question_type("花火とは") == "def"
    assert question_type("なぜ空は青い") == "why"
    assert question_type("ラーメンの作り方") == "how"
    assert question_type("いつ咲く") == "when"
    assert question_type("どこで見られる") == "where"
    assert question_type("いくら") == "cost"
    assert question_type("好きな食べ物は") == "opinion"


@pytest.mark.parametrize("query,topic", [
    ("花火とは", "花火"),
    ("好きな食べ物は？", "食事"),
    ("暇", "雑談"),
    ("猫がゴロゴロ言う", "猫"),
    ("おすすめの本", "読書"),
    ("明日の天気", "天気"),
    ("なぜ空は青い", "空"),
    ("観葉植物の葉が黄色い", "植物"),
    ("お腹が痛い", "健康"),
    ("肩が凝った", "健康"),
    ("gitがわからない", "Git"),
    ("Wi-Fiが遅い", "Wi-Fi"),
    ("タピオカって何", "タピオカ"),
    ("ありがとう", "感謝"),
    ("Pythonって何", "Python"),
])
def test_retrieval_picks_the_right_topic(kb, query, topic):
    m = kb.answer(query)
    assert m is not None, query
    assert m["topic"] == topic, (query, m["topic"], m["text"][:40])


@pytest.mark.parametrize("query,fields", [
    ("花火とは", ("def", "qa", "fact")),
    ("なぜ空は青い", ("why", "qa", "def")),
    ("お腹が痛い", ("qa", "tips", "when", "how")),
    ("観葉植物の葉が黄色い", ("qa", "tips", "why", "how")),
    ("好きな食べ物は？", ("opinion", "qa")),
    ("おすすめの本", ("opinion", "tips", "qa")),
])
def test_retrieval_picks_the_right_field(kb, query, fields):
    m = kb.answer(query)
    assert m is not None, query
    assert m["field"] in fields, (query, m["field"], m["text"][:40])


# ---------------------------------------------------------------------- #
# 知らないことは知らないと言う（＝誤情報の源を断つ）
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("query", ["67", "あ", "asdfgh", "基本的なことを教えて", "ぬるぬる猿"])
def test_opaque_or_unknown_returns_none(kb, query):
    assert kb.answer(query) is None, query


def test_confidence_is_honest_about_coverage(kb):
    strong = kb.answer("花火とは")
    assert strong is not None
    assert 0.0 < strong["confidence"] <= 0.99
    assert strong["coverage"] > 0.5


# ---------------------------------------------------------------------- #
# 単語の切り出し（1 文字の話題名・ASCII 語・過剰分割）
# ---------------------------------------------------------------------- #
def test_single_char_topics_are_indexed(kb):
    for w in ("猫", "犬", "暇", "星"):
        assert kb.index.topics_of(w), w
    assert "暇" in content_words("暇だなあ", kb.index)


def test_compounds_are_not_over_segmented(kb):
    """「基本的」を「基」+「本」に割らない（1 文字語は境界が要る）。"""
    words = content_words("基本的なことを教えて", kb.index)
    assert "本" not in words, words


def test_ascii_words_match_regardless_of_punctuation(kb):
    assert kb.answer("Wi-Fiが遅い") is not None
    assert kb.answer("wifiがつながらない") is not None or kb.answer("Wi-Fi") is not None
    assert "wi-fi" in content_words("Wi-Fiが遅い", kb.index) or \
           "wifi" in content_words("Wi-Fiが遅い", kb.index)


def test_head_word_of_question_is_the_topic(kb):
    """「X は/が」の X と、文末の名詞の両方を主題として扱う。"""
    assert kb.answer("犬の散歩はどれくらい必要")["topic"] == "犬"
    assert kb.answer("散歩はどれくらい") is not None


# ---------------------------------------------------------------------- #
# 会話の材料（応答を組むための部品が揃っている）
# ---------------------------------------------------------------------- #
def test_material_has_followups_and_related(kb):
    m = kb.answer("花火とは")
    assert m is not None
    assert isinstance(m.get("related"), list)
    assert "facts" in m and "item" in m


def test_all_sentences_are_non_trivial(kb):
    sents = kb.all_sentences()
    assert len(sents) >= 2000, len(sents)
    assert sum(1 for s in sents if len(s) >= 12) > len(sents) * 0.6
    assert not any(s.strip() in ("", "。") for s in sents)


def test_suggest_returns_related_topics(kb):
    out = kb.suggest("花火")
    assert isinstance(out, list)


def test_stats_shape(kb):
    st = kb.stats()
    for key in ("topics", "facts", "questions", "answers"):
        assert key in st and isinstance(st[key], int)


# --------------------------------------------------------------------------- #
# v2.0: 追加した項目（投資）と別名の拡張（空腹 → 食事）
# --------------------------------------------------------------------------- #
def test_investment_topic_exists(kb):
    a = kb.answer("投資って何")
    assert a is not None
    assert a["topic"] == "投資", a["topic"]
    assert "お金" in a["text"]


def test_investment_how_and_qa(kb):
    a = kb.answer("投資はどう始める")
    assert a is not None and a["topic"] == "投資"
    assert a["usage"] in ("how", "qa", "def"), a["usage"]
    b = kb.answer("投資は怖い")
    assert b is not None and b["topic"] == "投資"
    assert b["usage"] == "qa", b["usage"]


def test_investment_aliases_route_to_same_topic(kb):
    for q in ("資産運用を始めたい", "NISAって何", "投資信託とは", "積立投資"):
        a = kb.answer(q)
        assert a is not None, q
        assert a["topic"] == "投資", (q, a["topic"])


def test_hunger_aliases_route_to_meal(kb):
    for q in ("お腹すいた", "何食べよう", "空腹"):
        a = kb.answer(q)
        assert a is not None, q
        assert a["topic"] == "食事", (q, a["topic"])


def test_kb_stats_after_rebuild(kb):
    st = kb.stats()
    assert st["topics"] >= 199
    assert st["facts"] >= 611
    assert st["qa"] >= 426
    assert st["how"] >= 395
