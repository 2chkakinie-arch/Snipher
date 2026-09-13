"""v4 の *振る舞い* 契約 — 辞書引き・定型棚・でっち上げを戻さないための検査。

`tools/bench_v4.py` が広く浅く、ここで狭く深く（特定の壊れ方）を見ます。
"""

from __future__ import annotations

import re

import pytest

from snipher.composer import Composer
from snipher.instruction.answer import normalized_word, question_subject
from snipher.instruction.parser import parse, parse_question
from snipher.instruction.style import _plain_sentence, restyle, retime
from snipher.knowledge import KnowledgeBase
from snipher.mind import chat
from snipher.mind.parse import build_frame

BANNED = ("拍", "索引に", "品詞 ", "U+30", "読みは「", "語彙バンク", "できません", "出来ません",
          "分かりません", "わかりません", "手元に無い")


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase.shared()


@pytest.fixture(scope="module")
def composer(kb) -> Composer:
    return Composer(kb=kb)


def _text(composer: Composer, prompt: str, **kw) -> str:
    r = composer.compose(prompt, web=False, **kw)
    return r.text


# --------------------------------------------------------------------------- #
# 1) 指示の読み取り
# --------------------------------------------------------------------------- #
def test_length_spec_makes_it_an_instruction() -> None:
    d = parse("質問：Redis とは？200文字程度で。")
    assert d is not None and d.task == "answer" and d.fmt.target_chars == 200


def test_bare_question_is_not_an_instruction() -> None:
    assert parse("Redisとは？") is None
    assert parse("こんにちは") is None


def test_question_drops_trailing_format_words() -> None:
    assert parse_question("質問：Redis とは？200文字程度で。") == "Redis とは？"
    assert parse_question("質問：dockerとpodmanの違いを教えて。200文字程度、である調。") \
        == "dockerとpodmanの違いを教えて"


def test_subject_strips_parenthetical_note() -> None:
    assert question_subject("WebAssembly（Wasm）をブラウザで動かす一番のメリットは何ですか？") \
        == "WebAssembly"


def test_length_target_is_reached_with_same_topic_material(composer) -> None:
    got = _text(composer, "質問：Redis とは？200文字程度で。")
    assert 120 <= len(got.replace("\n", "")) <= 300, got
    for b in BANNED:
        assert b not in got


# --------------------------------------------------------------------------- #
# 2) 口調・文体の組み替え
# --------------------------------------------------------------------------- #
def test_question_marker_survives_plain_conversion() -> None:
    assert _plain_sentence("Docker で困っていることはありますか。").rstrip("？?").endswith("か")
    assert _plain_sentence("どの処理を速くしたいですか。").rstrip("？?").endswith("か")


def test_friendly_tone_keeps_questions_and_alternates_tails() -> None:
    got, _ = restyle("探索を削減できます。静的なサイトは速いです。", tone="friendly_professional")
    assert "だよ" in got and "だね" in got, got          # 指定された両方の語尾を使う
    q, _ = restyle("どの処理を速くしたいですか。", tone="friendly_professional")
    assert q.endswith("？") and "だよ" not in q and "だね" not in q, q


def test_retime_empty_tail_is_not_treated_as_unspecified() -> None:
    assert retime("どの処理を速くしたいですか。", "friendly_professional",
                  tail_override="").endswith("？")


# --------------------------------------------------------------------------- #
# 3) 会話の受け取り
# --------------------------------------------------------------------------- #
def test_predicate_utterance_keeps_the_users_own_word(kb) -> None:
    cl = chat.claims_for("疲れた", kb=kb, turn=2)
    text = " ".join(c.content for c in cl)
    assert "疲" in text, text


def test_unrelated_topic_words_are_not_dragged_in(kb) -> None:
    """語 1 文字の縁で別話題を混ぜない（「申年休假」にアルゴリズムの話は不要）。"""
    items = chat.topic_items("申年休假を申請する手順を教えて", kb=kb, limit=3, min_score=0.3)
    assert all(i.get("topic") not in ("アルゴリズム", "品詞", "Docker") for i in items), items


def test_leave_procedure_is_reachable_from_its_own_words(kb) -> None:
    topics = [i.get("topic") for i in chat.topic_items("休暇 申請 手順", kb=kb, limit=3, min_score=0.3)]
    assert "有給休暇" in topics, topics


def test_step_phrase_needs_two_items(kb) -> None:
    for turn in range(1, 9):
        for c in chat.claims_for("今日は天気が良いので公園に散歩に行こうと思います。", kb=kb, turn=turn):
            if "の順で" in c.content:
                assert c.content.count("、") >= 1, c.content


def test_reply_does_not_reuse_the_same_connector(composer) -> None:
    got = _text(composer, "今日は天気が良いので公園に散歩に行こうと思います。")
    for conn in ("実務的には、", "加えて、", "また、", "それと、", "参考までに、"):
        assert got.count(conn) <= 1, (conn, got)


# --------------------------------------------------------------------------- #
# 4) 知らない語
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("word,expect", [
    ("申年休假", ""),          # 假は別字なので推測しない
    ("プルントゥーラ", ""),     # カタカナ語を既知語に化かさない
    ("ゾルタクス＝ゼッカ", ""),
])
def test_normalization_does_not_invent_a_word(word, expect, kb) -> None:
    got, _note = normalized_word(word, kb=kb)
    assert got == expect, (word, got)


def test_unknown_word_answers_with_a_plan_not_a_refusal(composer) -> None:
    got = _text(composer, "プルントゥーラ・クラスターって導入する価値ある？")
    assert got and "できません" not in got and "無い" not in got
    for b in BANNED:
        assert b not in got, b


def test_no_dictionary_trivia_in_unknown_topic(composer) -> None:
    got = _text(composer, "ぬるぬる猿の生態について")
    for b in BANNED:
        assert b not in got, (b, got)


def test_field_is_not_invented_when_no_kb_slot(composer) -> None:
    """知識ベースの *欄* を引いていないときに field/usage を名乗らせない（v3 契約）。"""
    r = composer.compose("ぬるぬる猿の生態について", web=False)
    assert (r.knowledge or {}).get("field") is None, r.knowledge
    assert r.confidence <= 0.55, r.confidence


# --------------------------------------------------------------------------- #
# 5) 自己紹介
# --------------------------------------------------------------------------- #
def test_self_intro_names_the_assistant(composer) -> None:
    got = _text(composer, "自己是？名前とできることを短く教えて")
    assert "Snipher" in got, got
    assert len(got) <= 220, got


def test_capability_answer_does_not_brag_lexicon_counts(composer) -> None:
    got = _text(composer, "何ができるの？")
    assert re.search(r"\d", got), got
    assert "語彙バンク" not in got and "拍" not in got and "品詞" not in got

# --------------------------------------------------------------------------- #
# 6) 返答の形そのもの（ユーザーが禁じた出方）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prompt", [
    "WebAssembly（Wasm）をブラウザで動かす一番のメリットは何ですか？",
    "dockerとpodmanの違いは？",
    "空が青いのはなぜ？",
    "申年休假を申請する手順を教えて。",
    "プルントゥーラ・クラスターって導入する価値ある？",
    "今日は天気が良いので公園に散歩に行こうと思います。",
    "は？",
])
def test_replies_never_take_the_dictionary_shape(composer, prompt) -> None:
    got = _text(composer, prompt)
    assert got, prompt
    for b in BANNED:
        assert b not in got, (b, got)
    assert not re.search(r"表記\s*[：:]", got), got
    assert not re.search(r"同じ\s*\d+\s*拍", got), got


def test_same_connector_is_not_repeated(composer) -> None:
    got = _text(composer, "今日のニュースは？")
    for conn in ("また、", "加えて、", "それと、", "実務的には、", "なお、"):
        assert got.count(conn) <= 1, (conn, got)


def test_unknown_head_word_does_not_pump_confidence(composer) -> None:
    r = composer.compose("ぬるぬる猿の生態について", web=False)
    assert r.confidence <= 0.55, r.confidence


def test_procedure_ask_gets_procedure_like_answer(composer) -> None:
    got = _text(composer, "有給休暇を申請する手順を教えて。")
    assert re.search(r"申請|伝える|伝える|出す|決める", got), got
    assert "手順" in got or "\n" in got or "。" in got, got


def test_answer_leads_with_the_content_not_a_topic_label(composer) -> None:
    got = _text(composer, "WebAssembly（Wasm）をブラウザで動かす一番のメリットは何ですか？")
    first = got.split("\n")[0]
    assert not first.startswith("WebAssembly は"), first     # 見出しの書き出しを避ける
    assert re.search(r"速", first), first
