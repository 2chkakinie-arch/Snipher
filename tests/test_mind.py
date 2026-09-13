"""思考層（snipher.mind）— 分解 → 規則 → 状態 → 遊び → 合成 の各段。

ここでは「話題ごとの特例」を一切テストしない。同じ 1 本の道が
異なる主題（遊び・規則・暦・未知語）をどう横断するかだけを見る。
"""

from __future__ import annotations

import pytest

from snipher.mind import play, rules as R
from snipher.mind.frame import Claim, Frame
from snipher.mind.parse import build_frame, classify_ask, opaque_reason, rank_entities
from snipher.mind.state import ConversationState, turn_word
from snipher.mind.voice import render


# --------------------------------------------------------------------------- #
# 分解
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,ask", [
    ("猫とは", "definition"),
    ("なぜ空は青い", "reason"),
    ("卵の焼き方のコツを教えて", "procedure"),
    ("何ができるの", "capability"),
    ("今は西暦何年？", "now"),
    ("いくらかかる？", "price"),
    ("Pythonで素数判定を書いて", "code"),
    ("猫と犬はどっちが早い", "comparison"),
    ("7×8 を計算して", "compute"),
])
def test_classify_ask(text: str, ask: str) -> None:
    assert classify_ask(text) == ask


def test_frame_carries_act_topic_entities_and_scripts() -> None:
    f = build_frame("GLM5.3とは")
    assert f.act == "ask"
    assert f.ask == "definition"
    assert f.topic == "GLM5.3"
    assert f.entities[0].surface == "GLM5.3"
    assert f.entities[0].kind == "ascii"
    assert f.needs_web is True                  # 手元に無い語 → 裏取りに行く
    assert "GLM5.3" in f.flags["unknown_terms"]


def test_build_frame_records_a_chain_activity() -> None:
    f = build_frame("しりとりしよ")
    assert f.act == "invite"
    assert f.has_rule("chain")                  # 「前の語の最後の音」を規則として持つ
    assert f.rules[0].value == ""               # 前語がまだ無い


def test_opaque_reason_is_typed_not_boolean() -> None:
    assert opaque_reason("asdfgh") == "latin_noise"
    assert opaque_reason("67") == "digits_only"
    assert opaque_reason("あ") == "single_char"
    assert opaque_reason("猫とは") == ""


def test_rank_entities_prefers_the_queried_object() -> None:
    f = build_frame("三毛猫の性格は？")
    ents = rank_entities(f.entities, f.norm)
    assert ents and "三毛猫" in ents[0].surface


# --------------------------------------------------------------------------- #
# 規則コンパイラ
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,kind", [
    ("5文字で言って", "len"),
    ("ですます調で", "style"),
    ("箇条書きで3つ", "form"),
    ("ひらがなだけで", "charset"),
    ("ローマ字で", "lang"),
])
def test_compile_rules_recognises_style_conditions(text: str, kind: str) -> None:
    assert any(r.kind == kind for r in R.compile_rules(text)), text


def test_enforce_rewrites_what_it_can() -> None:
    polite = R.compile_rules("ですます調で")
    assert R.enforce("猫が好きだ", polite) == "猫が好きです。"
    kata = R.compile_rules("カタカナで")
    assert "ネコ" in R.enforce("猫", kata)
    fit = R.compile_rules("10文字以内で言って")
    out = R.enforce("今日はとても良い天気だったので散歩に行こうと思います。", fit)
    assert len(out) <= 16 and out.endswith("。")
    # 字数合わせのために語を足さない（盛るのが一番悪い嘘なので）
    short = R.compile_rules("40文字で言って")
    assert R.enforce("猫が好きだ", short) == "猫が好きだ"


def test_rule_check_rejects_a_broken_chain_link() -> None:
    rl = R.compile_rules("しりとりしよ", previous_word="りんご")
    assert R.check("ゴマ", rl)[0] is True
    ok, why = R.check("ネギ", rl)
    assert ok is False and "link" in why


def test_chain_ok_handles_sokuon_and_chouon() -> None:
    assert R.chain_ok("きつね", "ねぎ")
    assert R.chain_ok("コーヒー", "ひまわり")           # 長音は末尾の実音（ひ）で受ける
    assert not R.chain_ok("コーヒー", "いちご")
    assert R.chain_ok("コッパ", "ぱん") or R.chain_ok("コッパ", "パン")
    assert not R.chain_ok("りんご", "きつね")
    assert not R.chain_ok("", "りんご")


def test_from_definition_compiles_rules_of_an_explained_game() -> None:
    rules = R.from_definition("しりとりは、前の語の最後に続く音を次の語の頭にする", topic="しりとり")
    assert any(r.kind == "chain" for r in rules)


# --------------------------------------------------------------------------- #
# 会話状態
# --------------------------------------------------------------------------- #
def test_state_restores_an_open_activity_from_history() -> None:
    hist = [
        {"role": "user", "content": "しりとりしよ"},
        {"role": "assistant", "content": "アイス。"},
        {"role": "user", "content": "ソバ"},
    ]
    st = ConversationState(hist)
    assert st.activity is not None and st.activity.open
    assert st.activity.name == "しりとり"
    assert st.activity.our_word == "アイス"
    assert st.previous_token() == "アイス"          # 次の手は我々の語の送り音から
    assert st.expects_continuation("ソバ")
    assert not st.expects_continuation("猫とは")     # 質問は遊びの手ではない


def test_state_does_not_treat_the_invitation_as_a_word() -> None:
    st = ConversationState([{"role": "user", "content": "しりとりしよ"},
                            {"role": "assistant", "content": "りんご。"}])
    assert st.activity is not None
    assert "しりとりしよ" not in st.activity.used
    assert st.activity.used == ["りんご"]


def test_turn_word_strips_punctuation_and_particles() -> None:
    assert turn_word("ソバ！") == "ソバ"
    assert turn_word("ネクタイですね") == "ネクタイ"


def test_style_register_follows_the_user() -> None:
    st = ConversationState([{"role": "user", "content": "しりとりしような"},
                            {"role": "assistant", "content": "りんご！"}])
    assert st.register == "plain"


# --------------------------------------------------------------------------- #
# 語の遊び（しりとり等の一般実装）
# --------------------------------------------------------------------------- #
def test_pick_returns_a_legal_word_that_is_not_n_final() -> None:
    mv = play.pick("", R.compile_rules("しりとりしよ"))
    assert mv.legal and mv.word
    assert not mv.word.endswith("ん")
    assert mv.reading                                           # 読みも持つ（連鎖の鍵）


def test_pick_continues_from_the_previous_word() -> None:
    mv = play.pick("キツネ", R.compile_rules("しりとりしよ"))
    assert mv.legal and mv.word
    assert R.chain_ok("キツネ", mv.word)


def test_judge_flags_an_illegal_move_and_shows_the_reason() -> None:
    mv = play.judge("りんご", "キツネ", R.compile_rules("しりとりしよ"))
    assert mv.legal is False
    assert "りんご" in mv.violation and "キツネ" in mv.violation


def test_judge_accepts_a_legal_move() -> None:
    mv = play.judge("キツネ", "ネギ", R.compile_rules("しりとりしよ"))
    assert mv.legal is True and mv.word == "ネギ"


def test_word_from_turn_survives_politeness_and_emoji() -> None:
    assert play.word_from_turn("ゴマ！") == "ゴマ"
    assert play.word_from_turn("ネギです。") == "ネギ"


def test_claims_for_move_carry_the_explanation_as_evidence() -> None:
    mv = play.pick("", R.compile_rules("しりとりしよ"))
    claims = play.claims_for_move(mv, prev="", our_turn=True)
    assert claims and all(isinstance(c, Claim) for c in claims)
    assert any("アイス" in c.content or mv.word in c.content for c in claims)


# --------------------------------------------------------------------------- #
# 合成（voice.render）
# --------------------------------------------------------------------------- #
def test_render_turns_claims_into_verified_sentences() -> None:
    frame = Frame(raw="猫とは", norm="猫とは", act="ask", ask="definition", topic="猫")
    dossier = type("D", (), {})()
    dossier.claims = [Claim("definition", content="猫は小型の肉食動物です。", source="local:kb",
                            subject="猫", weight=0.9)]
    dossier.evidence = []
    dossier.coverage = 0.8
    dossier.topic = "猫"
    dossier.via = "kb"
    dossier.web_used = False
    dossier.sources = []
    dossier.notes = []
    out = render(dossier, frame)
    assert "猫" in out.text
    assert len(out.text) <= 215
    assert out.sentences and all(s.endswith(("。", "！", "？")) for s in out.sentences)


def test_render_drops_claims_without_content() -> None:
    frame = Frame(raw="あ", norm="あ", act="ask")
    dossier = type("D", (), {})()
    dossier.claims = [Claim("note", content="", source="local")]
    dossier.evidence = []
    dossier.coverage = 0.0
    dossier.topic = ""
    dossier.via = "none"
    dossier.web_used = False
    dossier.sources = []
    dossier.notes = []
    out = render(dossier, frame)
    assert out.text == ""
    assert out.plan == "empty"
