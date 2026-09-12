"""composer（文の設計図）のテスト。

ここで守るのは「定型文の引き当てでゴミを作らない」という契約です:

    * 知識ベースに材料があれば、その材料を日本語の応答に組み立てる
    * 材料が無ければ **無いと正直に言う**（それっぽい嘘を作らない）
    * 読めない入力（数字だけ・文字化け）には、読めなかったと言う
    * 同じ入力が続いても、毎回まったく同じ文を返さない（枠を回す）
    * 組み立てた文は必ず validate() を通る（助詞で切れた文・ループ・文体混在を出さない）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher import composer as comp_mod          # noqa: E402
from snipher.composer import Composer, Reply, detect_intent, detect_mood, validate  # noqa: E402
from snipher.knowledge import KnowledgeBase        # noqa: E402


@pytest.fixture(scope="module")
def composer() -> Composer:
    return Composer(kb=KnowledgeBase(), seed=7)


def _reply(composer: Composer, text: str, *, turn: int = 1) -> Reply:
    return composer.compose(text, turn=turn)


# ---------------------------------------------------------------------- #
# 1) 材料があるとき … 話題がずれていないこと（ユーザー報告の回帰テスト）
# ---------------------------------------------------------------------- #
def test_fireworks_question_returns_fireworks(composer):
    """「花火とは」に植物の世話の話が返っていた不具合の回帰テスト。"""
    r = _reply(composer, "花火とは")
    assert r.plan.startswith("knowledge")
    assert r.knowledge and r.knowledge.get("topic") == "花火"
    assert "花火" in r.text
    assert not any(w in r.text for w in ("水やり", "肥料", "観葉植物", "植え替え"))
    ok, why = validate(r.text, max_len=220)
    assert ok, why


def test_favorite_food_returns_opinion(composer):
    r = _reply(composer, "好きな食べ物は？")
    assert r.knowledge and r.knowledge.get("topic") in ("食事", "食べ物", "料理")
    assert r.text.endswith(("。", "？", "?", "！", "!"))


def test_plant_trouble_returns_care_advice(composer):
    """「葉が黄色い」は定義ではなく対処（tips / qa）を返す。"""
    r = _reply(composer, "観葉植物の葉が黄色い")
    assert r.knowledge is not None
    assert r.knowledge.get("field") in ("tips", "qa", "why", "how")
    assert "観葉植物は" not in r.text


def test_body_symptom_gets_care_not_definition(composer):
    r = _reply(composer, "お腹が痛い")
    assert r.knowledge is not None
    assert r.knowledge.get("field") in ("qa", "tips", "when", "how")
    assert any(w in r.text for w in ("受診", "休", "水分", "温かく", "無理"))


# ---------------------------------------------------------------------- #
# 2) 材料が無いとき … 正直であること
# ---------------------------------------------------------------------- #
def test_number_only_input_is_not_fabricated(composer):
    """「67」に定型文のゴミを返さない。数字だけなら読めなかったと言う。"""
    r = _reply(composer, "67")
    assert r.plan == "opaque_input"
    assert r.knowledge is None
    assert r.confidence < 0.4                      # 確信度を盛らない
    assert "67" in r.text                          # 相手の入力をそのまま返す
    ok, why = validate(r.text, max_len=220)
    assert ok, why


def test_ascii_garbage_is_rejected_politely(composer):
    r = _reply(composer, "asdfgh")
    assert r.plan == "opaque_input"
    assert r.knowledge is None
    assert any(w in r.text for w in ("読め", "分かり", "日本語", "意味"))


def test_single_char_input_is_opaque(composer):
    r = _reply(composer, "あ")
    assert r.plan == "opaque_input"
    assert r.knowledge is None


def test_unknown_topic_does_not_invent_facts(composer):
    r = _reply(composer, "ぬるぬる猿の生態について")
    assert r.knowledge is None
    assert r.plan in ("unknown_topic", "statement", "opaque_input")
    # 事実を主張する形（「〜です。」の断定＋数字）を捏造しない
    assert r.confidence <= 0.5


# ---------------------------------------------------------------------- #
# 3) 社会的な発話 … 挨拶・礼・謝罪・自分への質問
# ---------------------------------------------------------------------- #
def test_greeting_is_not_opaque(composer):
    r = _reply(composer, "こんにちは")
    assert r.plan != "opaque_input"
    assert any(w in r.text for w in ("こんにちは", "どうしました", "調子", "話"))


def test_identity_question_answers_about_self(composer):
    """「あなたは誰？」を読み取れない入力に落とさない。"""
    r = _reply(composer, "あなたは誰？")
    assert r.plan != "opaque_input"
    assert "Snipher" in r.text or "私" in r.text


def test_thanks_does_not_double_up(composer):
    r = _reply(composer, "ありがとう")
    assert detect_intent("ありがとう") == "thanks"
    # 「どういたしまして」系の応答に、相槌と問いを二重に付けない
    assert r.text.count("？") + r.text.count("?") <= 1
    assert not r.text.startswith("素敵です。")


def test_apology_and_praise_have_their_own_plans(composer):
    assert _reply(composer, "ごめんね").plan == "apology"
    assert _reply(composer, "すごいね").plan == "praise"
    assert _reply(composer, "さようなら").plan != "opaque_input"


def test_mood_detection():
    assert detect_mood("仕事が疲れた、つらい") == "negative"
    assert detect_mood("今日は楽しかった") == "positive"
    assert detect_mood("今日は水曜日") == "neutral"


# ---------------------------------------------------------------------- #
# 4) 多様性 … 同じ入力を繰り返しても同じ文を返さない
# ---------------------------------------------------------------------- #
def test_repeated_statement_varies(composer):
    texts = {composer.compose("ぬるぬる猿について話したい", turn=i).text for i in range(1, 9)}
    assert len(texts) >= 3, texts


def test_repeated_opaque_varies(composer):
    texts = {composer.compose("99", turn=i).text for i in range(1, 9)}
    assert len(texts) >= 3, texts


def test_history_avoids_repeating_previous_reply(composer):
    first = composer.compose("ぬるぬる猿について", turn=1)
    history = [{"role": "user", "content": "ぬるぬる猿について"},
               {"role": "assistant", "content": first.text}]
    second = composer.compose("ぬるぬる猿について", history=history, turn=2)
    assert second.text != first.text


# ---------------------------------------------------------------------- #
# 5) validate() … 壊れた日本語を通さない
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("bad,reason", [
    ("", "too_short"),
    ("短い", "too_short"),
    ("これは長すぎる文です" * 30, "too_long"),
    ("終点がない文です", "no_terminal"),
    ("これはですですおかしい。", "bad_pattern:ですです"),
    ("プレースホルダ {w} が残った文です。", "bad_pattern:{w}"),
    ("これは変な文ですを。", "dangling"),
    ("猫が好きです。猫が好きです。猫が好きです。猫が好きです。", "loop:猫が"),
])
def test_validate_rejects_broken_japanese(bad, reason):
    ok, why = validate(bad, max_len=170)
    assert ok is False
    assert why == reason or why.startswith(reason), (bad, why, reason)


def test_validate_accepts_greetings_and_polite_text():
    for good in ("こんにちは。", "おはようございます。今日はどんな一日にしますか。",
                 "花火は、火薬の燃焼で光と音を出す娯楽です。"):
        ok, why = validate(good, max_len=220)
        assert ok, (good, why)


def test_validate_catches_register_mix():
    # です/ます体に、だ・た体が混ざった文（組み立ての失敗）
    ok, why = validate("私は猫が好きです。今日は雨が降る。それで映画を見た。", max_len=170)
    assert ok is False and why == "register_mix"
    # 常体だけで統一されていれば通る（文体の一貫性を見ているのであって、敬体を強制しない）
    ok2, _ = validate("私は猫が好きだ。今日は雨が降る。", max_len=170)
    assert ok2 is True


# ---------------------------------------------------------------------- #
# 6) 採点・採用判定
# ---------------------------------------------------------------------- #
def test_rank_prefers_natural_japanese(composer):
    ranked = composer.rank(["今日はいい天気ですね。", "きょはいいんてきすあ。", "ですですます。"])
    assert ranked[0][1] == "今日はいい天気ですね。"
    assert ranked[0][0] >= ranked[-1][0]


def test_accept_rejects_broken_and_empty(composer):
    assert composer.accept("今日はいい天気ですね。") is True
    assert composer.accept("") is False
    assert composer.accept("ですです。") is False
    assert composer.accept("これは変な文ですを。") is False
    assert composer.accept("終点がない文です") is False


def test_rank_with_lm_uses_language_model():
    lm = comp_mod_lm = None
    try:
        from snipher import lm as lm_mod

        lm = comp_mod_lm = lm_mod.shared()
    except Exception:  # noqa: BLE001
        lm = None
    if lm is None or not lm.is_ready:
        pytest.skip("n-gram LM が未ビルド（python tools/build_lm.py）")
    c = Composer(kb=KnowledgeBase(), lm=lm)
    ranked = c.rank(["花火は、火薬の燃焼と爆発で光と音を出す娯楽です。", "ばびぶべぼは、。、"])
    assert ranked[0][1].startswith("花火")
    assert ranked[0][0] > ranked[-1][0]
    # LM が居るときは、文字をシャッフルした文を採用しない
    assert c.accept("きょはいいんてきすあ。") is False
    assert c.accept("花火は、火薬の燃焼と爆発で光と音を出す娯楽です。") is True


# ---------------------------------------------------------------------- #
# 7) 出力の形（プレースホルダや None を漏らさない）
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "花火とは", "67", "暇だなあ", "ありがとう", "お腹が痛い", "あなたは誰？", "こんにちは",
    "観葉植物の葉が黄色い", "asdfgh", "あ", "gitがわからない", "タピオカって何", "なぜ空は青い",
    "好きな食べ物は？", "明日の天気はどう？", "ぬるぬる猿", "Wi-Fiが遅い", "肩が凝った",
])
def test_every_reply_is_valid_japanese(composer, text):
    for turn in (1, 2, 3):
        r = composer.compose(text, turn=turn)
        assert r.text and r.text.strip(), (text, turn)
        assert "{" not in r.text and "}" not in r.text, (text, r.text)
        assert "None" not in r.text and "nan" not in r.text, (text, r.text)
        ok, why = validate(r.text, max_len=220)
        assert ok, (text, turn, r.text, why)
        assert 0.0 <= r.confidence <= 1.0
        assert r.plan


# --------------------------------------------------------------------------- #
# v2.0: 述語だけの発話・echo の選び方・機能語の主語化防止
# --------------------------------------------------------------------------- #
def _composer() -> Composer:
    """回転（turn）が必ず 1 から始まる、新しい Composer。"""
    return Composer(kb=KnowledgeBase(), seed=7)

class TestPredicateUtterances:
    """「疲れた」のように語彙に無い短文を、気持ちとして受けられるか。"""

    def test_tired_gets_empathy_not_unknown_topic(self):
        c = _composer()
        r = c.compose("疲れた")
        assert r.plan == "statement", r.plan
        assert "疲れた" in r.text
        assert "分かりません" not in r.text and "見つかりません" not in r.text

    def test_cold_past_tense_is_not_echoed_as_noun(self):
        """「寒かったのことですね」のような非文法を作らない。"""
        c = _composer()
        r = c.compose("寒かった")
        assert "のことですね" not in r.text
        ok, why = validate(r.text)
        assert ok, why

    def test_positive_feeling_is_acknowledged(self):
        c = _composer()
        r = c.compose("うれしい")
        assert r.plan == "statement"
        assert "うれしい" in r.text

    def test_actionable_kb_wins_over_feelings(self):
        """「お腹が痛い」は共感だけではなく対処（知識ベース qa）を返す。"""
        c = _composer()
        r = c.compose("お腹が痛い")
        assert r.plan.startswith("knowledge"), r.plan
        assert any(w in r.text for w in ("温かく", "水分", "受診"))

    def test_predicate_detection_needs_japanese_ending(self):
        u = _composer().analyze("67")
        assert u.pred == ""
        assert u.is_opaque
        u2 = _composer().analyze("疲れた")
        assert u2.pred == "疲れた"
        assert not u2.is_opaque

    def test_long_sentence_is_not_treated_as_bare_predicate(self):
        u = _composer().analyze("昨日は友達と夜遅くまで歩いていたら急に雨が降ってきた")
        assert u.pred == ""


class TestEchoSelection:
    """応答に埋め込む語（相手の言葉）の選び方。"""

    def test_particle_is_not_used_as_subject(self):
        c = _composer()
        r = c.compose("それについて教えて")
        assert "についてについて" not in r.text
        assert not r.text.startswith("について")
        ok, why = validate(r.text)
        assert ok, why

    def test_noun_beats_verb_and_adverb(self):
        """「歩いていたら急に雨が」→ 雨 であって 歩いて ではない。"""
        u = _composer().analyze("歩いていたら急に雨が")
        assert u.echo == "雨", u.echo

    def test_unknown_compound_noun_is_kept(self):
        u = _composer().analyze("ぬるぬる猿について語って")
        assert u.echo == "ぬるぬる猿", u.echo

    def test_echo_skip_words_are_excluded(self):
        from snipher.composer import _ECHO_SKIP

        u = _composer().analyze("ことについて")
        assert u.echo not in _ECHO_SKIP


class TestMaterialBranchGating:
    """知識ベースが当たったときに、生成文を安易に混ぜないこと。"""

    def test_kb_followup_suppresses_neural_followup(self):
        from snipher.core import SnipherCore

        core = SnipherCore(torch_provider=lambda: None)
        core.active_backend = lambda: None
        events = list(core.stream_reply([{"role": "user", "content": "花火とは"}]))
        st = events[-1]["stats"]
        text = "".join(e.get("text", "") for e in events if e.get("type") == "delta") or \
            str(events[-1].get("text") or "")
        assert st["route"] == "light"
        assert st["knowledge"]["topic"] == "花火"
        # 人が書いた followup があるので、ニューラル生成は呼ばれない（= 速い）
        assert not st.get("neural_used"), st
        assert text.endswith("？") or "か。" in text, text

    def test_no_material_branch_requires_lm_confidence(self):
        """内蔵コアの自信だけでなく n-gram LM の自然さも閾値を越える必要がある。"""
        import inspect

        from snipher.core import SnipherCore

        src = inspect.getsource(SnipherCore._light_reply)
        assert 'sc["confidence"] >= bar' in src
        assert '(sc["lm"] or 0.0) >= bar' in src


class TestMoodAndValidation:
    """気分の判定と文の検査（誤爆を潰す）。"""

    def test_desire_is_not_negative(self):
        """「飼いたい」を「痛い」と誤読して慰めない。"""
        assert detect_mood("犬を飼いたい") == "positive"
        assert detect_mood("お腹が痛い") == "negative"
        assert detect_mood("仕事がつらい") == "negative"
        assert detect_mood("今日は暑かった") == "neutral"

    def test_desire_gets_positive_ack(self):
        c = _composer()
        r = c.compose("犬を飼いたい")
        assert "つらい" not in r.text and "無理はしない" not in r.text

    def test_no_no_is_allowed_before_verb(self):
        """「脂ののった」は正当。「ののは」は不正。"""
        ok, _why = validate("最後に脂ののったものを食べる順番が好きです。")
        assert ok
        bad, why = validate("これはののは間違いです。")
        assert not bad and "のの" in why

    def test_generic_alias_word_is_still_echoed(self):
        """「歴史が好き」→ 好き ではなく 歴史 を話題として拾う。"""
        u = _composer().analyze("歴史が好き")
        assert u.echo == "歴史", u.echo

    def test_investment_reply_is_grounded(self):
        c = _composer()
        r = c.compose("投資って何")
        assert r.plan.startswith("knowledge"), r.plan
        assert r.knowledge["topic"] == "投資"
        ok, why = validate(r.text, max_len=240)
        assert ok, why
