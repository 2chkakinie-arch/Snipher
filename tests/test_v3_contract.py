"""v3 の受け入れ条件 — 「定型文ボットに見えない」ことと、2 つの評価プロンプト。

このファイルだけが、ユーザーが実際に投げた 2 つの発話
  「GLM5.3とは」 / 「しりとりしよ」
に対する契約を固定する。見るべきは *手段* ではなく *性質*:
どの話題でも (1) 手元の材料か検索で裏を取り (2) その場で組み立て (3) できないとは言わない。
特定タスクを強くする分岐を入れていないことは
`test_no_topic_specific_branches` で機械的に確かめる。
"""

from __future__ import annotations

import inspect
import os
import re

import pytest

from snipher.composer import Composer, validate
from web_fixtures import engine, grounding, server  # noqa: F401  (fixtures)

CAN_NOT_SAY = (
    "できません", "出来ません", "分かりません", "わかりません", "回答でき", "答えられ",
    "対応しておりません", "持ち合わせて", "学習されてい", "サポートして", "お答えでき",
    "わかりかね", "利用できません",
)

BATTERY = [
    "今は西暦何年？", "なにができますか", "67", "あ", "asdfgh", "疲れた", "ありがとう",
    "猫とは", "3×7は？", "GLM5.3とは", "しりとりしよ", "今日のニュースは？",
    "回文を作って", "Pythonで素数判定を書いて", "100 の素因数分解",
    "「うれしい」を英語にして", "私の名前は何？", "どうしようもない気分のときは",
]


@pytest.fixture(scope="module")
def composer() -> Composer:
    return Composer()


def _compose(c: Composer, text: str, *, history=None, turn: int = 1, web=None):
    return c.compose(text, history=history or [], turn=turn, web=web)


# --------------------------------------------------------------------------- #
# 評価プロンプト 1: 「GLM5.3とは」
# --------------------------------------------------------------------------- #
def test_unknown_latin_topic_does_not_fabricate(composer) -> None:
    r = _compose(composer, "GLM5.3とは", web=False)
    assert r.text and len(r.text) >= 6
    # 検索エンジン側の案内文（v2 が返していたゴミ）が混ざっていない
    for junk in ("リワード", "サインイン", "インテリジェント検索", "表現の自由", "アカウントを選択"):
        assert junk not in r.text
    # 手元に無い語を KB 由来として騙らない
    via = (r.knowledge or {}).get("via")
    assert via != "kb"
    assert r.confidence <= 0.85


def test_unknown_topic_is_grounded_from_the_web(composer, grounding) -> None:
    composer._web = grounding
    try:
        r = _compose(composer, "GLM5.3とは", web=True)
    finally:
        composer._web = None
    assert "GLM5.3" in r.text
    assert "2026" in r.text or "基盤モデル" in r.text       # 検索で取れた事実
    assert "出典" in r.text or "http" in r.text              # 参照元を示す
    for junk in ("リワード", "サインイン", "表現の自由"):
        assert junk not in r.text


# --------------------------------------------------------------------------- #
# 評価プロンプト 2: 「しりとりしよ」
# --------------------------------------------------------------------------- #
def test_shiritori_starts_a_game_not_an_encyclopedia_entry(composer) -> None:
    r = _compose(composer, "しりとりしよ", web=False)
    assert "検索エンジン" not in r.text and "表現の自由" not in r.text
    words = re.findall(r"[ァ-ヶーぁ-ん一-龯]{2,}", r.text)
    playable = [w for w in words if not w.endswith("ん")]
    assert playable, r.text                                  # 打ち手が 1 つは含まれる
    assert "しりとり" in r.text or "語" in r.text


def test_shiritori_accepts_a_legal_move(composer) -> None:
    hist = [{"role": "user", "content": "しりとりしよ"},
            {"role": "assistant", "content": "キツネ。"}]
    r = _compose(composer, "ネギ", history=hist, turn=2, web=False)
    assert "ネギ" in r.text
    assert "合いません" not in r.text                        # 正当な手を非法と言わない


def test_shiritori_flags_a_broken_link(composer) -> None:
    hist = [{"role": "user", "content": "しりとりしよ"},
            {"role": "assistant", "content": "りんご。"}]
    r = _compose(composer, "キツネ", history=hist, turn=2, web=False)
    # 文体は相手へ合わせる（くだけ口調なら「合わない」）。要るのは *非法だと指摘すること*。
    assert re.search(r"合(い)?ません|合わない|繋がりま|続きの音|始まりの音が", r.text), r.text
    assert "「りんご」" in r.text                                   # 根拠の語を明示する


def test_shiritori_does_not_repeat_the_word_already_said(composer) -> None:
    hist = [{"role": "user", "content": "しりとりしよ"},
            {"role": "assistant", "content": "アイス。"},
            {"role": "user", "content": "ソバ"},
            {"role": "assistant", "content": "いい手です。バソ。"},
            {"role": "user", "content": "しりとり続けて"}]
    r = _compose(composer, "続けて", history=hist, turn=3, web=False)
    assert r.text
    # 同じ手を繰り返さない（「アイス。」をそのまま戻さない）
    assert r.text.strip() != "アイス。"


# --------------------------------------------------------------------------- #
# 知識ベースの誤引き戻し（v2 の「しりとり → 鳥」型）
# --------------------------------------------------------------------------- #
def test_compound_topic_discloses_the_component_it_used(composer) -> None:
    r = _compose(composer, "三毛猫とは", web=False)
    assert "三毛猫" in r.text
    if (r.knowledge or {}).get("topic") == "猫":
        assert "単独の項目" in r.text or "構成語" in r.text    # 借りたことを明示する


def test_shiritori_does_not_retrieve_the_bird_entry(composer) -> None:
    r = _compose(composer, "しりとりとは", web=False)
    assert "しりとり" in r.text
    assert "とり" not in r.text.replace("しりとり", "").replace("取り", "")


# --------------------------------------------------------------------------- #
# 「できない」と言わない / 文章として整っている
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", BATTERY)
def test_never_says_cannot(composer, text: str) -> None:
    r = _compose(composer, text, web=False)
    assert r.text.strip(), f"{text!r} で空応答"
    for banned in CAN_NOT_SAY:
        assert banned not in r.text, f"{text!r} → {r.text} に {banned!r}"


@pytest.mark.parametrize("text", BATTERY)
def test_reply_is_tied_to_the_input(composer, text: str) -> None:
    """応答は「入力の実語」か「計算・検索で得た値」を含む。無関係な定型文を返さない。

    問いの言い回しそのものを復唱させるのが目的ではない（「猫とは」に
    「猫は…」と答えるほうが自然）。なので *内容語の 2 文字の手がかり* が
    応答にひとつも無ければ失敗、とする。
    """
    from snipher.lang.lex import bank

    r = _compose(composer, text, web=False)
    if r.plan in {"greet", "thanks", "apology", "farewell", "agree", "disagree", "praise"}:
        # 挨拶・礼・謝罪は語の重複で測れない（「ありがとう」→「いえいえ」が正しい応答）。
        # ここでは、無関係な百科記事を引きずっていないことだけを確かめる。
        for junk in ("検索エンジン", "表現の自由", "リワード", "サインイン"):
            assert junk not in r.text, (text, r.text)
        return
    keys = [w for w, pos in bank().segment(text)
            if str(pos).split("/")[0] in {"名詞", "動詞", "形容詞", "副詞"} and len(w) >= 1]
    keys += [k for k in re.findall(r"[ァ-ヶーぁ-んA-Za-z一-龯]{2,}", text) if k not in keys]
    if not keys:
        return
    hints: set[str] = set()
    for k in keys:
        if k in r.text:
            return
        for i in range(len(k) - 1):                      # 2 文字以上の続きの部分
            hints.add(k[i:i + 2])
        from snipher.lang.phonetics import to_hiragana
        kana = to_hiragana(k)
        if kana in r.text:
            return
        for i in range(len(kana) - 1):
            hints.add(kana[i:i + 2])
    assert any(h in r.text for h in hints) or re.search(r"\d", r.text), (text, r.text)


@pytest.mark.parametrize("text", BATTERY)
def test_prose_is_well_formed(composer, text: str) -> None:
    r = _compose(composer, text, web=False)
    if "```" in r.text:
        return                                              # コードは整形そのまま
    for s in r.sentences or [r.text]:
        body = s.strip()
        if not body:
            continue
        assert body[-1] in "。！？…）」』" or body.endswith(("：", "です")), (text, body)


def test_different_inputs_do_not_produce_the_same_reply(composer) -> None:
    got = [_compose(composer, q, web=False).text for q in
           ["猫とは", "犬とは", "車とは", "雨とは", "音楽とは"]]
    assert len(set(got)) >= 4, got                          # テンプレ一致がほぼ無い


def test_no_legacy_frame_tables_remain_in_composer() -> None:
    code = inspect.getsource(Composer.__module__ and __import__("snipher.composer",
                                                                fromlist=["Composer"]))
    for gone in ("UNKNOWN_FRAMES", "OPAQUE_FRAMES", "STATEMENT_FRAMES", "PRED_FRAMES",
                 "THANKS_FRAMES", "CANNOT_FRAMES", "OPEN_QUESTIONS", "FOLLOW_CONNECTORS",
                 "ACK_POS", "ACK_NEG"):
        assert gone not in code, f"{gone} がまだ composer に残っています"


def test_no_topic_specific_branches() -> None:
    """評価プロンプト向けの分岐を入れていないこと（一般層だけで解いている）。"""
    from snipher.mind import think as tk
    from snipher.mind import play as pl

    for mod in (tk, pl):
        code = inspect.getsource(mod)
        for topic in ("GLM", "ニュース", "西暦"):
            assert f'"{topic}"' not in code, f"{mod.__name__} に {topic} の特別分岐"
    # しりとりは「語の連鎖」という規則として扱う（ゲーム名で分岐しない）
    src = inspect.getsource(pl)
    assert 'name == "しりとり"' not in src


def test_safety_net_is_content_derived_too(composer) -> None:
    """mind を無効化しても（安全網でも）定型文は出てこない。"""
    os.environ["SNIPHER_MIND"] = "0"
    try:
        c = Composer()
        a = _compose(c, "しりとりしよ", web=False)
        b = _compose(c, "猫とは", web=False)
        assert a.text and b.text and a.text != b.text
        for banned in CAN_NOT_SAY:
            assert banned not in a.text and banned not in b.text
    finally:
        os.environ.pop("SNIPHER_MIND", None)


def test_capability_answer_uses_measured_numbers(composer) -> None:
    r = _compose(composer, "なにができますか", web=False)
    assert re.search(r"\d", r.text), r.text                  # 実測の語彙数・項目数など
    assert "語" in r.text or "件" in r.text


# --------------------------------------------------------------------------- #
# 文体の組み替え（語尾の文字差し替えではなく、活用を見てやる）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("polite,plain", [
    ("勉強しました", "勉強した。"),
    ("追加しました", "追加した。"),
    ("手があります", "手がある。"),
    ("猫が魚を食べます", "猫が魚を食べる。"),
    ("今日は寒かったです", "今日は寒かった。"),
    ("食べません", "食べない。"),
    ("好きです", "好きだ。"),
    ("来ます", "来る。"),
    ("始まりの音が合いません", "始まりの音が合わない。"),
])
def test_register_switch_keeps_grammar(polite: str, plain: str) -> None:
    from snipher.lang import morph

    assert morph.to_plain(polite) == plain
    assert morph.to_polite(plain) == f"{polite}。"


def test_register_switch_does_not_borrow_another_verb() -> None:
    """「書き」を「来る」に化けさせるサ変誤引きの再発防止（v2 の実不具合）。"""
    from snipher.lang import morph

    assert morph.to_plain("書きました") == "書いた。"
    assert morph.lemma_of("書き") == "書く"


def test_unbalanced_brackets_are_repaired_not_emitted() -> None:
    from snipher.composer import balance_quotes, validate

    fixed = balance_quotes("参考までに、りんご」の次なら「豪華」のような手がある。")
    assert fixed.count("「") == fixed.count("」")
    ok, _ = validate(fixed)
    assert ok, fixed
    assert validate("りんご」だけ閉じている文です。")[0] is False


# --------------------------------------------------------------------------- #
# 同じ発話を繰り返さない（定型ボットに見えないための最短検査）
# --------------------------------------------------------------------------- #
def test_repeated_statement_does_not_repeat_the_reply(composer) -> None:
    texts = [_compose(composer, "ぬるぬる猿について話したい", turn=i, web=False).text
             for i in range(1, 9)]
    assert len(set(texts)) >= 5, texts
    # 手元に無い話題では、相手の文をそのまま返さない・辞書引き応答もしない
    for t in set(texts):
        assert "ぬるぬる猿について話したい" not in t, t
        assert not re.search(r"(拍|索引|品詞は|読みは|U\+)", t), t
        assert len(t) >= 12, t


def test_thin_topic_moves_are_verifiable_facts(composer) -> None:
    """手元に無い話題での手は、すべて索引から数え直せる事実だけを述べる。"""
    from snipher.lang.lex import bank

    r = _compose(composer, "ぬるぬる猿について話したい", turn=5, web=False)
    b = bank()
    m = re.search(r"「猿」を含む語は手元の語彙に (\d+) 語見えて、例は (.+?) です", r.text)
    if m:
        real = [x.surface for x in b.containing("猿", limit=12) if x.surface != "猿"]
        assert int(m.group(1)) == len(real)
        for shown in m.group(2).split("、"):
            assert shown in real, (r.text, shown)


def test_first_person_question_reads_the_conversation(composer) -> None:
    """「私の名前は？」は自己紹介カードで返さない。会話を覚えていればそれを答える。"""
    known = [{"role": "user", "content": "私は健太です"},
             {"role": "assistant", "content": "承知しました。"}]
    a = _compose(composer, "私の名前は何？", history=known, turn=2, web=False)
    assert "健太" in a.text
    assert "Snipher という" not in a.text
    b = _compose(composer, "私の名前は何？", web=False)
    assert "名乗っ" in b.text or "記録がありません" in b.text
    assert not any(x in b.text for x in CAN_NOT_SAY)


def test_accept_rejects_gibberish_and_keeps_prose(composer) -> None:
    assert composer.accept("花火は、火薬の燃焼と爆発で光と音を出す娯楽です。") is True
    assert composer.accept("きょはいいんてきすあ。") is False


# --------------------------------------------------------------------------- #
# 文章としての仕上がり（整形・省略符・コードの扱い）
# --------------------------------------------------------------------------- #
def test_code_answer_is_not_penalised_for_lacking_a_period(composer) -> None:
    r = _compose(composer, "Pythonで素数判定を書いて", web=False)
    assert "```" in r.text and "def " in r.text
    ok, why = validate(r.text, max_len=400)
    assert ok, why


def test_ellipsis_survives_the_pipeline(composer) -> None:
    """NFKC は「…」を "..." に分解する。引用の省略は句読点として残す。"""
    r = _compose(composer, "しりとりしよ", web=False)
    assert "..." not in r.text
    assert validate(r.text, max_len=400)[0], r.text


def test_register_is_measured_without_counting_politeness(composer) -> None:
    """「選びました、「ん」で…」を常体混在と数えない（v2 の誤検知）。"""
    ok, why = validate("一般的な名詞を選びました、「ん」で終わる語は除きました。", max_len=400)
    assert ok, why


def test_translation_ask_names_the_word_it_read(composer) -> None:
    r = _compose(composer, "「うれしい」を英語にして", web=False)
    assert "うれしい" in r.text
    assert "ureshii" in r.text.replace("'", "")      # ローマ字表記まで示す
    assert "えいご" not in r.text.split("\n")[0]      # 目標言語を被説明語にしない


def test_demo_generation_endpoint_marks_broken_sentences() -> None:
    """/generate（確率生成デモ）は、壊れた文を *検査を通った顔で* 返さない。

    本体と同じ `validate()` を通し、落ちた文は理由と *整え案* を必ず添える。
    デモだからと言って非文をそのまま出さない、という取り決め。
    """
    import os

    os.environ.setdefault("SNIPHER_LFM_BACKEND", "off")
    os.environ.setdefault("SNIPHER_LFM_AUTO_FETCH", "0")
    import snipher.api as api_mod
    from fastapi.testclient import TestClient

    api_mod._core = None
    api_mod._lfm_engine = None
    seen_invalid = 0
    with TestClient(api_mod.app) as client:
        for seed in range(6):
            body = client.post("/generate", json={"n": 4, "seed": seed}).json()
            items = body.get("sentences") or [body]
            for s in items:
                assert "valid" in s, s
                if s["valid"] is False:
                    seen_invalid += 1
                    assert s.get("invalid_reason"), s
                    assert (s.get("repaired") or s.get("text")), s
    # 壊れた文が 1 つも無いならそれで良い。有ったなら、必ず理由と整え案が付いている。
