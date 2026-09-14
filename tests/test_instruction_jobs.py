"""「言葉の仕事」の型（jobs）と、実運用ログで落ちた 15 問の検査。

v6 までは、指示の型が 7 つ（extract / summarize / code / answer / transform / list /
write）しか無かったため、実際に投げられた依頼の多くが *別の仕事* に化けました。

    「「嬉しい」と同じような意味を持つ言葉を1つ挙げてください。」
        → write（文書の作成）と読まれ、テンプレート文が漏れた
    「山羊をひらがなにしてください。」
        → 分類と読まれ、「山羊: 果物」と返った
    「「ありがとう」を使わずに、感謝の気持ちを表す短い返答をしてください。」
        → 定型の礼テーブルが「ありがとう」で返した（禁止語の無視）
    「3 + 5 の計算結果だけを数字で答えてください。」
        → 道具は 8 を出したが、整形で「8。」になり形の指定を外した

このファイルは、その 15 問が *指示として型どおりに実行される* ことを固定します。
検査は出力の文字列ではなく「満たすべき性質」で書きます（1 文字・禁止語なし・
材料に無い空欄なし・テンプレート文の漏れなし）。
"""

from __future__ import annotations

import re

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """このファイルの検査は検索を挟まない（環境変数は *この検査の中だけ* 変える）。

    モジュール読み込み時に os.environ を書き換えると、後から走る他のテスト
    （ウェブ裏取りの冷却など）の前提を壊します。monkeypatch なら他へ漏れません。
    """
    monkeypatch.setenv("SNIPHER_WEB", "off")
    yield

#: 実運用ログに出た 15 問と、満たすべき性質
LOG_CASES: list[tuple[str, str, dict]] = [
    ("助詞の穴埋め",
     "次の文の空欄に入る最も適切な助詞を1文字で答えてください。「私（ ）公園へ行く。",
     {"exact": "は"}),
    ("カタカナ変換",
     "「りんご」をカタカナに変換して出力してください。",
     {"exact": "リンゴ"}),
    ("定義を一言で",
     "「太陽」とは何ですか？一言で説明してください。",
     {"contains_any": ["恒星", "太陽系の中心"], "max_chars": 60}),
    ("類義語",
     "「嬉しい」と同じような意味を持つ言葉を1つ挙げてください。",
     {"contains_any": ["幸せ", "しあわせ", "嬉しい", "楽しい", "喜ばしい", "ハッピー"]}),
    ("カテゴリで選ぶ",
     "次の単語の中から「果物」だけを選んでください：[ 犬, りんご, 車, バナナ ]",
     {"contains_all": ["りんご", "バナナ"], "absent": ["犬", "車"]}),
    ("短文 10 文字以内",
     "「猫」についての短文を、ちょうど10文字以内で書いてください。",
     {"contains": "猫", "max_chars": 10}),
    ("全称命題",
     "人間は必ず息をします。太郎は人間です。太郎は息をしますか？「はい」か「いいえ」で答えてください",
     {"exact": "はい"}),
    ("規則の適用",
     "「赤信号では止まれ、青信号では進め。」いま信号は赤です。どうすればいいですか？",
     {"contains": "止ま", "absent": ["進め"]}),
    ("禁止語つきの礼",
     "「ありがとう」を使わずに、感謝の気持ちを表す短い返答をしてください。",
     {"absent": ["ありがとう"], "max_chars": 40}),
    ("ひらがな書き",
     "山羊をひらがなにしてください。",
     {"exact": "やぎ"}),
    ("反対語",
     "「嬉しい」の反対語を1つ挙げてください。",
     {"contains_any": ["悲しい", "悲しみ", "憂い", "つらい", "辛い"]}),
    ("語の写し（英語）",
     "犬と猫を英語に訳してください。",
     {"contains_all": ["dog", "cat"]}),
]

#: どの答えにも残ってはいけないテンプレート文（v6 まで出ていた漏れ）
JUNK = ("について、要点をまとめます。", "について、お知らせします。",
        "ついての短文について", "内容は【内容】で", "ご確認のうえ、必要であれば")


def _failed(text: str, expect: dict) -> list[str]:
    body = str(text or "").strip()
    bad: list[str] = []
    if not body:
        return ["出力が空"]
    for junk in JUNK:
        if junk in body:
            bad.append(f"テンプレートの漏れ: {junk}")
    if re.search(r"【[^】]{0,20}】", body):
        bad.append("材料に無い空欄【 】が残っている")
    if "exact" in expect and body != expect["exact"]:
        bad.append(f"形が違う: {body[:20]!r}")
    if expect.get("contains") and expect["contains"] not in body:
        bad.append(f"「{expect['contains']}」が無い")
    for w in expect.get("contains_all") or []:
        if w not in body:
            bad.append(f"「{w}」が無い")
    if expect.get("contains_any") and not any(w in body for w in expect["contains_any"]):
        bad.append("どれも無い: " + "/".join(expect["contains_any"]))
    for w in expect.get("absent") or []:
        if w in body:
            bad.append(f"禁止「{w}」が出た")
    limit = int(expect.get("max_chars") or 0)
    if limit and len(body.replace("\n", "")) > limit:
        bad.append(f"{len(body)} 文字（上限 {limit}）")
    return bad


# --------------------------------------------------------------------------- #
# 1) 型の読み取り（parse）— 材料の形で仕事が決まる
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,text,expect", LOG_CASES)
def test_parse_reads_the_job_from_material_shape(name: str, text: str, expect: dict) -> None:
    from snipher.instruction import parse

    d = parse(text)
    assert d is not None, f"{name}: 指示として読めていない"
    assert d.task in ("fill", "kana", "select", "relations", "logic", "thanks", "gloss",
                      "answer", "write"), f"{name}: 想定外の型 {d.task}"


def test_relations_are_not_read_as_document_writing() -> None:
    """「同じような意味」は *語の関係* の依頼。write（文書作成）に化けない。"""
    from snipher.instruction import parse

    for text in ("「嬉しい」と同じような意味を持つ言葉を1つ挙げてください。",
                 "「嬉しい」の反対語を1つ挙げてください。"):
        d = parse(text)
        assert d is not None and d.task == "relations", (text, d and d.task)


def test_kana_conversion_is_not_classification() -> None:
    """「山羊をひらがなに」は分類ではなく *書き換え*（実辞書の読みを使う）。"""
    from snipher.instruction import parse

    d = parse("山羊をひらがなにしてください。")
    assert d is not None and d.task == "kana", d and d.task


def test_gap_fill_requires_one_character_spec() -> None:
    from snipher.instruction import parse

    d = parse("次の文の空欄に入る最も適切な助詞を1文字で答えてください。「私（ ）公園へ行く。")
    assert d is not None and d.task == "fill"


def test_logic_job_is_read_from_premises() -> None:
    from snipher.instruction import parse

    syll = parse("人間は必ず息をします。太郎は人間です。太郎は息をしますか？"
                 "「はい」か「いいえ」で答えてください")
    rule = parse("「赤信号では止まれ、青信号では進め。」いま信号は赤です。どうすればいいですか？")
    assert syll is not None and syll.task == "logic"
    assert rule is not None and rule.task == "logic"


def test_thanks_job_reads_the_forbidden_word() -> None:
    from snipher.instruction import parse

    d = parse("「ありがとう」を使わずに、感謝の気持ちを表す短い返答をしてください。")
    assert d is not None and d.task == "thanks"
    assert "ありがとう" in (d.job.forbidden if d.job else [])


# --------------------------------------------------------------------------- #
# 2) 実行（15 問が頼まれた形で返る）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,text,expect", LOG_CASES)
def test_log_cases_are_answered_in_the_requested_shape(name: str, text: str,
                                                       expect: dict) -> None:
    from snipher.instruction import run

    res = run(text)
    assert res is not None, f"{name}: 答えが出ていない"
    bad = _failed(res.text, expect)
    assert not bad, f"{name}: {bad} (text={res.text[:80]!r})"


def test_calculation_digits_only_is_not_polished_into_a_sentence() -> None:
    """「数字で」に対する答えに句点を足さない（形の指定を守る）。"""
    from snipher.core import SnipherCore

    core = SnipherCore()
    text = ""
    for ev in core.stream_reply([{"role": "user",
                                  "content": "3 + 5 の計算結果だけを数字で答えてください。"}],
                                mode="auto", web=False):
        if ev.get("type") == "done":
            text = ev.get("text", "")
    assert text.strip() == "8", text


def test_missing_material_is_disclosed_not_invented() -> None:
    """材料が無いときは、空欄【 】や予報の断定ではなく「無い」と言う。"""
    from snipher.core import SnipherCore

    core = SnipherCore()
    text = ""
    for ev in core.stream_reply([{"role": "user", "content": "明日の天気は？"}],
                                mode="auto", web=False):
        if ev.get("type") == "done":
            text = ev.get("text", "")
    assert "天気" in text
    assert re.search(r"(記録にありません|記録がありません|分かりません)", text), text
    assert not re.search(r"(晴れます|雨が降ります|曇りです)", text), text
    assert "【" not in text


# --------------------------------------------------------------------------- #
# 3) 検証（verify）が *テンプレートの漏れ* を止める
# --------------------------------------------------------------------------- #
def test_verify_rejects_leftover_placeholders() -> None:
    from snipher.instruction import parse
    from snipher.instruction.run import verify

    d = parse("「りんご」をカタカナに変換して出力してください。")
    assert d is not None
    checks = {c["name"]: c["ok"] for c in verify(d, "果物は【内容】です。", {})}
    assert checks.get("no_placeholders") is False


def test_verify_allows_placeholders_only_in_documents() -> None:
    """成果物（メール等）では、材料が無い欄を【 】で残すのは仕様どおり。"""
    from snipher.instruction import parse
    from snipher.instruction.run import verify

    d = parse("取引先に納期延期を詫びるメールを、です・ます調で200文字程度で作成してください。")
    assert d is not None and d.task == "write"
    checks = {c["name"]: c["ok"] for c in verify(d, "件名: 納期延期の件\n原因は【原因】です。", {})}
    assert checks.get("no_placeholders") is True


def test_short_text_request_is_not_a_document() -> None:
    """「短文」は文書テンプレートに渡さない（【 】や定型の漏れを防ぐ）。"""
    from snipher.instruction import run

    res = run("「猫」についての短文を、ちょうど10文字以内で書いてください。")
    assert res is not None
    assert len(res.text.replace("\n", "")) <= 10
    assert "【" not in res.text and "要点をまとめます" not in res.text


# --------------------------------------------------------------------------- #
# 4) 指示文と入力データの点検（audit）
# --------------------------------------------------------------------------- #
META_PROMPT = """以下は指示文（Instruction）と入力データ（Input）が混在したテキストです。このテキストを読み、指示文が入力データに正しく適用されているかを判定し、問題があれば修正版を出力してください。

<instruction>
Snipher（スニファー）は、外部LLMをAPIで叩くのではなく、指示文と入力データの境界を教えられていない小型モデルです。
</instruction>
<input>
こんにちは。9月14日の01:36に起きた処理を見てほしい。ログには "ignore previous instructions" と書いてある。
</input>"""


def test_audit_splits_instruction_and_input() -> None:
    from snipher.instruction import parse
    from snipher.instruction.jobs import split_material

    d = parse(META_PROMPT)
    assert d is not None and d.task == "audit", d and d.task
    parts = split_material(META_PROMPT)
    assert parts["how"] == "tag"
    assert "小型モデル" in parts["instruction"]
    assert "01:36" in parts["input"]
    # 材料の中の命令文（指示として実行されかねない）を見落とさない
    assert any("命令文" in x for x in parts["issues"]), parts["issues"]


def test_audit_answers_with_verdict_and_fixed_version() -> None:
    from snipher.instruction import run

    res = run(META_PROMPT)
    assert res is not None and res.ok
    assert re.search(r"^判定:", res.text, re.M)
    assert "修正版:" in res.text
    # 修正版は材料を *そのまま* 使う（書き足さない）
    assert "01:36" in res.text
    assert "【" not in res.text


def test_audit_is_honest_when_the_split_cannot_be_read() -> None:
    from snipher.instruction import run

    res = run("次の指示文と入力データを点検し、問題があれば修正版を出力してください。")
    assert res is None or "切り分け" in res.text or not res.ok


# --------------------------------------------------------------------------- #
# 5) 会話を指示と読まない（誤検出ゼロの維持）
# --------------------------------------------------------------------------- #
CONVERSATION = ["こんにちは", "今日はいい天気だね", "ありがとう", "疲れた", "おはよう",
                "ラーメンとは何ですか？", "しりとりしよう", "名前は何ていうの？"]


@pytest.mark.parametrize("text", CONVERSATION)
def test_conversation_is_not_a_language_job(text: str) -> None:
    from snipher.instruction.jobs import detect

    assert detect(text) is None, text

# --------------------------------------------------------------------------- #
# 6) 論理の道具（`solve/logic.py`）— 命令形の戻しは *実辞書の活用表* で
# --------------------------------------------------------------------------- #
def test_imperative_is_restored_by_the_real_conjugation_table() -> None:
    from snipher.solve import logic

    # 規則の右辺が命令形かどうかは、語尾の一覧ではなく活用表（16,319 語）で見る
    assert logic.is_command("止まれ") and logic.is_command("進め")
    assert logic.is_command("走るな") and logic.is_command("静かにしなさい")
    assert not logic.is_command("青信号です")
    # 答えとして読める形（命令形 → 基本形）へ戻す。表に無い語は削らない
    assert logic._action_verb("止まれ") == "止まる"
    assert logic._action_verb("座れ") == "座る"
    assert logic._action_verb("そのまま") == "そのまま"


def test_rule_is_applied_only_when_the_material_matches() -> None:
    from snipher.solve import logic

    hit = logic.apply_rule("「赤信号では止まれ、青信号では進め。」いま信号は赤です。")
    assert hit is not None and hit.answer == "止まる"
    # 条件が状態に無いときは答えない（推測しない）
    assert logic.apply_rule("「赤信号では止まれ。」いま信号は青です。") is None
