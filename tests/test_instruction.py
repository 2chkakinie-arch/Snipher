"""指示層（snipher.instruction）の実測テスト。

ここが確かめるのは「*指示された仕事が、指示された形で* 終わっているか」です。
読み上げ（「テキストを読みました」）や定型文が出てきたら失敗します。

    1. 読み取り  … 指示部／材料部／出力仕様（JSON スキーマ・件数・文字数・口調・役割）
    2. 実行      … 抽出・要約・コード・応答・変換
    3. 検証      … JSON はパースして欄を突き合わせ、箇条書きは本数を数え、
                   文字数は数え、コードは *実際に実行* する
    4. 経路      … core / composer / TaskRouter / think のどこから入っても同じ結果
"""

from __future__ import annotations

import json
import os
import re

import pytest

from snipher.instruction import explain, parse
from snipher.instruction import run as run_instruction
from snipher.instruction.answer import fallback_claims, question_subject
from snipher.instruction.run import CAN_NOT_SAY, verify
from snipher.instruction.style import count_chars, restyle
from web_fixtures import engine, grounding, server  # noqa: F401  (ローカル mini web)


@pytest.fixture()
def client(monkeypatch):
    """API テスト用のクライアント（ニューラルは無効＝決定論的に見る）。"""
    monkeypatch.setenv("SNIPHER_LFM_BACKEND", "off")
    monkeypatch.setenv("SNIPHER_LFM_AUTO_FETCH", "0")
    monkeypatch.setenv("SNIPHER_WEB", "off")
    import snipher.api as api_mod

    api_mod._core = None
    api_mod._lfm_engine = None
    from fastapi.testclient import TestClient

    with TestClient(api_mod.app) as c:
        yield c
    api_mod._core = None


def _sse(client, payload) -> list[dict]:
    events = []
    with client.stream("POST", "/api/chat", json=payload) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events

EXTRACT_PROMPT = """次のテキストから情報を抽出し、必ず指定のJSON形式のみで出力してください。余計な挨拶や解説は不要です。

テキスト: 「東京から京都まで新幹線で約2時間15分。料金は指定席で約14,000円です。」

JSONフォーマット:
{
  "origin": "出発地",
  "destination": "目的地",
  "duration": "所要時間",
  "fare": "料金"
}"""

SUMMARIZE_PROMPT = """以下の文章を読み、重要なポイントを3つの箇条書きで短く要約してください。

文章：
オセロや将棋などの完全情報ゲームにおいて、AIは探索アルゴリズムを用いて最適な手を選択します。ミニマックス法は相手が最善を尽くすことを前提に自分の利益を最大化する手法であり、α-β枝刈りを組み合わせることで無駄な探索を削減できます。さらに評価関数を工夫することで、深い読みを行わなくても強い着手を実現できます。"""

CODE_PROMPT = ("JavaScriptで、配列から重複した要素を取り除いて昇順にソートする関数 "
               "`uniqueSort(arr)` を作成してください。コードと簡単な解説を添えてください。")

ANSWER_PROMPT = """あなたは「頼れるベテランエンジニアのアシスタント」です。

以下の質問に対して、専門的でありながら親しみやすい口調（〜だよ、〜だね）で200文字程度で簡潔に答えてください。

質問：WebAssembly（Wasm）をブラウザで動かす一番のメリットは何ですか？"""

CONVERSATIONAL = [
    "こんにちは", "いま何時？", "ラーメンとは何ですか？", "しりとりしよう", "ありがとう",
    "12+7 はいくつ？", "今日はいい天気だね", "名前は何ていうの？", "もう少し詳しく",
    "pythonって何", "疲れた", "明日の予定は？",
]


# --------------------------------------------------------------------------- #
# 1. 読み取り
# --------------------------------------------------------------------------- #
def test_json_instruction_is_read_as_a_strict_extract_spec() -> None:
    d = parse(EXTRACT_PROMPT)
    assert d is not None and d.task == "extract"
    assert d.fmt.kind == "json" and d.fmt.strict
    assert [k for k, _ in d.fmt.schema_fields] == ["origin", "destination", "duration", "fare"]
    assert "東京" in d.payload and "14,000" in d.payload      # 材料は指示部と分離されている
    assert d.confidence >= 0.55
    assert "材料" in explain(d)


def test_bullet_count_and_length_are_read_from_the_instruction() -> None:
    d = parse(SUMMARIZE_PROMPT)
    assert d is not None and d.task == "summarize"
    assert d.fmt.bullets == 3
    assert d.payload and "ミニマックス法" in d.payload


def test_role_tone_and_length_are_read_as_an_answer_spec() -> None:
    d = parse(ANSWER_PROMPT)
    assert d is not None and d.task == "answer"
    assert d.role and "アシスタント" in d.role
    assert d.fmt.target_chars == 200
    assert d.fmt.tone in ("friendly", "friendly_professional")
    assert d.question and "WebAssembly" in d.question


def test_code_request_reads_language_and_function_name() -> None:
    d = parse(CODE_PROMPT)
    assert d is not None and d.task == "code"
    assert d.fmt.language == "javascript"
    name, _args = d.fmt.function_spec if isinstance(getattr(d.fmt, "function_spec", None), tuple) \
        else (None, None)
    from snipher.instruction.parser import function_spec

    assert function_spec(CODE_PROMPT)[0] == "uniqueSort"


def test_conversational_inputs_are_not_read_as_instructions() -> None:
    for text in CONVERSATIONAL:
        assert parse(text) is None, text


# --------------------------------------------------------------------------- #
# 2 + 3. 実行して検証する
# --------------------------------------------------------------------------- #
def test_extract_outputs_only_json_with_the_exact_values_from_the_text() -> None:
    res = run_instruction(EXTRACT_PROMPT)
    assert res is not None and res.task == "extract"
    data = json.loads(res.text)                     # JSON としてパースできる＝装飾が無い
    assert data == {"origin": "東京", "destination": "京都",
                    "duration": "約2時間15分", "fare": "約14,000円"}
    assert res.ok, [c for c in res.checks if not c["ok"]]
    assert not re.search(r"(?:はい|承知|以下|です。)", res.text.strip().splitlines()[0])
    assert all(c["ok"] for c in res.checks if c["name"] == "strict_json_only")


def test_extract_values_come_from_the_payload_not_from_a_template() -> None:
    other = EXTRACT_PROMPT.replace("東京から京都まで新幹線で約2時間15分。料金は指定席で約14,000円です。",
                                   "博多から熊本まで特急で約50分。料金は自由席で約5,000円です。")
    res = run_instruction(other)
    assert res is not None
    data = json.loads(res.text)
    assert data["origin"] == "博多" and data["destination"] == "熊本"
    assert data["duration"] == "約50分" and data["fare"] == "約5,000円"


def test_summarize_returns_the_requested_number_of_bullets_from_the_text() -> None:
    res = run_instruction(SUMMARIZE_PROMPT)
    assert res is not None and res.task == "summarize"
    lines = [x for x in res.text.split("\n") if x.strip()]
    assert len(lines) == 3
    assert all(x.startswith("・") for x in lines)
    joined = "".join(lines)
    assert "α-β" in joined or "枝刈り" in joined
    assert "評価関数" in joined
    assert res.ok, [c for c in res.checks if not c["ok"]]
    for banned in CAN_NOT_SAY:
        assert banned not in res.text


def test_summarize_does_not_fabricate_points_the_text_does_not_have() -> None:
    thin = "以下の文章を読み、重要なポイントを3つの箇条書きで短く要約してください。\n\n文章：\n猫は寝る。"
    res = run_instruction(thin)
    assert res is not None
    lines = [x for x in res.text.split("\n") if x.strip()]
    assert len(lines) <= 1                      # 材料に無い点をでっち上げない
    assert not res.ok                           # 仕様（3 件）を満たせなかったことは隠さない
    assert any(c["name"] == "bullet_count" and not c["ok"] for c in res.checks)


def test_code_answer_contains_an_executed_function_and_an_explanation() -> None:
    res = run_instruction(CODE_PROMPT)
    assert res is not None and res.task == "code"
    assert "```" in res.text and "uniqueSort" in res.text
    assert res.meta.get("language") == "javascript"
    if res.meta.get("ran"):                     # node がある環境では *実行して* 確かめる
        assert res.meta.get("ok") is True
    body = res.text.split("```")[-1].strip()
    assert body                                # コードの外に解説がある（指示どおり）
    assert res.ok, [c for c in res.checks if not c["ok"]]


def test_typescript_annotations_are_stripped_before_running() -> None:
    from snipher.instruction.code import strip_ts_types

    assert strip_ts_types("function total(nums: number[]): number {") \
        == "function total(nums) {"


def test_python_instruction_runs_the_composed_function() -> None:
    res = run_instruction("Pythonで、リストから重複を除いて降順に並べる関数 dedupeDesc(items) を"
                          "作成してください。")
    assert res is not None and res.task == "code"
    assert "```python" in res.text and "dedupeDesc" in res.text
    if res.meta.get("ran"):
        assert res.meta.get("ok") is True


def test_style_conversion_keeps_predicates_and_honours_the_requested_tone() -> None:
    got, _fixes = restyle("探索を削減できます。静的なサイトは速いです。", tone="friendly")
    assert "削減できるんだよ。" in got
    assert "速いんだね。" in got          # 語尾は だよ／だね を交互に載せる
    polite, _f = restyle("猫は寝る。速い。", register="polite")
    assert "寝ます" in polite and "速いです" in polite


# --------------------------------------------------------------------------- #
# 応答（役割・口調・文字数・材料）
# --------------------------------------------------------------------------- #
def test_answer_follows_role_tone_and_length_without_saying_it_cannot() -> None:
    res = run_instruction(ANSWER_PROMPT)
    assert res is not None and res.task == "answer"
    assert res.ok, [c for c in res.checks if not c["ok"]]
    assert "WebAssembly" in res.text
    assert re.search(r"(?:だよ|だね)[。！？!?]", res.text)       # 指定の語尾
    assert 200 * 0.55 <= count_chars(res.text) <= 200 * 1.55 + 20
    for banned in CAN_NOT_SAY:
        assert banned not in res.text
    assert res.authoritative is False           # 応答は「検証済みの仕事」ではない


def test_answer_uses_the_material_given_in_the_prompt() -> None:
    prompt = ("以下の質問に答えてください。\n\n"
              "資料：しりとりは、前の語の最後の音で始まる語を言う遊びです。\n\n"
              "質問：しりとりのルールは何ですか？")
    res = run_instruction(prompt)
    assert res is not None and res.task == "answer"
    assert "最後の音" in res.text or "しりとり" in res.text
    for banned in CAN_NOT_SAY:
        assert banned not in res.text


def test_answer_on_a_topic_outside_the_kb_counts_facts_instead_of_refusing() -> None:
    d = parse(ANSWER_PROMPT)
    assert d is not None
    claims = fallback_claims(d.question, None, subject="WebAssembly", web=None, kb=None,
                             notes=[])
    body = " ".join(c.content for c in claims)
    assert "WebAssembly" in body
    assert "語彙バンク" in body or "検索" in body
    for banned in CAN_NOT_SAY:
        assert banned not in body


def test_question_subject_reads_the_asked_word_not_a_fragment() -> None:
    assert question_subject("WebAssembly（Wasm）をブラウザで動かすメリットは何ですか？") \
        .startswith("WebAssembly")
    assert question_subject("質問：ラーメンとは何ですか？") == "ラーメン"


def test_answer_uses_web_evidence_when_search_is_available(grounding) -> None:
    """内部化されたネット裏取り（Edge scraping + html-fetch）で答えの材料を取る。"""
    prompt = ("以下の質問に、です・ます調で150文字程度で答えてください。\n\n"
              "質問：GLM5.3 とは何ですか？")
    res = run_instruction(prompt, web=grounding)
    assert res is not None and res.task == "answer"
    assert "GLM5.3" in res.text
    assert "2026 年 4 月" in res.text or "基盤モデル" in res.text
    srcs = res.meta.get("sources") or []
    assert srcs and str(srcs[0].get("url", "")).startswith("http://127.0.0.1")
    assert "出典" in res.text
    # 広告・検索エンジン側の案内文は答えに混ぜない
    assert "リワード" not in res.text and "サインイン" not in res.text
    assert "表現の自由" not in res.text
    for banned in CAN_NOT_SAY:
        assert banned not in res.text


def test_answer_on_a_kb_topic_is_grounded_in_the_kb() -> None:
    res = run_instruction("ラーメンとは何ですか？100文字程度で答えてください。")
    assert res is not None and res.task == "answer"
    assert "麺" in res.text or "スープ" in res.text
    for banned in CAN_NOT_SAY:
        assert banned not in res.text


# --------------------------------------------------------------------------- #
# 変換・列挙（solve 層が実際に計算する）
# --------------------------------------------------------------------------- #
def test_transform_returns_only_the_converted_text() -> None:
    res = run_instruction('次のテキストを大文字に変換してください。\n\nテキスト: "hello snipher"')
    assert res is not None and res.task == "transform"
    assert res.text.strip() == "HELLO SNIPHER"          # 指示文も解説も混ざらない
    assert res.ok, [c for c in res.checks if not c["ok"]]


def test_transform_reads_the_operation_word_not_the_instruction_noun() -> None:
    cases = {
        "次の文を逆順にしてください。\n\nテキスト: abc def": "fed cba",
        "次の行を並び替えてください。\n\nテキスト: banana, apple, cherry": "apple、banana、cherry",
        "次のリストの重複を取り除いてください。\n\nテキスト: a, b, a, c": "a、b、c",
    }
    for prompt, want in cases.items():
        res = run_instruction(prompt)
        assert res is not None, prompt
        assert res.text.strip() == want, f"{prompt} → {res.text!r}"


def test_list_returns_the_requested_number_of_items_from_the_material() -> None:
    prompt = ("以下の項目を3つの箇条書きで列挙してください。\n\n項目：\n"
              "りんごは赤い果物です。みかんは冬に出回る柑橘です。ぶどうは房になって実ります。")
    res = run_instruction(prompt)
    assert res is not None and res.task == "list"
    lines = [x for x in res.text.split("\n") if x.strip()]
    assert len(lines) == 3
    assert "りんご" in res.text and "ぶどう" in res.text


# --------------------------------------------------------------------------- #
# 検証器そのもの
# --------------------------------------------------------------------------- #
def test_verify_reports_every_unmet_part_of_the_spec() -> None:
    d = parse(SUMMARIZE_PROMPT)
    assert d is not None
    checks = verify(d, "・猫は寝る。", {"coverage": 1.0})
    names = {c["name"]: c["ok"] for c in checks}
    assert names["bullet_count"] is False
    assert names["non_empty"] is True
    assert names["no_refusal"] is True

    checks2 = verify(parse(EXTRACT_PROMPT), '{"origin": "東京"}', {"values": {"origin": "東京"},
                                                                  "missing": []})
    names2 = {c["name"]: c["ok"] for c in checks2}
    assert names2["json_schema"] is False          # 欄が足りない
    assert names2["strict_json_only"] is True


def test_verify_rejects_refusals() -> None:
    d = parse(ANSWER_PROMPT)
    assert d is not None
    checks = verify(d, "その質問には回答できません。", {})
    assert any(c["name"] == "no_refusal" and not c["ok"] for c in checks)


# --------------------------------------------------------------------------- #
# 4. 経路（core / composer / TaskRouter / think）
# --------------------------------------------------------------------------- #
def test_core_stream_reply_executes_the_instruction_and_labels_the_route() -> None:
    from snipher.core import SnipherCore

    core = SnipherCore()
    events = list(core.stream_reply([{"role": "user", "content": EXTRACT_PROMPT}],
                                    mode="auto", web=False))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "assist" and "start" in kinds and "delta" in kinds and kinds[-1] == "done"
    assist = events[0]
    assert assist["mode"] == "instruction" and assist["plan"] == "instruction:extract"
    done = events[-1]
    assert json.loads(done["text"])["fare"] == "約14,000円"
    assert done["stats"]["template_mode"] == "instruction"
    assert done["stats"]["task"]["verified"] is True
    assert done["stats"]["route"] == "instant"      # ニューラルを 1 トークンも使わない


def test_core_keeps_conversational_turns_on_the_usual_routes() -> None:
    from snipher.core import SnipherCore

    core = SnipherCore()
    for text in ("こんにちは", "12+7 はいくつ？"):
        events = list(core.stream_reply([{"role": "user", "content": text}],
                                        mode="auto", web=False))
        done = events[-1]
        assert done["stats"]["template_mode"] != "instruction", text
        assert done["text"].strip(), text


def test_composer_returns_the_instruction_result_as_the_reply() -> None:
    from snipher.composer import Composer

    c = Composer()
    reply = c.compose(EXTRACT_PROMPT, turn=1)
    assert reply.plan == "instruction:extract"
    assert json.loads(reply.text)["origin"] == "東京"
    assert reply.notes.get("authoritative") is True


def test_task_router_classifies_instructions_and_answers_them() -> None:
    from snipher.tasks import TaskRouter

    router = TaskRouter()
    assert router.classify(EXTRACT_PROMPT) == "instruction"
    assert router.classify(CODE_PROMPT) == "code"          # 既存の code 契約はそのまま
    assert router.classify("12+7 はいくつ？") == "arithmetic"
    ans = router.answer(SUMMARIZE_PROMPT, web=False)
    assert ans is not None and ans.kind == "instruction"
    assert ans.plan == "instruction:summarize"
    assert len([x for x in ans.text.split("\n") if x.strip()]) == 3


def test_think_returns_the_executed_instruction_as_authoritative() -> None:
    from snipher.knowledge import KnowledgeBase
    from snipher.mind.think import think

    out, thought = think(EXTRACT_PROMPT, kb=KnowledgeBase.shared(), web=None)
    assert out.plan == "instruction:extract"
    assert out.authoritative is True
    assert json.loads(out.text)["destination"] == "京都"
    assert any("指示" in s for s in thought.steps)
    assert thought.knowledge.get("via") == "instruction"


def test_api_chat_streams_the_instruction_result(client) -> None:
    """HTTP（/api/chat）から入っても同じ仕事を返す（SSE の形も同じ）。"""
    events = _sse(client, {"messages": [{"role": "user", "content": EXTRACT_PROMPT}],
                           "mode": "auto", "web": "off"})
    kinds = [e["type"] for e in events]
    assert kinds[0] == "assist" and kinds[-1] == "done" and "delta" in kinds
    done = events[-1]
    assert done["stats"]["template_mode"] == "instruction"
    assert json.loads(done["text"])["destination"] == "京都"


def test_think_does_not_read_a_greeting_as_an_instruction() -> None:
    from snipher.knowledge import KnowledgeBase
    from snipher.mind.think import think

    out, _thought = think("こんにちは", kb=KnowledgeBase.shared(), web=None)
    assert not out.plan.startswith("instruction")


# --------------------------------------------------------------------------- #
# 速度（指示経路はニューラルを起動しないので、ミリ秒台で返る）
# --------------------------------------------------------------------------- #
def test_instruction_route_is_fast() -> None:
    import time

    t0 = time.time()
    res = run_instruction(EXTRACT_PROMPT)
    dt = time.time() - t0
    assert res is not None and res.ok
    assert dt < 2.0, f"抽出に {dt:.2f}s かかった"


# --------------------------------------------------------------------------- #
# 難しい指示の形（欄の読み方・規則・翻訳・コードのみ）
# --------------------------------------------------------------------------- #
CSV_PROMPT = """次のテキストから情報を抽出し、CSV形式のみで出力してください。

テキスト: 「氏名: 山田太郎, 年齢: 34, 住所: 大阪市」

CSVフォーマット:
name,age,city"""

TABLE_PROMPT = """次のテキストから情報を抽出し、表形式で出力してください。

テキスト: 「りんごは1個120円。みかんは1個80円です。」

項目: 名前と値段"""

KV_PROMPT = """次のテキストから情報を抽出し、key: value 形式で出力してください。

テキスト: 「開始は10時、終了は17時です。」"""

RULE_REQUIRE_PROMPT = ("アルファベータ探索について、100文字以内で説明してください。"
                       "必ず「α-β枝刈り」という語を含めてください。")
RULE_FORBID_PROMPT = "Snipherの仕組みを3つの箇条書きで説明してください。「です・ます」は使わないこと。"
TRANSLATE_EN_PROMPT = """次の文章を英語に翻訳してください。

文章: 「今日は天気が良いので、公園に散歩に行きました。」"""
TRANSLATE_JA_PROMPT = ('Translate the following sentence into Japanese.\n\n'
                       'Text: "The weather is good today, so I went to the park for a walk."')
CODE_ONLY_PROMPT = """JavaScriptで、配列の合計を返す関数を書いてください。

コードのみを出力してください（解説は不要です）。"""


def test_parser_reads_csv_header_as_schema() -> None:
    d = parse(CSV_PROMPT)
    assert d is not None and d.task == "extract"
    assert d.fmt.kind == "csv"
    assert [k for k, _ in d.fmt.schema_fields] == ["name", "age", "city"]


def test_parser_reads_field_list_and_length_cap() -> None:
    d = parse(TABLE_PROMPT)
    assert d is not None and d.fmt.kind == "table"
    assert [k for k, _ in d.fmt.schema_fields] == ["名前", "値段"]

    d2 = parse(RULE_REQUIRE_PROMPT)
    assert d2 is not None and d2.fmt.max_chars == 100
    assert ("require", "α-β枝刈り") in [(r.kind, r.value) for r in d2.rules]


def test_parser_reads_a_forbidden_style_as_plain_not_polite() -> None:
    d = parse(RULE_FORBID_PROMPT)
    assert d is not None
    assert ("forbid", "です・ます") in [(r.kind, r.value) for r in d.rules]
    assert d.fmt.tone != "polite", "「使わない」を *使え* と読み替えてはいけない"


def test_csv_extract_fills_the_named_columns() -> None:
    res = run_instruction(CSV_PROMPT)
    assert res is not None and res.ok
    lines = [x for x in res.text.split("\n") if x.strip()]
    assert lines[0] == "name,age,city"
    assert "山田太郎" in lines[1] and "34" in lines[1]


def test_table_extract_makes_one_row_per_record() -> None:
    res = run_instruction(TABLE_PROMPT)
    assert res is not None and res.ok
    body = res.text
    assert "| 名前 | 値段 |" in body
    assert "りんご" in body and "みかん" in body
    assert "120円" in body and "80円" in body
    rows = [x for x in body.split("\n") if x.strip().startswith("|")]
    assert len(rows) == 4, f"ヘッダ + 区切り + 2 行のはず: {rows}"


def test_keyvalue_extract_uses_the_labels_in_the_material() -> None:
    res = run_instruction(KV_PROMPT)
    assert res is not None and res.ok
    assert "field1" not in res.text, "材料に書いてあるラベルを欄名にする"
    assert re.search(r"開始\s*[:：]\s*10時", res.text)
    assert re.search(r"終了\s*[:：]\s*17時", res.text)


def test_required_word_is_included_within_the_length_cap() -> None:
    res = run_instruction(RULE_REQUIRE_PROMPT)
    assert res is not None and res.ok, [(c["name"], c["why"]) for c in (res.checks if res else []) if not c["ok"]]
    assert "α-β枝刈り" in res.text
    assert count_chars(res.text) <= 100


def test_forbidden_style_is_not_used() -> None:
    res = run_instruction(RULE_FORBID_PROMPT)
    assert res is not None and res.ok
    bullets = [x for x in res.text.split("\n") if x.strip().startswith("・")]
    assert len(bullets) == 3
    assert not re.search(r"(?:です|ます)[。、]", res.text), "禁止された語尾が残っている"


def test_translation_into_english() -> None:
    res = run_instruction(TRANSLATE_EN_PROMPT)
    assert res is not None and res.ok
    assert res.meta.get("kind") == "translate"
    for word in ("weather", "park", "walk"):
        assert word in res.text.lower()
    assert not re.search(r"[ぁ-んァ-ヶー一-龯]", res.text), "英語の訳文に日本語が残っている"


def test_translation_into_japanese() -> None:
    res = run_instruction(TRANSLATE_JA_PROMPT)
    assert res is not None and res.ok
    assert res.meta.get("kind") == "translate"
    for word in ("天気", "公園", "散歩"):
        assert word in res.text
    assert re.search(r"(?:ました|です|だ)。?$", res.text.strip()[-4:]) or "行きました" in res.text


def test_code_only_output_has_no_prose() -> None:
    res = run_instruction(CODE_ONLY_PROMPT)
    assert res is not None and res.ok
    assert res.text.startswith("```") and res.text.rstrip().endswith("```")
    outside = re.sub(r"```.*?```", "", res.text, flags=re.DOTALL).strip()
    assert not outside, f"コードの外に文字がある: {outside[:60]}"
    assert "function" in res.text and "reduce" in res.text
