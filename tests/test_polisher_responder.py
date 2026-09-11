"""polisher(助動詞の補い)と responder(対話テーブル)のテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher.polisher import Polisher  # noqa: E402
from snipher.responder import Responder  # noqa: E402


# ---------------------------------------------------------------------- #
# polisher: 助動詞の補い
# ---------------------------------------------------------------------- #
def test_polish_terminal_punct():
    p = Polisher()
    out = p.polish("今日はいい天気ですね", register="polite")
    assert out["text"] == "今日はいい天気ですね。"
    assert "terminal_punct" in " ".join(out["fixes"])


def test_polish_duplicate_auxiliaries():
    p = Polisher()
    out = p.polish("そうですますです。", register="polite")
    assert "ますです" not in out["text"]
    assert out["fixes"]


def test_polish_verb_masu_repair():
    p = Polisher()
    # 飲む(五段) + ます は 飲みます になる
    out = p.polish("コーヒーを飲むます。", register="polite")
    assert "飲みます" in out["text"]
    assert "飲むます" not in out["text"]
    assert any(f.startswith("verb_masu") for f in out["fixes"])


def test_polish_polite_ending_da():
    p = Polisher()
    out = p.polish("それは名詞だ。", register="polite")
    assert out["text"].endswith("です。")
    assert any(f.startswith("polite_ending") for f in out["fixes"])
    # 普通体ターゲットなら触らない
    out2 = p.polish("それは名詞だ。", register="casual")
    assert out2["text"].endswith("だ。")


def test_polish_adj_da():
    p = Polisher()
    out = p.polish("この本は面白いだ。", register="polite")
    assert "面白いです" in out["text"]


def test_polish_keeps_clean_text():
    p = Polisher()
    out = p.polish("今日もいい一日になりますね。", register="polite")
    assert out["text"] == "今日もいい一日になりますね。"
    assert out["changed"] is False
    assert out["fixes"] == []


# ---------------------------------------------------------------------- #
# responder: 対話テーブル
# ---------------------------------------------------------------------- #
def test_responder_greeting():
    r = Responder()
    out = r.reply("こんにちは")
    assert out["intent"] == "greeting"
    assert out["text"]
    assert out["confidence"] >= 0.9
    assert out["use_generator"] is False


def test_responder_thanks():
    r = Responder()
    out = r.reply("ありがとうございます！")
    assert out["intent"] == "thanks"


def test_responder_identity():
    r = Responder()
    out = r.reply("あなたは誰ですか")
    assert out["intent"] == "identity"
    assert "Snipher" in out["text"]


def test_responder_fallback_uses_generator_and_reports_confidence():
    r = Responder()
    out = r.reply("へえ、そうなんだ")
    assert out["intent"] in ("agreement", "fallback")
    assert out["base_text"]  # 安全な骨子がある
    if out["use_generator"]:
        assert 0.0 < out["confidence"] <= 1.0
        # base_text は生成文を含まない(短い)
        assert len(out["base_text"]) <= len(out["text"])


def test_responder_topic_reflection():
    r = Responder()
    out = r.reply("昨日、コーヒーを飲みました")
    assert out["topic"] is None or out["topic"]  # 解析次第だが壊れない
    assert out["text"]


def test_responder_reply_is_fast():
    import time

    r = Responder()
    t0 = time.time()
    for i in range(20):
        r.reply(f"テストの入力{i}です")
    elapsed = time.time() - t0
    assert elapsed < 2.0  # 20回で2秒未満(1応答あたり 100ms 未満)
