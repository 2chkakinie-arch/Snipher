"""v8 再帰的思考ループ（RecurrentMind）のテスト。

* <think>…</think> の思考区間の分離と終端検出
* 思考 → 下書き → 検証 → 却下なら再下書き、が回る
* 校正（モデルC）が反復・文体混在・定型表現を却下する
* テンプレート無しの長文生成が 4-gram の周回なしで続く
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEMPLATE_MARKERS = ("の知識で答えます", "としてお答えします", "お役に立てれば幸いです")


def _mind(tiny_moe):
    from snipher.mind.recurrent import RecurrentMind

    core = tiny_moe["core"]
    return RecurrentMind(cores={"A": core, "B": core, "C": core})


class _StubCore:
    """generate_ids だけ差し替えるコアのスタブ。思考と本文を返し分けられる。"""

    def __init__(self, tok, chunk_ids, answer_ids=None):
        self.tok = tok
        self.is_ready = True
        self.chunk_ids = chunk_ids
        self.answer_ids = list(answer_ids) if answer_ids is not None else list(chunk_ids)
        self.calls = []

    def generate_ids(self, prompt_ids, *, max_new=64, controls=None, seed=None,
                     forbid=(), stop_ids=(), stop_text=None, ctx=128,
                     stop_on_sentence=True, history=None):
        self.calls.append({"max_new": max_new, "controls": controls, "stop_text": stop_text})
        # stop_text が指定されている = 思考区間（</think> で閉じる）
        return list(self.chunk_ids if stop_text else self.answer_ids)

    def score(self, text):
        return {"mean_logprob": -1.0, "ppl": 2.7, "chars": len(text)}


def test_available_and_status(tiny_moe):
    mind = _mind(tiny_moe)
    assert mind.available
    st = mind.status()
    assert st["kind"] == "recurrent-v8"
    assert st["core_A"]["kind"] == "moe"


def test_think_region_splits_on_close_marker(tiny_moe):
    """モデルが </think> を出したら、そこで思考区間を閉じて本文から外す。"""
    from snipher.mind.recurrent import RecurrentMind

    tok = tiny_moe["tok"]
    outline_ids = tok.encode("信号の意味を思い出す。") + tok.encode("</think>")
    stub = _StubCore(tok, outline_ids)
    mind = RecurrentMind(cores={"A": stub, "B": tiny_moe["core"], "C": tiny_moe["core"]})
    th = mind.think("赤信号ではどうする？", seed=0)
    assert th["closed"] is True
    assert th["text"] == "信号の意味を思い出す。"
    assert "</think>" not in th["text"]


def test_think_returns_outline_without_markers(tiny_moe):
    mind = _mind(tiny_moe)
    th = mind.think("赤信号ではどうする？", seed=1)
    assert th["text"] != ""                     # 何かしらの思考は出る
    assert "<think>" not in th["text"] and "</think>" not in th["text"]


def test_respond_produces_text_with_thought(tiny_moe):
    from snipher.mind.recurrent import RecurrentMind

    tok = tiny_moe["tok"]
    stub = _StubCore(tok, tok.encode("信号の意味を思い出す。</think>"),
                     tok.encode("赤信号では止まります。"))
    mind = RecurrentMind(cores={"A": stub, "B": stub, "C": tiny_moe["core"]})
    res = mind.respond("赤信号ではどうする？", max_chars=32, seed=1)
    assert res["route"] == "recurrent"
    assert res["thought"]
    assert res["text"] == "赤信号では止まります。"
    assert res["rounds"] >= 1
    assert "<think>" not in res["text"]


def test_verify_rejects_loops_and_templates(tiny_moe):
    mind = _mind(tiny_moe)
    v = mind.verify("こんにちは。こんにちは。こんにちは。こんにちは。")
    assert v["accepted"] is False
    assert any("反復" in r for r in v["reasons"])
    v2 = mind.verify("私の知識で答えます。猫は動物です。")
    assert v2["accepted"] is False
    assert any("定型表現" in r for r in v2["reasons"])


def test_verify_accepts_clean_sentence(tiny_moe):
    mind = _mind(tiny_moe)
    v = mind.verify("猫は小さな動物です。")
    assert v["accepted"] is True, v["reasons"]


def test_respond_stream_emits_contract(tiny_moe):
    mind = _mind(tiny_moe)
    types = []
    for ev in mind.respond_stream("3×7は？", max_chars=32, seed=1):
        types.append(ev.get("type"))
    assert types[0] == "start"
    assert "thought" in types
    assert "draft" in types
    assert "verify" in types
    assert "delta" in types
    assert types[-1] == "done"


def test_single_core_fallback_covers_all_roles(tiny_moe):
    from snipher.mind.recurrent import RecurrentMind

    tok = tiny_moe["tok"]
    stub = _StubCore(tok, tok.encode("果物を思い浮かべる。</think>"),
                     tok.encode("果物が好きです。"))
    mind = RecurrentMind(cores={"B": stub})
    assert mind.available
    assert mind.cores["A"] is stub and mind.cores["C"] is stub  # 役割を 1 コアで埋める
    res = mind.respond("好きな食べ物は？", max_chars=32, seed=1)
    assert res["text"] == "果物が好きです。"


def test_generate_novel_loop_accumulates_to_max(tiny_moe):
    """ループ論理: 空でないチャンクを max_chars まで積み、反復禁止を通す。"""
    from snipher.mind.recurrent import RecurrentMind

    tok = tiny_moe["tok"]
    chunk = tok.encode("あいうえおかきくけこさしすせそたちつてと")  # 20 文字
    stub = _StubCore(tok, chunk)
    mind = RecurrentMind(cores={"A": stub, "B": stub, "C": stub})
    novel = mind.generate_novel("テスト", max_chars=123, with_think=False)
    assert len(novel) >= 123
    for m in TEMPLATE_MARKERS:
        assert m not in novel
    # no_repeat_ngram=5 が渡っている
    assert any(c["controls"].no_repeat_ngram == 5 for c in stub.calls)


def test_generate_novel_real_model_is_template_free(tiny_moe):
    mind = _mind(tiny_moe)
    novel = mind.generate_novel("小さな町の話", max_chars=120, temperature=0.9, seed=3)
    assert len(novel) > 20
    for m in TEMPLATE_MARKERS:
        assert m not in novel
    assert "<think>" not in novel and "</think>" not in novel
    assert "<user>" not in novel and "<asst>" not in novel
    # 5-gram の周回がない（generate_novel は no_repeat_ngram=5 で物理的に封印）
    seen = set()
    for i in range(len(novel) - 4):
        seen.add(novel[i:i + 5])
    assert len(seen) == len(novel) - 4, "同一 5-gram が反復されている"


def test_rejected_draft_is_redrafted(tiny_moe, monkeypatch):
    """初回下書きが定型表現で却下されたら、再下書きが走ることを保証。"""
    mind = _mind(tiny_moe)
    calls = {"n": 0}

    def fake_draft(user_text, outline_ids=None, *, wave=None, max_chars=None, seed=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"text": "私の知識で答えます。猫は動物です。", "ids": []}
        return {"text": "猫は小さな動物です。", "ids": []}

    monkeypatch.setattr(mind, "draft", fake_draft)
    res = mind.respond("猫とは？", max_chars=32, seed=1)
    assert calls["n"] >= 2
    assert res["accepted"] is True
