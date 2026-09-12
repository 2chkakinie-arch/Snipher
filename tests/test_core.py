"""Snipher Core（内部構造としての LFM2.5）のテスト。

ニューラルコアはフェイクで差し替え、経路判定（instant / neural / fallback）・
助動詞の補い・イベント契約を検証する。torch 不要。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher.core import (  # noqa: E402
    CANNED_INTENTS,
    ROUTE_FALLBACK,
    ROUTE_INSTANT,
    ROUTE_NEURAL,
    SnipherCore,
    _looks_incomplete,
)


class FakeNeural:
    """LfmEngine / GgufBackend と同じイベント契約のフェイク。"""

    kind = "fake"

    def __init__(self, reply: str = "はい、それは良い考え方ですね。"):
        self.reply = reply
        self.calls: list[dict] = []
        self.fail = False

    @property
    def is_ready(self) -> bool:
        return True

    def engine_name(self) -> str:
        return "LFM2.5-FAKE (test)"

    def status(self) -> dict:
        return {"state": "ready", "error": None, "learned_chars": [],
                "is_local": True, "backend": "fake"}

    def scan_unknown(self, text: str):
        return None

    def stream_chat(self, messages, **opts):
        self.calls.append({"messages": messages, **opts})
        if self.fail:
            yield {"type": "error", "message": "fake failure"}
            return
        yield {"type": "start", "engine": self.engine_name(), "template_mode": "native",
               "prompt_tokens": 10, "max_new_tokens": 64}
        for piece in (self.reply[: len(self.reply) // 2], self.reply[len(self.reply) // 2:]):
            yield {"type": "delta", "text": piece}
        yield {"type": "done", "text": self.reply, "stats": {
            "new_tokens": 8, "tokens_per_second": 42.0, "template_mode": "native",
            "engine": self.engine_name(),
        }}


def _core_with(fake: FakeNeural | None) -> SnipherCore:
    core = SnipherCore(torch_provider=lambda: None)
    if fake is not None:
        core.backend_kind = "fake"
        core.active_backend = lambda: fake  # type: ignore[method-assign]
    else:
        core.backend_kind = None
        core.active_backend = lambda: None  # type: ignore[method-assign]
    return core


def _events(core: SnipherCore, text: str, **kw) -> list[dict]:
    return list(core.stream_reply([{"role": "user", "content": text}], **kw))


# ---------------------------------------------------------------------- #
# 経路判定
# ---------------------------------------------------------------------- #
def test_route_instant_for_canned_intents():
    fake = FakeNeural()
    core = _core_with(fake)
    draft = core.assist.draft("こんにちは")
    assert draft["intent"] in CANNED_INTENTS
    assert core.route_of(draft, "auto") == ROUTE_INSTANT
    assert fake.calls == []  # ニューラルコアは 1 トークンも消費しない


def test_route_neural_for_uncertain():
    fake = FakeNeural()
    core = _core_with(fake)
    draft = core.assist.draft("量子コンピュータの仕組みってどうなってるの")
    assert core.route_of(draft, "auto") == ROUTE_NEURAL


def test_route_fast_mode_never_neural():
    fake = FakeNeural()
    core = _core_with(fake)
    draft = core.assist.draft("量子コンピュータの仕組みってどうなってるの")
    assert core.route_of(draft, "fast") == ROUTE_INSTANT


def test_route_fallback_without_neural():
    core = _core_with(None)
    core.disable_light()          # 内蔵蒸留コアも無い（＝ルールのみ）環境の契約
    draft = core.assist.draft("こんにちは")
    assert core.route_of(draft, "auto") == ROUTE_FALLBACK


# ---------------------------------------------------------------------- #
# イベント契約
# ---------------------------------------------------------------------- #
def test_instant_path_events():
    fake = FakeNeural()
    core = _core_with(fake)
    evs = _events(core, "こんにちは")
    kinds = [e["type"] for e in evs]
    assert kinds[0] == "assist" and kinds[1] == "start"
    assert "delta" in kinds and kinds[-1] == "done"
    done = evs[-1]
    assert done["stats"]["assist"] == "rule"
    assert done["stats"]["route"] == "instant"
    assert done["text"]
    assert fake.calls == []


def test_neural_path_streams_and_postprocesses():
    fake = FakeNeural(reply="そうですか、それは面白いですね")
    core = _core_with(fake)
    evs = _events(core, "最近ハマっていることについてどう思う？")
    kinds = [e["type"] for e in evs]
    assert kinds[0] == "assist" and evs[0]["mode"] == "lfm"
    assert "start" in kinds and kinds.count("delta") >= 2 and kinds[-1] == "done"
    done = evs[-1]
    assert done["stats"]["assist"] == "lfm"
    assert done["stats"]["route"] == "neural"
    assert done["stats"]["tokens_per_second"] == 42.0
    # ニューラル出力も内部で助動詞の補い(polisher)を通る → 句点が補われる
    assert done["text"] == "そうですか、それは面白いですね。"
    assert any("terminal_punct" in f for f in done["stats"]["fixes"])
    # 内部プロンプト: Snipher の人格が system_prompt として神経系に渡される
    call = fake.calls[0]
    assert call["system_prompt"] and "会話パートナー" in call["system_prompt"]
    assert any(m["role"] == "user" for m in call["messages"])


def test_neural_failure_falls_back_to_draft():
    fake = FakeNeural()
    fake.fail = True
    core = _core_with(fake)
    evs = _events(core, "幽体離脱のやり方を教えて")
    kinds = [e["type"] for e in evs]
    assert kinds[-1] == "done"
    done = evs[-1]
    assert done["text"]  # 必ず何か返す
    assert done["stats"]["neural_fallback"] is True
    assert "delta" in kinds  # フォールバック文もストリームされる


def test_forced_lfm_mode_skips_assist_event():
    fake = FakeNeural()
    core = _core_with(fake)
    evs = _events(core, "こんにちは", mode="lfm")
    kinds = [e["type"] for e in evs]
    assert "assist" not in kinds          # 直接経路では assist を出さない
    assert kinds[0] == "start"
    assert evs[-1]["stats"]["engine"].startswith("LFM")


def test_fallback_uses_base_text_for_uncertain():
    core = _core_with(None)
    core.disable_light()
    evs = _events(core, "意味不明きょくせんぷる語の羅列はどう？")
    done = evs[-1]
    assert done["stats"]["route"] == "fallback"
    assert done["stats"]["fallback_reason"]
    assert done["text"]


# ---------------------------------------------------------------------- #
# 助動詞の補い
# ---------------------------------------------------------------------- #
def test_looks_incomplete():
    assert _looks_incomplete("私は猫が好き")
    assert not _looks_incomplete("私は猫が好きです。")
    assert not _looks_incomplete("行くよ")


def test_complete_fragment_rules_only():
    core = _core_with(None)
    r = core.complete_fragment("私は猫が好き", use_neural=False)
    assert r["engine"] == "rule"
    assert r["text"].rstrip().endswith(("。", "です", "ます")) or len(r["text"]) >= len("私は猫が好き")


def test_complete_fragment_with_neural():
    fake = FakeNeural(reply="私は猫が好きです。")
    core = _core_with(fake)
    r = core.complete_fragment("私は猫が好き")
    assert r["engine"] == fake.engine_name()
    assert "neural_completion" in r["fixes"]
    assert r["text"] == "私は猫が好きです。"


def test_complete_fragment_skips_neural_for_complete_text():
    fake = FakeNeural()
    core = _core_with(fake)
    r = core.complete_fragment("今日は良い天気です。")
    assert r["engine"] == "rule"
    assert fake.calls == []


# ---------------------------------------------------------------------- #
# 状態
# ---------------------------------------------------------------------- #
def test_status_shape():
    fake = FakeNeural()
    core = _core_with(fake)
    st = core.status()
    assert st["neural_ready"] is True
    assert st["engine_label"] == fake.engine_name()
    assert "acquire" in st
    core2 = _core_with(None)
    st2 = core2.status()
    assert st2["neural_ready"] is False
