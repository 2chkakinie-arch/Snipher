"""v6 契約 — 「確率の波」が実際に生成へ干渉し、チャットラインへ可視化されること。

3 本柱のどれも *動いていること* を機械的に確かめる:
  1. リアルタイム・ステアリング — 生成中でもプロンプトを受け取り、
     ロジット・バイアスとして次のサンプリングから干渉する（出力は止まらない）
  2. リアルタイム Web 検索 — 挨拶・相槌以外のあらゆるプロンプトで出力中に走り、
     結果が確率波として注入され、`web` イベントとしてチャットラインへ流れる
  3. 並列熟考 — 裏で推論が走り、`thought` イベントとして可視化される

加えて Gemma 2 準拠の確率的生成経路（top_p / 文字単位サンプリング）と、
done イベントの waves 統計・出典の合流を検査する。
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from snipher.lfm import steering as steering_mod
from snipher.lfm.steering import LogitModulator, SteeringBus


# --------------------------------------------------------------------------- #
# 1) SteeringBus の単体契約
# --------------------------------------------------------------------------- #
class _Tok:
    """文字単位の最小トークナイザ（DistilledCore と同じ顔）。"""

    def __init__(self, vocab: str):
        self.vocab = vocab
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = {i: c for i, c in enumerate(vocab)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids)

    def size(self) -> int:
        return len(self.vocab)


def test_steer_creates_a_decaying_probability_wave() -> None:
    tok = _Tok("abcでをは")
    bus = SteeringBus(LogitModulator(tokenizer=tok))
    sid = bus.steer("abc", strength=2.0, ttl=8)
    assert sid >= 0
    biases = bus.active_biases()
    assert biases, "介入プロンプトがロジット・バイアスに変換されていない"
    a_id = tok.stoi["a"]
    first = biases[a_id]
    assert first > 0
    # トークンを進めると波は減衰する（確率の波は時間とともに弱まる）
    for _ in range(4):
        bus.advance()
    assert bus.active_biases()[a_id] < first
    # TTL を過ぎると消える
    for _ in range(20):
        bus.advance()
    assert bus.active_biases() == {}


def test_steer_records_application_events_for_the_timeline() -> None:
    tok = _Tok("abc")
    bus = SteeringBus(LogitModulator(tokenizer=tok))
    bus.steer("abc")
    logits = np.zeros(len(tok.vocab), dtype=np.float32)
    out = bus.apply_to_logits(logits)
    assert out[tok.stoi["a"]] > 0            # ロジットが実際に曲がった
    events = bus.take_events()
    assert any(e["type"] == "steer" and e["text"] == "abc" for e in events)
    assert bus.take_events() == []           # 取り出し後は空（二重表示しない）


def test_evidence_wave_injection() -> None:
    tok = _Tok("量子計算機は速い")
    bus = SteeringBus(LogitModulator(tokenizer=tok))
    assert bus.inject_evidence(["量子計算機は速い"]) >= 0
    assert bus.active_biases()


# --------------------------------------------------------------------------- #
# 2) 生成中のステアリング — 出力を止めずに確率分布が曲がる
# --------------------------------------------------------------------------- #
def _core_with_distilled():
    from snipher.core import SnipherCore

    c = SnipherCore()
    light = c.light_core()
    assert light is not None and light.is_ready, "内蔵蒸留コアがロードできない"
    return c, light


def test_steer_mid_generation_changes_the_output() -> None:
    """生成の途中で steer() すると、以降のトークンの確率分布が実際に変わる。"""
    from snipher.lfm.steering import get_steering_bus

    c, light = _core_with_distilled()
    bus = get_steering_bus(tokenizer=light.tok)
    bus.clear()

    msgs = [{"role": "user", "content": "なにか話して"}]

    def run() -> str:
        out = []
        for ev in light.stream_chat(msgs, max_new_tokens=14, temperature=0.9, top_k=30):
            if ev.get("type") == "delta":
                out.append(ev["text"])
        return "".join(out)

    baseline = run()

    # 生成を別スレッドで走らせ、途中で「波」を投げ込む
    results: dict = {}
    first_token = threading.Event()

    def worker() -> None:
        out = []
        for ev in light.stream_chat(msgs, max_new_tokens=14, temperature=0.9, top_k=30):
            if ev.get("type") == "delta":
                out.append(ev["text"])
                first_token.set()          # 1 トークン目が出た = 生成はまだ続いている
        results["text"] = "".join(out)

    t = threading.Thread(target=worker)
    t.start()
    # 1 トークン目の直後に介入する（固定 sleep だと、速い環境では生成が終わってから
    # 波を投げることになり、このテストが時々落ちる）。
    assert first_token.wait(timeout=30), "最初のトークンが出ない"
    sid = bus.steer("猫猫猫猫猫猫猫", strength=6.0, ttl=40)
    t.join(timeout=30)
    assert sid >= 0
    steered = results.get("text", "")
    # バイアスが発動したイベントが記録されている
    events = bus.take_events()
    assert any(e["type"] == "steer" for e in events)
    # 強い波は出力を実際に曲げる（baseline と違う / あるいは介入文字を含む）
    assert steered != baseline or "猫" in steered
    bus.clear()


def test_api_steer_endpoint_queues_a_wave() -> None:
    from fastapi.testclient import TestClient

    from snipher.api import app

    with TestClient(app) as client:
        r = client.post("/api/steer", json={"text": "もっと短く", "strength": 1.5})
        body = r.json()
        assert r.status_code == 200
        assert body["ok"] is True, body
        assert body["id"] >= 0
        assert body["active_waves"] >= 1


# --------------------------------------------------------------------------- #
# 3) stream_reply の Agent イベント契約
# --------------------------------------------------------------------------- #
def _events(core, text: str, **kw):
    return list(core.stream_reply([{"role": "user", "content": text}], **kw))


def test_non_greeting_runs_realtime_web_and_emits_events(monkeypatch) -> None:
    """挨拶以外は検索が出力中に走り、web イベントがチャットラインへ流れる。"""
    from snipher.core import SnipherCore

    c = SnipherCore()

    class _FakeRealtime:
        def __init__(self):
            self.queries: list[str] = []

        def search_async(self, query, *, on_result=None, extra_queries=()):
            self.queries.append(query)

            def _do():
                if on_result:
                    on_result(["証拠文その1。", "証拠文その2。"],
                              [{"url": "https://example.com/a", "title": "例",
                                "snippet": "証拠"}])
                return None

            return 1, _submit(_do)

    def _submit(fn):
        import concurrent.futures as cf

        ex = cf.ThreadPoolExecutor(max_workers=1)
        return ex.submit(fn)

    fake = _FakeRealtime()
    monkeypatch.setattr(c, "_realtime_web", fake)
    evs = _events(c, "量子アニーリングの利点は何？", wave_wait_ms=2000)
    kinds = [e["type"] for e in evs]
    assert "web" in kinds, kinds
    web_events = [e for e in evs if e["type"] == "web"]
    assert web_events[0]["state"] == "start"
    done_events = [e for e in web_events if e.get("state") == "done"]
    assert done_events and done_events[0]["sentences"] == 2
    assert done_events[0]["sources"][0]["url"] == "https://example.com/a"
    # 検索は実際に出力中に走った
    assert fake.queries == ["量子アニーリングの利点は何？"]
    # done の stats に波の記録と出典が合流する
    stats = evs[-1]["stats"]
    assert stats["waves"]["web_search"] is True
    urls = [s.get("url") for s in stats.get("sources") or []]
    assert "https://example.com/a" in urls
    # done は必ず最後
    assert evs[-1]["type"] == "done"


def test_greeting_skips_web_search_entirely(monkeypatch) -> None:
    from snipher.core import SnipherCore

    c = SnipherCore()

    class _FakeRealtime:
        def __init__(self):
            self.queries: list[str] = []

        def search_async(self, query, *, on_result=None, extra_queries=()):
            self.queries.append(query)
            return -1, None

    fake = _FakeRealtime()
    monkeypatch.setattr(c, "_realtime_web", fake)
    for greet in ("こんにちは", "ありがとう", "おはよう"):
        evs = _events(c, greet)
        assert not [e for e in evs if e["type"] == "web"], greet
        assert evs[-1]["stats"]["waves"]["web_search"] is False


def test_thought_events_are_visible_in_the_stream() -> None:
    from snipher.core import SnipherCore

    c = SnipherCore()
    evs = _events(c, "なぜ空は青いの？", wave_wait_ms=2000)
    thoughts = [e for e in evs if e["type"] == "thought"]
    assert any(t.get("state") == "start" for t in thoughts)
    done_thoughts = [t for t in thoughts if t.get("state") == "done"]
    assert done_thoughts, "熟考の完了イベントが流れない"
    t = done_thoughts[0]
    assert t["steps"] and t["conclusion"]
    assert 0 <= t["confidence"] <= 1
    assert evs[-1]["stats"]["waves"]["deliberate"] is True


def test_wave_events_never_precede_start_and_done_is_last() -> None:
    from snipher.core import SnipherCore

    c = SnipherCore()
    evs = _events(c, "光合成の仕組みを教えて", wave_wait_ms=2000)
    kinds = [e["type"] for e in evs]
    assert kinds[0] in ("assist", "start")
    # start より前に web/thought/steer を出さない（UI の契約）
    first_start = kinds.index("start")
    for w in ("web", "thought", "steer"):
        if w in kinds:
            assert kinds.index(w) > first_start, kinds
    assert kinds[-1] == "done"


def test_realtime_web_evidence_reaches_the_steering_bus(monkeypatch) -> None:
    """検索結果は SteeringBus へ注入され、次の生成の確率に干渉できる。"""
    from snipher.core import SnipherCore
    from snipher.lfm.steering import get_steering_bus

    c = SnipherCore()
    light = c.light_core()
    bus = get_steering_bus(tokenizer=light.tok)
    bus.clear()

    class _FakeGrounding:
        enabled = True

        def gather(self, query, *, explicit=None, extra_queries=()):
            class _Ev:
                def __init__(self, text):
                    self.text = text

            class _G:
                evidence = [_Ev("量子もつれは遠く離れた粒子の相関である。")]
                sources = [{"url": "https://example.com/q", "title": "量子"}]

            return _G()

    from snipher.ground.realtime import RealtimeWebGrounding

    monkeypatch.setattr(c, "_realtime_web", RealtimeWebGrounding(web_grounding=_FakeGrounding()))
    evs = _events(c, "量子もつれとは何ですか", wave_wait_ms=3000)
    assert evs[-1]["stats"]["waves"]["web_search"] is True
    # 証拠文が確率波としてバスに載った（または既に消費された）
    st = bus.status()
    injected = any(s["kind"] == "evidence" for s in st["signals"]) or st["applied_total"] > 0
    assert injected or bus.take_events(), "検索結果が確率波になっていない"
    bus.clear()


# --------------------------------------------------------------------------- #
# 4) Gemma 2 準拠の確率的生成経路
# --------------------------------------------------------------------------- #
def test_gemma_engine_generates_probabilistically_with_top_p() -> None:
    from snipher.core import SnipherCore

    c = SnipherCore()
    gemma = c.gemma_engine()
    assert gemma is not None and gemma.is_ready, "Gemma エンジンが準備できない"
    msgs = [{"role": "user", "content": "今日の夕飯について話そう"}]
    outs = set()
    for _ in range(3):
        text = "".join(ev["text"] for ev in gemma.stream_chat(msgs, max_new_tokens=10)
                       if ev.get("type") == "delta")
        outs.add(text)
    # 確率的サンプリング（temperature/top_p）なので揺れが出る
    assert len(outs) >= 2, outs


def test_gemma_digest_long_reads_beyond_the_context_window() -> None:
    from snipher.core import SnipherCore

    c = SnipherCore()
    gemma = c.gemma_engine()
    assert gemma is not None
    long_text = "。".join(f"これは{i}番目の文です" for i in range(120)) + "。"
    digest = gemma.digest_long(long_text, budget=120)
    assert 0 < len(digest) <= 120
    assert digest.startswith("これは0番目の文です")   # 先頭は必ず保持
    assert long_text[-12:] in digest or "119" in digest  # 末尾も保持


def test_forced_neural_without_heavy_backend_uses_gemma(monkeypatch) -> None:
    from snipher.core import ROUTE_NEURAL, SnipherCore

    c = SnipherCore()
    monkeypatch.setattr(c, "active_backend", lambda: None)
    monkeypatch.setattr(c, "neural_available", lambda: False)
    evs = _events(c, "最近ハマっている趣味について語って", mode="neural", wave_wait_ms=0)
    done = evs[-1]
    assert done["type"] == "done"
    assert done["stats"]["route"] == ROUTE_NEURAL
    assert done["stats"]["gemma"] is True
    assert "Gemma" in done["stats"]["engine"]
