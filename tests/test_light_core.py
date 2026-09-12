"""内蔵ニューラルコア（LFM2.5 蒸留スナップショット）と階層ルーティングのテスト。

重い LFM2.5 が居ない環境（Vercel 等）で知能を担うのは次の 2 つ:

    * `snipher/neural/*` … NumPy だけで動く文字レベル LM（int8 量子化・リポジトリ同梱）
    * `snipher/knowledge.py` … 語彙テーブルから生成した知識ベース（BM25）

ここではフェイクのコアを注入して経路契約を検証し、実重みが同梱されている場合だけ
追加のスモーク検査を走る（`pytest -m slow` 不要・訓練は行わない）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher.core import (  # noqa: E402
    ROUTE_FALLBACK,
    ROUTE_INSTANT,
    ROUTE_LIGHT,
    ROUTE_NEURAL,
    SnipherCore,
)


class FakeLight:
    """DistilledCore と同じ顔のフェイク。"""

    kind = "distilled"

    def __init__(self, reply="猫を飼うのは楽しいですね。", *, score_conf=0.9, ppl=4.0):
        self._reply = reply
        self._conf = score_conf
        self._ppl = ppl
        self.reply_calls: list[dict] = []
        self.complete_calls: list[str] = []
        self.score_calls: list[str] = []

    @property
    def is_ready(self) -> bool:
        return True

    def engine_name(self) -> str:
        return "Snipher 内蔵ニューラルコア (test)"

    def status(self) -> dict:
        return {"kind": "distilled", "state": "ready", "engine": self.engine_name(),
                "params": 1234, "vocab": 300, "runtime": "numpy", "download_required": False}

    def reply(self, user_text, **kw):
        self.reply_calls.append({"user": user_text, **kw})
        return self._reply

    def generate(self, prompt="", **kw):
        return self._reply

    def score(self, text, **kw):
        self.score_calls.append(text)
        return {"perplexity": self._ppl, "mean_logprob": -1.4, "confidence": self._conf, "ok": True}

    def complete(self, fragment, **kw):
        self.complete_calls.append(fragment)
        return {"text": fragment + "です。", "added": "です。", "changed": True, "confidence": 0.8}


def _core_with_light(light: FakeLight | None) -> SnipherCore:
    core = SnipherCore(torch_provider=lambda: None)
    core.active_backend = lambda: None          # type: ignore[method-assign]
    if light is None:
        core.disable_light()
    else:
        core._light = light
        core._light_state = "ready"
        core.cfg.light_core = "on"
    return core


def _events(core, text, **kw):
    return list(core.stream_reply([{"role": "user", "content": text}], **kw))


# ---------------------------------------------------------------------- #
# 経路
# ---------------------------------------------------------------------- #
def test_distilled_core_enables_light_route():
    core = _core_with_light(FakeLight())
    draft = core.assist.draft("量子コンピュータの仕組みってどうなってるの")
    assert core.route_of(draft, "auto") == ROUTE_LIGHT
    assert core.neural_available() is False      # フルウェイトは居ない
    assert core.any_neural() is True


def test_canned_intents_stay_instant_even_with_light_core():
    """挨拶などの確実な応答は、蒸留コアがあってもニューラルを通さない（速度維持）。"""
    core = _core_with_light(FakeLight())
    draft = core.assist.draft("こんにちは")
    assert core.route_of(draft, "auto") == ROUTE_INSTANT


def test_fast_mode_ignores_light_core():
    light = FakeLight()
    core = _core_with_light(light)
    evs = _events(core, "量子コンピュータって何？", mode="fast")
    assert evs[-1]["stats"]["route"] == ROUTE_INSTANT
    assert light.reply_calls == []


def test_forced_neural_mode_uses_light_when_heavy_absent():
    light = FakeLight()
    core = _core_with_light(light)
    evs = _events(core, "最近嵌っているぬるぬる猿について語って", mode="lfm")
    done = evs[-1]
    assert done["stats"]["route"] == ROUTE_LIGHT
    assert light.reply_calls                      # 内蔵コアに本文生成を依頼している
    assert done["stats"]["neural_used"] is True   # 候補採点も内蔵コアが担う
    assert done["stats"]["engine"].startswith("Snipher 内蔵ニューラルコア")
    assert done["stats"]["neural_confidence"] > 0


def test_route_fallback_without_any_neural():
    core = _core_with_light(None)
    draft = core.assist.draft("量子コンピュータの仕組み")
    assert core.route_of(draft, "auto") == ROUTE_FALLBACK


# ---------------------------------------------------------------------- #
# 知識ベースとの連携
# ---------------------------------------------------------------------- #
def test_knowledge_hit_is_used_in_light_path():
    core = _core_with_light(FakeLight())
    evs = _events(core, "雨ってなぜ降るの？")
    done = evs[-1]
    assert done["stats"]["route"] == ROUTE_LIGHT
    kn = done["stats"]["knowledge"]
    assert kn and kn.get("topic"), "知識ベースのトピックが引用される"
    assert done["text"]


def test_status_reports_tiers():
    core = _core_with_light(FakeLight())
    st = core.status()
    assert st["light_ready"] is True
    assert st["tiers"]["distilled"] in ("ready", "unchecked")
    assert st["knowledge"]["facts"] > 100
    assert st["neural_ready_any"] is True
    assert st["engine_label"] is None            # 重い LFM2.5 は未ロード


# ---------------------------------------------------------------------- #
# 助動詞の補い
# ---------------------------------------------------------------------- #
def test_complete_fragment_uses_light_core():
    light = FakeLight()
    core = _core_with_light(light)
    # ルール(polisher)では断定できない断片文だけニューラルに回る
    r = core.complete_fragment("歩いていたら急に雨が")
    assert r["engine"] == light.engine_name()
    assert len(light.complete_calls) == 1
    assert r["text"].endswith("です。")
    assert "neural_completion" in r["fixes"]


def test_complete_fragment_rule_path_first():
    light = FakeLight()
    core = _core_with_light(light)
    r = core.complete_fragment("私は猫が好き")     # polisher が確実に直せる
    assert r["engine"] in ("rule", light.engine_name())
    assert r["text"].endswith(("です。", "ます。", "。"))


def test_complete_fragment_skips_light_for_finished_text():
    light = FakeLight()
    core = _core_with_light(light)
    r = core.complete_fragment("今日は良い天気です。")
    assert r["engine"] == "rule"
    assert light.complete_calls == []


# ---------------------------------------------------------------------- #
# 昇格（確信度が低いときだけフルウェイトの起動を試みる）
# ---------------------------------------------------------------------- #
def test_low_confidence_queues_escalation():
    light = FakeLight(score_conf=0.05, ppl=30.0)
    core = _core_with_light(light)
    core.ensure_started = lambda: setattr(core, "_boot_called", True)  # type: ignore[method-assign]
    done = _events(core, "意味不明きょくせんぷる語の羅列はどう？")[-1]
    assert done["stats"]["escalation"] == "queued_full_weights"
    assert core._boot_called is True
    assert done["text"]


def test_high_confidence_does_not_escalate():
    light = FakeLight(score_conf=0.9, ppl=4.0)
    core = _core_with_light(light)
    core.ensure_started = lambda: setattr(core, "_boot_called", True)  # type: ignore[method-assign]
    done = _events(core, "週末は映画を見ました")[-1]
    assert "escalation" not in done["stats"]
    assert not getattr(core, "_boot_called", False)


# ---------------------------------------------------------------------- #
# 実装の単体（訓練不要で走る）
# ---------------------------------------------------------------------- #
def test_tokenizer_roundtrip_and_roles():
    from snipher.neural.tokenizer import CharTokenizer

    tok = CharTokenizer.from_text("今日は天気がいいですね。今日は", max_vocab=60)
    ids = tok.encode("今日は天気がいいですね。")
    assert tok.decode(ids) == "今日は天気がいいですね。"
    assert tok.has_unknown("今日は天気がいいですね") is False
    assert tok.has_unknown("鼕𠮷") is True          # 語彙外の文字は unk に寄せる
    assert tok.role_ids("user") and tok.role_ids("assistant")
    from snipher.neural.tokenizer import BOS as _BOS
    assert tok.encode("今日", add_bos=True)[0] == _BOS


def test_store_quantization_roundtrip():
    import numpy as np

    from snipher.neural.nn import MicroNet, NNConfig, forward_backward
    from snipher.neural.store import load, save

    cfg = NNConfig(n_vocab=64, d_model=16, n_layers=2, n_heads=2, max_pos=64)
    net = MicroNet.random(cfg, seed=3)
    tmp = Path(".pytest_store_tmp")
    tmp.mkdir(exist_ok=True)
    out = tmp / "core.npz"
    info = save(out, net, list("あいうえお"), extra={"hello": "world"})
    try:
        net2, vocab, extra = load(out)
        assert extra["hello"] == "world"
        assert vocab == list("あいうえお")
        assert set(net2.params) == set(net.params)
        for k, v in net.params.items():
            d = float(np.abs(v - net2.params[k]).max())
            assert d <= abs(float(v.max())) * 0.05 + 1e-3, f"{k} の量子化誤差が大きすぎ: {d}"
        x = np.array([[1, 5, 9, 12, 3]], dtype=np.int64)
        lg, _ = net2.forward(x)
        assert lg.shape == (1, 5, 64)
        assert np.isfinite(lg).all()
        assert info["params"] == net.n_params()
    finally:
        (tmp / "core.npz").unlink(missing_ok=True)
        tmp.rmdir()


def test_corpus_generator_is_grammatical():
    """コーパス生成品質（助動詞の形が崩れた文を学習させない）。"""
    from snipher.neural.corpus import CorpusBuilder, _clean

    b = CorpusBuilder(seed=5)
    sents = [s for s in (b.sentence() for _ in range(600)) if s]
    assert len(sents) > 400
    assert not any("ことができます" in s and "ますこと" in s for s in sents)
    assert not any("ますので" in s and "ますます" in s for s in sents)
    assert all(_clean(s) for s in sents)
    assert any(s.endswith("ません。") for s in sents)
    assert any(s.endswith("たいです。") for s in sents)
    assert any(s.endswith("ください。") for s in sents)


def test_trainer_reduces_loss():
    """少量データでも loss が下がること（訓練パイプラインの健全性）。"""
    pytest.importorskip("numpy")
    from snipher.neural.nn import MicroNet, NNConfig
    from snipher.neural.tokenizer import BOS, EOS, CharTokenizer
    from snipher.neural.train import TextDataset, TrainConfig, Trainer

    texts = [f"私は毎日{n}を勉強します。" for n in ("日本語", "数学", "音楽", "料理", "英語", "歴史")]
    tok = CharTokenizer.from_text("".join(texts), max_vocab=80)
    ids: list[int] = []
    for _ in range(12):
        for t in texts:
            ids += [BOS] + tok.encode(t) + [EOS]
    import numpy as np

    cfg = NNConfig(n_vocab=tok.size(), d_model=24, n_layers=2, n_heads=3, max_pos=64)
    net = MicroNet.random(cfg, seed=1)
    ds = TextDataset(np.array(ids, dtype=np.int64), 24, np.random.default_rng(0))
    tc = TrainConfig(seq_len=24, batch=16, epochs=3, lr=8e-3, warmup_ratio=0.0, min_lr=1e-4)
    base = Trainer(net, ds, ds, tc)._eval(ds, 2)["loss"]
    res = Trainer(net, ds, ds, tc).train()
    assert res["best_val"]["loss"] < base * 0.9, (base, res["best_val"])
    assert res["best_val"]["ppl"] > 1.0


# ---------------------------------------------------------------------- #
# 同梱スナップショット（ビルド済みなら実ファイルでスモーク）
# ---------------------------------------------------------------------- #
def test_bundled_snapshot_smoke():
    import numpy as np

    from snipher.neural.core import DistilledCore, available

    if not available():
        pytest.skip("内蔵ニューラルコアの重みが未ビルド（python tools/distill_neural.py）")
    core = DistilledCore()
    assert core.is_ready
    st = core.status()
    assert st["state"] == "ready" and st["params"] > 10_000 and st["download_required"] is False
    # seed 固定で決定的に検査する（温度>0 でも再現できるように）
    txt = core.generate("今日は天気が", max_chars=24, temperature=0.6, seed=1234)
    assert isinstance(txt, str) and len(txt) >= 1
    sc = core.score("今日はいい天気ですね。")
    assert sc["ok"] and sc["confidence"] > 0.3 and sc["perplexity"] < 60
    comp = core.complete("私は毎日朝に", seed=1234)
    assert comp["text"].startswith("私は毎日朝に")
    assert comp["changed"] is True
    assert comp["added"]
    # KV キャッシュ: 逐次デコードと一括計算は同じ logits になる
    from snipher.neural.nn import DecodeCache

    ids = [2] + core.tok.encode("今日は天気が")
    full, _ = core.net.forward(np.array([ids], dtype=np.int64))
    cache = DecodeCache(core.net, batch=1)
    last = cache.prefill(core.net, np.array([ids[:-1]], dtype=np.int64))
    step = cache.step(core.net, np.array([ids[-1]], dtype=np.int64))
    assert float(np.abs(last - full[0, -2]).max()) < 1e-3
    assert float(np.abs(step - full[0, -1]).max()) < 1e-3
    a = core.generate_ids(ids, max_new=12, temperature=0.9, seed=7, use_cache=True)
    b = core.generate_ids(ids, max_new=12, temperature=0.9, seed=7, use_cache=False)
    assert a == b
    evs = list(core.stream_chat([{"role": "user", "content": "こんにちは"}], max_new_tokens=16))
    kinds = [e["type"] for e in evs]
    assert "start" in kinds and kinds[-1] == "done"
    assert evs[-1]["stats"]["engine"].startswith("Snipher")
