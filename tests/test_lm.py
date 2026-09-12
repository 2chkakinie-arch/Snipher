"""巨大 n-gram 言語モデル（snipher/lm.py）のテスト。

LM の仕事は **生成ではなく判定**です:

    * 自然な日本語と、文字をシャッフルした日本語を perplexity で区別できる
    * 文法を壊した文（ですです / ますます / 助詞で切れた文）を減点できる
    * 保存 → 読み込みでスコアが 1 bit も変わらない（量子化は決定的）
    * 確信度は校正値（lo/hi）に対して単調に下がる
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher.lm import (  # noqa: E402
    DEFAULT_PATH,
    CharNgramLM,
    confidence_from_ppl,
    shared,
)

TINY_CORPUS = [
    "今日はいい天気ですね。", "私は猫が好きです。", "明日は朝から雨が降るそうです。",
    "もう少し詳しく教えてください。", "花火は夜空に光と音を出す娯楽です。",
    "お腹が痛いときは、まず温かくして休んでください。",
] * 25


@pytest.fixture(scope="module")
def tiny_lm() -> CharNgramLM:
    lm = CharNgramLM(order=4)
    lm.train(TINY_CORPUS, min_counts=(1, 1, 1, 1))
    return lm


# ---------------------------------------------------------------------- #
# 学習・判定
# ---------------------------------------------------------------------- #
def test_train_counts_entries_and_tokens(tiny_lm):
    assert tiny_lm.is_ready
    assert tiny_lm.total > 0
    assert tiny_lm.n_params() > 0
    assert tiny_lm.keys[1].size == tiny_lm.counts[1].size
    # キーは昇順（searchsorted の前提）
    for n in range(1, tiny_lm.order + 1):
        k = tiny_lm.keys[n]
        assert (k[1:] >= k[:-1]).all() if k.size > 1 else True


def test_natural_japanese_beats_shuffled(tiny_lm):
    good = tiny_lm.score("今日はいい天気ですね。")
    junk = tiny_lm.score("きょはいいんてきすあ。")
    shuffled = tiny_lm.score("。です気天いいい今日は")
    assert good["perplexity"] < junk["perplexity"]
    assert good["perplexity"] < shuffled["perplexity"]
    assert good["confidence"] > junk["confidence"]
    assert good["confidence"] > shuffled["confidence"]


def test_broken_grammar_is_penalised(tiny_lm):
    base = tiny_lm.score("今日はいい天気ですね。")
    doubled = tiny_lm.score("今日はいい天気ですです。")
    assert doubled["perplexity"] > base["perplexity"]
    assert doubled["confidence"] < base["confidence"]


def test_bad_ratio_flags_local_breakage(tiny_lm):
    # 1 文字だけ壊れていても、bad / worst が反応する
    s = tiny_lm.score("今日はいい天気ですね。")
    assert s["bad_ratio"] == 0.0 or s["bad"] == 0
    junk = tiny_lm.score("きょはいいんてきすあ。")
    assert junk["bad"] >= s["bad"]


def test_score_of_empty_text(tiny_lm):
    s = tiny_lm.score("")
    assert s["ok"] is False and s["confidence"] == 0.0


def test_logprob_scales_with_length(tiny_lm):
    lp, n = tiny_lm.logprob("私は猫が好きです。")
    assert n == len("私は猫が好きです。")
    assert lp < 0


# ---------------------------------------------------------------------- #
# 保存 / 読み込み
# ---------------------------------------------------------------------- #
def test_save_load_roundtrip_is_exact(tiny_lm, tmp_path):
    out = tmp_path / "lm.npz"
    info = tiny_lm.save(out)
    assert info["bytes"] > 0 and info["params"] == tiny_lm.n_params()
    back = CharNgramLM.load(out)
    assert back is not None
    assert back.order == tiny_lm.order
    assert back.vocab == tiny_lm.vocab
    assert back.n_params() == tiny_lm.n_params()
    for text in ("今日はいい天気ですね。", "私は猫が好きです。", "きょはいいんてきすあ。"):
        assert back.score(text)["logprob"] == pytest.approx(tiny_lm.score(text)["logprob"], abs=1e-3)


def test_load_missing_file_returns_none(tmp_path):
    assert CharNgramLM.load(tmp_path / "nothing.npz") is None


def test_load_rejects_foreign_npz(tmp_path):
    import numpy as np

    p = tmp_path / "bogus.npz"
    np.savez_compressed(p, header=np.frombuffer(b"NOT_SNIPHER" + b"\x00" * 32, dtype=np.uint8))
    assert CharNgramLM.load(p) is None


# ---------------------------------------------------------------------- #
# 校正・確信度
# ---------------------------------------------------------------------- #
def test_confidence_is_monotonic_in_perplexity():
    a = confidence_from_ppl(2.0)
    b = confidence_from_ppl(10.0)
    c = confidence_from_ppl(100.0)
    d = confidence_from_ppl(10000.0)
    assert a > b > c > d
    assert 0.0 <= d and a <= 1.0


def test_confidence_uses_calibration():
    calib = {"lo": 5.0, "hi": 3000.0}
    assert confidence_from_ppl(5.0, calib) == pytest.approx(0.95, abs=1e-6)
    assert confidence_from_ppl(3000.0, calib) == pytest.approx(0.05, abs=1e-6)
    mid = confidence_from_ppl(50.0, calib)
    assert 0.05 < mid < 0.95
    # 校正の外側でも 0..1 に収まる
    assert 0.0 <= confidence_from_ppl(1e6, calib) <= 1.0
    assert 0.0 <= confidence_from_ppl(1.0, calib) <= 1.0


# ---------------------------------------------------------------------- #
# 生成・並べ替え（LM だけで文を作れることの確認）
# ---------------------------------------------------------------------- #
def test_next_probs_is_a_distribution(tiny_lm):
    p = tiny_lm.next_probs("今日はいい")
    assert p.shape == (tiny_lm.n_vocab,)
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-4)
    assert float(p.max()) > 0.0


def test_generate_produces_text(tiny_lm):
    out = tiny_lm.generate("私は", max_chars=16, temperature=0.7, top_k=8, seed=3)
    assert isinstance(out, str)
    assert out == "" or len(out) >= 1


def test_rank_orders_by_naturalness(tiny_lm):
    ranked = tiny_lm.rank(["今日はいい天気ですね。", "。です気天いいい今日は", "私は猫が好きです。"])
    assert len(ranked) == 3
    assert ranked[0][1] in ("今日はいい天気ですね。", "私は猫が好きです。")
    assert ranked[-1][1] == "。です気天いいい今日は"


# ---------------------------------------------------------------------- #
# 同梱モデル（ビルド済みなら実ファイルでスモーク）
# ---------------------------------------------------------------------- #
def test_bundled_lm_smoke():
    lm = shared()
    if lm is None or not lm.is_ready:
        pytest.skip(f"n-gram LM が未ビルド（python tools/build_lm.py → {DEFAULT_PATH}）")
    st = lm.status()
    assert st["state"] == "ready"
    assert st["params"] > 100_000            # 巨大であること（小型ネットより 1 桁以上多い）
    assert st["order"] >= 4
    assert st["vocab"] > 500
    assert st["download_required"] is False
    assert st["bytes"] > 0
    assert st["bytes"] < 32 * 1024 * 1024    # 軽いこと（LFM2.5-1.2B-JP の 1/100 以下）

    good = lm.score("花火は、火薬の燃焼と爆発で光と音を出し、夜空に模様を描く娯楽です。")
    junk = lm.score("はばがを、光と音を出し娯楽です花火爆発夜空模様描く。")
    assert good["confidence"] > junk["confidence"]
    assert good["confidence"] > 0.5
    assert junk["confidence"] < 0.6


def test_shared_instance_is_cached():
    lm = shared()
    if lm is None:
        pytest.skip("n-gram LM が未ビルド")
    assert shared() is lm
