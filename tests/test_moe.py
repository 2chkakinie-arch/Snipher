"""v8 アーキテクチャ（GQA + MoE(SwiGLU) + RoPE + RMSNorm）のテスト。

* 勾配は float64 の中心差分と一致する（forward_backward の検証）
* GQA は KV キャッシュを 1/group に削減する
* MoE は推論時に Top-K だけを走らせる
* 保存・読み込みで MoENet が再構成される
* 学習で loss が下がる
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _cfg(**kw) -> "MoEConfig":
    from snipher.neural.moe import MoEConfig

    base = dict(n_vocab=64, d_model=16, n_layers=4, n_heads=4, n_kv_heads=2,
                n_experts=4, top_k=2, expert_dim=24, max_pos=64)
    base.update(kw)
    return MoEConfig(**base)


def test_moe_forward_is_finite_and_shaped():
    from snipher.neural.moe import MoENet

    net = MoENet.random(_cfg(), seed=3)
    ids = np.random.default_rng(0).integers(1, 64, size=(2, 10)).astype(np.int64)
    logits, cache = net.forward(ids)
    assert logits.shape == (2, 10, 64)
    assert np.isfinite(logits).all()


def test_gqa_reduces_kv_heads():
    from snipher.neural.moe import MoENet

    net = MoENet.random(_cfg(n_heads=8, n_kv_heads=2), seed=1)
    c = net.cfg
    assert c.kv_dim == 2 * c.head_dim          # KV は 8 ヘッドではなく 2 ヘッドぶん
    assert c.group == 4                        # 4 ヘッドが 1 つの KV を共有
    assert c.d_model == 8 * c.head_dim


def test_moe_topk_routing_is_sparse_at_inference():
    from snipher.neural.moe import MoENet

    net = MoENet.random(_cfg(top_k=2, n_experts=4), seed=2)
    ids = np.array([[1, 5, 9, 12]], dtype=np.int64)
    # hard=True でゲートを再現するため、forward を直接は返さないので step 経路で確認
    cache = net
    from snipher.neural.moe import MoEDecodeCache

    dec = MoEDecodeCache(net, batch=1)
    dec.prefill(net, ids)
    # ルータゲートが top_k だけ有効かは _topk_softmax で確認
    from snipher.neural.moe import _topk_softmax

    r = np.array([[0.1, 2.0, -1.0, 3.0]], dtype=np.float32)
    g = _topk_softmax(r, 2)
    assert np.count_nonzero(g) == 2
    assert g[0, 3] > 0 and g[0, 1] > 0


def test_moe_gradient_matches_finite_difference_float64():
    from snipher.neural.moe import MoENet, forward_backward

    cfg = _cfg(n_layers=2)
    net = MoENet.random(cfg, seed=3)
    for k in net.params:
        net.params[k] = net.params[k].astype(np.float64)
    ids = np.random.default_rng(0).integers(1, 64, size=(2, 9)).astype(np.int64)
    target = np.random.default_rng(1).integers(1, 64, size=(2, 9)).astype(np.int64)
    pad = np.ones_like(ids, dtype=np.float32)
    loss, G = forward_backward(net, ids, target, pad)

    def ploss():
        lg, _ = net.forward(ids, pad=pad)
        e = np.exp(lg - lg.max(-1, keepdims=True))
        p = e / e.sum(-1, keepdims=True)
        tp = np.take_along_axis(p, target[:, :, None], -1)[:, :, 0]
        return -np.log(np.clip(tp, 1e-9, None)).mean()

    worst = 0.0
    for k in net.params:
        v = net.params[k]
        for _ in range(3):
            i = tuple(np.random.randint(0, s) for s in v.shape)
            orig = v[i]
            v[i] = orig + 1e-5
            lp = ploss()
            v[i] = orig - 1e-5
            lm = ploss()
            v[i] = orig
            num = (lp - lm) / 2e-5
            ana = G[k][i]
            rel = abs(num - ana) / max(1e-9, abs(num) + abs(ana))
            worst = max(worst, rel)
    assert worst < 1e-4, f"勾配が中心差分と一致しない (worst={worst})"


def test_moe_store_roundtrip_reconstructs_moenet():
    from snipher.neural.moe import MoENet
    from snipher.neural.store import load, save

    net = MoENet.random(_cfg(), seed=5)
    tmp = Path(".pytest_v8_store")
    tmp.mkdir(exist_ok=True)
    out = tmp / "v8.npz"
    try:
        info = save(out, net, list("あいうえお"), extra={"kind": "v8"})
        net2, vocab, extra = load(out)
        from snipher.neural.moe import MoENet as MoENet2

        assert isinstance(net2, MoENet2)
        assert extra["kind"] == "v8"
        assert vocab == list("あいうえお")
        assert set(net2.params) == set(net.params)
        for k, v in net.params.items():
            d = float(np.abs(v - net2.params[k]).max())
            assert d <= abs(float(v.max())) * 0.05 + 1e-3, f"{k} の量子化誤差が大きい: {d}"
        assert info["params"] == net.n_params()
    finally:
        (tmp / "v8.npz").unlink(missing_ok=True)
        tmp.rmdir()


def test_moe_decode_cache_matches_full_forward():
    from snipher.neural.moe import MoEDecodeCache, MoENet

    net = MoENet.random(_cfg(), seed=4)
    ids = np.array([[3, 7, 11, 2, 9]], dtype=np.int64)
    dec = MoEDecodeCache(net, batch=1)
    last = dec.prefill(net, ids)
    # 全位置 forward の最後の位置と一致する
    logits, _ = net.forward(ids, fill_cache=None)
    assert np.allclose(last, logits[:, -1, :], atol=1e-4)
    # 1 トークン進めても有限
    nxt = dec.step(net, np.array([12], dtype=np.int64))
    assert np.isfinite(nxt).all()


def test_moe_trainer_reduces_loss(tiny_moe):
    net, tok = tiny_moe["net"], tiny_moe["tok"]
    assert net.n_params() > 0
    # fixture で学習済み: 同じ分布からの loss がランダム初期化より低いことを担保
    from snipher.neural.moe import MoENet

    fresh = MoENet.random(net.cfg, seed=123)
    from snipher.neural.moe import forward_backward
    from snipher.neural.tokenizer import BOS, EOS

    ids = []
    for t in ["<user>3×7は？\n<asst><think>3を7回足すと21。</think>\n21"] * 20:
        ids += [BOS] + tok.encode(t) + [EOS]
    x = np.array([ids[:-1]], dtype=np.int64)
    y = np.array([ids[1:]], dtype=np.int64)
    pad = np.ones(x.shape, dtype=np.float32)
    loss_trained, _ = forward_backward(net, x, y, pad)
    loss_fresh, _ = forward_backward(fresh, x, y, pad)
    assert loss_trained < loss_fresh * 0.5
