"""Phase 3（DPO / GRPO-lite）のテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _net():
    from snipher.neural.moe import MoEConfig, MoENet

    cfg = MoEConfig(n_vocab=48, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2,
                    n_experts=4, top_k=2, expert_dim=24, max_pos=64)
    return MoENet.random(cfg, seed=5)


def test_logprob_and_grad_matches_finite_difference():
    from snipher.neural.align import logprob, logprob_and_grad

    net = _net()
    ids = [3, 7, 11, 2, 9, 4]
    lp, G = logprob_and_grad(net, ids)
    assert lp == pytest.approx(logprob(net, ids), rel=1e-4)
    # 勾配は順方向の 1 点で中心差分と一致（float32 なので緩め）
    v = net.params["emb"]
    i = (10, 5)
    orig = v[i]
    eps = 1e-3
    v[i] = orig + eps
    lp_p = logprob(net, ids)
    v[i] = orig - eps
    lp_m = logprob(net, ids)
    v[i] = orig
    num = (lp_p - lp_m) / (2 * eps)
    assert num == pytest.approx(G["emb"][i], rel=1e-2, abs=1e-3)


def test_dpo_prefers_think_over_no_think():
    from snipher.neural.align import dpo_step

    net = _net()
    import copy

    ref = copy.deepcopy(net)
    ids_c = [3, 1, 2, 4, 5, 6, 7]     # ダミー系列（chosen）
    ids_r = [3, 1, 2, 4, 8, 9, 10]    # rejected
    losses = []
    for _ in range(6):
        res = dpo_step(net, ref, ids_c, ids_r, beta=0.5)
        losses.append(res["loss"])
        assert np.isfinite(res["grad_norm"])
    # 選好のマージンは学習とともに改善する（loss は下がる傾向）
    assert losses[-1] <= losses[0] + 1e-6


def test_grpo_reward_signals():
    from snipher.neural.align import grpo_reward

    good = grpo_reward("<think>3を7回足すと21。</think>\n21", expected="21")
    assert good["reward"] >= 1.5 and good["checks"]["correct"]
    no_think = grpo_reward("21", expected="21")
    assert no_think["reward"] < good["reward"]
    empty = grpo_reward("", expected="21")
    assert empty["reward"] < 0
    looped = grpo_reward("<think>あ</think>\n" + "ああああああああ" * 5, expected=None)
    assert looped["checks"]["no_loop"] is False


def test_grpo_step_runs_and_updates_params():
    from snipher.neural.align import grpo_step

    net = _net()
    before = net.params["emb"].copy()
    from snipher.neural.tokenizer import CharTokenizer

    tok = CharTokenizer.from_vocab(["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<asst>",
                                    "<sys>", "あ", "い", "う", "え", "お", "か", "き", "く",
                                    "こ", "答", "え"])

    counter = {"n": 0}

    def gen(_net, prompt):
        counter["n"] += 1
        # 良/不良が混ざるように（報酬にばらつきが出る）
        if counter["n"] % 3 == 0:
            return "<think>考える。</think>\n答え"
        return "ええ"

    res = grpo_step(net, ["問1", "問2"], tok=tok, k=3, expected=["答え", "答え"], gen=gen)
    assert np.isfinite(res["mean_reward"])
    assert res["grad_norm"] > 0
    assert not np.array_equal(before, net.params["emb"])
