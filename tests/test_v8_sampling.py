"""v8 サンプラ（反復の物理的禁止・確率の波）のテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher.neural.sample import (  # noqa: E402
    SamplingControls,
    apply_biases,
    no_repeat_ban,
    presence_frequency_penalize,
    repetition_penalize,
    sample_token,
    top_k_filter,
    top_p_filter,
)


def test_top_k_keeps_only_k():
    lg = np.array([0.0, 5.0, 3.0, 1.0], dtype=np.float32)
    out = top_k_filter(lg, 2)
    assert (out == -np.inf).sum() == 2
    assert out[1] == 5.0 and out[2] == 3.0


def test_top_p_keeps_high_mass():
    lg = np.array([10.0, 9.0, 0.0, 0.0], dtype=np.float32)
    out = top_p_filter(lg, 0.8)
    assert out[0] == 10.0 and out[1] == 9.0


def test_no_repeat_ban_blocks_completed_ngram():
    # 履歴: a b c d a b c → 4-gram "a b c d" は既出。次が d なら (a,b,c,d) を再形成 → 禁止
    a, b, c, d, e = 10, 11, 12, 13, 14
    history = [a, b, c, d, a, b, c]
    lg = np.zeros(64, dtype=np.float32)
    lg[d] = 100.0            # 反復になる d が最有力
    out = no_repeat_ban(lg, history, ngram=4)
    assert out[d] == -np.inf
    assert out[e] == 0.0     # 新規トークンは禁止されない


def test_no_repeat_ban_allows_first_occurrence():
    history = [10, 11, 12, 13]
    lg = np.zeros(64, dtype=np.float32)
    lg[14] = 50.0
    out = no_repeat_ban(lg, history, ngram=4)
    assert np.isfinite(out[14])


def test_repetition_penalty_reduces_seen():
    lg = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    out = repetition_penalize(lg, [0, 1], 1.5)
    assert out[0] < 2.0 and out[1] < 2.0 and out[2] == 2.0


def test_presence_frequency_penalize():
    lg = np.array([5.0, 5.0], dtype=np.float32)
    out = presence_frequency_penalize(lg, [0, 0, 1], 1.0, 0.5)
    assert out[0] < out[1]   # 2 回出た方がより減衰


def test_apply_biases_boosts_and_bans():
    lg = np.zeros(64, dtype=np.float32)
    out = apply_biases(lg, boost={5: 3.0}, ban=[7, 9])
    assert out[5] == 3.0
    assert out[7] == -np.inf and out[9] == -np.inf


def test_sample_token_respects_ban():
    rng = np.random.default_rng(0)
    lg = np.zeros(64, dtype=np.float32)
    lg[13] = 100.0
    ctrl = SamplingControls(temperature=1.0, top_k=64, ban=[13])
    got = sample_token(lg, ctrl, [10, 11, 12], rng)
    assert got != 13


def test_sample_token_never_loops_4gram():
    """4-gram 禁止を有効にすると、同一 4-gram の周回を延々と出力しない。"""
    rng = np.random.default_rng(0)
    history = []
    ctrl = SamplingControls(temperature=1.0, top_k=16, no_repeat_ngram=4, repetition_penalty=1.0)
    # 語彙 16 個で 400 トークン生成しても、同じ 4-gram が 2 回は出ない
    for _ in range(400):
        lg = np.array([float(i + 1) for i in range(16)], dtype=np.float32)
        t = sample_token(lg, ctrl, history, rng)
        assert t is not None
        history.append(t)
    seen = set()
    for i in range(len(history) - 3):
        seen.add(tuple(history[i:i + 4]))
    assert len(seen) == len(history) - 3  # 全 4-gram が一意


def test_sample_token_returns_none_when_all_banned():
    rng = np.random.default_rng(0)
    lg = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    # 4 語彙しかなく 4-gram が尽きると、全候補が禁止されて None になる
    history = []
    got_none = False
    for _ in range(300):
        t = sample_token(lg, SamplingControls(temperature=1.0, top_k=4, no_repeat_ngram=4),
                         history, rng)
        if t is None:
            got_none = True
            break
        history.append(t)
    assert got_none
