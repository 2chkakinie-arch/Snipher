"""対話整列（Phase 3）: DPO と GRPO-lite の numpy 実装。

ユーザー設計図の「褒める / 怒る」強化学習を、外部ライブラリ無しで行います。

* ``dpo_step`` … 選好ペア (chosen, rejected) から、参照モデルとの差で
  「<think> で整理して答えた回答」を直接確率調整する（Direct Preference Optimization）。
* ``grpo_step`` … 同一プロンプトから K 本を生成し、グループ相対アドバンテージ
  （ルール報酬: フォーマット / 正解 / 反復なし）で方策勾配を積む（GRPO-lite）。

勾配は ``moe.forward_backward``（検証済みの解析勾配）の線形結合なので、
float64 の中心差分と一致する。小規模なら CPU だけで回ります。
"""

from __future__ import annotations

import math

import numpy as np

from .moe import MoENet, clip_grads, forward_backward


# ---------------------------------------------------------------------- #
# 対数尤度とその勾配
# ---------------------------------------------------------------------- #
def logprob_and_grad(net: MoENet, ids: list[int]) -> tuple[float, dict[str, np.ndarray]]:
    """系列 ids の総対数尤度 Σ_t log p(ids[t+1] | ids[:t+1]) とその勾配。"""
    x = np.array([ids[:-1]], dtype=np.int64)
    y = np.array([ids[1:]], dtype=np.int64)
    pad = np.ones(x.shape, dtype=np.float32)
    loss, G = forward_backward(net, x, y, pad)     # loss = 平均 NLL
    n = int(x.shape[1])
    lp = -loss * n
    G_lp = {k: -n * v for k, v in G.items()}
    return lp, G_lp


def logprob(net: MoENet, ids: list[int]) -> float:
    """勾配不要の対数尤度（参照モデル用）。"""
    x = np.array([ids[:-1]], dtype=np.int64)
    y = np.array([ids[1:]], dtype=np.int64)
    logits, _ = net.forward(x)
    lg = logits[0].astype(np.float64)
    lg = lg - lg.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return float(logp[np.arange(y.shape[1]), y[0]].sum())


# ---------------------------------------------------------------------- #
# 簡易 Adam
# ---------------------------------------------------------------------- #
class _Adam:
    def __init__(self, net: MoENet, lr: float = 3e-4, beta1: float = 0.9,
                 beta2: float = 0.98, eps: float = 1e-8, wd: float = 0.0):
        self.net = net
        self.lr = lr
        self.b1, self.b2, self.eps, self.wd = beta1, beta2, eps, wd
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in net.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in net.params.items()}

    def step(self, G: dict[str, np.ndarray]) -> None:
        self.t += 1
        P = self.net.params
        for k, g in G.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            upd = self.lr * mh / (np.sqrt(vh) + self.eps)
            if self.wd:
                upd = upd + self.lr * self.wd * P[k]
            P[k] -= upd.astype(P[k].dtype)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


# ---------------------------------------------------------------------- #
# DPO
# ---------------------------------------------------------------------- #
def dpo_step(net: MoENet, ref: MoENet, ids_chosen: list[int], ids_rejected: list[int],
             beta: float = 0.2) -> dict:
    """1 ペアぶんの DPO 更新。loss と勾配ノルムを返す。

    L = -log σ( β · (logπ(y_w) - logπ_ref(y_w) - logπ(y_l) + logπ_ref(y_l)) )
    """
    lp_c, G_c = logprob_and_grad(net, ids_chosen)
    lp_r, G_r = logprob_and_grad(net, ids_rejected)
    ref_c = logprob(ref, ids_chosen)
    ref_r = logprob(ref, ids_rejected)
    margin = (lp_c - ref_c) - (lp_r - ref_r)
    loss = float(-math.log(max(1e-9, float(_sigmoid(np.float64(beta * margin))))))
    w = float(-beta * _sigmoid(np.float64(-beta * margin)))
    G = {k: w * (G_c.get(k, 0.0) - G_r.get(k, 0.0)) for k in net.params}
    norm = clip_grads(G, 1.0)
    return {"loss": loss, "margin": float(margin), "weight": w, "grad_norm": norm,
            "grad": G}


# ---------------------------------------------------------------------- #
# GRPO-lite（グループ相対報酬による方策勾配）
# ---------------------------------------------------------------------- #
def grpo_reward(completion: str, expected: str | None = None,
                require_think: bool = True) -> dict:
    """ルール報酬。報酬は -1..+2 の範囲で、明確な信号だけを与える。

    * フォーマット … <think>…</think> を閉じている: +0.5
    * 正解 … 期待文字列と一致（算数など検算できるタスク）: +1.0
    * 反復なし … 4-gram の周回が無い: +0.5
    * 空 / 思考マーカーだけ: -1.0
    """
    reward = 0.0
    checks = {}
    if not completion or completion.strip() in ("<think>", "</think>", "<think></think>"):
        return {"reward": -1.0, "checks": {"empty": True}}
    if require_think:
        ok = "<think>" in completion and "</think>" in completion
        checks["think_closed"] = bool(ok)
        if ok:
            reward += 0.5
    answer = completion.split("</think>", 1)[-1] if "</think>" in completion else completion
    if expected is not None:
        checks["correct"] = bool(expected in answer)
        if expected in answer:
            reward += 1.0
    # 4-gram 周回の検出
    n = 4
    ids = list(answer)
    looped = any(
        answer.count(answer[i:i + n]) >= 2
        for i in range(len(answer) - n + 1) if n >= 3
    ) if len(answer) >= 2 * n else False
    checks["no_loop"] = not looped
    if not looped:
        reward += 0.5
    return {"reward": reward, "checks": checks}


def grpo_step(net: MoENet, prompts: list[str], *, tok, k: int = 4, lr: float = 2e-4,
              beta_kl: float = 0.01, ref: MoENet | None = None,
              expected: list[str | None] | None = None,
              gen) -> dict:
    """1 ステップの GRPO。``gen`` は (net, prompt) → completion 文字列 を返す関数。

    A_i = (r_i - mean(r)) / (std(r) + 1e-6)、KL 正則化は参照モデルとの差で軽く入れる。
    """
    expected = expected or [None] * len(prompts)
    G = {key: np.zeros_like(v) for key, v in net.params.items()}
    rewards: list[float] = []
    losses: list[float] = []
    for prompt, exp in zip(prompts, expected):
        group_rewards: list[float] = []
        group: list[str] = []
        for _ in range(k):
            comp = gen(net, prompt)
            group.append(comp)
            group_rewards.append(grpo_reward(comp, exp)["reward"])
        mean_r = float(np.mean(group_rewards))
        std_r = float(np.std(group_rewards)) + 1e-6
        for comp, r in zip(group, group_rewards):
            ids = completion_ids(net, tok, prompt, comp)
            if len(ids) < 2:
                continue
            adv = (r - mean_r) / std_r
            _, G_lp = logprob_and_grad(net, ids)
            for key in G:
                G[key] = G[key] + (adv / (k or 1)) * G_lp[key]
            losses.append(-adv * math.log(max(1e-9, 1.0)))  # 監視用
        rewards.extend(group_rewards)
    norm = clip_grads(G, 1.0)
    # KL 正則化（参照モデルから離れすぎない）
    if ref is not None and beta_kl:
        for key in net.params:
            G[key] = G[key] + beta_kl * (ref.params[key] - net.params[key])
    opt = _Adam(net, lr=lr)
    opt.step(G)
    return {"mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "grad_norm": norm}


def completion_ids(net: MoENet, tok, prompt: str, completion: str) -> list[int]:
    from .tokenizer import BOS

    return [BOS] + tok.encode(prompt) + tok.encode(completion)
