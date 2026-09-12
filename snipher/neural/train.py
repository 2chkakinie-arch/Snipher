"""内蔵ニューラルコアの訓練（蒸留）ループ。numpy のみ。

    python tools/distill_neural.py          # 本番スナップショットを再生成
    python -c "from snipher.neural.train import ..."  # テストから小さな学習も可

教師データは語彙テーブルと kb.json から自動生成するため、外部データも
ダウンロードも不要です（`= ユーザーが何もアップロードしなくていい`）。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .nn import NNConfig, MicroNet, clip_grads, forward_backward


@dataclass
class TrainConfig:
    seq_len: int = 64
    batch: int = 96
    epochs: int = 8
    lr: float = 1.2e-3
    min_lr: float = 8e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    seed: int = 7
    eval_batches: int = 24
    log_every: int = 40


class TextDataset:
    """トークン列を 1 本に繋いで、ランダム窓を返す小さなデータセット。"""

    def __init__(self, ids: np.ndarray, seq_len: int, rng: np.random.Generator):
        self.ids = ids.astype(np.int64, copy=False)
        self.seq_len = seq_len
        self.rng = rng

    def batch(self, size: int, rng: np.random.Generator | None = None):
        rng = rng or self.rng
        T = self.seq_len
        n = len(self.ids) - T - 1
        if n <= 1:
            raise ValueError("コーパスが短すぎます")
        starts = rng.integers(0, n, size=size)
        idx = starts[:, None] + np.arange(T + 1)[None, :]
        window = self.ids[idx]
        x, y = window[:, :-1], window[:, 1:]
        return x, y

    def fixed_batch(self, size: int, seed: int = 99):
        rng = np.random.default_rng(seed)
        return self.batch(size, rng)


class Trainer:
    def __init__(self, net: MicroNet, train_ds: TextDataset, val_ds: TextDataset | None = None,
                 cfg: TrainConfig | None = None):
        self.net = net
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.cfg = cfg or TrainConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.step = 0
        self.history: list[dict] = []
        self._adam = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in net.params.items()}

    # ------------------------------------------------------------------ #
    def _lr(self) -> float:
        c = self.cfg
        total = max(1, self.total_steps)
        warm = max(1, int(total * c.warmup_ratio))
        if self.step < warm:
            return c.lr * (self.step + 1) / warm
        prog = (self.step - warm) / max(1, total - warm)
        return c.min_lr + 0.5 * (c.lr - c.min_lr) * (1 + math.cos(math.pi * min(1.0, prog)))

    def _apply(self, G: dict) -> None:
        c = self.cfg
        b1, b2, eps = c.beta1, c.beta2, c.eps
        t = self.step + 1
        lr = self._lr()
        P = self.net.params
        for k, g in G.items():
            m, v = self._adam[k]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * (g * g)
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            upd = lr * mh / (np.sqrt(vh) + eps)
            if c.weight_decay:
                upd = upd + lr * c.weight_decay * P[k]
            P[k] -= upd.astype(P[k].dtype)

    def _eval(self, ds: TextDataset, batches: int) -> dict:
        x, y = ds.fixed_batch(self.cfg.batch)
        pad = np.ones(x.shape, dtype=np.float32)
        loss, _ = forward_backward(self.net, x, y, pad)
        logits, _ = self.net.forward(x, pad=pad)
        acc = float((logits.argmax(-1) == y).astype(np.float32).mean())
        return {"loss": loss, "acc": acc, "ppl": float(math.exp(min(20.0, loss)))}

    # ------------------------------------------------------------------ #
    def train(self, on_log=None, on_epoch=None) -> dict:
        """学習ループ。`on_epoch(net, metrics)` は各 epoch の検証後に呼ばれる
        （重い学習でも途中のスナップショットを保存できるようにするため）。"""
        c = self.cfg
        steps_per_epoch = max(1, len(self.train_ds.ids) // (c.batch * c.seq_len))
        self.total_steps = steps_per_epoch * c.epochs
        best = {"loss": float("inf")}
        t0 = time.time()
        for ep in range(c.epochs):
            for _ in range(steps_per_epoch):
                x, y = self.train_ds.batch(c.batch)
                pad = np.ones(x.shape, dtype=np.float32)
                loss, G = forward_backward(self.net, x, y, pad)
                clip_grads(G, c.grad_clip)
                self.step += 1
                self._apply(G)
                if c.log_every and self.step % c.log_every == 0:
                    cur = {"loss": round(loss, 4), "ppl": round(math.exp(min(20.0, loss)), 2),
                           "step": self.step, "lr": round(self._lr(), 6),
                           "sec": round(time.time() - t0, 1)}
                    self.history.append(cur)
                    if on_log:
                        on_log("train", cur)
            m = self._eval(self.val_ds or self.train_ds, c.eval_batches)
            m["epoch"] = ep + 1
            m["sec"] = round(time.time() - t0, 1)
            self.history.append({"eval": m})
            if on_log:
                on_log("eval", m)
            if m["loss"] < best["loss"]:
                best = dict(m)
            if on_epoch:
                try:
                    on_epoch(self.net, m)
                except Exception:  # noqa: BLE001 - 保存失敗で学習は止めない
                    pass
        return {"best_val": best, "final": self.history[-1] if self.history else {},
                "steps": self.step, "seconds": round(time.time() - t0, 1)}
