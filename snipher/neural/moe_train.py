"""MoE コアの訓練ループ（numpy のみ）。train.py と同じ契約・同じ流儀。"""

from __future__ import annotations

import math
import time

import numpy as np

from .moe import MoENet, clip_grads, forward_backward
from .train import TextDataset, TrainConfig


class MoETrainer:
    def __init__(self, net: MoENet, train_ds: TextDataset, val_ds: TextDataset | None = None,
                 cfg: TrainConfig | None = None):
        self.net = net
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.cfg = cfg or TrainConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.step = 0
        self.history: list[dict] = []
        self._adam = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in net.params.items()}

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

    def train(self, on_log=None, on_epoch=None) -> dict:
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
                except Exception:  # noqa: BLE001
                    pass
        return {"best_val": best, "final": self.history[-1] if self.history else {},
                "steps": self.step, "seconds": round(time.time() - t0, 1)}
