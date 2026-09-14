"""MoECore — MoENet（v8 コア）を生成・評価するラッパ。

``DistilledCore``（既存の内蔵コア）と対になる v8 版。numpy のみで動き、
* KV キャッシュ + Top-K ルーティングによる逐次デコード
* ``SamplingControls``（反復封印・プレゼンス/頻度・top-k/top-p）を使った生成
* テキストの対数尤度スコア（校正モデルC の検証に使う）
を提供する。重みは ``store.py`` の ``kind: moe`` 形式で ``data/neural/v8*.npz`` に置く。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .moe import MoEConfig, MoEDecodeCache, MoENet
from .sample import SamplingControls, sample_token
from .tokenizer import BOS, EOS, CharTokenizer, PAD, UNK

DEFAULT_PATH = "snipher/data/neural/v8.npz"


@dataclass
class MoECore:
    """v8 の思考コア。``net`` + ``tok``、または ``path``（npz）から作る。"""

    net: MoENet | None = None
    tok: CharTokenizer | None = None
    path: str | Path | None = None

    def __post_init__(self) -> None:
        if self.net is None and self.path is not None:
            from .store import load

            net, vocab, _extra = load(self.path)
            if not isinstance(net, MoENet):
                raise TypeError(f"{self.path} は MoENet ではありません ({type(net).__name__})")
            self.net = net
            self.tok = CharTokenizer.from_vocab(vocab)

    @property
    def is_ready(self) -> bool:
        return self.net is not None and self.tok is not None

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "MoECore":
        from .store import load

        net, vocab, _extra = load(path)
        if not isinstance(net, MoENet):
            raise TypeError(f"{path} は MoENet ではありません ({type(net).__name__})")
        return cls(net=net, tok=CharTokenizer.from_vocab(vocab))

    # ------------------------------------------------------------------ #
    def decode(self, ids: list[int]) -> str:
        """特殊トークンを除いて文字に戻す。"""
        skip = {PAD, UNK, BOS, EOS}
        return "".join(self.tok.itos[i] for i in ids if i not in skip and i < len(self.tok.itos))

    def _prompt_ids(self, prompt: str | list[int] | None, with_bos: bool = True) -> list[int]:
        if prompt is None:
            return [BOS] if with_bos else []
        if isinstance(prompt, str):
            ids = self.tok.encode(prompt)
        else:
            ids = [int(i) for i in prompt]
        if with_bos and (not ids or ids[0] != BOS):
            ids = [BOS] + ids
        return ids

    # ------------------------------------------------------------------ #
    _SENT_END = "。！？!?"

    def generate_ids(self, prompt: str | list[int] | None, *,
                     max_new: int = 256, controls: SamplingControls | None = None,
                     forbid: tuple[int, ...] = (PAD, UNK), stop_ids: tuple[int, ...] = (EOS,),
                     stop_text: str | None = None, seed: int = 0,
                     ctx: int = 256, stop_on_sentence: bool = True,
                     history: list[int] | None = None) -> list[int]:
        """生成トークン列（プロンプトを除く）を返す。

        ``stop_text`` を渡すと、デコード済み末尾にその文字列が現れたら停止する
        （``</think>`` で思考区間を閉じるのに使う）。``stop_on_sentence=False``
        なら句点では止めない（思考・長文生成用）。``history`` は反復封印・
        反復ペナルティの既出トークン列（チャンクを跨いだ周回も封じる）。
        """
        ctrl = controls or SamplingControls()
        rng = np.random.default_rng(seed)
        ids = self._prompt_ids(prompt)
        cache = MoEDecodeCache(self.net, batch=1)
        if len(ids) > ctx:
            ids = [BOS] + ids[-ctx + 1:]
        if len(ids) > 1:
            logits = cache.prefill(self.net, ids)
        else:
            logits = np.zeros((1, self.net.cfg.n_vocab), dtype=np.float32)
            logits[0, BOS] = 1.0
        V = self.net.cfg.n_vocab
        generated: list[int] = []
        hist = list(history or [])
        tail: str = ""
        stop_len = (len(stop_text) if stop_text else 0) * 2 + 4
        for _ in range(max_new):
            tok = sample_token(logits[0], ctrl, hist, rng)
            if tok is None or tok >= V:
                break
            if tok == EOS or tok in forbid or (stop_ids and tok in stop_ids):
                break
            generated.append(tok)
            hist.append(tok)
            tail += self.decode([tok])
            if stop_len:
                tail = tail[-stop_len:]
            if stop_text and stop_text in tail:
                break
            if stop_on_sentence and any(s in tail for s in self._SENT_END):
                break
            logits = cache.step(self.net, np.asarray([tok], dtype=np.int64))
        return generated

    def generate(self, prompt: str | list[int] | None = None, *,
                 max_new: int = 256, max_chars: int | None = None,
                 controls: SamplingControls | None = None,
                 temperature: float | None = None,
                 forbid: tuple[int, ...] = (PAD, UNK), stop_text: str | None = None,
                 seed: int = 0, ctx: int = 256, strip_think: bool = True) -> str:
        """生成トークン（プロンプトの続き）を文字列として返す。

        ``max_chars`` は ``max_new`` の別名。``temperature`` を渡すと簡易の
        ``SamplingControls`` を組む（``controls`` と同時指定は不可）。
        """
        if controls is None and temperature is not None:
            controls = SamplingControls(temperature=temperature)
        generated = self.generate_ids(prompt, max_new=max_chars or max_new,
                                      controls=controls, forbid=forbid,
                                      stop_text=stop_text, seed=seed, ctx=ctx)
        text = self.decode(generated)
        if strip_think:
            text = text.replace("<think>", "").replace("</think>", "")
        return text.strip()

    # ------------------------------------------------------------------ #
    def score(self, text: str, prompt: str | None = None) -> dict:
        """テキストの対数尤度とパープレキシティ（モデルC の検証に使う）。"""
        if not text:
            return {"mean_logprob": -1e9, "ppl": 1e9, "chars": 0}
        ids = self._prompt_ids(prompt)
        ids += self.tok.encode(text)
        pad = np.ones((1, len(ids)), dtype=np.float32)
        x = np.asarray([ids], dtype=np.int64)
        logits, _ = self.net.forward(x, pad=pad)
        tgt = np.asarray(ids[1:], dtype=np.int64)[None, :]
        lg = logits[:, :-1]
        e = np.exp(lg - lg.max(-1, keepdims=True))
        p = e / e.sum(-1, keepdims=True)
        lp = np.log(np.clip(np.take_along_axis(p, tgt[:, :, None], axis=-1)[:, :, 0], 1e-9, None))
        mean_lp = float(lp.mean())
        return {"mean_logprob": mean_lp, "ppl": float(np.exp(-mean_lp)),
                "chars": len(text)}

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        cfg = self.net.cfg
        return {
            "kind": "moe",
            "params": self.net.n_params(),
            "vocab": cfg.n_vocab,
            "d_model": cfg.d_model,
            "layers": cfg.n_layers,
            "blocks": list(cfg.blocks),
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "n_experts": cfg.n_experts,
            "top_k": cfg.top_k,
            "expert_dim": cfg.expert_dim,
            "max_pos": cfg.max_pos,
            "rope": True,
            "rmsnorm": True,
        }
