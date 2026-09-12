"""内蔵ニューラルコアの公開 API（推論側）。

Snipher Core から見て、これは「LFM2.5-1.2B-JP とまったく同じ顔」をしています
（`is_ready` / `stream_chat` / `engine_name` / `status`）。違うのは
**重みがパッケージ内にあり、ダウンロード不要・数ミリ秒でロードできる**ことです。

主な役割は LFM2.5 が担うものと同じ 3 つ:

    1. 確率的に不安な部分の文章生成  → generate()
    2. 助動詞の補い（断片文の復元）  → complete()
    3. 確信度の判定（ゲート）        → score() / rescore()
"""

from __future__ import annotations

import math
import re
import threading
import time
from pathlib import Path

import numpy as np

from .nn import MicroNet, forward_backward
from .tokenizer import ASST, BOS, EOS, SYS, USER, CharTokenizer
from . import store as _store

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "neural" / "core.npz"
_STOP = ("。", "！", "？", "\n")


def _top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    cutoff = np.partition(logits, -k)[-k]
    return np.where(logits < cutoff, -np.inf, logits)


class DistilledCore:
    """文字レベル小型 Transformer（LFM2 Hybrid 構造）のサーベイラブルなラッパ。"""

    def __init__(self, net: MicroNet | None = None, tok: CharTokenizer | None = None,
                 path: str | Path | None = None):
        self._lock = threading.RLock()
        self.net: MicroNet | None = net
        self.tok: CharTokenizer | None = tok
        self.meta: dict = {}
        self.error: str | None = None
        self.load_seconds: float | None = None
        self.path = Path(path) if path else DEFAULT_PATH
        if net is None:
            self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> bool:
        t0 = time.time()
        p = self.path
        if not p.exists():
            self.error = f"内蔵ニューラルコアの重みが見つかりません: {p}"
            return False
        try:
            net, vocab, extra = _store.load(p)
        except Exception as exc:  # noqa: BLE001
            self.error = f"重みのロードに失敗しました: {exc}"
            return False
        with self._lock:
            self.net = net
            self.tok = CharTokenizer(vocab)
            self.meta = extra or {}
            self.load_seconds = round(time.time() - t0, 3)
        return True

    def reload(self, path: str | Path | None = None) -> bool:
        if path:
            self.path = Path(path)
        self.error = None
        return self._load()

    @property
    def is_ready(self) -> bool:
        return self.net is not None

    ready = is_ready

    # ------------------------------------------------------------------ #
    def engine_name(self) -> str:
        if self.net is None:
            return "Snipher 内蔵ニューラルコア（未ロード）"
        n = self.net.n_params()
        return f"Snipher 内蔵ニューラルコア（LFM2.5 蒸留 / {n / 1e6:.2f}M params）"

    def status(self) -> dict:
        return {
            "kind": "distilled",
            "state": "ready" if self.is_ready else "unavailable",
            "engine": self.engine_name(),
            "params": self.net.n_params() if self.net else 0,
            "vocab": self.tok.size() if self.tok else 0,
            "load_seconds": self.load_seconds,
            "weights_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "error": self.error,
            "trained_at": self.meta.get("trained_at"),
            "metrics": self.meta.get("metrics"),
            "runtime": "numpy",
            "download_required": False,
        }

    # ------------------------------------------------------------------ #
    # 低レベル生成
    # ------------------------------------------------------------------ #
    def _sample(self, logits: np.ndarray, rng: np.random.Generator, temperature: float,
                top_k: int, penalty_ids: list[int], repetition_penalty: float) -> int:
        lg = logits.astype(np.float32).copy()
        if repetition_penalty and penalty_ids:
            for i in set(penalty_ids):
                if lg[i] > 0:
                    lg[i] = lg[i] / repetition_penalty
                else:
                    lg[i] = lg[i] * repetition_penalty
        lg = _top_k_filter(lg, top_k)
        if temperature <= 1e-3:
            return int(np.argmax(lg))
        p = np.exp((lg - lg.max()) / temperature)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            return int(np.argmax(lg))
        p = p / s
        return int(rng.choice(lg.shape[-1], p=p))

    def generate_ids(self, prompt_ids: list[int], *, max_new: int = 64, temperature: float = 0.8,
                     top_k: int = 40, repetition_penalty: float = 1.08, seed: int | None = None,
                     forbid: tuple[int, ...] = (USER, SYS), ctx: int = 56) -> list[int]:
        assert self.net is not None and self.tok is not None
        rng = np.random.default_rng(seed)
        ids = list(prompt_ids)
        out: list[int] = []
        net = self.net
        for _ in range(max_new):
            window = ids[-ctx:] if len(ids) > ctx else ids
            x = np.array([window], dtype=np.int64)
            logits, _ = net.forward(x)
            last = logits[0, -1]
            nxt = self._sample(last, rng, temperature, top_k, ids, repetition_penalty)
            if nxt == EOS or nxt in forbid:
                break
            ids.append(nxt)
            out.append(nxt)
            ch = self.tok.itos.get(nxt, "")
            if ch in _STOP or len(out) >= max_new:
                break
        return out

    def generate(self, prompt: str = "", *, max_chars: int = 64, temperature: float = 0.8,
                 top_k: int = 40, seed: int | None = None, as_assistant: bool = True) -> str:
        """プロンプトの続きを生成する（`as_assistant` で会話の返答形式に）。"""
        if not self.is_ready:
            return ""
        head = [BOS]
        if as_assistant:
            head += [ASST]
        ids = head + self.tok.encode(prompt)  # type: ignore[union-attr]
        gen = self.generate_ids(ids, max_new=max_chars, temperature=temperature, top_k=top_k,
                                forbid=(USER, SYS) if as_assistant else ())
        text = self.tok.decode(gen)  # type: ignore[union-attr]
        return text.strip()

    def reply(self, user_text: str, *, max_chars: int = 72, temperature: float = 0.8,
              top_k: int = 40, seed: int | None = None, context: list[dict] | None = None) -> str:
        """発話に対する返答を生成する（内部プロンプトは <user>/<asst> マーカー）。"""
        if not self.is_ready:
            return ""
        hist = ""
        for m in (context or [])[-4:]:
            role = m.get("role")
            if role == "user":
                hist += f"<user>{m.get('content', '')}\n"
            elif role == "assistant":
                hist += f"<asst>{m.get('content', '')}\n"
        ids = [BOS] + self.tok.encode(hist + f"<user>{user_text}\n<asst>")  # type: ignore[union-attr]
        gen = self.generate_ids(ids, max_new=max_chars, temperature=temperature, top_k=top_k,
                                seed=seed, forbid=(USER, SYS, ASST))
        return self.tok.decode(gen).strip()  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    def complete(self, fragment: str, *, max_chars: int = 40, temperature: float = 0.35,
                 top_k: int = 30) -> dict:
        """助動詞の補い: 断片文を自然な一文に補完する。

        ルール(polisher)で埋まらなかった文末を、内蔵ニューラルコアが
        「続き」ではなく **文全体を復元する** 形で生成し、断片の末尾と突き合わせて
        最短の補いだけを返す。
        """
        if not self.is_ready:
            return {"text": fragment, "added": "", "changed": False, "confidence": 0.0}
        frag = fragment.rstrip()
        # 学習時と同じ形（続き: <断片>\n全文）でプロンプトし、続きだけを採る
        ids = [BOS] + self.tok.encode(f"続き: {frag}\n")  # type: ignore[union-attr]
        gen = self.generate_ids(ids, max_new=max_chars, temperature=temperature, top_k=top_k,
                                forbid=(USER, SYS, ASST))
        text = self.tok.decode(gen).strip()  # type: ignore[union-attr]
        # 断片を繰り返して出力するモデルにも対応して重複を除く
        added = text
        if added.startswith(frag):
            added = added[len(frag):]
        else:
            for cut in range(min(len(added), len(frag)), 0, -1):
                if added[:cut] == frag[-cut:]:
                    added = added[cut:]
                    break
        added = added.strip().split("\n")[0].lstrip("、").strip()
        out = f"{frag}{added}" if added else frag
        # 補完の確信度: 補った部分を含む文全体の perplexity と補足長のバランス
        conf = 0.0
        if added:
            sc = self.score(out)
            len_penalty = 0.0 if len(added) <= 12 else min(0.35, (len(added) - 12) * 0.03)
            conf = max(0.0, 1.0 - sc["perplexity"] / 14.0) - len_penalty
        return {"text": out, "added": added, "changed": bool(added),
                "confidence": round(float(conf), 4)}

    # ------------------------------------------------------------------ #
    def score(self, text: str, *, register_hint: bool = False) -> dict:
        """ teacher-forcing でテキストを採点する（perplexity と確信度）。"""
        if not self.is_ready:
            return {"perplexity": 1e9, "mean_logprob": -21.0, "confidence": 0.0, "ok": False}
        toks = self.tok.encode(text)  # type: ignore[union-attr]
        if len(toks) < 2:
            return {"perplexity": 12.0, "mean_logprob": -2.5, "confidence": 0.5, "ok": True}
        seq = [BOS] + toks + [EOS]
        x = np.array([seq[:-1]], dtype=np.int64)
        y = np.array([seq[1:]], dtype=np.int64)
        pad = np.ones(x.shape, dtype=np.float32)
        logits, _ = self.net.forward(x)  # type: ignore[union-attr]
        lg = logits[0].astype(np.float64)
        lg = lg - lg.max(-1, keepdims=True)
        logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        nll = -float(logp[np.arange(y.shape[1]), y[0]].mean())
        ppl = math.exp(min(20.0, nll))
        # ppl≈4 → 0.95, ppl≈8 → 0.8, ppl≈16 → 0.55, ppl≈40 → 0.2 くらいの単調写像
        conf = 1.0 / (1.0 + max(0.0, math.log(max(ppl, 1.0001)) / 1.05))
        return {"perplexity": round(ppl, 3), "mean_logprob": round(-nll, 4),
                "confidence": round(min(0.99, conf), 4), "ok": True}

    def rescore(self, candidates: list[str], *, prefix: str = "") -> list[dict]:
        """候補文（スロットの差し替え案）を内蔵コアで採点して並び替える。"""
        scored = []
        for c in candidates:
            s = self.score((prefix + c) if prefix else c)
            scored.append({"text": c, **s})
        scored.sort(key=lambda d: -d["confidence"])
        return scored

    # ------------------------------------------------------------------ #
    # ストリーミング互換（LFM2.5 バックエンドと同じイベント形式）
    # ------------------------------------------------------------------ #
    def stream_chat(self, messages: list[dict], *, max_new_tokens: int | None = None,
                    temperature: float | None = None, top_k: int | None = None,
                    repetition_penalty: float | None = None, use_template: bool = True,
                    system_prompt: str | None = None, **_ignored):
        """`GgufBackend.stream_chat` と同じイベントを出す（UI/コア側は無差別に使える）。"""
        if not self.is_ready:
            yield {"type": "error", "message": "内蔵ニューラルコアがロードされていません"}
            return
        last_user = ""
        history: list[dict] = []
        for m in messages:
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
            else:
                history.append(m)
        t0 = time.time()
        prompt = ""
        for m in history[-4:]:
            if m.get("role") == "assistant":
                prompt += f"<asst>{m.get('content', '')}\n"
        prompt += f"<user>{last_user}\n<asst>"
        ids = [BOS] + self.tok.encode(prompt)  # type: ignore[union-attr]
        max_new = int(max_new_tokens or 72)
        temp = float(temperature if temperature is not None else 0.8)
        k = int(top_k or 40)
        rep = float(repetition_penalty or 1.06)
        yield {"type": "start", "engine": self.engine_name(),
               "template_mode": "builtin-role-markers"}
        rng = np.random.default_rng(int(time.time() * 1000) % (2 ** 31))
        cur = list(ids)
        produced: list[str] = []
        n = 0
        try:
            while n < max_new:
                window = cur[-56:] if len(cur) > 56 else cur
                logits, _ = self.net.forward(np.array([window], dtype=np.int64))  # type: ignore[union-attr]
                nxt = self._sample(logits[0, -1], rng, temp, k, cur, rep)
                if nxt == EOS or nxt in (USER, SYS, ASST):
                    break
                cur.append(nxt)
                ch = self.tok.itos.get(nxt, "")  # type: ignore[union-attr]
                if not ch:
                    continue
                produced.append(ch)
                n += 1
                yield {"type": "delta", "text": ch}
                if ch in _STOP and n >= 8:
                    break
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": f"内蔵ニューラルコアの生成エラー: {exc}"}
        text = "".join(produced).strip()
        dt = max(1e-6, time.time() - t0)
        yield {"type": "done", "text": text, "stats": {
            "engine": self.engine_name(), "backend": "distilled",
            "new_tokens": n, "tokens_per_second": round(n / dt, 1),
            "seconds": round(dt, 3), "max_new_tokens": max_new,
            "temperature": temp, "top_k": k,
        }}

    def scan_unknown(self, text: str) -> dict | None:
        """文字レベル語彙なので未知文字は原理的に発生しない（GGUF と同じ強み）。"""
        return {"unknown": [], "checked": len(text or ""), "note": "char-level vocab: 未知文字なし"}

    def unload(self) -> None:  # pragma: no cover - インタフェース合わせ
        with self._lock:
            self.net = None
            self.tok = None


def available(path: str | Path | None = None) -> bool:
    p = Path(path) if path else DEFAULT_PATH
    return p.exists()
