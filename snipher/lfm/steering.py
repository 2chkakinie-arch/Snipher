"""リアルタイム・モデル・ステアリング AI (ロジット・モジュレーション)。

要件:
    AI が出力中の時でも、プロンプトを AI がノンストップで受け取り、
    プロンプトを確率の波にして生成に干渉させることで出力を止めることなく
    プロンプトを送信できる技術。

実装:
    - 生成ループは 1 トークンごとに `SteeringBus.active_biases()` を参照する
    - 外部から `steer(text)` で投げられた介入プロンプトは、即座に
      トークン化 → 埋め込み → ロジットバイアス へ変換される
    - バイアスは減衰付きで加算され、次のサンプリングに確率の波として乗る
    - Web 検索結果も同様に `inject_evidence(texts)` で流せる
    - バイアスが実際にロジットへ加算された瞬間を **適用イベント** として記録し、
      ストリーム側は `take_events()` で取り出して UI に可視化できる（Agent 化）

これにより、ユーザーが生成中に追加指示を送っても生成を中断せず、
確率分布を滑らかに曲げて出力をステアできる。出力が止まることはない。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SteeringSignal:
    """1 発の介入プロンプト（確率の波）。"""

    id: int
    text: str
    bias: dict[int, float]          # token_id -> logit bias
    kind: str = "prompt"            # prompt / evidence / deliberation
    strength: float = 1.0
    decay: float = 0.92
    ttl: int = 16                   # 何トークン有効か
    created_at: float = field(default_factory=time.time)
    applied_tokens: int = 0         # 実際に何トークンのロジットに乗ったか

    def factor(self, age: int) -> float:
        return (self.decay ** age) * self.strength

    def effective_bias(self, age: int) -> dict[int, float]:
        f = self.factor(age)
        if f < 0.05:
            return {}
        return {tid: v * f for tid, v in self.bias.items()}


class LogitModulator:
    """ロジットに確率の波として干渉するモジュール。

    介入プロンプト → キーワード抽出 → トークン ID → バイアス
    の変換を行い、生成時の logits に加算する。
    """

    def __init__(self, tokenizer=None, vocab_size: int = 640):
        self.tokenizer = tokenizer
        tok_size = 0
        if tokenizer is not None:
            try:
                tok_size = int(tokenizer.size())
            except Exception:  # noqa: BLE001
                tok_size = 0
        self.vocab_size = tok_size or vocab_size

    def text_to_bias(self, text: str, *, strength: float = 2.2) -> dict[int, float]:
        text = str(text or "").strip()
        if not text or self.tokenizer is None:
            return {}
        bias: dict[int, float] = {}
        try:
            # トークナイズして各トークンにバイアスを付与
            ids = self.tokenizer.encode(text)
            for tid in ids:
                if 0 <= tid < self.vocab_size:
                    bias[tid] = bias.get(tid, 0.0) + strength * 0.6
            # 文字レベルでもバイアス (日本語の文字ごと)
            for ch in text:
                if ch.strip():
                    try:
                        cid = self.tokenizer.encode(ch)
                        for tid in cid[:2]:
                            if 0 <= tid < self.vocab_size:
                                bias[tid] = bias.get(tid, 0.0) + strength * 0.3
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            # フォールバック: 文字コードベースの簡易バイアス
            for ch in text[:16]:
                tid = ord(ch) % self.vocab_size
                bias[tid] = bias.get(tid, 0.0) + strength * 0.4
        return bias

    def evidence_to_bias(self, sentences: list[str], *, strength: float = 1.6) -> dict[int, float]:
        bias: dict[int, float] = {}
        for sent in sentences[:4]:
            b = self.text_to_bias(sent, strength=strength * 0.7)
            for tid, v in b.items():
                bias[tid] = bias.get(tid, 0.0) + v * 0.5
        return bias

    def token_text(self, tid: int) -> str:
        """token_id → 文字列 (GGUF など別語彙のバックエンドへの橋渡し用)。"""
        if self.tokenizer is None:
            return ""
        try:
            return str(self.tokenizer.itos.get(tid, ""))
        except Exception:  # noqa: BLE001
            return ""


class SteeringBus:
    """ノンストップで介入を受け付けるバス。

    生成スレッドは `active_biases()` で最新の合成バイアスを取得する。
    UI / API スレッドは `steer(text)` / `inject_evidence(sentences)` で
    いつでも介入できる。生成を止める必要はない。

    適用イベント:
        バイアスが実際に生成へ乗ると `note_applied` が記録され、
        `take_events()` がそれを SSE イベントとして取り出せる。
    """

    def __init__(self, modulator: LogitModulator | None = None, max_events: int = 64):
        self.modulator = modulator or LogitModulator()
        self._lock = threading.RLock()
        self._signals: dict[int, SteeringSignal] = {}
        self._age: dict[int, int] = {}
        self._next_id = 0
        self._events: list[dict] = []
        self._max_events = max_events
        self._applied_total = 0

    # ------------------------------------------------------------------ #
    # 介入の受付（生成中でもいつでも呼べる）
    # ------------------------------------------------------------------ #
    def steer(self, text: str, *, strength: float = 2.2, ttl: int = 16,
              kind: str = "prompt", tokenizer=None) -> int:
        text = str(text or "").strip()
        if not text:
            return -1
        mod = self.modulator
        if tokenizer is not None and mod.tokenizer is None:
            mod.tokenizer = tokenizer
            mod.vocab_size = int(tokenizer.size())
        bias = mod.text_to_bias(text, strength=strength)
        if not bias:
            return -1
        sig = SteeringSignal(id=-1, text=text, bias=bias, kind=kind,
                             strength=strength, ttl=ttl)
        with self._lock:
            sig.id = self._next_id
            self._next_id += 1
            self._signals[sig.id] = sig
            self._age[sig.id] = 0
        return sig.id

    def inject_evidence(self, sentences: list[str], *, strength: float = 1.6) -> int:
        sentences = [s for s in sentences if s and s.strip()][:6]
        if not sentences:
            return -1
        bias = self.modulator.evidence_to_bias(sentences, strength=strength)
        if not bias:
            return -1
        sig = SteeringSignal(id=-1, text=" | ".join(sentences[:2]), bias=bias,
                             kind="evidence", strength=strength, ttl=20)
        with self._lock:
            sig.id = self._next_id
            self._next_id += 1
            self._signals[sig.id] = sig
            self._age[sig.id] = 0
        return sig.id

    # ------------------------------------------------------------------ #
    # 生成ループ側
    # ------------------------------------------------------------------ #
    def active_signals(self) -> list[SteeringSignal]:
        """生きているシグナル（TTL/時間で失効したものを掃除して返す）。"""
        now = time.time()
        with self._lock:
            for sid in list(self._signals.keys()):
                sig = self._signals[sid]
                if self._age.get(sid, 0) >= sig.ttl or now - sig.created_at > 30:
                    self._signals.pop(sid, None)
                    self._age.pop(sid, None)
            return [self._signals[sid] for sid in sorted(self._signals)]

    def active_biases(self) -> dict[int, float]:
        """減衰を適用した合成バイアス（生成ループが毎トークン呼ぶ）。"""
        out: dict[int, float] = {}
        for sig in self.active_signals():
            for tid, v in sig.effective_bias(self._age.get(sig.id, 0)).items():
                out[tid] = out.get(tid, 0.0) + v
        return out

    def advance(self) -> None:
        """生成トークン 1 発ぶん age を進める（生成ループが呼ぶ）。"""
        with self._lock:
            for sid in list(self._age.keys()):
                self._age[sid] += 1

    def apply_to_logits(self, logits: np.ndarray) -> np.ndarray:
        """ロジットへ確率の波を加算し、適用を記録する。"""
        biases = self.active_biases()
        if not biases:
            return logits
        out = logits.astype(np.float32).copy()
        n = out.shape[-1]
        hit = 0
        for tid, v in biases.items():
            if 0 <= tid < n:
                out[tid] += v
                hit += 1
        if hit:
            self.note_applied(n_tokens=hit)
        return out

    def active_text_biases(self) -> dict[str, float]:
        """合成バイアスを「トークン文字列 → 重み」で返す。

        別語彙のバックエンド（GGUF の BPE など）では文字単位の token_id が
        一致しないため、文字列を橋渡しに自前のトークナイザで再エンコードする。
        """
        out: dict[str, float] = {}
        biases = self.active_biases()
        for tid, v in biases.items():
            s = self.modulator.token_text(tid)
            if s:
                out[s] = out.get(s, 0.0) + v
        return out

    def note_applied(self, *, n_tokens: int = 0) -> None:
        """バイアスが発動したことを適用イベントとして記録する。"""
        with self._lock:
            self._applied_total += 1
            sigs = sorted(self._signals.values(), key=lambda s: s.id)
            for sig in sigs:
                sig.applied_tokens += max(1, n_tokens)
            if sigs:
                latest = sigs[-1]
                self._events.append({
                    "type": "steer",
                    "kind": latest.kind,
                    "text": latest.text[:80],
                    "signals": len(sigs),
                    "tokens_biased": n_tokens,
                    "t": time.time(),
                })
                if len(self._events) > self._max_events:
                    del self._events[: len(self._events) - self._max_events]

    # ------------------------------------------------------------------ #
    # 可視化（SSE イベント化）
    # ------------------------------------------------------------------ #
    def take_events(self) -> list[dict]:
        """適用イベントを取り出す。同一シグナルの連打は 1 件に集約する。

        トークンごとに適用を記録すると UI が埋まるので、(kind, text) 単位で
        合算し「何トークンに干渉したか」を 1 行にまとめて返す。
        """
        with self._lock:
            ev, self._events = self._events, []
        merged: dict[tuple, dict] = {}
        order: list[tuple] = []
        for e in ev:
            key = (e.get("kind"), e.get("text"))
            if key in merged:
                merged[key]["tokens_biased"] += int(e.get("tokens_biased") or 0)
                merged[key]["applies"] += 1
            else:
                merged[key] = {**e, "applies": 1}
                order.append(key)
        return [merged[k] for k in order]

    def snapshot_new(self, after_id: int) -> list[SteeringSignal]:
        """after_id より後に届いたシグナル（リモート転送などに使う）。"""
        with self._lock:
            return [s for sid, s in sorted(self._signals.items()) if sid > after_id]

    def last_id(self) -> int:
        with self._lock:
            return self._next_id - 1

    def clear(self):
        with self._lock:
            self._signals.clear()
            self._age.clear()
            self._events.clear()
            self._applied_total = 0

    def status(self) -> dict:
        with self._lock:
            return {
                "pending": len(self._signals),
                "applied_total": self._applied_total,
                "signals": [{"id": s.id, "text": s.text[:48], "kind": s.kind,
                             "ttl": s.ttl, "strength": s.strength,
                             "applied_tokens": s.applied_tokens}
                            for s in sorted(self._signals.values(), key=lambda s: s.id)[-4:]],
            }


# シングルトン (プロセス内で 1 つのバスを共有)
_GLOBAL_BUS: SteeringBus | None = None
_BUS_LOCK = threading.Lock()


def get_steering_bus(tokenizer=None) -> SteeringBus:
    global _GLOBAL_BUS
    with _BUS_LOCK:
        if _GLOBAL_BUS is None:
            mod = LogitModulator(tokenizer=tokenizer)
            _GLOBAL_BUS = SteeringBus(mod)
        elif tokenizer is not None and _GLOBAL_BUS.modulator.tokenizer is None:
            _GLOBAL_BUS.modulator.tokenizer = tokenizer
            _GLOBAL_BUS.modulator.vocab_size = int(tokenizer.size())
        return _GLOBAL_BUS


def reset_steering_bus() -> None:
    """テスト用のリセット（シングルトンを破棄する）。"""
    global _GLOBAL_BUS
    with _BUS_LOCK:
        _GLOBAL_BUS = None
