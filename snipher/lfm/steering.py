"""リアルタイム・モデル・ステアリング AI (ロジット・モジュレーション)。

要件:
    AI が出力中の時でも、プロンプトを AI がノンストップで受け取り、
    プロンプトを確率の波にして生成に干渉させることで出力を止めることなく
    プロンプトを送信できる技術。

実装:
    - 生成ループは 1 トークンごとに `SteeringBus` をポーリングする
    - 外部から `steer(text)` で投げられた介入プロンプトは、即座に
      トークン化 → 埋め込み → ロジットバイアス へ変換される
    - バイアスは減衰付きで加算され、次のサンプリングに確率の波として乗る
    - Web 検索結果も同様に `inject_evidence(texts)` で流せる

これにより、ユーザーが生成中に追加指示を送っても生成を中断せず、
確率分布を滑らかに曲げて出力をステアできる。出力が止まることはない。
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from collections import deque

import numpy as np


@dataclass
class SteeringSignal:
    text: str
    bias: dict[int, float]  # token_id -> logit bias
    strength: float = 1.0
    decay: float = 0.92
    ttl: int = 16  # 何トークン有効か
    created_at: float = field(default_factory=time.time)

    def effective_bias(self, age: int) -> dict[int, float]:
        factor = (self.decay ** age) * self.strength
        if factor < 0.05:
            return {}
        return {tid: v * factor for tid, v in self.bias.items()}


class LogitModulator:
    """ロジットに確率の波として干渉するモジュール。

    介入プロンプト → キーワード抽出 → トークン ID → バイアス
    の変換を行い、生成時の logits に加算する。
    """

    def __init__(self, tokenizer=None, vocab_size: int = 640):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def text_to_bias(self, text: str, *, strength: float = 2.2) -> dict[int, float]:
        text = str(text or "").strip()
        if not text or self.tokenizer is None:
            return {}
        bias: dict[int, float] = {}
        # 介入テキスト中の内容語をバイアス対象にする
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
                    except Exception:
                        pass
        except Exception:
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

    def apply(self, logits: np.ndarray, signals: list[SteeringSignal],
              age_map: dict[int, int] | None = None) -> np.ndarray:
        if not signals:
            return logits
        out = logits.astype(np.float32).copy()
        for idx, sig in enumerate(signals):
            age = age_map.get(idx, 0) if age_map else 0
            eff = sig.effective_bias(age)
            for tid, v in eff.items():
                if 0 <= tid < out.shape[-1]:
                    out[tid] += v
        return out


class SteeringBus:
    """ノンストップで介入を受け付けるバス。

    生成スレッドは `poll()` で最新のバイアスを取得する。
    UI / API スレッドは `steer(text)` / `inject_evidence(sentences)` で
    いつでも介入できる。生成を止める必要はない。
    """

    def __init__(self, modulator: LogitModulator | None = None):
        self.modulator = modulator or LogitModulator()
        self._signals: deque[SteeringSignal] = deque(maxlen=16)
        self._lock = threading.RLock()
        self._age: dict[int, int] = {}
        self._next_id = 0
        self._id_map: dict[int, SteeringSignal] = {}

    def steer(self, text: str, *, strength: float = 2.2, ttl: int = 16) -> int:
        text = str(text or "").strip()
        if not text:
            return -1
        bias = self.modulator.text_to_bias(text, strength=strength)
        if not bias:
            return -1
        sig = SteeringSignal(text=text, bias=bias, strength=strength,
                             ttl=ttl, created_at=time.time())
        with self._lock:
            sid = self._next_id
            self._next_id += 1
            self._signals.append(sig)
            self._id_map[sid] = sig
            self._age[sid] = 0
        return sid

    def inject_evidence(self, sentences: list[str], *, strength: float = 1.6) -> int:
        sentences = [s for s in sentences if s and s.strip()][:6]
        if not sentences:
            return -1
        bias = self.modulator.evidence_to_bias(sentences, strength=strength)
        if not bias:
            return -1
        sig = SteeringSignal(text="|".join(sentences[:2]), bias=bias,
                             strength=strength, ttl=20, created_at=time.time())
        with self._lock:
            sid = self._next_id
            self._next_id += 1
            self._signals.append(sig)
            self._id_map[sid] = sig
            self._age[sid] = 0
        return sid

    def poll(self) -> list[SteeringSignal]:
        with self._lock:
            # TTL 切れを除去
            alive: deque[SteeringSignal] = deque(maxlen=16)
            new_age: dict[int, int] = {}
            for sid, sig in list(self._id_map.items()):
                age = self._age.get(sid, 0)
                if age >= sig.ttl:
                    continue
                # 時間でも減衰 (30秒で消える)
                if time.time() - sig.created_at > 30:
                    continue
                alive.append(sig)
                new_age[sid] = age
            self._signals = alive
            # age を進める
            for sid in list(new_age.keys()):
                new_age[sid] += 1
            self._age = new_age
            return list(self._signals)

    def apply_to_logits(self, logits: np.ndarray) -> np.ndarray:
        signals = self.poll()
        if not signals:
            return logits
        age_map = {i: self._age.get(sid, 0)
                   for i, sid in enumerate(list(self._id_map.keys())[-len(signals):])}
        # 簡易: 全シグナルを合成
        out = logits
        for sig in signals:
            # 各シグナルの age を取得
            eff = sig.effective_bias(0)  # 近似
            for tid, v in eff.items():
                if 0 <= tid < out.shape[-1]:
                    out[tid] += v * 0.5
        return out

    def clear(self):
        with self._lock:
            self._signals.clear()
            self._id_map.clear()
            self._age.clear()

    def status(self) -> dict:
        with self._lock:
            return {
                "pending": len(self._signals),
                "signals": [{"text": s.text[:48], "ttl": s.ttl, "strength": s.strength}
                            for s in list(self._signals)[-4:]],
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
        return _GLOBAL_BUS
