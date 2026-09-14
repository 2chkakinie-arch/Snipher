"""v8 サンプリング制御（繰り返し封印・プレゼンス/頻度ペナルティ・top-k/top-p）。

反復ループ（同じ単語・定型文の周回）を「物理的に」禁止するため、
- ``no_repeat_ngram`` で同一 n-gram を 2 度生成しない（logit を -inf に潰す）
- プレゼンス/頻度ペナルティで既出トークンの再登場を抑制する
これらはテンプレートや後処理ではなく、サンプリングの logit 段階で働く。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_NEG = -np.inf


@dataclass
class SamplingControls:
    """生成時の logit 制御。すべて省略可（既定値は汎用の日本語向け）。"""

    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 1.0
    repetition_penalty: float = 1.08
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    no_repeat_ngram: int = 4          # 同じ n-gram を 2 度出さない（0 で無効）
    no_repeat_window: int = 512       # 反復判定に使う直近トークン数
    boost: dict[int, float] = field(default_factory=dict)   # 確率の波: 強調トークン
    ban: set[int] = field(default_factory=set)              # 確率の波: 禁止トークン


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """k 未満の logit を -inf に潰す（k<=0 ならそのまま）。"""
    if k and k > 0 and k < logits.shape[-1]:
        th = np.partition(logits, -k, axis=-1)[..., -k:]
        cutoff = th.min(axis=-1, keepdims=True)
        return np.where(logits < cutoff, _NEG, logits)
    return logits


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """累積確率 p を超える裾を落とす nucleus sampling（p>=1 ならそのまま）。

    既存コア（core.py の ``_top_p_filter``）と同じ規則: 累積確率が p に達した
    ところまで残す（``searchsorted(cum, p) + 1`` 個）。
    """
    if p >= 1.0:
        return logits
    order = np.argsort(logits, axis=-1)[..., ::-1]
    sorted_l = np.take_along_axis(logits, order, axis=-1)
    e = np.exp(sorted_l - sorted_l.max(axis=-1, keepdims=True))
    probs = e / np.maximum(1e-12, e.sum(axis=-1, keepdims=True))
    cum = np.cumsum(probs, axis=-1)
    cutoff = np.clip(np.searchsorted(cum, p, side="left") + 1, 1, logits.shape[-1])
    idx = np.arange(logits.shape[-1])[None, :]
    mask = idx < cutoff[..., None]
    kept_sorted = np.where(mask, sorted_l, _NEG)
    out = np.empty_like(logits)
    np.put_along_axis(out, order, kept_sorted, axis=-1)
    return out


def repetition_penalize(logits: np.ndarray, history: list[int], penalty: float) -> np.ndarray:
    """既出トークンの logit を割り引く（1.0 で無効）。"""
    if penalty == 1.0 or not history:
        return logits
    out = logits.copy()
    for t in set(history):
        if out[t] > 0:
            out[t] /= penalty
        else:
            out[t] *= penalty
    return out


def presence_frequency_penalize(logits: np.ndarray, history: list[int],
                                presence: float, frequency: float) -> np.ndarray:
    """OpenAI 型のプレゼンス/頻度ペナルティ。"""
    if not history or (presence == 0.0 and frequency == 0.0):
        return logits
    out = logits.copy()
    counts: dict[int, int] = {}
    for t in history:
        counts[t] = counts.get(t, 0) + 1
    for t, c in counts.items():
        out[t] -= presence + frequency * c
    return out


def no_repeat_ban(logits: np.ndarray, history: list[int], ngram: int,
                  window: int = 512) -> np.ndarray:
    """同一 n-gram の再生成を禁止する。

    ``history`` 末尾の (ngram-1) トークン + 候補トークン でできる n-gram が、
    直近 ``window`` トークン内に既に現れていれば、その候補の logit を -inf にする。
    これにより「同じ語列の周回」はサンプリング時に物理的に起きない。
    """
    if ngram <= 1 or len(history) < ngram:
        return logits
    out = logits.copy()
    h = history[-window:]
    prefix = tuple(h[-(ngram - 1):]) if ngram > 1 else ()
    if len(prefix) != ngram - 1:
        return out
    seen: set[tuple[int, ...]] = set()
    for i in range(len(h) - ngram + 1):
        seen.add(tuple(h[i:i + ngram]))
    for t in range(out.shape[0]):
        if (prefix + (t,)) in seen:
            out[t] = _NEG
    return out


def apply_biases(logits: np.ndarray, boost: dict[int, float] | None,
                 ban: set[int] | list[int] | None) -> np.ndarray:
    """確率の波（強調/禁止）を logit に合成する。None は空扱い。"""
    out = logits.copy()
    for t, w in (boost or {}).items():
        if 0 <= t < out.shape[0]:
            out[t] += float(w)
    for t in (ban or ()):
        if 0 <= t < out.shape[0]:
            out[t] = _NEG
    return out


def sample_token(logits: np.ndarray, controls: SamplingControls,
                 history: list[int], rng: np.random.Generator) -> int | None:
    """logit に温度・top-k・top-p・反復封印・プレゼンス/頻度を適用して 1 トークン選ぶ。

    戻り値は ``int``（選ばれたトークン）。全候補が -inf（反復封印で尽きた等）の
    場合は ``None`` を返す（呼び出し側は停止とみなす）。
    """
    lg = logits.astype(np.float32)
    lg = apply_biases(lg, controls.boost, controls.ban)
    lg = presence_frequency_penalize(lg, history, controls.presence_penalty,
                                     controls.frequency_penalty)
    lg = repetition_penalize(lg, history, controls.repetition_penalty)
    lg = no_repeat_ban(lg, history, controls.no_repeat_ngram, controls.no_repeat_window)
    if not np.isfinite(lg).any():
        return None
    if controls.temperature and controls.temperature > 0:
        lg = lg / controls.temperature
    lg = top_k_filter(lg, controls.top_k)
    lg = top_p_filter(lg, controls.top_p)
    finite = np.isfinite(lg)
    if not finite.any():
        return None
    e = np.exp(lg - lg.max())
    e = np.where(finite, e, 0.0)
    s = e.sum()
    if s <= 0:
        return None
    probs = e / s
    return int(rng.choice(logits.shape[0], p=probs))
