"""Snipher の独自確率計算式。

LFM2.5 1.2B JP(1.2Bパラメータの日本語モデル)の振る舞いを観察して得た知見を、
極小のパラメータで近似するために設計したスコアリングモデル。

----------------------------------------------------------------------
LFM2.5 1.2B JP の解析から導いた知見(設計根拠)
----------------------------------------------------------------------
1. 大規模モデルでも、日本語の「次のトークン」は
   (a) 高頻度の文法パターン(文型) と
   (b) 動詞が要求する格助詞(を/に/が/へ) と
   (c) 話題の意味クラスの連続性
   に強く支配される。→ これを低次元の特徴量に落とし込める。

2. 助詞・助動詞・活用語尾は出現分布が鋭い(エントロピーが低い)ため、
   小さいテーブルで高精度に決められる。→ テーブルで「事前に決める」。

3. 述語(動詞・形容詞)の選択が文の骨格を決め、その他の要素は述語に
   従属する傾向がある。→ 述語を中心にしたスコアリングが有効。

----------------------------------------------------------------------
独自確率式
----------------------------------------------------------------------
候補 w(トークン) のスコア S(w) を次の和で定義する:

    S(w) = alpha * log P_freq(w | 直前)        … 頻度事前分布(コーパス)
         + beta  * log P_trans(pos_w | pos_prev) … 品詞遷移(マルコフ)
         + gamma * Q_role(w, 述語)             … 格・役割適合
         + delta * Q_inflect(w, 前後)          … 活用整合
         + epsilon * Q_register(w, 文体)       … 文体整合
         + zeta   * Q_topic(w, 話題)           … トピック一貫性

     P(w) = softmax(S / temperature)

パラメータは {alpha, beta, gamma, delta, epsilon, zeta} の6個のスカラーのみ。
(重みは config.json に格納。頻度は corpus.json の25文から統計を取る。)
----------------------------------------------------------------------
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from .lexicon import Lexicon

# 品詞の内部ラベル
_POS_VERB = "動詞"
_POS_ADJ = "形容詞"
_POS_NOUN = "名詞"
_POS_PART = "助詞"
_POS_AUX = "助動詞"
_POS_ADV = "副詞"
_POS_CONJ = "接続詞"
_POS_PUNCT = "句点"

# 品詞遷移: 「直前の品詞 → 取りうる次の品詞」の重み(1.2B JP の統計的傾向を簡約)
_TRANSITIONS = {
    _POS_NOUN: {_POS_PART: 1.0, _POS_AUX: 0.5, _POS_VERB: 0.2, _POS_NOUN: 0.1},
    _POS_PART: {_POS_NOUN: 0.9, _POS_VERB: 0.7, _POS_ADJ: 0.5, _POS_ADV: 0.3, _POS_AUX: 0.2},
    _POS_VERB: {_POS_AUX: 0.9, _POS_PUNCT: 0.8, _POS_PART: 0.4, _POS_CONJ: 0.2},
    _POS_ADJ: {_POS_AUX: 0.9, _POS_NOUN: 0.6, _POS_PUNCT: 0.7},
    _POS_AUX: {_POS_PUNCT: 0.9, _POS_PART: 0.3, _POS_CONJ: 0.2, _POS_AUX: 0.2},
    _POS_ADV: {_POS_VERB: 0.8, _POS_ADJ: 0.6, _POS_NOUN: 0.3},
    _POS_CONJ: {_POS_NOUN: 0.8, _POS_VERB: 0.6, _POS_ADV: 0.3},
    _POS_PUNCT: {_POS_NOUN: 0.6, _POS_VERB: 0.3, _POS_ADV: 0.3, _POS_ADJ: 0.2, _POS_CONJ: 0.4},
}

# 動詞が要求する助詞(自動詞なら「が」、他動詞なら「を」など)の整合スコア
_ROLE_BONUS = {
    ("名詞", "を", "他動詞"): 1.0,
    ("名詞", "を", "自動詞"): -1.0,
    ("名詞", "が", "自動詞"): 1.0,
    ("名詞", "が", "他動詞"): 0.2,
    ("名詞", "に", "自動詞"): 0.6,
    ("名詞", "へ", "自動詞"): 0.6,
    ("名詞", "で", "自動詞"): 0.4,
    ("名詞", "は", "他動詞"): 0.3,
    ("名詞", "は", "自動詞"): 0.4,
}


class ProbabilityModel:
    """極小パラメータの確率的スコアリングモデル。"""

    def __init__(self, lexicon: Lexicon | None = None, seed: int | None = None):
        self.lex = lexicon or Lexicon()
        cfg = self.lex.config.get("weights", {})
        self.alpha = cfg.get("alpha", 1.2)
        self.beta = cfg.get("beta", 1.0)
        self.gamma = cfg.get("gamma", 1.0)
        self.delta = cfg.get("delta", 0.8)
        self.epsilon = cfg.get("epsilon", 0.6)
        self.zeta = cfg.get("zeta", 1.5)
        self.temperature = self.lex.config.get("temperature", 0.8)
        self.max_candidates = self.lex.config.get("max_candidates", 64)
        self.rng = random.Random(seed)
        self._build_freq()

    # ------------------------------------------------------------------ #
    # コーパスからの頻度統計(alpha 項)
    # ------------------------------------------------------------------ #
    def _build_freq(self) -> None:
        self.unigram = Counter()
        self.bigram = Counter()
        self.pos_freq = Counter()
        for sent in self.lex.corpus:
            toks = sent.get("tokens", [])
            poss = sent.get("pos", [])
            for t in toks:
                self.unigram[t] += 1
            for a, b in zip(toks, toks[1:]):
                self.bigram[(a, b)] += 1
            for p in poss:
                self.pos_freq[p] += 1
        self.total_unigram = sum(self.unigram.values()) or 1

    def freq_logprob(self, token_surface: str, prev_surface: str | None) -> float:
        """log P(token | prev) の簡易推定。"""
        if prev_surface and (prev_surface, token_surface) in self.bigram:
            return math.log((self.bigram[(prev_surface, token_surface)] + 1.0) / self.total_unigram)
        if token_surface in self.unigram:
            return math.log((self.unigram[token_surface] + 1.0) / self.total_unigram)
        return math.log(0.5 / self.total_unigram)  # スムージング

    # ------------------------------------------------------------------ #
    # 品詞遷移(beta 項)
    # ------------------------------------------------------------------ #
    def transition_logprob(self, pos: str, prev_pos: str | None) -> float:
        if prev_pos is None:
            return 0.0
        table = _TRANSITIONS.get(prev_pos, {})
        w = table.get(pos, 0.02)
        return math.log(w + 1e-6)

    # ------------------------------------------------------------------ #
    # 役割適合(gamma 項)
    # ------------------------------------------------------------------ #
    @staticmethod
    def role_score(pos: str, particle: str, verb_trans: str | None) -> float:
        if pos != _POS_NOUN or not particle or verb_trans is None:
            return 0.0
        key = (_POS_NOUN, particle, verb_trans)
        if key in _ROLE_BONUS:
            return _ROLE_BONUS[key]
        # 動詞が決めた助詞に一致するか
        if particle == "を" and verb_trans == "他動詞":
            return 0.5
        if particle == "が" and verb_trans == "自動詞":
            return 0.5
        return 0.0

    # ------------------------------------------------------------------ #
    # 活用整合(delta 項)
    # ------------------------------------------------------------------ #
    @staticmethod
    def inflection_score(pos: str, prev_pos: str | None) -> float:
        """助動詞「ます/た/ない」は動詞の連用形・形容詞の後に付きやすい。"""
        if pos == _POS_AUX and prev_pos in (_POS_VERB, _POS_ADJ):
            return 1.0
        if pos == _POS_AUX and prev_pos == _POS_NOUN:
            return 0.2
        if pos == _POS_VERB and prev_pos == _POS_NOUN:
            return -0.2  # 名詞の直後に動詞はやや不自然
        return 0.0

    # ------------------------------------------------------------------ #
    # 文体整合(epsilon 項)
    # ------------------------------------------------------------------ #
    @staticmethod
    def register_score(surface: str, register: str) -> float:
        polite = register == "polite"
        polite_marks = ("です", "ます", "ました", "ません", "でした")
        casual_marks = ("だ", "た", "ない", "だった")
        if polite and surface in polite_marks:
            return 1.0
        if polite and surface in casual_marks:
            return -0.6
        if not polite and surface in casual_marks:
            return 1.0
        if not polite and surface in polite_marks:
            return -0.6
        return 0.0

    # ------------------------------------------------------------------ #
    # トピック一貫性(zeta 項)
    # ------------------------------------------------------------------ #
    @staticmethod
    def topic_score(tags: list[str], topic_tags: list[str] | None) -> float:
        if not tags or not topic_tags:
            return 0.0
        common = set(tags) & set(topic_tags)
        return min(1.0, 0.5 * len(common))

    # ------------------------------------------------------------------ #
    # 統合スコア
    # ------------------------------------------------------------------ #
    def score(
        self,
        surface: str,
        pos: str,
        tags: list[str],
        prev_surface: str | None,
        prev_pos: str | None,
        particle: str | None,
        verb_trans: str | None,
        register: str,
        topic_tags: list[str] | None,
    ) -> float:
        s = 0.0
        s += self.alpha * self.freq_logprob(surface, prev_surface)
        s += self.beta * self.transition_logprob(pos, prev_pos)
        s += self.gamma * self.role_score(pos, particle or "", verb_trans)
        s += self.delta * self.inflection_score(pos, prev_pos)
        s += self.epsilon * self.register_score(surface, register)
        s += self.zeta * self.topic_score(tags, topic_tags)
        return s

    # ------------------------------------------------------------------ #
    # 候補選択(softmax サンプリング)
    # ------------------------------------------------------------------ #
    def sample(self, candidates: list[dict], temperature: float | None = None) -> dict:
        """候補リストからスコアに基づき確率的に1つ選ぶ。"""
        if not candidates:
            raise ValueError("候補が空です")
        temp = self.temperature if temperature is None else temperature
        if len(candidates) == 1:
            return candidates[0]
        scores = [c.get("score", -1e9) for c in candidates]
        # 数値安定化のため最大値を引く
        mx = max(scores)
        if temp <= 0:
            best = max(candidates, key=lambda c: c.get("score", -1e9))
            return best
        exps = [math.exp((sc - mx) / temp) for sc in scores]
        total = sum(exps) or 1.0
        probs = [e / total for e in exps]
        r = self.rng.random()
        acc = 0.0
        for cand, p in zip(candidates, probs):
            acc += p
            if r <= acc:
                return cand
        return candidates[-1]
