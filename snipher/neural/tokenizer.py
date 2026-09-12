"""文字レベルトークナイザ（内蔵ニューラルコア用）。

BPE と違い**未知文字が原理的に発生しません**。絵文字・人名用漢字（𠮷 など）・
異体字もそのまま 1 トークンになり、語彙に無い文字は ``<unk>`` ではなく
学習済みの文字埋め込みの近傍（同じ Unicode 範囲の平均）で扱います。

役割マーカー（``<user>`` / ``<asst>``）を文中に置くことで、会話の続きとして
生成できる（テンプレートが無くても動く = LFM2.5 と同じ方針）。
"""

from __future__ import annotations

import re

# 意味を持つ制御トークン。ID は固定（学習時・推論時でズレないようにする）。
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<asst>", "<sys>"]

PAD, UNK, BOS, EOS, USER, ASST, SYS = range(len(SPECIAL_TOKENS))

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_MULTISPACE = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """推論時と学習時で必ず同じ正規化を通す（ズレると生成が壊れる）。"""
    t = str(text or "")
    t = t.replace("\u3000", " ")
    t = _WHITESPACE.sub(" ", t)
    t = _MULTISPACE.sub("\n\n", t)
    # 疑問符・感嘆符の揺れを正規化（? ？ ! ！）
    t = t.replace("？", "?").replace("！", "!")
    return t.strip()


class CharTokenizer:
    """1 文字 = 1 トークンのごく単純なトークナイザ。"""

    def __init__(self, vocab: list[str]):
        self.vocab = list(vocab)
        self.itos = {i: c for i, c in enumerate(self.vocab)}
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.n_vocab = len(self.vocab)
        # <unk> の代替として使う「同じ範囲の平均」の候補（学習済み文字の集合）
        self._letters = [
            i for c, i in self.stoi.items() if len(c) == 1 and c.isalpha() and i >= len(SPECIAL_TOKENS)
        ]

    # ------------------------------------------------------------------ #
    @classmethod
    def from_text(cls, text: str, max_vocab: int = 1200) -> "CharTokenizer":
        """コーパスから語彙を作る（頻度順・特殊トークンは常に先頭）。"""
        from collections import Counter

        cnt = Counter(normalize(text))
        common = [c for c, _ in cnt.most_common(max_vocab - len(SPECIAL_TOKENS))]
        return cls(SPECIAL_TOKENS + common)

    @classmethod
    def from_vocab(cls, vocab: list[str]) -> "CharTokenizer":
        return cls(vocab)

    # ------------------------------------------------------------------ #
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        s = normalize(text)
        ids = [self.stoi.get(ch, UNK) for ch in s]
        if add_bos:
            ids = [BOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        out: list[str] = []
        for i in ids:
            ch = self.itos.get(int(i))
            if ch is None or ch in ("<pad>", "<unk>", "<bos>", "<eos>"):
                continue
            out.append(ch)
        return "".join(out)

    # ------------------------------------------------------------------ #
    def unknown_positions(self, text: str) -> list[int]:
        return [i for i, c in enumerate(self.encode(text)) if c == UNK]

    def has_unknown(self, text: str) -> bool:
        return UNK in self.encode(text)

    # ------------------------------------------------------------------ #
    def role_ids(self, role: str) -> list[int]:
        """会話マーカーのトークン列。"""
        tok = {"user": USER, "assistant": ASST, "system": SYS}[role]
        return [tok]

    def size(self) -> int:
        return self.n_vocab
