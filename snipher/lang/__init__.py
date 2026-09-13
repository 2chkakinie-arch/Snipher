"""Snipher の言語層 — 音韻・語彙・形態論の共通土台。

タスク用の分岐を増やさず、**言葉そのものを扱う能力**を一か所に集める。
しりとり（拍の連鎖）・韻・文字数指定・表記変換・品詞照会・分かち書きは、
みなこの三つのモジュールの上で組み立つ。
"""

from __future__ import annotations

from . import lex, morph, phonetics
from .lex import Word, WordBank, bank, lookup
from .morph import conjugate, ctype_of, detect_pos, inflect, lemma_of, stem, to_plain, to_polite
from .phonetics import (
    chain_key,
    chain_match,
    chain_start,
    char_count,
    ends_with_n,
    kana_ratio,
    kana_to_ro,
    mora_count,
    mora_split,
    normalize,
    ro_to_kana,
    to_hiragana,
    to_katakana,
)

__all__ = [
    "lex", "morph", "phonetics", "Word", "WordBank", "bank", "lookup",
    "conjugate", "inflect", "stem", "lemma_of", "detect_pos", "ctype_of", "to_polite", "to_plain",
    "normalize", "to_hiragana", "to_katakana", "kana_to_ro", "ro_to_kana",
    "mora_split", "mora_count", "char_count", "chain_key", "chain_start", "chain_match",
    "ends_with_n", "kana_ratio",
]
