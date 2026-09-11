"""未知文字（トークナイザが苦手な文字）の検出。

トークナイザが 1 文字をどう処理するかを調べ、次のように分類する:

- ``unk``        … UNK トークンに落ちる（最悪）
- ``bytes``      … バイトフォールバック断片（<0xNN> など）に分解される
- ``fragmented`` … 2 個以上のサブワードに分解される
- ``ok``         … 1 トークンで保持できている

"unk" / "bytes" を未知文字として学習対象に推奨する。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# バイトフォールバック系の特殊断片の目安
_BYTE_RE = re.compile(r"^(<0x[0-9A-Fa-f]{2}>|<\|?byte[_-]?\d+\|?>|<bytes_.*>)$")

_SKIP = set(" \t\r\n　、。！？!?\n")


@dataclass
class CharReport:
    char: str
    codepoint: str
    status: str                 # unk | bytes | fragmented | ok
    pieces: list[str] = field(default_factory=list)
    n_pieces: int = 0

    @property
    def is_unknown(self) -> bool:
        return self.status in ("unk", "bytes", "fragmented")

    def to_dict(self) -> dict:
        return {
            "char": self.char,
            "codepoint": self.codepoint,
            "status": self.status,
            "pieces": self.pieces,
            "n_pieces": self.n_pieces,
            "unknown": self.is_unknown,
        }


def classify_char(tokenizer, ch: str) -> CharReport | None:
    """1 文字を単独トークナイズして状態を判定する。"""
    if ch in _SKIP:
        return None
    try:
        ids = tokenizer.encode(ch, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(ch)
    if not ids:
        return None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    try:
        raw_tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        raw_tokens = [str(i) for i in ids]

    if unk_id is not None and unk_id in ids:
        return CharReport(ch, f"U+{ord(ch):04X}", "unk", raw_tokens, len(ids))
    if any(_BYTE_RE.match(t or "") for t in raw_tokens):
        return CharReport(ch, f"U+{ord(ch):04X}", "bytes", raw_tokens, len(ids))
    if len(ids) >= 2:
        shown = [t for t in (_display(tokenizer, i) for i in ids) if t]
        return CharReport(ch, f"U+{ord(ch):04X}", "fragmented", shown or raw_tokens, len(ids))
    return CharReport(ch, f"U+{ord(ch):04X}", "ok", raw_tokens, 1)


def _display(tokenizer, token_id: int) -> str:
    try:
        s = tokenizer.decode([token_id])
        return s if s.strip() else ""
    except Exception:
        return ""


def scan_text(tokenizer, text: str, max_chars: int = 200) -> dict:
    """テキスト中の未知文字を報告する。"""
    reports: list[CharReport] = []
    seen: set[str] = set()
    for ch in text:
        if ch in seen or ch in _SKIP:
            continue
        seen.add(ch)
        if len(seen) > max_chars:
            break
        try:
            rep = classify_char(tokenizer, ch)
        except Exception:
            rep = None
        if rep is not None:
            reports.append(rep)

    unknown = [r for r in reports if r.is_unknown]
    return {
        "unknown": [r.to_dict() for r in unknown],
        "all": [r.to_dict() for r in reports],
        "counts": {
            "chars": len(seen),
            "unknown": len(unknown),
        },
    }


def describe(ch: str) -> str:
    """人間向けの 1 行説明（UI 表示用）。"""
    name = ""
    try:
        name = unicodedata.name(ch, "")
    except Exception:
        pass
    return f"{ch} (U+{ord(ch):04X} {name})".strip()
