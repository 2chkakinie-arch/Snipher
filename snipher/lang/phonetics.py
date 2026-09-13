"""音韻・かな・拍（モーラ）の共通語彙 — Snipher が「言葉の形」を数えるための層。

しりとり系の手遊び・韻・文字数指定・読み替え・ローマ字変換・全角半角の正規化は、
どれも **同じ一本の土台**（文字種変換 + 拍の分解 + 母音/子音の同定）の上で動く。
タスクごとに特別な分岐を作らないためのモジュール。

    normalize()   … NFKC + 濁点の統合 + 空白正規化（比較・検索の直前に必ず通す）
    mora_split()  … 「やっぱり」→ ['や', 'っ', 'ぱ', 'り']（拍の列）
    kana_to_ro()  … かな → ローマ字（ヘボン式）
    ro_to_kana()  … ローマ字 → かな（別綴り・子音重複＝促音にも対応）
    chain_key()   … 「次の語の頭」に合わせる音（しりとり・けとり等の共通判定）
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
# 文字集合・正規化
# --------------------------------------------------------------------------- #
_HIRA_SMALL = "ぁぃぅぇぉゃゅょゎっ"
_VOICED_HEAD = {
    "か": "が", "き": "ぎ", "く": "ぐ", "け": "げ", "こ": "ご",
    "さ": "ざ", "し": "じ", "す": "ず", "せ": "ぜ", "そ": "ぞ",
    "た": "だ", "ち": "ぢ", "つ": "づ", "て": "で", "と": "ど",
    "は": "ば", "ひ": "び", "ふ": "ぶ", "へ": "べ", "ほ": "ぼ",
    "う": "ゔ", "わ": "ゐ",
}
_KATA_SPECIAL = {"ヶ": "け", "ヷ": "わ", "ヸ": "い", "ヹ": "え", "ヺ": "を", "ヽ": "", "ヾ": ""}

HIRAGANA = ("ぁあぃいぅうぇえぉおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむ"
            "めもゃゅょやゆよりるれろわをん")
KATAKANA = ("ァアィイゥウェエォオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミム"
            "メモャュョヤユヨラリルレロワヲン")

_RE_KANA = re.compile(r"[ぁ-んァ-ヶ]")
_RE_HIRA = re.compile(r"[぀-ゟ]")
_RE_KATA = re.compile(r"[゠-ヿ]")
_RE_DAKUTEN = re.compile(r"[゙゚]")


def normalize(text: str) -> str:
    """比較・検索用の正規形（NFKC + NFC + 半角カナの統合 + 空白）。"""
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[ \t\u3000]+", " ", t)
    # NFKC は省略符「…」を ASCII の三時点に分解する。文章の末尾・引用符の直前に
    # 来るものだけ、もとの省略符に戻す（コードの print(...) を壊さないため条件を付ける）。
    t = re.sub(r"\.\.\.(?=[」』。、！？\s]|$)", "…", t)
    return t.strip()


def to_hiragana(text: str) -> str:
    """カタカナ → ひらがな（長音「ー」は残す）。"""
    out = []
    for ch in str(text or ""):
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        elif ch == "ー":
            out.append("ー")
        elif ch in _KATA_SPECIAL:
            out.append(_KATA_SPECIAL[ch])
        elif ch == "・":
            continue
        else:
            out.append(ch)
    return "".join(out)


def to_katakana(text: str) -> str:
    """ひらがな → カタカナ。"""
    out = []
    for ch in str(text or ""):
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            out.append(chr(code + 0x60))
        else:
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------- #
# ローマ字 ⇔ かな（対の表を一本だけ持ち、両方向をそこから作る）
# --------------------------------------------------------------------------- #
_KANA_RO: tuple[tuple[str, str], ...] = (
    ("かゃ", "kya"),
    ("がゃ", "gya"),
    ("きぃ", "kyi"),
    ("きぇ", "kye"),
    ("きゅ", "kyu"),
    ("きょ", "kyo"),
    ("ぎぃ", "gyi"),
    ("ぎぇ", "gye"),
    ("ぎゅ", "gyu"),
    ("ぎょ", "gyo"),
    ("さゃ", "sya"),
    ("ざゃ", "zya"),
    ("しぃ", "syi"),
    ("しぇ", "sye"),
    ("しゃ", "sha"),
    ("しゅ", "shu"),
    ("しょ", "sho"),
    ("じぃ", "zyi"),
    ("じぇ", "zye"),
    ("じゃ", "ja"),
    ("じゅ", "ju"),
    ("じょ", "jo"),
    ("たゃ", "tya"),
    ("だゃ", "dya"),
    ("ちぃ", "tyi"),
    ("ちぇ", "tye"),
    ("ちゃ", "cha"),
    ("ちゅ", "chu"),
    ("ちょ", "cho"),
    ("ぢぃ", "dyi"),
    ("ぢぇ", "dye"),
    ("ぢゃ", "ja"),
    ("ぢゅ", "ju"),
    ("ぢょ", "jo"),
    ("なゃ", "nya"),
    ("にぃ", "nyi"),
    ("にぇ", "nye"),
    ("にゅ", "nyu"),
    ("にょ", "nyo"),
    ("はゃ", "hya"),
    ("ばゃ", "bya"),
    ("ぱゃ", "pya"),
    ("ひぃ", "hyi"),
    ("ひぇ", "hye"),
    ("ひゅ", "hyu"),
    ("ひょ", "hyo"),
    ("びぃ", "byi"),
    ("びぇ", "bye"),
    ("びゅ", "byu"),
    ("びょ", "byo"),
    ("ぴぃ", "pyi"),
    ("ぴぇ", "pye"),
    ("ぴゅ", "pyu"),
    ("ぴょ", "pyo"),
    ("まゃ", "mya"),
    ("みぃ", "myi"),
    ("みぇ", "mye"),
    ("みゅ", "myu"),
    ("みょ", "myo"),
    ("らゃ", "rya"),
    ("りぃ", "ryi"),
    ("りぇ", "rye"),
    ("りゅ", "ryu"),
    ("りょ", "ryo"),
    ("ゔぃ", "vi"),
    ("ゔぅ", "vu"),
    ("ゔぇ", "ve"),
    ("ゔぉ", "vo"),
    ("ゔゃ", "vya"),
    ("ゔゅ", "vyu"),
    ("ゔょ", "vyo"),
    ("、", ","),
    ("。", "."),
    ("〇", "0"),
    ("ぁ", "la"),
    ("あ", "a"),
    ("ぃ", "li"),
    ("い", "i"),
    ("ぅ", "lu"),
    ("う", "u"),
    ("ぇ", "le"),
    ("え", "e"),
    ("ぉ", "lo"),
    ("お", "o"),
    ("か", "ka"),
    ("が", "ga"),
    ("き", "ki"),
    ("ぎ", "gi"),
    ("く", "ku"),
    ("ぐ", "gu"),
    ("け", "ke"),
    ("げ", "ge"),
    ("こ", "ko"),
    ("ご", "go"),
    ("さ", "sa"),
    ("ざ", "za"),
    ("し", "shi"),
    ("じ", "ji"),
    ("す", "su"),
    ("ず", "zu"),
    ("せ", "se"),
    ("ぜ", "ze"),
    ("そ", "so"),
    ("ぞ", "zo"),
    ("た", "ta"),
    ("だ", "da"),
    ("ち", "chi"),
    ("ぢ", "ji"),
    ("っ", "ltu"),
    ("つ", "tsu"),
    ("づ", "zu"),
    ("て", "te"),
    ("で", "de"),
    ("と", "to"),
    ("ど", "do"),
    ("な", "na"),
    ("に", "ni"),
    ("ぬ", "nu"),
    ("ね", "ne"),
    ("の", "no"),
    ("は", "ha"),
    ("ば", "ba"),
    ("ぱ", "pa"),
    ("ひ", "hi"),
    ("び", "bi"),
    ("ぴ", "pi"),
    ("ふ", "fu"),
    ("ぶ", "bu"),
    ("ぷ", "pu"),
    ("へ", "he"),
    ("べ", "be"),
    ("ぺ", "pe"),
    ("ほ", "ho"),
    ("ぼ", "bo"),
    ("ぽ", "po"),
    ("ま", "ma"),
    ("み", "mi"),
    ("む", "mu"),
    ("め", "me"),
    ("も", "mo"),
    ("ゃ", "lya"),
    ("や", "ya"),
    ("ゅ", "lyu"),
    ("ゆ", "yu"),
    ("ょ", "lyo"),
    ("よ", "yo"),
    ("ら", "ra"),
    ("り", "ri"),
    ("る", "ru"),
    ("れ", "re"),
    ("ろ", "ro"),
    ("ゎ", "lwa"),
    ("わ", "wa"),
    ("ゐ", "wi"),
    ("ゑ", "we"),
    ("を", "wo"),
    ("ん", "n"),
    ("ゔ", "vu"),
    ("ヶ", "ka"),
    ("ヷ", "wa"),
    ("ー", "-"),
)

_RO_ALIAS: tuple[tuple[str, str], ...] = (
    ("si", "し"),
    ("syi", "し"),
    ("ti", "ち"),
    ("tc", "ち"),
    ("tyi", "ち"),
    ("tu", "つ"),
    ("tsi", "つ"),
    ("hu", "ふ"),
    ("zi", "じ"),
    ("zyi", "じ"),
    ("di", "ぢ"),
    ("du", "づ"),
    ("dzu", "づ"),
    ("nn", "ん"),
    ("n'", "ん"),
    ("xtu", "っ"),
    ("ltu", "っ"),
    ("xtsu", "っ"),
    ("wi", "うぃ"),
    ("we", "うぇ"),
    ("wu", "う"),
    ("yi", "い"),
    ("ye", "いぇ"),
    ("xa", "ぁ"),
    ("xi", "ぃ"),
    ("xu", "ぅ"),
    ("xe", "ぇ"),
    ("xo", "ぉ"),
    ("la", "ぁ"),
    ("li", "ぃ"),
    ("lu", "ぅ"),
    ("le", "ぇ"),
    ("lo", "ぉ"),
    ("fa", "ふぁ"),
    ("fi", "ふぃ"),
    ("fe", "ふぇ"),
    ("fo", "ふぉ"),
    ("fya", "ふゃ"),
    ("fyu", "ふゅ"),
    ("fyo", "ふょ"),
    ("kwa", "くぁ"),
    ("kwi", "くぃ"),
    ("kwe", "くぇ"),
    ("kwo", "くぉ"),
    ("gwa", "ぐぁ"),
    ("gwi", "ぐぃ"),
    ("gwe", "ぐぇ"),
    ("gwo", "ぐぉ"),
    ("tha", "てゃ"),
    ("tcha", "っちゃ"),
    ("tchi", "っち"),
    ("tchu", "っちゅ"),
    ("va", "ゔぁ"),
    ("vi", "ゔぃ"),
    ("ve", "ゔぇ"),
    ("vo", "ゔぉ"),
    ("--", "ー"),
    ("-", "ー"),
    ("sha", "しゃ"),
    ("shu", "しゅ"),
    ("sho", "しょ"),
    ("sya", "しゃ"),
    ("syu", "しゅ"),
    ("syo", "しょ"),
    ("cha", "ちゃ"),
    ("chu", "ちゅ"),
    ("cho", "ちょ"),
    ("tya", "ちゃ"),
    ("tyu", "ちゅ"),
    ("tyo", "ちょ"),
    ("cya", "ちゃ"),
    ("cyu", "ちゅ"),
    ("cyo", "ちょ"),
    ("ja", "じゃ"),
    ("ju", "じゅ"),
    ("jo", "じょ"),
    ("jya", "じゃ"),
    ("jyu", "じゅ"),
    ("jyo", "じょ"),
    ("zya", "じゃ"),
    ("zyu", "じゅ"),
    ("zyo", "じょ"),
)


_KANA_RO_MAP: dict[str, str] = {}
_RO_TABLE: dict[str, str] = {}
for _kana, _ro in _KANA_RO:
    if not _ro:
        continue
    _KANA_RO_MAP.setdefault(_kana, _ro)
    _RO_TABLE.setdefault(_ro, _kana)
for _ro, _kana in _RO_ALIAS:
    _RO_TABLE.setdefault(_ro, _kana)
for _kana, _ro in (("ぁ", "a"), ("ぃ", "i"), ("ぅ", "u"), ("ぇ", "e"), ("ぉ", "o"),
                   ("ゃ", "ya"), ("ゅ", "yu"), ("ょ", "yo")):
    _KANA_RO_MAP.setdefault(_kana, _ro)
_MAX_RO = max(len(_r) for _r in _RO_TABLE)




def kana_to_ro(text: str) -> str:
    """かな → ローマ字（ヘボン式）。拗音・促音・撥音まで处理的に処理する。"""
    t = to_hiragana(normalize(text))
    out: list[str] = []
    i = 0
    while i < len(t):
        two = t[i:i + 2]
        if len(two) == 2 and two in _KANA_RO_MAP:
            out.append(_KANA_RO_MAP[two])
            i += 2
            continue
        kana = t[i]
        if kana == "っ":
            nxt = _KANA_RO_MAP.get(t[i + 1], "")
            out.append(nxt[:1] if nxt else "")
            i += 1
            continue
        if kana == "ん":
            after = t[i + 1:i + 2]
            out.append("n" if (not after or after in "ー。、！？ｎ" or after[0] in "nbmyp") else "n'")
            i += 1
            continue
        out.append(_KANA_RO_MAP.get(kana, kana))
        i += 1
    return "".join(out)


def ro_to_kana(text: str) -> str:
    """ローマ字 → かな（longest-match。子音の重複は促音）。"""
    t = normalize(text).lower()
    out: list[str] = []
    i = 0
    n = len(t)
    while i < n:
        ch = t[i]
        if ch not in "abcdefghijklmnopqrstuvwxyz'":
            out.append("ー" if ch == "-" else ch)
            i += 1
            continue
        if i + 1 < n and t[i] == t[i + 1] and ch not in "aeiou" and ch != "n":
            out.append("っ")
            i += 1
            continue
        hit = ""
        for ln in range(min(_MAX_RO, n - i), 0, -1):
            chunk = t[i:i + ln]
            if chunk in _RO_TABLE:
                hit = chunk
                break
        if hit:
            out.append(_RO_TABLE[hit])
            i += len(hit)
            continue
        out.append(ch)
        i += 1
    return "".join(out)





def reading_hint(text: str) -> str:
    """漢字混じりの表記を、語彙バンクの読み（かな）に直す。

    拍・しりとり・字数を *表記* で数えると日本語の音が壊れます
    （「日本語」を 3 と数えてしまう等）。辞書に有る語は読みを使って数えます。
    """
    t = str(text or "")
    if not t or not any("\u4e00" <= c <= "\u9fff" for c in t):
        return t
    try:
        from .lex import bank

        w = bank().entry(t)
        if w is not None and w.reading:
            return w.reading
    except Exception:  # noqa: BLE001
        pass
    return t


def mora_split(text: str) -> list[str]:
    """文字列を拍に分解する（拗音 1 拍・促音 1 拍・長音 1 拍）。"""
    kana = to_hiragana(normalize(reading_hint(text)))
    out: list[str] = []
    i = 0
    while i < len(kana):
        ch = kana[i]
        nxt = kana[i + 1] if i + 1 < len(kana) else ""
        if ch in _HIRA_SMALL:
            out.append(ch)
            i += 1
            continue
        if nxt in "ゃゅょ":
            out.append(kana[i:i + 2])
            i += 2
            continue
        if ch == "ー":
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return out


def mora_count(text: str) -> int:
    return len(mora_split(text))


def char_count(text: str) -> int:
    return len(normalize(text))


def first_mora(text: str) -> str:
    ms = mora_split(text)
    return ms[0] if ms else ""


def last_mora(text: str) -> str:
    ms = mora_split(text)
    return ms[-1] if ms else ""


def ends_with_n(text: str) -> bool:
    """拍の最後が「ん」か（拗音・長音は考慮しない。みけん→○、ミー→×）。"""
    return last_mora(text) == "ん"


def chain_key(text: str) -> str:
    """次の語の頭に渡す音。しりとり系の「送り手」。

    - 末尾が「ー」→ 直前の母音拍（コーヒー → ひ）
    - 末尾が促音・小文字だけ → 直前の正常な拍
    - それ以外 → 末尾の拍
    """
    ms = mora_split(text)
    while ms and ms[-1] in ("ー", "・") or (ms and ms[-1] in _HIRA_SMALL):
        ms.pop()
    if not ms:
        return ""
    last = ms[-1]
    if len(last) == 2:                      # 拗音（きゃ → き の段で受ける）
        return last
    return last


def chain_start(text: str) -> str:
    """語の頭の拍（しりとりで「受け手」が確かめる位置）。"""
    ms = [m for m in mora_split(text) if m not in ("っ", "ー")]
    return ms[0] if ms else ""


def chain_match(prev: str, nxt: str) -> bool:
    """prev → nxt が拍の連鎖として成立するか（しりとり一般形）。"""
    a, b = chain_key(prev), chain_start(nxt)
    # 同段の清濁・拗音ずらしも一応受けつける（例: ぎ ← き）
    if not a or not b:
        return False
    if a == b:
        return True
    # 濁音・半濁音・拗音の同段（例: ぎ ← き、ぱ ← は）も許す手があるが、
    # 既定は厳密に拍で一致させる（判定根拠をあいまいにしないため）。
    return False


def voicing_pair(mora: str) -> tuple[str, str]:
    """清音 ↔ 濁音（判定の緩め方に使う）。"""
    rev = {v: k for k, v in _VOICED_HEAD.items()}
    return (rev.get(mora, mora), _VOICED_HEAD.get(mora, mora))


# --------------------------------------------------------------------------- #
# 文字種の判定・補助
# --------------------------------------------------------------------------- #
def is_kana(text: str) -> bool:
    return bool(_RE_KANA.search(str(text or "")))


def is_all_kana(text: str) -> bool:
    t = normalize(text)
    return bool(t) and all(_RE_KANA.match(c) for c in t)


def is_hiragana(text: str) -> bool:
    t = normalize(text)
    return bool(t) and all(_RE_HIRA.match(c) for c in t)


def is_katakana(text: str) -> bool:
    t = normalize(text)
    return bool(t) and all(_RE_KATA.match(c) for c in t)


def is_kanji(ch: str) -> bool:
    return bool(ch) and "一" <= ch <= "鿿"


def ascii_ratio(text: str) -> float:
    t = normalize(text)
    if not t:
        return 1.0
    return sum(1 for c in t if c.isascii()) / len(t)


def kana_ratio(text: str) -> float:
    t = normalize(text)
    if not t:
        return 0.0
    return sum(1 for c in t if _RE_KANA.match(c)) / len(t)


def count_syllables(text: str) -> int:
    return mora_count(text)


def strip_punct(text: str) -> str:
    return re.sub(r"[\s。、！？!?・…「」『』()（）\[\]｛｝{}:：;；〜~\-_,.\"'’‘]+", "",
                  normalize(text))


def reverse_text(text: str) -> str:
    return normalize(text)[::-1]


def has_small_kana(text: str) -> bool:
    return any(c in _HIRA_SMALL for c in to_hiragana(normalize(text)))


def long_vowel_end(text: str) -> bool:
    return normalize(text).endswith(("ー", "〜"))


__all__ = [
    "normalize", "to_hiragana", "to_katakana", "kana_to_ro", "ro_to_kana", "mora_split",
    "mora_count", "char_count", "first_mora", "last_mora", "chain_key", "chain_start",
    "chain_match", "ends_with_n", "is_kana", "is_all_kana", "is_hiragana", "is_katakana",
    "is_kanji", "ascii_ratio", "kana_ratio", "strip_punct", "reverse_text", "has_small_kana",
    "long_vowel_end", "voicing_pair", "HIRAGANA", "KATAKANA",
]
