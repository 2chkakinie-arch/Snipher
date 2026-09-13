"""規則コンパイラ — 発話に書かれた条件を「検証できる述語」に変える。

「5文字で」「ですます調で」「箇条書きで」「A で始めて」「A は使わないで」「3 つ」…、
そして「前の語の最後の音から始める」（＝語の連鎖）まで、
**タスク専用の分岐を作らず**に同じ 1 本のパイプラインで受け止めるための層。

    compile_rules(text)  → [Rule, …]
    check(text, rules)   → (ok, 落ちた理由)
    enforce(text, rules) → 可能なものは書き換えて満たす（文字数・文体・形式）
"""

from __future__ import annotations

import re

from ..lang.phonetics import (
    chain_key,
    chain_start,
    char_count,
    is_all_kana,
    is_hiragana,
    is_katakana,
    mora_count,
    normalize,
    to_hiragana,
)
from .frame import Rule

_NUM = "([0-9０-９]+|[一二三四五六七八九十]+)"
_CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
       "十": 10, "十一": 11, "十二": 12}

_NUM_RE = re.compile(_NUM)


def to_int(value: str) -> int | None:
    v = normalize(str(value)).replace(",", "")
    if v.isdigit():
        return int(v)
    if v in _CN:
        return _CN[v]
    if re.fullmatch(r"[十百千]", v or ""):
        return _CN.get(v)
    m = re.fullmatch(r"([一二三四五六七八九])十([一二三四五六七八九])?", v or "")
    if m:
        return _CN[m.group(1)] * 10 + (_CN.get(m.group(2) or "", 0))
    return None


# --------------------------------------------------------------------------- #
# 抽出
# --------------------------------------------------------------------------- #
_PATTERNS: tuple[tuple[str, re.Pattern, tuple[str, ...]], ...] = (
    ("len", re.compile(rf"{_NUM}\s*(?:文字|文字数|キャラ|字)\s*(?:以内|ほど|前後|で|に)?"), ("num",)),
    ("len", re.compile(r"(?:短め|短く|短文|一句|1 文|一文)で"), ("short",)),
    ("len", re.compile(r"(?:長め|長く|たっぷ)り?で"), ("long",)),
    ("morae", re.compile(rf"{_NUM}\s*(?:音|モーラ|拍)\s*(?:で|以内)?"), ("num",)),
    ("count", re.compile(rf"{_NUM}\s*(?:つ|個|件|例|件分|パターン|語|ワード)\s*(?:あげて|ください|ちょうだい|ある|列出し)?"),
     ("num",)),
    ("line", re.compile(rf"{_NUM}\s*行"), ("num",)),
    ("start", re.compile(r"[「『]?(.{1,6}?)[」』]?\s*(?:で|から)\s*(?:始めて|始め|開始して|スタートして|切って|きって)"),
     ["word"]),
    ("start", re.compile(r"(?:頭|最初|冒頭|先頭)\s*は\s*[「『]?(.{1,6}?)[」』]?(?:に|と)?\s*(?:する|してください)"),
     ["word"]),
    ("end", re.compile(r"[「『]?(.{1,6}?)[」』]?\s*(?:で|に)\s*(?:終わって|終わる|締め|締めて|終わり)"), ["word"]),
    ("end", re.compile(r"(?:語尾|最後|末尾|尻)は\s*[「『]?(.{1,6}?)[」』]?\s*(?:にして|にしてください|と書いて)"),
     ["word"]),
    ("forbid", re.compile(r"[「『]?(.{1,10}?)[」』]?\s*(?:は|を)?\s*(?:使わないで|使わない|使うな|禁止|抜きで|除いて|なしで|入れないで)"),
     ["word"]),
    ("require", re.compile(r"[「『]?(.{1,10}?)[」』]?\s*(?:を|は)?\s*(?:含めて|入れて|入れる|必ず|忘れずに|必ず入れて)"), ["word"]),
    ("style", re.compile(r"(ですます|敬語|丁寧|です・ます)(?:調|体|口)?(で|に)?"), ["polite"]),
    ("style", re.compile(r"(だ・である|である|常体|タメ口|カジュアル|くだけた|親しい)(?:調|体|口)?(で|に)?"), ["plain"]),
    ("form", re.compile(r"(?:箇条書き|項目立て|リスト|bullets?|bullet ?points?)(?:で|にして)?"), ["bullets"]),
    ("form", re.compile(r"(?:番号付き|番号を振って|1. 2. の形式|ナンバリング)(?:で|にして)?"), ["numbered"]),
    ("form", re.compile(r"(?:表|table)\s*(?:形式|で|にして)"), ["table"]),
    ("form", re.compile(r"(?:一行|1 行|1行)\s*(?:で|に)?"), ["oneline"]),
    ("form", re.compile(r"(?:JSON|json)\s*(?:で|形式で)?"), ["json"]),
    ("form", re.compile(r"(?:コードブロック|コードで|コードとして)(?:書いて|出して)?"), ["code"]),
    ("charset", re.compile(r"(?:ひらがな|平仮名)\s*(?:だけ|のみ|で|にして)"), ["hiragana"]),
    ("charset", re.compile(r"(?:カタカナ|片仮名)\s*(?:だけ|のみ|で|にして)"), ["katakana"]),
    ("charset", re.compile(r"(?:漢字)\s*(?:は)?\s*(?:使わない|抜いて|なしで|禁止)"), ["no_kanji"]),
    ("charset", re.compile(r"(?:ひらがな|平仮名)\s*(?:は)?\s*(?:使わない|抜いて|なし)"), ["no_hiragana"]),
    ("lang", re.compile(r"(?:英語|日本語|中国語|韓国語|フランス語|ドイツ語|スペイン語)\s*(?:で|に)?\s*(?:訳して|翻訳して|言い換えて|書いて|教えて|出して)"),
     ["lang"]),
    ("lang", re.compile(r"(?:ローマ字|ローマ字表記|roumaji|romaji)\s*(?:で|にして)"), ["romaji"]),
)

_CHAIN_HINTS = (
    "しりとり", "語の連鎖", "しりとりする", "取りしりとり", "ケツのカ", "最後の一音",
    "最後の音", "最後の文字", "最初の文字", "前の語の最後", "前の言葉の最後", "次の言葉の頭",
    "尾首合わせ", "文末を頭", "語尾を頭",
)
_GAME_HINTS = ("しよう", "遊ぼ", "あそぼ", "やろう", "やりましょ", "しませんか", "したい",
               "つきあって", "付き合って")


def _words(text: str) -> list[str]:
    """引用符に囲まれた語・かな/漢字の塊・英数字列を「語の候補」として拾う。"""
    out: list[str] = []
    for m in re.finditer(r"[「『]([^」』\n]{1,12})[」』]", text):
        out.append(m.group(1).strip())
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_+\-.]*", text):
        out.append(m.group())
    return [x for x in out if x]


def compile_rules(text: str, *, previous_word: str = "") -> list[Rule]:
    """発話から条件を取り出す。無いなら空（＝制約なし）。"""
    t = normalize(str(text or ""))
    low = t.lower()
    rules: list[Rule] = []
    seen: set[tuple[str, object]] = set()

    def add(kind: str, value: object, raw: str, hard: bool = True) -> None:
        key = (kind, value if isinstance(value, (str, int, float, tuple)) else id(value))
        if key in seen:
            return
        seen.add(key)
        rules.append(Rule(kind=kind, value=value, raw=raw, hard=hard))

    for kind, pat, modes in _PATTERNS:
        for m in pat.finditer(t):
            raw = m.group(0)
            if modes[0] == "num":
                n = to_int(m.group(1))
                if n:
                    add(kind, n, raw)
            elif modes[0] == "short":
                add("len", 26, raw)
            elif modes[0] == "long":
                add("len", 220, raw)
            elif modes[0] == "word":
                cand = (m.group(1) if m.lastindex else "").strip("。、,.!?！？ ")
                if cand:
                    add(kind, cand, raw)
            elif modes[0] in ("polite", "plain", "bullets", "numbered", "table", "oneline",
                              "json", "code", "hiragana", "katakana", "no_kanji", "no_hiragana",
                              "lang", "romaji"):
                add(kind, modes[0], raw)

    # 語の連鎖（しりとり系）: 「語の最後の音から始める」という関係として捉える
    if any(h in low for h in _CHAIN_HINTS) or (previous_word and _is_invite(low) and _names_chain(low)):
        add("chain", to_hiragana(previous_word) if previous_word else "", raw="語の連鎖")
    return rules


def _is_invite(t: str) -> bool:
    return any(h in t for h in _GAME_HINTS)


def _names_chain(t: str) -> bool:
    return "しりとり" in t or "語" in t and any(h in t for h in ("連鎖", "頭", "最後"))


def from_definition(def_text: str, *, topic: str = "") -> list[Rule]:
    """話題の *説明文* から遊び方・手順の規則を読む（＝知識を見て動けるための入口）。

    しりとりを特別扱いするのではなく、「前の語の最後の拍を、次の語の頭にする」と
    説明されているもの全般を、同じ規則（chain）にまとめる。
    """
    t = normalize(str(def_text or "")) + " " + normalize(str(topic or ""))
    out: list[Rule] = []
    chain_reads = (
        r"(最後|末尾|尻|語尾)[の]?(?:一?音|文字|拍|句|音|ことば|語)?[^。]{0,18}(頭|始め|続き)",
        r"前(?:の)?(?:語|言葉|ことば|ワード)[^。]{0,18}(頭|始め|続き)",
        r"(頭|最初|冒頭)[の]?(?:音|文字|拍)?[^。]{0,14}(前|次)",
    )
    if "しりとり" in t or any(re.search(pat, t) for pat in chain_reads):
        out.append(Rule(kind="chain", value="", raw="定義から読んだ規則: 語の連鎖"))
    if re.search(r"(交互|順番|交代)", t):
        out.append(Rule(kind="form", value="turn-taking", raw="定義から読んだ規則: 交互に"))
    if re.search(r"(ん[で]?\s*(?:負|敗|アウト|終わ))|(最後に.{0,6}ん)", t):
        out.append(Rule(kind="forbid", value="ん", raw="定義から読んだ規則: ん で終わらない"))
    return out


# --------------------------------------------------------------------------- #
# 検証
# --------------------------------------------------------------------------- #
def check(text: str, rules: list[Rule]) -> tuple[bool, str]:
    """生成物が条件を満たしているか。1 つでも落ちて False。"""
    t = str(text or "")
    for r in rules:
        ok, why = _check_one(t, r)
        if not ok and r.hard:
            return False, f"{r.kind}:{why}"
    return True, "ok"


def _check_one(t: str, r: Rule) -> tuple[bool, str]:
    body = t.strip()
    if r.kind == "chain":
        # 規則として前語を持っているときだけ検証する（打ち手 1 語が渡された場合）。
        prev = str(r.value or "")
        if not prev:
            return True, ""
        core = re.sub(r"[。、！？!?.…]+\s*$", "", body).strip()
        if 1 <= len(core) <= 8 and not re.search(r"(が|を|に|は|の|で|と)\s*$", core):
            return (chain_ok(prev, core), "link")
        return True, ""
    if r.kind == "len":
        n = int(r.value or 0)
        if r.raw and "以内" in r.raw:
            return (char_count(body) <= n, "over") if n else (True, "")
        return (abs(char_count(body) - n) <= max(2, int(n * 0.55)), "away") if n > 2 else (True, "")
    if r.kind == "morae":
        n = int(r.value or 0)
        core = re.sub(r"[。、！？!?\s]+$", "", body)
        return (abs(mora_count(core) - n) <= 1, "morae")
    if r.kind == "start":
        w = normalize(str(r.value))
        return (normalize(body).replace("「", "").startswith(w), "head")
    if r.kind == "end":
        w = normalize(str(r.value))
        return (normalize(body).rstrip("。！？!?").endswith(w), "tail")
    if r.kind == "forbid":
        w = normalize(str(r.value))
        return (w not in normalize(body), "used")
    if r.kind == "require":
        w = normalize(str(r.value))
        return (w in normalize(body), "missing")
    if r.kind == "charset":
        v = str(r.value)
        kana_body = to_hiragana(re.sub(r"[。、！？!?「」『』\s]", "", body))
        if v == "hiragana":
            return (bool(kana_body) and is_hiragana(kana_body), "not-hira")
        if v == "katakana":
            return (is_katakana(kana_body), "not-kata")
        if v == "no_kanji":
            return (not re.search(r"[一-龯]", body), "kanji")
        if v == "no_hiragana":
            return (not re.search(r"[ぁ-ん]", body), "kana")
        return (True, "")
    if r.kind == "style":
        v = str(r.value)
        polite = len(re.findall(r"(です|ます|ません|ください|でしょう)", body))
        plain = len(re.findall(r"(だ。|である|た。|ない。|だ、)", body))
        if v == "polite":
            return (polite >= 1 and plain == 0, "style")
        if v == "plain":
            return (plain >= 1, "style")
        return (True, "")
    if r.kind == "form":
        v = str(r.value)
        if v in ("bullets", "numbered"):
            n = len([x for x in re.split(r"\n", body) if x.strip()])
            return (n >= 2, "not-list")
        if v == "oneline":
            return ("\n" not in body.rstrip(), "multiline")
        if v == "json":
            return (body.lstrip().startswith(("{", "[")), "not-json")
        if v == "code":
            return ("```" in body, "no-block")
        if v == "table":
            return ("|" in body, "no-table")
        return (True, "")
    if r.kind == "count":
        n = int(r.value or 0)
        items = [x for x in re.split(r"\n|、|,|・", body) if x.strip()]
        return (abs(len(items) - n) <= max(1, n // 2), f"count={len(items)}")
    if r.kind == "lang":
        v = str(r.value)
        if v == "romaji":
            return (not re.search(r"[ぁ-んァ-ヶ一-龯]", body), "kana-left")
        return (True, "")
    return (True, "")


# --------------------------------------------------------------------------- #
# 書式の強制（可能なものは書き換えて満たす）
# --------------------------------------------------------------------------- #
def _restyle_sentence(t: str, mode: str) -> str:
    """1 文の文体を敬体/常体へ組み替える（活用エンジンに一手任せる）。"""
    from ..lang.morph import to_plain, to_polite

    body = t.rstrip("。！？!?…")
    mark = next((c for c in reversed(t) if c in "。！？!?…"), "。")
    if not body:
        return t
    out = to_polite(body) if mode == "polite" else to_plain(body)
    out = out.rstrip("。")
    return out + (mark if mark in "！？?" else "。")


def _re_script(t: str, mode: str) -> str:
    """文字種の規則を、読み（語彙バンク）を使って満たす。"""
    from ..lang.lex import bank as _bank
    from ..lang.phonetics import kana_to_ro, to_hiragana, to_katakana

    if not t:
        return t
    try:
        seg = _bank().segment(t)
    except Exception:  # noqa: BLE001
        seg = [(t, "")]
    out: list[str] = []
    for surface, _pos in seg:
        w = _bank().entry(surface) if surface else None
        read = (w.reading if w and w.reading else to_hiragana(surface))
        if mode == "hiragana":
            out.append(to_hiragana(read) if re.match(r"[ぁ-ん]", read) else surface)
        elif mode in ("katakana", "no_kanji"):
            out.append(to_katakana(read) if re.match(r"[ぁ-ん]", read) else surface)
        elif mode == "romaji":
            out.append(kana_to_ro(to_hiragana(read)) if re.match(r"[ぁ-んァ-ヶ]", read)
                       else surface)
        elif mode == "no_hiragana":
            out.append(to_katakana(surface))
        else:
            out.append(surface)
    return "".join(out)


def _fit_length(t: str, n: int) -> str:
    """指定文字数に収まるよう、文の区切りから短くする（語を足して盛らない）。"""
    body = t.strip()
    if n <= 0 or char_count(body) <= n + max(2, int(n * 0.55)):
        return body
    # 文末の句读点で切ってから、それでも長ければ記号跨ぎで切る
    for cut in [s for s in re.split(r"(?<=[。！？!?])", body) if s.strip()]:
        if char_count(cut) >= n:
            body = cut
            break
    if char_count(body) > n:
        head = body[:n]
        # 助詞で切れた語はそこごと落とす（「猫が」→「猫」）
        head = re.sub(r"(が|を|に|は|の|で|と|も|へ)\s*$", "", head)
        body = head
    body = body.strip("、,。・ ")
    return (body + "。") if body and not body.endswith(("。", "！", "？")) else body


def enforce(text: str, rules: list[Rule], *, kb_split: bool = True) -> str:
    """スタイル・形式・字数の規則を本文に適用する。

    ここは「規則を検証するだけ」で終わらせないための層で、書き換え可能なものは
    実際に組み替える（文体・文字種・箇条書き・字数）。語の内容そのものは
    絶対に足さない — 盛って字数を合わせるのは捏造だから。
    """
    out = str(text or "").strip()
    for r in rules:
        if r.kind == "style" and r.value in ("polite", "plain"):
            out = "".join(_restyle_sentence(s, str(r.value))
                          for s in re.split(r"(?<=[。！？!?])", out) if s.strip())
        elif r.kind == "form" and r.value in ("bullets", "numbered"):
            out = _to_list(out, numbered=r.value == "numbered")
        elif r.kind == "form" and r.value == "oneline":
            out = re.sub(r"\s*\n\s*", " ", out)
        elif r.kind == "charset" and r.value in ("hiragana", "katakana", "no_kanji",
                                                 "no_hiragana"):
            out = _re_script(out, str(r.value))
        elif r.kind == "lang" and r.value == "romaji":
            out = _re_script(out, "romaji")
        elif r.kind == "len":
            out = _fit_length(out, int(r.value or 0))
    return out.strip()


def _to_list(t: str, *, numbered: bool = False) -> str:
    parts = [p.strip(" 。") for p in re.split(r"(?<=[。])\s*", t) if p.strip(" 。")]
    if len(parts) < 2:
        parts = [p.strip() for p in re.split(r"[、,・]", t) if p.strip()]
    if len(parts) < 2:
        return t
    lines = []
    for i, p in enumerate(parts[:6], 1):
        lines.append(f"{i}. {p}。" if numbered else f"・{p}。")
    return "\n".join(lines)


def chain_ok(prev_word: str, candidate: str) -> bool:
    """語の連鎖の成立判定。

    前語の **送り音（末尾の拍）** と、次の語の **頭（最初の拍）** を比べます。
    長音・促音は実質の音に直す（コーヒー → ひ、キツネ ← ご なら不一致）。
    拗音は親字まで許す（「きょ」には「きょ」も「き」も来る）、清濁は厳密に見る。
    """
    if not prev_word or not candidate:
        return False
    a = to_hiragana(chain_key(prev_word))
    b = to_hiragana(chain_start(candidate))
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) == 2 and b == a[0]:          # きゃ ← き（親字で受ける）
        return True
    if len(b) == 2 and a == b[0]:          # き ← きゃ
        return True
    return False


__all__ = ["compile_rules", "from_definition", "check", "enforce", "chain_ok", "to_int"]
