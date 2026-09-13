"""テキスト・言葉の一般操作 — 数える・変える・調べる・並べる。

「文字数を数えて」「ローマ字にして」「頻度を出して」「逆から」「しりとりで〇で始まる語」
「読みは」「品詞は」「活用を見せて」… 言葉に関する仕事は、タスク分岐を作らず
この 1 本の入口で受ける。辞書引きは `snipher.lang`（13 万語の実語彙）を使う。
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections import Counter

from ..lang import lex, morph
from ..lang.phonetics import (
    char_count,
    kana_to_ro,
    mora_count,
    mora_split,
    normalize,
    ro_to_kana,
    to_hiragana,
    to_katakana,
)
from .math import Solution


def quoted_terms(text: str) -> list[str]:
    out = [m.group(1).strip() for m in re.finditer(r"[「『]([^」』\n]{1,24})[」』]", text)]
    return [x for x in out if x]


_INSTRUCTION_TAIL = re.compile(
    r"(?:を|は|に|の|をそれぞれ)?\s*(?:大文字|小文字|全角|半角|キャメルケース|スネークケース|ケバブケース|"
    r"スラッグ|ソート|並び替え(?:て)?|整列(?:して)?|重複を(?:削除|取り)?|ベース64|base64|Base64|16進|"
    r"シーザー暗号|rot1?3|ローマ字|かな|ひらがな|カタカナ|頻度|回数|出現回数|文字数|語数|行数|逆(?:から)?|"
    r"リバース|reverse|変換(?:して)?|CSV|json|JSON|表(?:形式)?|読み方?|品詞|活用形?|拍数|何文字|何音|"
    r"を含む[^。]*|が始まる[^。]*|で始まる[^。]*|で終わる[^。]*|を教えて|ください|して|くださいな)\s*"
    r"(?:に|で|して|くれて|くれますか|もらえますか|できますか|お願い|教えて)?[^。]*[。？?！!]?\s*$")


def _subject(text: str) -> str:
    """操作対象の語/文を取り出す（引用 → 指示語を落とした前部 → 全文の順）。"""
    t = normalize(text)
    q = quoted_terms(t)
    if q:
        return q[0]
    m = re.search(r"[「『](.{1,40}?)[」』]", t)
    if m:
        return m.group(1)
    m = re.match(r"^(.{1,40}?)(?:を|は|の)?\s*(?:何文字|ローマ字|逆|大文字|小文字|全角|半角|ソート|"
                 r"頻度|数えて|変換)", t)
    base = m.group(1) if m else t
    # 指示語（「〜にして」「〜を教えて」）を落としてから、対象だけを残す
    base = _INSTRUCTION_TAIL.sub("", base).strip("、。 ：:・ ")
    base = re.sub(r"^(?:次の文|以下の文|この文|この言葉を?|文を)\s*", "", base).strip("、。 ")
    return (base or t).strip("。？！?! ")


# --------------------------------------------------------------------------- #
# 個別の操作
# --------------------------------------------------------------------------- #
def count_words(text: str) -> Solution:
    t = _subject(text) or normalize(text)
    kana_words = [w for w, _ in lex.bank().segment(t)]
    latin = re.findall(r"[A-Za-z']+", t)
    lines = [x for x in re.split(r"\n", t) if x.strip()]
    steps = [f"文字数 {char_count(t)}（空白含む）、うちかな・漢字の語 {len(kana_words)} 語、"
             f"欧文の語 {len(latin)} 語、行数 {len(lines)}"]
    return Solution(answer=f"{len(kana_words)} 語（欧文 {len(latin)} 語）", steps=steps, kind="count",
                    detail={"chars": char_count(t), "words": len(kana_words), "latin": len(latin),
                            "lines": len(lines)})


def char_stats(text: str) -> Solution:
    t = _subject(text)
    kana = to_hiragana(t)
    kinds = Counter()
    for ch in t:
        if ch.isspace():
            kinds["空白"] += 1
        elif "ぁ" <= ch <= "ん":
            kinds["ひらがな"] += 1
        elif "ァ" <= ch <= "ヶ":
            kinds["カタカナ"] += 1
        elif "一" <= ch <= "鿿":
            kinds["漢字"] += 1
        elif ch.isdigit():
            kinds["数字"] += 1
        elif ch.isascii() and ch.isalpha():
            kinds["欧文"] += 1
        else:
            kinds["記号"] += 1
    body = "、".join(f"{k} {v}" for k, v in kinds.most_common())
    return Solution(answer=f"{char_count(t)} 文字（{body}）",
                    steps=[f"拍数は {mora_count(kana)} 拍: {'・'.join(mora_split(kana)[:14])}"],
                    kind="count", detail={"chars": char_count(t), "morae": mora_count(kana),
                                         "kinds": dict(kinds)})


def frequency(text: str, *, top: int = 8) -> Solution:
    body = _INSTRUCTION_TAIL.sub("", normalize(text)).strip("、。 ")
    body = body or normalize(text)
    words = [w for w, pos in lex.bank().segment(body) if pos.split("/")[0] in {"名詞", "動詞", "形容詞"}]
    counts = Counter(words).most_common(top)
    if not counts:
        counts = Counter(re.findall(r"[A-Za-z]+", body.lower())).most_common(top)
    answer = "、".join(f"{w} {c}" for w, c in counts)
    return Solution(answer=answer or "数えられる語がありません",
                    steps=[f"語に割って数えました（{len(words)} 語）"], kind="frequency",
                    detail={"counts": dict(counts)})


def transliterate(text: str) -> Solution | None:
    t = _subject(text)
    if not t:
        return None
    low = normalize(text).lower()
    if any(k in low for k in ("ローマ字", "roumaji", "romaji", "ラテン文字")) and \
            not re.search(r"(大文字|小文字)", low):
        kana = to_hiragana(t)
        return Solution(answer=kana_to_ro(kana), steps=[f"「{t}」→ 拍 {'・'.join(mora_split(kana))}"],
                        kind="transliterate")
    latin_only = bool(re.fullmatch(r"[a-z'\- ]+", t))
    if latin_only or any(k in low for k in ("かなにして", "ひらがなに", "カタカナにして", "かな表記")):
        kana = ro_to_kana(t)
        if any(c.isdigit() or c.isalpha() and c.isascii() for c in kana):
            return None
        if "カタカナ" in low:
            kana = to_katakana(kana)
        return Solution(answer=kana, steps=[f"{t} → {kana}（{mora_count(kana)} 拍）"],
                        kind="transliterate")
    if any(k in low for k in ("全角", "半角")):
        conv = "NFKC" if "全角" in low else "NFKC"
        import unicodedata

        out = unicodedata.normalize(conv, t)
        if "半角" in low:
            out = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in out)
        return Solution(answer=out, steps=["全角/半角を揃えました"], kind="transform")
    return None


def reverse_text(text: str) -> Solution | None:
    if not re.search(r"(逆から|逆に|反転|リバース|reverse)", normalize(text)):
        return None
    t = _subject(text)
    return Solution(answer=t[::-1], steps=[f"{len(t)} 文字を後ろから並びました"], kind="transform")


def case_convert(text: str) -> Solution | None:
    low = normalize(text).lower()
    t = _subject(text)
    if not re.search(r"[A-Za-z]", t):
        return None
    if "大文字" in low or "upper" in low:
        return Solution(answer=t.upper(), steps=["英字を大文字にしました"], kind="transform")
    if "小文字" in low or "lower" in low:
        return Solution(answer=t.lower(), steps=["英字を小文字にしました"], kind="transform")
    if "キャメル" in low or "camelCase" in low:
        parts = re.split(r"[\s_\-]+", t.lower())
        out = parts[0] + "".join(x.capitalize() for x in parts[1:])
        return Solution(answer=out, steps=["語をSplitして 2 つ目以降を大文字開始に"], kind="transform")
    if "スネーク" in low or "snake_case" in low:
        out = re.sub(r"(?<!^)(?=[A-Z])", "_", t).lower().replace(" ", "_").replace("-", "_")
        return Solution(answer=out, steps=["大文字の前に _ を入れて小文字化"], kind="transform")
    if "ケバブ" in low or "kebab" in low or "スラッグ" in low or "slug" in low:
        out = re.sub(r"[\s_]+", "-", t.lower())
        out = re.sub(r"[^a-z0-9\-]", "", out).strip("-")
        return Solution(answer=out, steps=["英数と - だけに残して小文字化"], kind="transform")
    return None


def sort_dedupe(text: str) -> Solution | None:
    low = normalize(text)
    if not re.search(r"(並び替|整列|ソート|sort|重複を|ユニーク|dedupe|uniq)", low):
        return None
    t = _subject(text)
    items = [x.strip() for x in re.split(r"[,、\n;；|・\s]+", t) if x.strip()]
    if not items:
        return None
    numbered = all(re.fullmatch(r"-?\d+(\.\d+)?", x) for x in items)
    if "重複" in low or "ユニーク" in low or "dedupe" in low.lower():
        out = list(dict.fromkeys(items))
        return Solution(answer="、".join(out),
                        steps=[f"{len(items)} 件から {len(out)} 件へ（{len(items) - len(out)} 件が重複）"],
                        kind="dedupe")
    reverse = bool(re.search(r"(降順|大きく|後ろから|reverse)", low))
    key = (lambda x: float(x)) if numbered else None
    out = sorted(items, key=key, reverse=reverse)
    return Solution(answer="、".join(out),
                    steps=[f"{len(items)} 件を{'数値の降順' if reverse and numbered else '数値の昇順' if numbered else '文字順'}でならべました"],
                    kind="sort")


def encode_decode(text: str) -> Solution | None:
    low = normalize(text).lower()
    t = _subject(text)
    if "base64" in low or "ｂａｓｅ64" in low:
        if "デコード" in low or "戻して" in low or "復号" in low:
            try:
                return Solution(answer=base64.b64decode(t + "=" * (-len(t) % 4)).decode("utf-8"),
                                steps=["base64 から文字列に戻しました"], kind="decode")
            except (binascii.Error, UnicodeDecodeError):
                return None
        raw = t.encode("utf-8")
        return Solution(answer=base64.b64encode(raw).decode("ascii"), steps=["UTF-8 → base64"],
                        kind="encode")
    if "16進" in low or "hex" in low:
        if "に戻して" in low or "デコード" in low:
            try:
                return Solution(answer=bytes.fromhex(re.sub(r"\s", "", t)).decode("utf-8", "replace"),
                                steps=["16 進 → UTF-8"], kind="decode")
            except ValueError:
                return None
        return Solution(answer=t.encode("utf-8").hex(), steps=["UTF-8 → 16 進"], kind="encode")
    if "rot" in low or "シーザー" in low:
        n = re.search(r"rot\s*(\d+)", low)
        k = int(n.group(1)) if n else 13
        out = "".join(
            chr((ord(c) - (97 if c.islower() else 65) + k) % 26 + (97 if c.islower() else 65))
            if c.isalpha() and c.isascii() else c for c in t)
        return Solution(answer=out, steps=[f"アルファベットを {k} 文字ずつずらしました"], kind="cipher")
    return None


def json_csv(text: str) -> Solution | None:
    """JSON → 表 / 表 → JSON など、構造化された変換。"""
    low = normalize(text).lower()
    t = _subject(text)
    body = text
    if "csv" in low and ("json" in low):
        if "csv" in low.split("json")[0]:        # JSON → CSV
            data = _load_json(body)
            if data:
                keys = list({k for row in data for k in row})
                lines = [",".join(keys)] + [",".join(str(r.get(k, "")) for k in keys) for r in data]
                return Solution(answer="\n".join(lines), steps=[f"{len(data)} 行を変換"],
                                kind="convert")
        data = _parse_csv(body)
        if data:
            return Solution(answer=json.dumps(data, ensure_ascii=False, indent=2),
                            steps=[f"{len(data)} 行を変換"], kind="convert")
    if "markdown" in low or "table" in low or "表に" in low:
        data = _load_json(body)
        if data:
            keys = list({k for row in data for k in row})
            head = "| " + " | ".join(keys) + " |"
            sep = "|" + "|".join(["---"] * len(keys)) + "|"
            rows = ["| " + " | ".join(str(r.get(k, "")) for k in keys) + " |" for r in data]
            return Solution(answer="\n".join([head, sep] + rows), steps=["表組みにしました"],
                            kind="convert")
    return None


def _load_json(body: str) -> list[dict]:
    m = re.search(r"[\[{].*[\]}]", body, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return []
    if isinstance(data, dict):
        return [data]
    return [x for x in data if isinstance(x, dict)]


def _parse_csv(body: str) -> list[dict]:
    lines = [ln for ln in body.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    keys = [k.strip() for k in re.split(r"[,、\t]", lines[0])]
    out: list[dict] = []
    for ln in lines[1:]:
        vals = [v.strip() for v in re.split(r"[,、\t]", ln)]
        out.append({k: (vals[i] if i < len(vals) else "") for i, k in enumerate(keys)})
    return out


# --------------------------------------------------------------------------- #
# 辞書的な照会（読み・拍・品詞・活用・語列表現）
# --------------------------------------------------------------------------- #
def dictionary_query(text: str) -> Solution | None:
    t = _subject(text)
    low = normalize(text).lower()
    bank = lex.bank()
    ent = bank.entry(t)
    if ent is None and len(t) > 1:
        # 「三毛猫とは」の主語部分など、末尾の助詞を落として再試行
        for cut in ("とは", "って何", "の読み", "の意味", "を", "は", "が", "の"):
            if t.endswith(cut):
                ent = bank.entry(t[: -len(cut)])
                t = t[: -len(cut)]
                break
    if ent is None:
        return None
    if any(k in low for k in ("読み", "ふりがな", "何と読む", "なんて読む", "よみ")):
        return Solution(answer=f"「{ent.surface}」は「{ent.reading or to_hiragana(t)}」と読みます",
                        steps=[f"品詞 {ent.pos or '不明'}・{mora_count(ent.reading or t)} 拍"],
                        kind="lexicon", detail={"reading": ent.reading, "pos": ent.pos})
    if any(k in low for k in ("何文字", "何音", "拍", "文字数", "morae")):
        rd = ent.reading or to_hiragana(ent.surface)
        return Solution(answer=f"{char_count(ent.surface)} 文字・{mora_count(rd)} 拍",
                        steps=[f"読み「{rd}」を拍に割ると {'・'.join(mora_split(rd))}"],
                        kind="lexicon", detail={"chars": char_count(ent.surface),
                                               "morae": mora_count(rd)})
    if any(k in low for k in ("ローマ字", "romaji")):
        return Solution(answer=kana_to_ro(ent.reading or to_hiragana(ent.surface)),
                        steps=[f"読み {ent.reading or to_hiragana(ent.surface)} から"],
                        kind="lexicon")
    if any(k in low for k in ("活用", "活用形", "動詞の", "変形")):
        forms = {}
        for f in ("dictionary", "masu", "mashita", "te", "ta", "nai", "nakatta",
                  "conditional", "imperative", "potential", "volitional"):
            try:
                forms[f] = morph.inflect(ent.dictform or ent.surface, f)
            except Exception:  # noqa: BLE001
                continue
        body = "／".join(f"{k}:{v}" for k, v in forms.items() if v)
        return Solution(answer=body, steps=[f"活用型 {morph.ctype_of(ent.dictform) or '不明'}"],
                        kind="lexicon", detail={"forms": forms})
    if any(k in low for k in ("品詞", "pos", "動詞？", "名詞？")):
        return Solution(answer=f"{ent.pos or '不明'}", steps=[f"見出し {ent.surface}（読み {ent.reading}）"],
                        kind="lexicon")
    return None


_EN_HINT = ("英単語", "えいたんご", "english word", "words", "word", "アルファベット")


def word_length_query(text: str) -> Solution | None:
    """「N 文字 / N 音の語を M 個」— 欧文も和語も、辞書を実際に引いて出す。"""
    low = normalize(text).lower()
    m_len = re.search(r"(\d+)\s*(?:文字|字|音|拍|letters?|レター)", low)
    m_cnt = re.search(r"(\d+)\s*(?:個|つ|件|words?)", low)
    asks_word = re.search(r"(単語|ことば|言葉|ワード|名詞|英単語|words?\b|word\b)", low)
    if not (m_len and asks_word) and not (m_cnt and asks_word):
        return None
    n = int(m_len.group(1)) if m_len else 0
    cnt = max(1, min(12, int(m_cnt.group(1)) if m_cnt else 5))
    want_en = any(h in low for h in _EN_HINT)
    bank = lex.bank()
    if want_en and n:
        got = lex.english_by_length(n, limit=cnt)
        if got:
            return Solution(answer="、".join(got),
                            steps=[f"欧文語彙 {len(lex.english_words()):,} 語から {n} 文字の語を "
                                   f"{len(got)} 個引き出しました"],
                            kind="wordlist", detail={"words": got, "language": "en"})
    if want_en and not n:
        got = [w for w in lex.english_words()[:20000] if cnt - 1 <= len(w) <= cnt + 1][:cnt]
        if got:
            return Solution(answer="、".join(got), steps=["欧文語彙から語数で引きました"],
                            kind="wordlist", detail={"words": got, "language": "en"})
    if n:
        got = bank.by_morae(n, limit=cnt * 4)[:cnt]
        if got:
            return Solution(answer="、".join(w.surface for w in got),
                            steps=[f"語彙バンク {len(bank):,} 語から {n} 拍の語を {len(got)} 個選びました"],
                            kind="wordlist", detail={"words": [w.surface for w in got]})
    return None


def word_chain_query(text: str) -> Solution | None:
    """「A で始まる語」「A で終わる語」「O 音の名詞」など、語の在庫から出す仕事。"""
    low = normalize(text)
    bank = lex.bank()
    m = re.search(r"[「『]?(.{1,8}?)[」』]?\s*で始まる(?:ことば|語|単語|ワード)?", low)
    if m:
        head = m.group(1).strip("。、 ")
        got = bank.by_reading_prefix(to_hiragana(head), limit=8)
        if got:
            return Solution(answer="、".join(w.surface for w in got),
                            steps=[f"読み {head} 始まりで {len(got)} 語"], kind="wordlist",
                            detail={"words": [w.surface for w in got]})
    m = re.search(r"[「『]?(.{1,8}?)[」』]?\s*(?:で終わる|語尾が|終わりの語)", low)
    if m:
        tail = m.group(1).strip("。、 ")
        got = bank.by_reading_suffix(to_hiragana(tail), limit=8)
        if got:
            return Solution(answer="、".join(w.surface for w in got),
                            steps=[f"読み {tail} 終わりで {len(got)} 語"], kind="wordlist",
                            detail={"words": [w.surface for w in got]})
    m = re.search(r"(\d+)\s*(?:音|拍|モーラ)\s*(?:の|以内)[^。]*(?:名詞|ことば|語|単語)", low)
    if m:
        n = int(m.group(1))
        got = bank.by_morae(n, limit=10)
        if got:
            return Solution(answer="、".join(w.surface for w in got),
                            steps=[f"{n} 音の名詞を {len(got)} 語"], kind="wordlist",
                            detail={"words": [w.surface for w in got]})
    m = re.search(r"[「『](.{1,10})[」』]\s*(?:を含む|に入っている|部分に持つ|が使われた)", low)
    if m:
        got = bank.containing(m.group(1), limit=10)
        if got:
            return Solution(answer="、".join(w.surface for w in got),
                            steps=[f"語彙バンクから {len(got)} 語"], kind="wordlist",
                            detail={"words": [w.surface for w in got]})
    return None


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
# 対象を絞った処理（辞書・語彙）→ 明示的な指示語が要る変換 → 汎用変換 の順。
# transliterate を case_convert より後ろに置くと「hello を大文字に」がkana化してしまいます。
HANDLERS = (dictionary_query, word_length_query, word_chain_query, case_convert, sort_dedupe,
            encode_decode, json_csv, reverse_text, transliterate)


def handle(text: str) -> Solution | None:
    low = normalize(text).lower()
    if re.search(r"(頻度|回数|よく使う語)", low):
        return frequency(text)
    if re.search(r"(文字数を?数えて|語数|何語|いくつに数える|行数)", low) and "何文字" not in low:
        return count_words(text)
    if re.search(r"(どんな文字|文字の種類|内訳)", low):
        return char_stats(text)
    for fn in HANDLERS:
        try:
            sol = fn(text)
        except Exception:  # noqa: BLE001
            sol = None
        if sol is not None:
            return sol
    if re.search(r"(しりとり|語の連鎖|で始まる|で終わる|音の|韻)", low):
        sol = word_chain_query(text)
        if sol is not None:
            return sol
    return None


__all__ = ["handle", "count_words", "char_stats", "frequency", "transliterate", "reverse_text",
           "case_convert", "sort_dedupe", "encode_decode", "json_csv", "dictionary_query",
           "word_chain_query", "quoted_terms"]
