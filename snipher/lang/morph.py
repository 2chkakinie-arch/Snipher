"""形態論（活用エンジン）— 語を組み替えて書き出す部品。

相手の言いかけた動詞を別の活用に組み替える（「食べた」→「食べてみてください」）、
活用形から辞書形に戻して語を引く、文末の文体ぞろえ — すべてここを通す。

活用語幹は `data/conjugation.bin.gz`（`tools/build_wordbank.py` が辞書から抽出した実データ）を
先に使い、辞書に無い語だけ音韻ルールで縮退する。形容詞・名詞（断定に落とす）も同じ入口。

    inflect("書く", "te")     → 書いて
    inflect("書いた", "nai")  → 書かない
    inflect("高い", "ta")     → 高かった
    inflect("猫", "masu")     → 猫です
    lemma_of("走り")          → 走る
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

from .phonetics import normalize, to_hiragana

DATA = Path(__file__).resolve().parents[1] / "data"

_ROWS = {"う": "あいうえお", "く": "かきくけこ", "ぐ": "がぎぐげご", "す": "さしすせそ",
         "ず": "ざじずぜぞ", "つ": "たちつてと", "ぬ": "なにぬねの", "ぶ": "ばびぶべぼ",
         "む": "まみむめも", "る": "らりるれろ"}
_TE = {"う": "って", "く": "いて", "ぐ": "いで", "す": "して", "つ": "って", "ぬ": "んで",
       "ぶ": "んで", "む": "んで", "る": "って", "ず": "じで"}
_NAI = {"う": "わ", "く": "か", "ぐ": "が", "す": "さ", "つ": "た", "ぬ": "な", "ぶ": "ば",
        "む": "ま", "る": "ら", "ず": "ざ"}
# 連用タ接続の末尾 → 辞書形語尾（逆写しの候補）
_FROM_TA = {"い": ("う", "く", "ぐ"), "し": ("す",), "っ": ("つ", "る"), "ん": ("ぬ", "ぶ", "む"),
            "じ": ("ず",)}
_FROM_NAI = {v: k for k, v in _NAI.items()}

_TABLE: dict[str, dict[str, str]] | None = None
_TYPE: dict[str, str] = {}


def table() -> dict[str, dict[str, str]]:
    """活用パラダイム表（辞書形 → {活用形ラベル: 表記}）。遅延ロードで 1 回だけ読む。"""
    global _TABLE, _TYPE
    if _TABLE is not None:
        return _TABLE
    out: dict[str, dict[str, str]] = {}
    path = DATA / "conjugation.bin.gz"
    if path.exists():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    lemma, ctype, pairs = parts[0], parts[1], parts[2]
                    forms: dict[str, str] = {}
                    for pair in pairs.split(";"):
                        if "=" not in pair:
                            continue
                        f, s = pair.split("=", 1)
                        if f and s:
                            forms[f] = s
                    if not forms:
                        continue
                    cur = out.get(lemma)
                    if cur is None:
                        out[lemma] = forms
                        _TYPE[lemma] = ctype
                        continue
                    # 同じ辞書形に複数の活用型がある（五段の音便の別立てなど）。
                    # 「ウ音便」は文語的な並びなので、现代形（促音便・イ音便・一段）を優先する。
                    if "ウ音便" in _TYPE[lemma] and "ウ音便" not in ctype:
                        out[lemma] = forms
                        _TYPE[lemma] = ctype
        except Exception:  # noqa: BLE001
            out = {}
    _TABLE = out
    return out


def _forms(lemma: str) -> dict[str, str]:
    return table().get(lemma) or {}


def ctype_of(lemma: str) -> str:
    """活用型ラベル（"五段・カ行イ音便" など）。無ければ ""。"""
    table()
    k = to_hiragana(normalize(lemma))
    return _TYPE.get(normalize(lemma)) or _TYPE.get(k, "")


def is_ichidan(lemma: str) -> bool:
    ct = ctype_of(lemma)
    if ct:
        return "一段" in ct or "変格" in ct
    s = to_hiragana(normalize(lemma))
    return len(s) >= 3 and s.endswith("る") and s[-2] in "いえ"


def is_adjective(lemma: str) -> bool:
    ct = ctype_of(lemma)
    if ct:
        return "形容詞" in ct
    k = to_hiragana(normalize(lemma))
    return len(k) >= 3 and k.endswith("い") and k[-2] not in "ゃゅょ" and not k.endswith(("るい",))


def detect_pos(surface: str) -> str:
    """活用語尾からの品詞推定（辞書に無い語にも効く）。"""
    k = to_hiragana(normalize(surface))
    if not k:
        return ""
    if is_adjective(k):
        return "形容詞"
    if k.endswith("る") and len(k) >= 2:
        return "動詞/一段" if is_ichidan(k) else "動詞/五段"
    if len(k) >= 2 and k[-1] in _ROWS:
        return "動詞/五段"
    return ""


_BACK = {
    "い": ("う", "く", "ぐ"), "き": ("く",), "ぎ": ("ぐ",), "し": ("す",), "ち": ("つ",),
    "に": ("ぬ",), "び": ("ぶ",), "み": ("む",), "り": ("る",), "っ": ("う", "つ", "る"),
    "ん": ("む", "ぬ", "ぶ"), "じ": ("ず",), "ぢ": ("づ",), "づ": ("づく",),
}
_DICT_END = tuple("いうくぐすつぬぶむる")


def _looks_dict_form(k: str) -> bool:
    return bool(k) and (k.endswith("る") or (k[-1] in "いうくぐすつぬぶむ" and len(k) >= 2))


def lemma_of(surface: str) -> str:
    """活用形 → 辞書形。語彙バンクの辞書形欄を先に使い、無ければ語尾から復元する。"""
    raw = normalize(surface)
    if not raw:
        return ""
    try:
        from .lex import bank

        w = bank().entry(raw)
        if w is not None and w.pos.startswith(("動詞", "形容詞")) and w.dictform:
            return w.dictform
    except Exception:  # noqa: BLE001
        pass
    k = to_hiragana(raw)
    for tail in ("ました", "ません", "なさい", "たい", "ます", "でした", "です", "だった", "だ"):
        if k.endswith(tail) and len(k) > len(tail) + 1:
            k = k[: -len(tail)]
            break
    k = k.rstrip("。、")
    if not k:
        return raw
    if k in table() or raw in table():
        return raw if raw in table() else k
    if k.endswith("ない") and len(k) >= 4:
        # 「食べない・合わない」は語尾だけ見ると形容詞に擬ける。動詞として
        # 引けるかを先に確かめて、そちらが通ればこちらを採用する。
        hit = _verify_lemma(_euphony_lemmas(k), k[:-2], raw)
        if hit:
            return hit
    if is_adjective(k):
        return k
    # サ変・カ変（勉強した → 勉強する / 来た → 来る）: 語幹 + する が辞書にあればそれ。
    base = k[:-1] if (len(k) >= 3 and k[-1] in "たて") else k
    try:
        from .lex import bank as _bank

        def _known(cand: str) -> bool:
            return cand in table() or _bank().has(cand)
    except Exception:  # noqa: BLE001
        def _known(cand: str) -> bool:
            return cand in table()
    if len(base) >= 2 and base.endswith("し"):
        stem = base[:-1]
        for cand in (stem + "する", stem + "為る"):
            if _known(cand) and _fits(cand, raw):
                return cand
        # 辞書に見出しの無いサ変でも、「名詞/サ変接続」なら する を attach してよい
        try:
            spos = _bank().pos(stem)
        except Exception:  # noqa: BLE001
            spos = ""
        if spos.startswith("名詞") and "サ変" in spos and _fits(stem + "する", raw):
            return stem + "する"
    if len(base) >= 2 and base.endswith("き"):
        for cand in (base[:-1] + "来る", base[:-1] + "くる", "来る"):
            if _known(cand) and _fits(cand, raw):
                return cand
    cands: list[str] = _euphony_lemmas(k)          # 音便（行った・食べない・寒かった）
    if len(k) >= 3 and k[-1] in "たて":
        k = k[:-1]
    if k.endswith("な") and len(k) >= 3:
        cands.append(k[:-1] + "る")
    cands.append(k + "る")                       # 一段・る 動詞の連用形 → 辞書形
    if k and k[-1] in _BACK:
        cands += [k[:-1] + x for x in _BACK[k[-1]]]
    if _looks_dict_form(k):
        cands.insert(0, k)
    hit = _verify_lemma(cands, k, raw)
    if hit:
        return hit
    if _looks_dict_form(k):
        return k
    return raw


# 五段活用の「音便」で消える語尾 → 辞書形の語尾の候補
_EUPHONY = {"っ": ("く", "ぐ", "う", "つ", "る", "す"), "ん": ("む", "ぬ", "ぶ"),
            "じ": ("ず",)}
# 未然形（〜ない）の語尾 → 辞書形
_MIZEN = {"あ": "う", "か": "く", "が": "ぐ", "さ": "す", "ざ": "ず", "た": "つ", "だ": "ず",
          "な": "ぬ", "ば": "ぶ", "ま": "む", "ら": "る", "わ": "う", "ぬ": "ぬ", "ぶ": "ぶ"}
# 連用形（〜た／〜て）の語尾 → 辞書形（音便で消えないもの）
_RENYOU = {"き": "く", "ぎ": "ぐ", "し": "す", "ち": "つ", "じ": "ず", "に": "ぬ",
           "ひ": "ふ", "び": "ぶ", "み": "む", "り": "る", "い": "う", "ぜ": "ず",
           "で": "る", "て": "る"}


def _mizen_lemmas(stem: str) -> list[str]:
    out = []
    if stem and stem[-1] in _MIZEN:
        out.append(stem[:-1] + _MIZEN[stem[-1]])
    return out


def _renyou_lemmas(stem: str) -> list[str]:
    out = []
    if stem and len(stem) >= 2 and stem[-1] in _RENYOU:
        out.append(stem[:-1] + _RENYOU[stem[-1]])
    return out


def _bank_pos(surface: str) -> str:
    try:
        from .lex import bank as _bank

        return str(_bank().pos(surface) or "")
    except Exception:  # noqa: BLE001
        return ""


def _euphony_lemmas(k: str) -> list[str]:
    """「行った・食べなかった・寒かった」のような音便込みの形から辞書形候補を作る。

    語尾の置換だけだと誤りが出るので、ここでは *候補を出すだけ* にして、
    実際にその活用形を再生できるかの確認は `_fits`（＝`inflect` で戻して一致を見る）に譲る。
    """
    out: list[str] = []
    if len(k) < 3:
        return out
    for tail, kind in (("なかった", "mizen"), ("なくて", "mizen"), ("ない", "mizen"),
                       ("かった", "adj"), ("くない", "adj"), ("くて", "adj"),
                       ("た", "ta"), ("て", "te")):
        if not k.endswith(tail) or len(k) <= len(tail):
            continue
        stem = k[: len(k) - len(tail)]
        if kind == "adj":                                   # 寒かっ / 寒く
            out.append((stem[:-1] if stem.endswith("か") else stem) + "い")
        elif kind == "mizen":                               # 食べ[な]い / 行か[な]い
            out += _mizen_lemmas(stem)
            out.append(stem + "る")                          # 下一段・カ変
            if stem.endswith("し"):
                out.append(stem[:-1] + "する")               # サ変
        else:                                               # 食べ[た] / 話[し]た / 行[っ]た
            if stem and stem[-1] in _EUPHONY:                # 促音便・撥音便
                out += [stem[:-1] + x for x in _EUPHONY[stem[-1]]]
            out += _renyou_lemmas(stem)
            out.append(stem + "る")
            if stem.endswith("し"):
                out.append(stem + "する")                    # サ変（勉強した）
        break
    return list(dict.fromkeys(x for x in out if x and len(x) >= 2))


def _verify_lemma(cands: list[str], kana_stem: str, surface: str = "") -> str:
    """候補の活用形が本当に辞書にあるか（品詞＋成本）で選ぶ。

    `surface` を渡すと、その表記を本当に再生成できる候補だけを残す。
    「行った」に 行く と 行う の両方が引けるような二重解は、成本（使用頻度）で選ぶ。
    """
    try:
        from .lex import bank

        b = bank()
        scored: list[tuple[int, str]] = []
        for cand in dict.fromkeys(cands):
            w = b.entry(cand)
            if w is not None and w.pos.startswith(("動詞", "形容詞")):
                if surface and not _fits(cand, surface):
                    continue
                scored.append((w.cost, cand))
        scored.sort()
        if scored:
            return scored[0][1]
    except Exception:  # noqa: BLE001
        pass
    for cand in cands:
        if cand in table():
            return cand
    return ""


def _adj_infix(k: str, form: str) -> str:
    head = k[:-1] if k.endswith("い") else k
    table_ = {
        "dictionary": k, "renyou": head + "く", "stem": head + "く", "te": head + "くて",
        "adverb": head + "く", "adv": head + "く", "mizen": head + "くな",
        "ta": head + "かった", "mashita": head + "かったです", "masu": head + "いです",
        "masen": head + "くないです", "nai": head + "くない", "nakatta": head + "くなかった",
        "conditional": head + "ければ", "imperative": head + "かれ",
        "volitional": head + "かろう", "sou": head + "かろう", "potential": head + "い",
        "rentai": head + "き",
    }
    return table_.get(form, k)


def _sahen(head: str, form: str) -> str:
    """サ変（〜する）の活用。head は語幹（「追加」など）。"""
    table_ = {"dictionary": head + "する", "renyou": head + "し", "stem": head + "し",
              "masu": head + "します", "mashita": head + "しました", "masen": head + "しません",
              "te": head + "して", "ta": head + "した", "nai": head + "しない",
              "nakatta": head + "しなかった", "conditional": head + "すれば",
              "imperative": head + "しろ", "potential": head + "できる",
              "volitional": head + "しよう", "sou": head + "しよう", "rentai": head + "する"}
    return table_.get(form, head + "する")


def _kahen(head: str, form: str) -> str:
    """カ変（〜来る）の活用。"""
    table_ = {"dictionary": head + "来る", "renyou": head + "来", "stem": head + "来",
              "masu": head + "来ます", "mashita": head + "来ました", "masen": head + "来ません",
              "te": head + "来て", "ta": head + "来た", "nai": head + "来ない",
              "nakatta": head + "来なかった", "conditional": head + "来れば",
              "imperative": head + "来い", "potential": head + "来られる",
              "volitional": head + "来よう", "sou": head + "来よう", "rentai": head + "来る"}
    return table_.get(form, head + "来る")


def _ichidan(k: str, form: str) -> str:
    """上一段・下一段（辞書に無い語の縮退）。"""
    stem = k[:-1]
    table_ = {"dictionary": k, "renyou": stem, "stem": stem, "masu": stem + "ます",
              "mashita": stem + "ました", "masen": stem + "ません", "te": stem + "て",
              "ta": stem + "た", "nai": stem + "ない", "nakatta": stem + "なかった",
              "conditional": stem + "れば", "imperative": stem + "ろ",
              "potential": stem + "られる", "volitional": stem + "よう", "sou": stem + "よう",
              "rentai": k}
    return table_.get(form, k)


def _kana_inflect(k: str, form: str) -> str:
    """音韻ルールだけで五段活用を組み替える（辞書に無い語の縮退）。"""
    last = k[-1]
    row = _ROWS[last]
    stem = k[:-1]
    table_ = {
        "dictionary": k, "renyou": stem + row[1], "stem": stem + row[1],
        "te": stem + _TE[last], "ta": (stem + _TE[last]).replace("て", "た"),
        "masu": stem + row[1] + "ます", "masen": stem + row[1] + "ません",
        "mashita": (stem + _TE[last]).replace("て", "た") + "です",
        "nai": stem + _NAI[last] + "ない", "nakatta": stem + _NAI[last] + "なかった",
        "conditional": stem + row[3] + "ば", "imperative": stem + row[3],
        "potential": stem + row[3] + "る", "volitional": stem + row[0] + "う",
        "sou": stem + row[0] + "う", "rentai": k,
    }
    return table_.get(form, k)


def inflect(surface: str, form: str = "dictionary") -> str:
    """活用を組み替える。未知の語は音韻ルール、活用しない語は断定に落とす。"""
    raw = normalize(surface)
    if not raw:
        return ""
    lemma = raw if form == "dictionary" else lemma_of(raw)
    k = to_hiragana(lemma)
    forms = _forms(lemma)

    # ---------- イ形容詞 ----------
    if is_adjective(lemma):
        if "基本形" in forms or lemma in table():
            f = {"dictionary": "基本形", "renyou": "連用形", "adverb": "連用形",
                 "adv": "連用形", "stem": "語幹", "conditional": "仮定形",
                 "imperative": "命令形", "volitional": "未然ウ接続", "sou": "未然ウ接続",
                 "mizen": "未然形", "rentai": "体言接続"}.get(form)
            if f and f in forms and form in ("renyou", "adverb", "adv", "conditional",
                                              "imperative", "volitional", "sou", "mizen",
                                              "rentai"):
                tail = {"renyou": "", "adverb": "", "adv": "", "conditional": "ば",
                        "imperative": "", "volitional": "う", "sou": "う", "mizen": "",
                        "rentai": ""}[form]
                return forms[f] + tail
        return _adj_infix(k, form)

    # ---------- 動詞 ----------
    verbish = bool(k) and (k[-1] in _ROWS or k.endswith("る"))
    if verbish:
        renyou = forms.get("連用形")
        te_stem = forms.get("連用タ接続")
        mizen = forms.get("未然形")
        katei = forms.get("仮定形")
        meirei = next((forms.get(x) for x in ("命令ｒｏ", "命令ｅ", "命令ｉ", "命令ｙｏ")
                       if forms.get(x)), None)
        u_stem = forms.get("未然ウ接続")
        if renyou or mizen:
            if form == "dictionary":
                return lemma
            if form in ("renyou", "stem"):
                return renyou or _kana_inflect(k, form)
            if form in ("adverb", "adv"):
                return renyou or _kana_inflect(k, "renyou")
            if form == "mizen":
                if mizen:
                    return mizen
                row = _ROWS.get(k[-1])
                return (k[:-1] + row[0]) if row else k
            if form == "masu":
                return (renyou or _kana_inflect(k, "renyou")) + "ます"
            if form == "masen":
                return (renyou or _kana_inflect(k, "renyou")) + "ません"
            if te_stem:
                # 五段の音便: 連用タ接続の後続は ぐ/ぬ/ぶ/む なら 「で」、それ以外 「て」
                glue = "で" if (k[-1] in ("ぐ", "ぬ", "ぶ", "む") or te_stem.endswith("ん")) else "て"
                te_body = te_stem + glue
            else:
                te_body = (renyou or "") + "て"
            if te_body.endswith("で"):
                ta_body = te_body[:-1] + "だ"
            else:
                ta_body = te_body[:-1] + "た"
            if form == "te":
                return te_body
            if form == "ta":
                return ta_body
            if form == "mashita":
                return (renyou or _kana_inflect(k, "renyou")) + "ました"
            if form == "nai":
                return (mizen + "ない") if mizen else _kana_inflect(k, "nai")
            if form == "nakatta":
                return (mizen + "なかった") if mizen else _kana_inflect(k, "nakatta")
            if form == "conditional":
                return (katei + "ば") if katei else _kana_inflect(k, "conditional")
            if form == "imperative":
                return meirei or _kana_inflect(k, "imperative")
            if form == "potential":
                if lemma in ("する", "為る", "為る"):
                    return "できる"
                if to_hiragana(lemma) in ("くる", "来る"):
                    return "来られる"
                if is_ichidan(lemma):
                    return (renyou or "") + "られる"
                if renyou and renyou.endswith("い"):
                    return renyou[:-1] + "える"
                return _kana_inflect(k, "potential")
            if form in ("volitional", "sou"):
                return (u_stem + "う") if u_stem else _kana_inflect(k, "volitional")
            if form == "rentai":
                return lemma
        # 表にない動詞は「サ変 → カ変 → 一段 → 五段」の順で縮退する
        if k.endswith("する") and len(k) >= 3 and "五段" not in ctype_of(lemma):
            return _restore_kanji(raw, lemma, _sahen(k[:-2], form), kana_only=False)
        if (k.endswith("来る") or k.endswith("くる")) and len(k) >= 3 and "五段" not in ctype_of(lemma):
            head = k[:-2]
            return _restore_kanji(raw, lemma, _kahen(head if head else "来", form), kana_only=False)
        if k.endswith("る") and len(k) >= 3 and is_ichidan(k) and "五段" not in ctype_of(lemma):
            return _restore_kanji(raw, lemma, _ichidan(k, form), kana_only=False)
        if k[-1] in _ROWS:
            out = _kana_inflect(k, form)
            return out if form == "dictionary" else _restore_kanji(raw, lemma, out)

    # ---------- 活用しない語（名詞など） ----------
    plain = {"dictionary": "", "renyou": "", "stem": "", "masu": "です", "mashita": "でした",
             "masen": "ではありません", "te": "で", "ta": "だった", "nai": "ではない",
             "nakatta": "ではなかった", "conditional": "なら", "imperative": "にしなさい",
             "potential": "ができる", "volitional": "にしよう", "sou": "にしよう",
             "rentai": "な"}
    return raw + plain.get(form, "です")


def _restore_kanji(raw: str, lemma: str, out_kana: str, *, kana_only: bool = True) -> str:
    """音韻ルールで組み立てたかな形に、元の漢字部を戻す（書く→書いて を保つ）。
    語彙バンクの表記（活用形の語彙エントリ）があればそれを優先する。"""
    try:
        from .lex import bank

        w = bank().entry(out_kana)
        if w is not None and w.pos.startswith(("動詞", "形容詞")):
            return w.surface
        lk = to_hiragana(lemma)
        if lk and out_kana.startswith(lk) and lemma != out_kana and not kana_only:
            # 「追加します」のように語幹が漢字表記なら、語幹だけ元の表記に戻す
            head_len = 0
            for ch in lemma:
                if head_len >= len(lk):
                    break
                head_len += 1
            return lemma[:head_len] + out_kana[head_len:]
    except Exception:  # noqa: BLE001
        pass
    return out_kana


# 後方互換・別名
conjugate = inflect


def stem(surface: str) -> str:
    """連用形（「書き」「食べ」）。"""
    return inflect(surface, "renyou")


def _split_last_word(area: str) -> tuple[str, str]:
    """文末の語 1 つを取り出す（(前なりゆき, 文末語)）。語彙バンクの分かち書きを使う。"""
    if not area:
        return "", ""
    head = area
    try:
        from .lex import bank as _bank

        seg = _bank().segment(area)
        if seg:
            last = seg[-1][0]
            if last and last != area and area.endswith(last) and len(last) <= 8:
                head = last
    except Exception:  # noqa: BLE001
        pass
    return area[: len(area) - len(head)], head


def _fits(lemma: str, surface: str) -> bool:
    """lemma が surface を再生成できるか（活用表で確かめる）。

    「書き」から「来る」を引くような暴走を止める弁です。活用のいずれかの形が
    surface と一致しなければ、その辞書形は使いません。
    """
    if not lemma or not surface:
        return False
    want = to_hiragana(surface)
    try:
        for f in ("renyou", "dictionary", "ta", "te", "nai"):
            if to_hiragana(inflect(lemma, f)) == want:
                return True
        forms = _forms(lemma)
    except Exception:  # noqa: BLE001
        return False
    return any(to_hiragana(v) == want for v in forms.values())


# 敬語の尾巴 → 組み替える語の形（語の種類もここで決める）
def fits_inflection(lemma: str, surface: str) -> bool:
    """`lemma` を活用させた形のどれかが `surface` と一致するか。

    未知語の列を動詞だと誤って決めないための確認弁に使う。
    """
    return bool(lemma) and _fits(lemma, surface)


_PLAIN_TAILS: tuple[tuple[str, str, str], ...] = (
    ("かったです", "ta", "adj"),
    ("ません", "nai", "verb"),
    ("ました", "ta", "verb"),
    ("ます", "dictionary", "verb"),
    ("でした", "ta", "noun"),
    ("です", "plain", "noun"),
)


def to_plain(text: str) -> str:
    """敬体 → 常体。述語 1 語を辞書形に戻してから、必要な形に組み直す。

    語尾だけを機械的に差し替えると「合い+ない」「勉強し+る」のような壊れた語が
    出ます。なので (1) 文末の語を切り出し (2) 辞書形に引いて (3) 活用で組み直す。
    """
    t = normalize(text).rstrip("。")
    if not t:
        return text
    # 「食べたいです」「美味しくないです」は述語そのものが常体なので、
    # 敬語部分だけ外すのが正しい。活用を組み替えると「食べたいだ」になる。
    if re.search(r"(たくない|たい|ない)です$", t):
        return t[: -len("です")] + "。"
    if t.endswith("たくありません"):
        return t[: -len("たくありません")] + "たくない。"
    for tail, form, kind in _PLAIN_TAILS:
        if not t.endswith(tail) or len(t) <= len(tail):
            continue
        area = t[: len(t) - len(tail)]
        prefix, head = _split_last_word(area)
        if not head:
            break
        if kind == "adj":
            stem = head[:-1] + "い" if head.endswith("かっ") else head
            lem = lemma_of(stem) if stem else ""
            new = inflect(lem or stem, "ta") if form == "ta" else (lem or stem)
            if new:
                return t[: len(t) - len(tail) - len(head)] + new + "。"
            break
        if kind == "noun":
            if form == "ta":                                  # 勉強でした → 勉強だった
                return t[: len(t) - len(tail) - len(head)] + head + "だった。"
            if is_adjective(head):                            # 可愛いです → 可愛い
                return t[: len(t) - len(tail) - len(head)] + head + "。"
            return t[: len(t) - len(tail) - len(head)] + head + "だ。"
        # 動詞: 直前が名詞ならサ変複合（勉強+し → 勉強する）を先に、それから語単体。
        # 助詞まで巻き込むと「音が合い」を 1 語と誤読するので、名詞に限定する。
        spans = [head]
        if prefix:
            try:
                from .lex import bank as _bank

                _, plast = _split_last_word(prefix)
                if plast and str(_bank().pos(plast)).startswith("名詞"):
                    spans.insert(0, plast + head)
            except Exception:  # noqa: BLE001
                pass
        for span in spans:
            if not span:
                continue
            lem = lemma_of(span)
            if not lem or not _fits(lem, span):
                continue
            new = inflect(lem, form)
            if new and new != span:
                return t[: len(t) - len(tail) - len(span)] + new + "。"
    return t + "。"


_POLITE_TAILS = (                       # (平文の尾巴, 時制・否定の区別)
    ("なかった", "nata"),
    ("たい", "tai"),
    ("ない", "nai"),
    ("た", "ta"),
)
# 単独では述語にならない付属語。この形のまま敬体に換えるとおかしくなるので触らない。
_AUX_SPANS = frozenset({"ない", "ぬ", "たい", "た", "だ", "です", "ます", "した", "だった",
                        "である", "はず", "べき", "ところ", "もの", "の", "な", "やん", "っ"})


def _polite_reinflect(span: str) -> str:
    """動詞・形容詞の連なり 1 かたまりを敬体に組み替える。

    語尾の文字を差し替えると「合わない」→「合わないです」のように崩れるので、
    (1) 辞書形に引き直す (2) その辞書形から平文を作ってもとが一致するときだけ
    組み替える、という順で確かめる。一致が取れない場合は触らないのが安全。
    """
    if not span or len(span) < 2 or span in _AUX_SPANS:
        return ""
    lem = lemma_of(span)
    if not lem:
        return ""
    try:
        from .lex import bank as _bank

        pos = str(_bank().pos(lem) or _bank().pos(span) or "")
    except Exception:  # noqa: BLE001
        pos = ""
    major = pos.split("/")[0]
    if span.endswith(("て", "で")) and not span.endswith("いて"):
        return ""                                  # 接続助詞のて形は敬体に変換しない
    for tail, tag in _POLITE_TAILS:
        if not span.endswith(tail) or len(span) <= len(tail):
            continue
        if tag in ("tai",) or major == "形容詞":
            # 「〜たいです」「〜かったです」は助動詞・助辞を足すだけで揃う。
            return span + "です"
        try:
            plain = {"nata": inflect(lem, "nai") + "かった",
                     "nai": inflect(lem, "nai"),
                     "ta": inflect(lem, "ta")}[tag]
        except Exception:  # noqa: BLE001
            plain = ""
        if plain != span:
            return ""
        base = inflect(lem, "masu")
        if not base.endswith("ます"):
            return ""
        stem = base[:-2]
        return {"nata": stem + "ませんでした", "nai": stem + "ません",
                "ta": stem + "ました"}[tag]
    if inflect(lem, "dictionary") != span:
        if major == "形容詞" and is_adjective(span):
            return span + "です"
        return ""
    base = inflect(lem, "masu")
    return base if base.endswith("ます") else ""


def to_polite(text: str) -> str:
    """文末を敬体にそろえる（to_plain と対称に、語を引いてから組み替える）。"""
    t = normalize(text)
    if not t:
        return t
    core = t.rstrip("。！？!?")
    if not core:
        return t
    if re.search(r"(でした|ません|ました|ます|です|でしょう|ください)$", core):
        return t if t[-1] in "。！？!?" else t + "。"
    if core.endswith("である"):
        return core[: -len("である")] + "です。"
    if core.endswith("だ") and len(core) >= 3:
        # 「選んだ」の だ は助動詞なので触らない。直前が体言のときだけ断定の だ とみる。
        try:
            from .lex import bank as _bank

            prev = _bank().segment(core[:-1])
            pos = str(prev[-1][1]) if prev else ""
        except Exception:  # noqa: BLE001
            pos = ""
        if pos.split("/")[0] in ("名詞", "代名詞", "副詞") or "サ変接続" in pos \
                or pos.startswith("形容動詞"):
            return core[:-1] + "です。"
    try:
        from .lex import bank as _bank

        toks = [x for x, _p in _bank().segment(core)]
    except Exception:  # noqa: BLE001
        toks = [core]
    for k in range(1, min(4, len(toks)) + 1):
        span = "".join(toks[-k:])
        # 助詞は語の境界なので、そこに掛かる前までしか述語として数えない
        if str(_bank_pos(toks[-k])).split("/")[0] in ("助詞", "助動詞", "記号") or \
                toks[-k] in ("は", "が", "を", "に", "で", "と", "も", "の", "へ", "や", "か"):
            break
        if not span or span in _AUX_SPANS or not core.endswith(span):
            continue
        polite = _polite_reinflect(span)
        if polite and polite != span:
            return core[: len(core) - len(span)] + polite + "。"
    # 形容詞・形容動詞の語幹しか拾えなかったときは、敬体を足すにとどめる
    if toks:
        last = toks[-1]
        if last and not last.endswith("です") and (
                str(_bank_pos(last)).split("/")[0] in ("形容詞", "形容動詞")
                or is_adjective(last)):
            return core + "です。"
    return t if t[-1] in "。！？!?" else t + "。"


def split_predicate(text: str) -> tuple[str, str]:
    """文を「主題部」と「述語部」に割る（述語を受け直すための簡易係り受け）。"""
    t = normalize(text).strip("。！？!?、 ")
    parts = re.split(r"(が|は|を|に|で|と|も|へ)", t, maxsplit=1)
    if len(parts) >= 3:
        return parts[0].strip(), "".join(parts[1:]).strip()
    return "", t


def negative_of(pred: str) -> str:
    """述語の否定形（「忙しい」→「忙しくない」）。"""
    return inflect(pred, "nai")


__all__ = ["inflect", "conjugate", "stem", "lemma_of", "detect_pos", "ctype_of", "is_ichidan",
           "is_adjective", "to_polite", "to_plain", "split_predicate", "negative_of", "table"]
