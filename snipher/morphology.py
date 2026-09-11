"""日本語の活用処理(形態論)を if 構文ベースで実装するモジュール。

動詞(五段/一段/サ変/カ変)・形容詞(い/な)・助動詞の活用を、
辞書エントリの表記(漢字かな交じり)とローマ字読みから機械的に生成する。
音便(買った/書いた/泳いだ/飲んだ/待ったなど)にも対応している。
"""

from __future__ import annotations

# 五段動詞の活用語尾(行ごと)  a/i/u/e/o の各段
_GODAN_ROWS = {
    "k": ("か", "き", "く", "け", "こ"),
    "g": ("が", "ぎ", "ぐ", "げ", "ご"),
    "s": ("さ", "し", "す", "せ", "そ"),
    "t": ("た", "ち", "つ", "て", "と"),
    "n": ("な", "に", "ぬ", "ね", "の"),
    "b": ("ば", "び", "ぶ", "べ", "ぼ"),
    "m": ("ま", "み", "む", "め", "も"),
    "r": ("ら", "り", "る", "れ", "ろ"),
    "w": ("わ", "い", "う", "え", "お"),
}

# 音便(た形/て形)の語尾
_GODAN_TE = {
    "k": ("いて", "いた"),
    "g": ("いで", "いだ"),
    "s": ("して", "した"),
    "t": ("って", "った"),
    "n": ("んで", "んだ"),
    "b": ("んで", "んだ"),
    "m": ("んで", "んだ"),
    "r": ("って", "った"),
    "w": ("って", "った"),
}


# 五段動詞の語尾かな → 活用行
_GODAN_KANA_ROWS = {
    "く": "k",  # 書く・歩く・行く
    "ぐ": "g",  # 泳ぐ
    "す": "s",  # 話す・探す
    "つ": "t",  # 待つ・勝つ
    "ぬ": "n",  # 死ぬ
    "ぶ": "b",  # 遊ぶ・選ぶ
    "む": "m",  # 飲む・読む・休む
    "る": "r",  # 取る・走る・座る・帰る
    "う": "w",  # 買う・会う・言う・思う(わ行)
}


def _godan_row(r: str) -> str:
    """五段動詞の読み(かな)の末尾から活用する行を判定する。"""
    if not r:
        return "r"
    return _GODAN_KANA_ROWS.get(r[-1], "r")


class Morphology:
    """語の活用形を生成する。"""

    # ------------------------------------------------------------------ #
    # 動詞
    # ------------------------------------------------------------------ #
    def verb_stem(self, verb: dict) -> str:
        """活用語尾を取り除いた漢字かな表記の語幹を返す。"""
        s = verb["s"]
        c = verb["c"]
        if c == "サ変":
            return s[:-2] if s.endswith("する") else s
        if c == "カ変":
            return s[:-1] if s.endswith("る") else s
        # 五段・一段は末尾1文字(活用語尾)を除く
        return s[:-1]

    def inflect_verb(self, verb: dict, form: str) -> str:
        """動詞を指定の形に活用させる。

        form:
            "dict"        辞書形(終止形)
            "masu"        連用形(ます接続: 書き/食べ/し/来)
            "te"          て形
            "ta"          た形(過去/完了)
            "nai"         ない形(未然+ない)
            "ba"          ば形(仮定)
            "volitional"  意志形
        """
        c = verb["c"]
        stem = self.verb_stem(verb)
        r = verb["r"]

        if form == "dict":
            return verb["s"]

        if c == "サ変":
            suffix = {
                "masu": "し",
                "te": "して",
                "ta": "した",
                "nai": "しない",
                "ba": "すれば",
                "volitional": "しよう",
            }.get(form, "")
            return stem + suffix

        if c == "カ変":
            suffix = {
                "masu": "",
                "te": "て",
                "ta": "た",
                "nai": "ない",
                "ba": "れば",
                "volitional": "よう",
            }.get(form, "")
            return stem + suffix

        if c == "一段":
            suffix = {
                "masu": "",
                "te": "て",
                "ta": "た",
                "nai": "ない",
                "ba": "れば",
                "volitional": "よう",
            }.get(form, "")
            return stem + suffix

        # 五段
        row = _godan_row(r)
        a, i, u, e, o = _GODAN_ROWS[row]
        if form == "masu":
            return stem + i
        if form == "nai":
            return stem + a + "ない"
        if form == "ba":
            return stem + e + "ば"
        if form == "volitional":
            return stem + o + "う"
        if form in ("te", "ta"):
            # 行く は例外的に促音便(行った/行って)
            if r == "いく":
                return stem + ("って" if form == "te" else "った")
            te, ta = _GODAN_TE[row]
            return stem + (te if form == "te" else ta)
        return verb["s"]

    # ------------------------------------------------------------------ #
    # 形容詞
    # ------------------------------------------------------------------ #
    def inflect_adjective(self, adj: dict, form: str) -> str:
        """形容詞(い/な)を指定の形に活用させる。

        form:
            "dict"         終止形(高い / 静か)
            "attributive"  連体形(高い / 静かな)
            "adverbial"    連用形(高く / 静かに)
            "past"         過去形(高かった / 静かだった)
            "te"           て形(高くて / 静かで)
            "negative"     否定形(高くない / 静かではない)
        """
        s = adj["s"]
        c = adj["c"]

        if c == "な形容詞":
            return {
                "dict": s,
                "attributive": s + "な",
                "adverbial": s + "に",
                "past": s + "だった",
                "te": s + "で",
                "negative": s + "ではない",
            }.get(form, s)

        # い形容詞: 語尾「い」を取り除く
        stem = s[:-1]
        return {
            "dict": s,
            "attributive": s,
            "adverbial": stem + "く",
            "past": stem + "かった",
            "te": stem + "くて",
            "negative": stem + "くない",
        }.get(form, s)

    # ------------------------------------------------------------------ #
    # 助動詞
    # ------------------------------------------------------------------ #
    def aux_form(self, aux_surface: str, form: str, register: str = "polite") -> str:
        """助動詞の活用形(です/ます/だ/た/ない など)を返す。"""
        polite = register == "polite"
        table = {
            "です": {
                "present": "です" if polite else "だ",
                "past": "でした" if polite else "だった",
                "negative": "ではありません" if polite else "ではない",
                "neg_past": "ではありませんでした" if polite else "ではなかった",
            },
            "ます": {
                "present": "ます",
                "past": "ました",
                "negative": "ません",
                "neg_past": "ませんでした",
            },
            "だ": {
                "present": "です" if polite else "だ",
                "past": "でした" if polite else "だった",
                "negative": "ではありません" if polite else "ではない",
                "neg_past": "ではありませんでした" if polite else "ではなかった",
            },
            "た": {
                "present": "た",
                "past": "た",
                "negative": "",
                "neg_past": "",
            },
            "ない": {
                "present": "ません" if polite else "ない",
                "past": "ませんでした" if polite else "なかった",
                "negative": "",
                "neg_past": "",
            },
        }
        entry = table.get(aux_surface)
        if entry is None:
            return aux_surface
        return entry.get(form, aux_surface)
