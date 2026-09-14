"""日本語文の事後整形(助動詞の補い・文体の統一)。

Snipher-mini やハイブリッド補正で作った下書きを、if 構文ベースの
ルールだけで最小限の安全な修理をする。ニューラルモデルが無い環境でも
常に動作し、処理は数マイクロ秒(テーブル引き + 文字列操作)で済む。

修理ルール(保守的 = 壊さないことを最優先):
    1. 文末の句点補完        「〜ます」→「〜ます。」
    2. 助動詞の二重          「ますです」→「ます」
    3. 丁寧体の文末断定      「〜だ。」→「〜です。」(丁寧体ターゲット時)
    4. 動詞終止形+ますの修復 「飲むます」→「飲みます」(辞書と活用表で修理)
    5. い形容詞+だの修復     「いいだ」→「いいです」
    6. 終助詞後の句点        「〜ですね」→「〜ですね。」
"""

from __future__ import annotations

import re

from .lexicon import Lexicon
from .morphology import Morphology

# 丁寧体マーカー / 普通体マーカー
_POLITE_ENDS = ("ます。", "です。", "ました。", "ません。", "でしょうか。", "ますか", "ですか")
_CASUAL_ENDS = ("だ。", "た。", "ない。", "だった。", "だね", "だよ")


class Polisher:
    """文を最小限ルールで修理する。"""

    def __init__(self, lexicon: Lexicon | None = None):
        self.lex = lexicon or Lexicon()
        self.morph = Morphology()
        self._verb_by_surface = {v["s"]: v for v in self.lex.verbs}
        self._adj_by_surface = {a["s"]: a for a in self.lex.adjectives}

    # ------------------------------------------------------------------ #
    def polish(self, text: str, register: str = "polite", *, bare: bool = False) -> dict:
        """text を修理する。→ {"text": str, "fixes": [ルール名, ...]}

        `bare=True` は「答えだけをそのまま返す」指定（数字だけ／1 語だけ等）のとき。
        文末の句点補完と敬体化は行いません（"8" を "8。" にすると指定した形ではなくなる）。
        """
        original = text
        fixes: list[str] = []
        if not text or not text.strip():
            return {"text": text, "fixes": fixes}

        # 2. 助動詞の二重 「ますです」「ましたです」
        new = re.sub(r"ましたです", "ました", new_text := text)
        if new != new_text:
            fixes.append("aux_duplicate: ましたです→ました")
        text = new
        new = re.sub(r"ますです", "ます", new_text := text)
        if new != new_text:
            fixes.append("aux_duplicate: ますです→ます")
        text = new
        new = re.sub(r"ですです", "です", new_text := text)
        if new != new_text:
            fixes.append("aux_duplicate: ですです→です")
        text = new

        # 4. 動詞の終止形 + ます → 連用形 + ます(辞書にある動詞のみ)
        text = self._fix_verb_masu(text, fixes)

        # 3/5. 丁寧体ターゲットなら文末の「だ。」を「です。」に
        if register == "polite" and not bare:
            text = self._fix_polite_ending(text, fixes)

        # 1/6. 文末の句点補完（形を指定された答えには足さない）
        if not bare:
            text = self._fix_terminal_punct(text, fixes)

        # 7. 日本語と英数字のあいだの空白（技術系の文の読みやすさ）
        text = self._fix_ascii_spacing(text, fixes)

        return {"text": text, "fixes": fixes, "changed": text != original}

    # ------------------------------------------------------------------ #
    # 日本語と英数字の境界に半角空白を 1 つ入れる
    #   「最後にgit push で共有する」→「最後に git push で共有する」
    #   句読点・括弧・既存の空白はそのまま（二重には入れない）
    # ------------------------------------------------------------------ #
    _JP = r"\u3041-\u309f\u30a1-\u30f6\u30fc\u4e00-\u9fff"
    _ASCII_WORD = r"[A-Za-z0-9]"

    def _fix_ascii_spacing(self, text: str, fixes: list[str]) -> str:
        if not text:
            return text
        before = text
        text = re.sub(f"([{self._JP}])({self._ASCII_WORD})", r"\1 \2", text)
        text = re.sub(f"({self._ASCII_WORD})([{self._JP}])", r"\1 \2", text)
        if text != before:
            fixes.append("ascii_spacing")
        return text

    # ------------------------------------------------------------------ #
    def _fix_verb_masu(self, text: str, fixes: list[str]) -> str:
        """「飲むます」のような 終止形+ます を 連用形+ます に直す。"""

        def repl(m: re.Match) -> str:
            body = m.group(1)
            # 末尾から、辞書にある動詞(終止形)になる最長の切れ目を探す
            for cut in range(len(body)):
                surface = body[cut:]
                verb = self._verb_by_surface.get(surface)
                if verb is None:
                    continue
                masu_stem = self.morph.inflect_verb(verb, "masu")
                fixed = body[:cut] + masu_stem + "ます"
                fixes.append(f"verb_masu: {surface}ます→{fixed}")
                return fixed
            return m.group(0)

        # 活用語尾(る/う/く/... )+ます を一括修理
        return re.sub(r"([一-龯ぁ-んァ-ヶ]{1,6}[るくぐすつぬぶむう])ます", repl, text)

    def _fix_polite_ending(self, text: str, fixes: list[str]) -> str:
        """丁寧体で文末が普通体の断定だったら直す(文全体の末尾のみ)。"""
        stripped = text.rstrip("。！？\n ")
        if not stripped:
            return text
        # い形容詞 + だ
        m = re.search(r"([一-龯ぁ-ん]{1,8}い)だ$", stripped)
        if m and m.group(1) in self._adj_by_surface:
            adj = self._adj_by_surface[m.group(1)]
            if adj.get("c") == "い形容詞":
                fixed = stripped[: -1] + "です" + text[len(stripped):]
                fixes.append(f"polite_ending: {m.group(1)}だ→{m.group(1)}です")
                return fixed
        # 一般名詞/な形容詞 + だ(「ただ」「〜のだ」などは除外)
        if stripped.endswith("だ") and not stripped.endswith(("ただ", "のだ", "んだ")):
            body = stripped[:-1]
            # ます/です の後ろに「だ」が続く異常は直さない(上位ルールで処理済み)
            fixed = body + "です" + text[len(stripped):]
            fixes.append(f"polite_ending: {stripped[-8:]}→{body[-7:]}です")
            return fixed
        return text

    def _fix_terminal_punct(self, text: str, fixes: list[str]) -> str:
        """文末に句点が無ければ補う(疑問詞の後は ? でも可)。"""
        t = text.rstrip()
        if not t:
            return text
        last = t[-1]
        if last in "。！？!?、":
            return text
        fixes.append("terminal_punct: 。を補完")
        return t + "。"
