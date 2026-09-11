"""日本語の構文解析: 文構造・助動詞・要点(キーポイント)の抽出。

外部の形態素解析器に依存せず、辞書テーブルと if 構文によるルールで
トークン化・品詞推定・係り受けの大まかな推定・要点抽出を行う。
"""

from __future__ import annotations

import re
from collections import Counter

from .lexicon import Lexicon
from .morphology import Morphology

# 句読点などの記号
_SYMBOLS = set("。、！？!?…「」『』（）()・")


class ParsedToken:
    """解析済みトークン。"""

    __slots__ = ("surface", "lemma", "pos", "sub", "tags", "reading")

    def __init__(self, surface, lemma, pos, sub="", tags=None, reading=""):
        self.surface = surface
        self.lemma = lemma
        self.pos = pos
        self.sub = sub
        self.tags = tags or []
        self.reading = reading

    def as_dict(self):
        return {
            "surface": self.surface,
            "lemma": self.lemma,
            "pos": self.pos,
            "sub": self.sub,
            "tags": self.tags,
            "reading": self.reading,
        }

    def __repr__(self):  # pragma: no cover
        return f"<Token {self.surface} {self.pos}/{self.sub}>"


class Parser:
    """辞書ベースの日本語解析器。"""

    # 活用形の候補を辞書形へ引き直すための逆引き接尾辞
    def __init__(self, lexicon: Lexicon | None = None):
        self.lex = lexicon or Lexicon()
        self.morph = Morphology()

    # ------------------------------------------------------------------ #
    # トークン化
    # ------------------------------------------------------------------ #
    def tokenize(self, text: str) -> list[str]:
        """辞書の最長一致で単語分割する(未知語は1文字ずつ)。"""
        if not text:
            return []
        # 記号を分離
        text = re.sub(r"([。、！？!?…])", r" \1 ", text)
        words = []
        i = 0
        n = len(text)
        surfaces = self._all_surfaces()
        max_len = max((len(s) for s in surfaces), default=0)
        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue
            matched = None
            for length in range(min(max_len, n - i), 0, -1):
                cand = text[i : i + length]
                if cand in surfaces:
                    matched = cand
                    break
            if matched:
                words.append(matched)
                i += len(matched)
            else:
                words.append(ch)
                i += 1
        return words

    def _all_surfaces(self) -> set[str]:
        s = set()
        for group in (
            self.lex.verbs,
            self.lex.adjectives,
            self.lex.nouns,
            self.lex.particles,
            self.lex.auxiliaries,
            self.lex.adverbs,
            self.lex.conjunctions,
            self.lex.interjections,
            self.lex.adnominals,
        ):
            for e in group:
                s.add(e["s"])
        # 活用形も分割対象に含める(「見ました」→「見」+「ました」等)
        for v in self.lex.verbs:
            for form in ("masu", "te", "ta", "nai", "ba", "volitional"):
                s.add(self.morph.inflect_verb(v, form))
        for a in self.lex.adjectives:
            # な形容詞の「〜な/〜で」は助詞・コピュラと区別するため、
            # 活用形の分割対象には「い形容詞」のみを加える。
            if a.get("c") != "い形容詞":
                continue
            for form in ("adverbial", "past", "te", "negative"):
                s.add(self.morph.inflect_adjective(a, form))
        return s

    # ------------------------------------------------------------------ #
    # 品詞推定
    # ------------------------------------------------------------------ #
    def tag_token(self, surface: str) -> ParsedToken:
        """1トークンの品詞・語彙情報を推定する。"""
        if surface in _SYMBOLS:
            pos = "句点" if surface in "。．" else "読点" if surface in "、，" else "記号"
            return ParsedToken(surface, surface, pos)

        # 動詞の活用形を辞書形に引き戻して判定する
        verb = self._match_verb(surface)
        if verb:
            return ParsedToken(surface, verb["s"], "動詞", verb.get("c", ""), verb.get("t", []), verb.get("r", ""))

        adj = self._match_adjective(surface)
        if adj:
            return ParsedToken(surface, adj["s"], "形容詞", adj.get("c", ""), adj.get("t", []), adj.get("r", ""))

        noun = self.lex.find_noun(surface)
        if noun:
            return ParsedToken(surface, surface, "名詞", noun.get("c", ""), noun.get("t", []), noun.get("r", ""))

        particle = self.lex.find_particle(surface)
        if particle:
            return ParsedToken(surface, surface, "助詞", particle.get("f", ""), [], surface)

        aux = self.lex.find_auxiliary(surface)
        if aux:
            return ParsedToken(surface, aux["s"], "助動詞", aux.get("k", ""), [], aux.get("r", ""))

        for adv in self.lex.adverbs:
            if adv["s"] == surface:
                return ParsedToken(surface, surface, "副詞", "", [], adv.get("r", ""))

        for conj in self.lex.conjunctions:
            if conj["s"] == surface:
                return ParsedToken(surface, surface, "接続詞", "", [], conj.get("r", ""))

        for interj in self.lex.interjections:
            if interj["s"] == surface:
                return ParsedToken(surface, surface, "感動詞", "", [], interj.get("r", ""))

        for adn in self.lex.adnominals:
            if adn["s"] == surface:
                return ParsedToken(surface, surface, "連体詞", "", [], adn.get("r", ""))

        # 未知語のヒューリスティック
        if re.fullmatch(r"[0-9０-９]+", surface):
            return ParsedToken(surface, surface, "数詞", "", ["数"], surface)
        if re.fullmatch(r"[ぁ-ん]+", surface):
            return ParsedToken(surface, surface, "未知語", "かな", [], surface)
        return ParsedToken(surface, surface, "未知語", "", [], surface)

    def _match_verb(self, surface: str) -> dict | None:
        if self.lex.find_verb(surface):
            return self.lex.find_verb(surface)
        for verb in self.lex.verbs:
            for form in ("masu", "te", "ta", "nai", "ba", "volitional"):
                if self.morph.inflect_verb(verb, form) == surface:
                    return verb
        return None

    def _match_adjective(self, surface: str) -> dict | None:
        if self.lex.find_adjective(surface):
            return self.lex.find_adjective(surface)
        for adj in self.lex.adjectives:
            for form in ("attributive", "adverbial", "past", "te", "negative"):
                if self.morph.inflect_adjective(adj, form) == surface:
                    return adj
        return None

    # ------------------------------------------------------------------ #
    # 全文解析
    # ------------------------------------------------------------------ #
    def parse(self, text: str) -> dict:
        """文を解析し、トークン列・品詞・文構造・助動詞・要点を返す。"""
        tokens = [self.tag_token(t) for t in self.tokenize(text)]
        pos_seq = [t.pos for t in tokens]
        particles = [t for t in tokens if t.pos == "助詞"]
        auxiliaries = [t for t in tokens if t.pos == "助動詞"]
        verbs = [t for t in tokens if t.pos == "動詞"]
        adjectives = [t for t in tokens if t.pos == "形容詞"]
        nouns = [t for t in tokens if t.pos == "名詞"]
        adverbs = [t for t in tokens if t.pos == "副詞"]

        structure = self._detect_structure(tokens)

        return {
            "text": text,
            "tokens": [t.as_dict() for t in tokens],
            "pos_sequence": pos_seq,
            "structure": structure,
            "particles": [p.as_dict() for p in particles],
            "auxiliaries": [a.as_dict() for a in auxiliaries],
            "verbs": [v.as_dict() for v in verbs],
            "adjectives": [a.as_dict() for a in adjectives],
            "nouns": [n.as_dict() for n in nouns],
            "adverbs": [a.as_dict() for a in adverbs],
            "key_points": self._extract_key_points(tokens),
            "topic": self._detect_topic(tokens),
        }

    def _detect_structure(self, tokens: list[ParsedToken]) -> dict:
        """文構造(文型)を推定する。"""
        if not tokens:
            return {"pattern": "empty", "name": "空文", "order": []}

        surfaces = [t.surface for t in tokens]

        if any(s in surfaces for s in ("か", "？", "?")):
            return {"pattern": "question", "name": "疑問文", "order": self._order(tokens)}

        has_aux_past = any(a.surface in ("た", "ました", "でした") for a in tokens if a.pos == "助動詞")
        has_verb = any(t.pos == "動詞" for t in tokens)
        has_adj = any(t.pos == "形容詞" for t in tokens)
        has_noun = any(t.pos == "名詞" for t in tokens)
        has_topic_p = any(p.surface == "は" for p in tokens if p.pos == "助詞")
        has_subject_p = any(p.surface == "が" for p in tokens if p.pos == "助詞")
        has_object_p = any(p.surface == "を" for p in tokens if p.pos == "助詞")

        if has_verb and has_object_p:
            pattern = "plain"
            name = "平叙文(他動詞構文)"
        elif has_verb:
            pattern = "topic_subject"
            name = "平叙文(自動詞構文)"
        elif has_adj or (has_noun and not has_verb):
            pattern = "state" if has_adj else "nominal"
            name = "状態文" if has_adj else "名詞述語文"
        else:
            pattern = "plain"
            name = "平叙文"

        return {
            "pattern": pattern,
            "name": name,
            "order": self._order(tokens),
            "tense": "past" if has_aux_past else "nonpast",
            "has_verb": has_verb,
            "has_adjective": has_adj,
            "has_noun": has_noun,
            "topic_particle": has_topic_p,
            "subject_particle": has_subject_p,
            "object_particle": has_object_p,
        }

    @staticmethod
    def _order(tokens: list[ParsedToken]) -> list[str]:
        return [t.pos for t in tokens if t.pos not in ("記号", "句点", "読点")]

    def _extract_key_points(self, tokens: list[ParsedToken]) -> list[dict]:
        """要点(キーポイント)を抽出する。

        判定ロジック:
            1. 動詞・形容詞(述語) → 核となる要点
            2. 助詞「は/が/を/に/へ/で」が受ける名詞 → 主題・対象・場所
            3. 助動詞 → 時制・モダリティの要点
        """
        points: list[dict] = []
        for i, t in enumerate(tokens):
            if t.pos == "動詞":
                points.append({"kind": "述語", "value": t.lemma, "tags": t.tags})
            elif t.pos == "形容詞":
                points.append({"kind": "述語", "value": t.lemma, "tags": t.tags})
            elif t.pos == "助動詞":
                points.append({"kind": "助動詞", "value": t.surface, "sub": t.sub})
            elif t.pos == "助詞" and t.surface in ("は", "が", "を", "に", "へ", "で", "から", "まで"):
                # 直前の名詞を役割付きで拾う
                for prev in reversed(tokens[:i]):
                    if prev.pos == "名詞":
                        role = {
                            "は": "主題", "が": "主語", "を": "対象",
                            "に": "対象/場所", "へ": "方向", "で": "場所/手段",
                            "から": "起点", "まで": "終点",
                        }.get(t.surface, t.surface)
                        points.append({"kind": "役割", "role": role, "value": prev.surface, "tags": prev.tags})
                        break
        return points

    def _detect_topic(self, tokens: list[ParsedToken]) -> str | None:
        """「は」が受ける名詞を話題(トピック)として返す。"""
        for i, t in enumerate(tokens):
            if t.pos == "助詞" and t.surface == "は":
                for prev in reversed(tokens[:i]):
                    if prev.pos == "名詞":
                        return prev.surface
        # 「は」がなければ先頭の名詞を話題とみなす
        for t in tokens:
            if t.pos == "名詞":
                return t.surface
        return None

    # ------------------------------------------------------------------ #
    # 助動詞の完全な意味解釈(カテゴリ判定)
    # ------------------------------------------------------------------ #
    @staticmethod
    def aux_category(aux_surface: str) -> str:
        """助動詞の意味カテゴリを返す。"""
        categories = {
            "です": "断定/丁寧", "だ": "断定",
            "でした": "断定/丁寧(過去)",
            "ます": "丁寧", "ました": "丁寧(過去)", "ません": "丁寧(否定)",
            "ましょう": "丁寧(勧誘)", "でしょう": "丁寧(推量)",
            "ではない": "否定", "ではありません": "否定(丁寧)",
            "た": "過去/完了",
            "ない": "否定", "ぬ": "否定", "ず": "否定",
            "れる": "受身/可能/尊敬", "られる": "受身/可能/尊敬",
            "せる": "使役", "させる": "使役",
            "たい": "希望", "う": "意志/推量", "よう": "意志/推量",
            "まい": "否定推量", "らしい": "推量", "ようだ": "比況/推量",
            "そうだ": "様態/伝聞", "みたいだ": "比況", "べきだ": "義務",
        }
        return categories.get(aux_surface, "その他")
