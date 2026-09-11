"""確率的な日本語文の生成器。

あらかじめ決めておいた文型(patterns.json)のスロットを、
ProbabilityModel のスコアに従って単語テーブルから確率的に埋めることで
文章を組み立てる。「述語(動詞/形容詞)を先に決める → その要求に合わせて
名詞と助詞を選ぶ」という述語中心の生成を行う。
"""

from __future__ import annotations

import random

from .lexicon import Lexicon
from .morphology import Morphology
from .probability import ProbabilityModel

_POS = {
    "noun": "名詞",
    "verb": "動詞",
    "adj": "形容詞",
    "part": "助詞",
    "aux": "助動詞",
    "adv": "副詞",
    "conj": "接続詞",
    "punct": "句点",
    "reading": "読点",
}


class Generator:
    """文スロットを確率的に埋めて文を生成する。"""

    def __init__(self, lexicon: Lexicon | None = None, model: ProbabilityModel | None = None, seed: int | None = None):
        self.lex = lexicon or Lexicon()
        self.morph = Morphology()
        self.model = model or ProbabilityModel(self.lex, seed=seed)
        self.rng = random.Random(seed)
        self.ctx: dict = {}

    # ================================================================== #
    # 公開 API
    # ================================================================== #
    def generate(self, prompt: str | None = None, register: str | None = None,
                 tense: str = "nonpast", seed: int | None = None) -> dict:
        """プロンプト(任意)から1文を確率的に生成する。"""
        if seed is not None:
            self.model.rng = random.Random(seed)
            self.rng = random.Random(seed)

        register = register or self.lex.config.get("register", "polite")
        topic = self._pick_topic(prompt)

        self.ctx = {
            "tokens": [],          # [{"surface","pos","tags"}, ...]
            "topic": topic,        # 話題の名詞エントリ
            "topic_tags": topic.get("t", []) if topic else [],
            "register": register,
            "tense": tense,
            "predicate": None,     # 述語(動詞/形容詞/名詞)エントリ
            "predicate_pos": None,
        }

        pattern = self._pick_pattern(prompt)
        self._select_predicate(pattern)

        for slot in pattern["structure"]:
            self._fill_slot(slot)

        self._ensure_sentence_end()
        surface = self._render()
        return {
            "text": surface,
            "pattern": pattern["id"],
            "pattern_name": pattern["name"],
            "topic": topic["s"] if topic else None,
            "register": register,
            "tense": tense,
            "tokens": list(self.ctx["tokens"]),
        }

    def generate_many(self, n: int = 5, **kwargs) -> list[dict]:
        """複数文を生成する(独立な文のリスト)。"""
        return [self.generate(**kwargs) for _ in range(n)]

    # ================================================================== #
    # 話題・文型・述語の決定
    # ================================================================== #
    def _pick_topic(self, prompt: str | None) -> dict | None:
        """プロンプト中の名詞を話題にする。無ければランダムに選ぶ。"""
        if prompt:
            surfaces = sorted(
                (n for n in self.lex.nouns),
                key=lambda n: -len(n["s"]),
            )
            for n in surfaces:
                if n["s"] in prompt:
                    return n
        if not self.lex.nouns:
            return None
        return self.rng.choice(self.lex.nouns)

    def _pick_pattern(self, prompt: str | None) -> dict:
        is_question = bool(prompt and any(c in prompt for c in ("?", "？", "か")))
        if is_question:
            for p in self.lex.patterns:
                if p["id"] == "question":
                    return p
        weighted = [(p, p.get("weight", 1.0)) for p in self.lex.patterns]
        return self._weighted_choice(weighted)

    def _weighted_choice(self, weighted: list[tuple[dict, float]]) -> dict:
        total = sum(w for _, w in weighted)
        r = self.rng.random() * total
        acc = 0.0
        for item, w in weighted:
            acc += w
            if r <= acc:
                return item
        return weighted[-1][0]

    def _select_predicate(self, pattern: dict) -> None:
        """文型に応じて述語(動詞/形容詞/名詞)を先に決める。"""
        pid = pattern["id"]
        structure = pattern["structure"]
        self.ctx["wants_transitive"] = "object" in structure
        self.ctx["wants_intransitive"] = pid in ("topic_subject",)
        self.ctx["plain_verb"] = "conjecture_aux" in structure
        # 状態文/感嘆文では形容詞が主述語。理由/逆接文では形容詞は従属節の述語。
        self.ctx["adj_is_predicate"] = pid in ("state", "exclamation")

        if pid in ("state", "exclamation"):
            self.ctx["predicate"] = self._pick_adjective()
            self.ctx["predicate_pos"] = _POS["adj"]
        elif pid == "nominal":
            self.ctx["predicate"] = self._pick_topic_noun()
            self.ctx["predicate_pos"] = _POS["noun"]
        else:
            self.ctx["predicate"] = self._pick_verb()
            self.ctx["predicate_pos"] = _POS["verb"]

    def _pick_verb(self) -> dict | None:
        generic = set(self.lex.config.get("generic_verbs", []))
        affinity = self.lex.config.get("topic_affinity", {})
        preferred = set()
        for tag in self.ctx["topic_tags"]:
            preferred.update(affinity.get(tag, []))

        # 文構造をあらかじめ決める: 目的語を要求する文型では他動詞のみ、
        # 自動詞文型では自動詞のみに候補を絞り込む(if 構文による制約)。
        pool = self.lex.verbs
        if self.ctx.get("wants_transitive"):
            pool = [v for v in pool if v.get("tr") == "他動詞"]
        elif self.ctx.get("wants_intransitive"):
            pool = [v for v in pool if v.get("tr") == "自動詞"]
        if not pool:
            pool = self.lex.verbs

        cands = []
        for v in pool:
            s = self.model.score(
                v["s"], _POS["verb"], v.get("t", []), None, None,
                None, v.get("tr"), self.ctx["register"], self.ctx["topic_tags"])
            if v["s"] in preferred:
                s += 0.8  # 話題と親和的な述語を優先
            if v["s"] in generic:
                s -= 1.2  # 「する/来る」のような汎用動詞を抑制
            cands.append({"entry": v, "score": s})
        best = self.model.sample(cands)
        return best["entry"] if best else None

    def _pick_adjective(self, only_i: bool = False) -> dict | None:
        pool = self.lex.adjectives
        if only_i:
            pool = [a for a in pool if a.get("c") == "い形容詞"]
        if not pool:
            return None
        cands = [
            {"entry": a, "score": self.model.score(
                a["s"], _POS["adj"], a.get("t", []), None, None,
                None, None, self.ctx["register"], self.ctx["topic_tags"])}
            for a in pool
        ]
        best = self.model.sample(cands)
        return best["entry"] if best else None

    def _pick_topic_noun(self) -> dict | None:
        return self._pick_noun("は")

    # ================================================================== #
    # スロットの埋め込み
    # ================================================================== #
    def _fill_slot(self, slot: str) -> None:
        handlers = {
            "topic": self._slot_topic,
            "subject": self._slot_subject,
            "object": self._slot_object,
            "adverb": self._slot_adverb,
            "verb": self._slot_verb,
            "adjective": self._slot_adjective,
            "auxiliary": self._slot_auxiliary,
            "sentence_end": self._slot_sentence_end,
            "exclamation_end": self._slot_exclamation_end,
            "noun_predicate": self._slot_noun_predicate,
            "reason_conj": self._slot_reason_conj,
            "contrast_conj": self._slot_contrast_conj,
            "time": self._slot_time,
            "place": self._slot_place,
            "clause1": self._slot_clause,
            "clause2": self._slot_clause,
            "conjunction": self._slot_conjunction,
            "question_mark": self._slot_question_mark,
            "conjecture_aux": self._slot_conjecture_aux,
        }
        fn = handlers.get(slot)
        if fn:
            fn()

    # ---- 名詞 + 助詞 ---- #
    def _slot_topic(self) -> None:
        noun = self._pick_noun("は")
        if noun:
            self._emit(noun["s"], _POS["noun"], noun.get("t", []))
            self._emit("は", _POS["part"], [])

    def _slot_subject(self) -> None:
        noun = self._pick_noun("が")
        if noun:
            self._emit(noun["s"], _POS["noun"], noun.get("t", []))
            self._emit("が", _POS["part"], [])

    def _slot_object(self) -> None:
        noun = self._pick_noun("を")
        if noun:
            self._emit(noun["s"], _POS["noun"], noun.get("t", []))
            self._emit("を", _POS["part"], [])

    def _slot_time(self) -> None:
        times = self.lex.nouns_by_tag("時間")
        if not times:
            times = self.lex.nouns
        noun = self.rng.choice(times)
        self._emit(noun["s"], _POS["noun"], noun.get("t", []))
        # 相対的な時(今/今日/明日/昨日)には「に」を付けない
        particle = "は" if noun["s"] in ("今", "今日", "明日", "昨日") else "に"
        self._emit(particle, _POS["part"], [])

    def _slot_place(self) -> None:
        places = self.lex.nouns_by_tag("場所")
        if not places:
            places = self.lex.nouns
        noun = self.rng.choice(places)
        self._emit(noun["s"], _POS["noun"], noun.get("t", []))
        self._emit("で", _POS["part"], [])

    def _pick_noun(self, particle: str) -> dict | None:
        pred = self.ctx.get("predicate")
        verb_trans = pred.get("tr") if pred and self.ctx.get("predicate_pos") == _POS["verb"] else None
        pred_tags = pred.get("t", []) if pred else []
        cands = []
        for n in self.lex.nouns:
            s = self.model.score(
                n["s"], _POS["noun"], n.get("t", []),
                self._prev_surface(), self._prev_pos(),
                particle, verb_trans, self.ctx["register"], self.ctx["topic_tags"])
            # 述語動詞と同じ意味クラスの名詞を優先する(コーヒー+飲む など)
            overlap = bool(pred_tags and set(n.get("t", [])) & set(pred_tags))
            if particle == "を" and pred_tags:
                s += 1.2 if overlap else -1.0
            elif particle == "が" and pred_tags:
                s += 0.5 if overlap else -0.2
            cands.append({"entry": n, "score": s})
        best = self.model.sample(cands)
        return best["entry"] if best else None

    # ---- 述語 ---- #
    def _slot_verb(self) -> None:
        verb = self.ctx.get("predicate")
        if not verb:
            verb = self._pick_verb()
            self.ctx["predicate"] = verb
        surface = self._verb_surface(verb)
        self._emit(surface, _POS["verb"], verb.get("t", []))

    def _verb_surface(self, verb: dict) -> str:
        tense = self.ctx.get("tense", "nonpast")
        polite = self.ctx.get("register", "polite") == "polite"
        # 推量文(〜でしょう)の前では終止形(辞書形)を使う
        if self.ctx.get("plain_verb"):
            if tense == "past":
                return self.morph.inflect_verb(verb, "ta")
            return verb["s"]
        if polite:
            stem = self.morph.inflect_verb(verb, "masu")
            return stem + ("ました" if tense == "past" else "ます")
        if tense == "past":
            return self.morph.inflect_verb(verb, "ta")
        return verb["s"]

    def _slot_adjective(self) -> None:
        if self.ctx.get("adj_is_predicate"):
            adj = self.ctx.get("predicate")
            if not adj:
                adj = self._pick_adjective()
                self.ctx["predicate"] = adj
                self.ctx["predicate_pos"] = _POS["adj"]
        else:
            # 理由/逆接節の述語として独立に選ぶ(い形容詞が自然に接続する)
            adj = self._pick_adjective(only_i=True)
        surface = self._adj_predicate_surface(adj)
        self._emit(surface, _POS["adj"], adj.get("t", []))

    def _adj_predicate_surface(self, adj: dict) -> str:
        tense = self.ctx.get("tense", "nonpast")
        c = adj.get("c", "")
        if c == "い形容詞" and tense == "past":
            return self.morph.inflect_adjective(adj, "past")
        # な形容詞はコピュラ(です/でした)が時制を担うため語幹のまま
        return adj["s"]

    def _slot_auxiliary(self) -> None:
        """述語に接続するコピュラ(です/だ/でした)を出力する。"""
        pred = self.ctx.get("predicate")
        pred_pos = self.ctx.get("predicate_pos")
        polite = self.ctx.get("register", "polite") == "polite"
        tense = self.ctx.get("tense", "nonpast")

        if pred_pos == _POS["adj"] and pred and pred.get("c") == "な形容詞":
            copula = "でした" if tense == "past" else ("です" if polite else "だ")
        elif pred_pos == _POS["adj"]:
            # い形容詞: 丁寧体では「です」、普通体ではコピュラ無し
            copula = "です" if polite else ""
        else:
            copula = "です" if polite else "だ"
        if copula:
            self._emit(copula, _POS["aux"], [])

    def _slot_noun_predicate(self) -> None:
        noun = self.ctx.get("predicate")
        if not noun:
            noun = self._pick_topic_noun()
            self.ctx["predicate"] = noun
        self._emit(noun["s"], _POS["noun"], noun.get("t", []))
        polite = self.ctx.get("register", "polite") == "polite"
        tense = self.ctx.get("tense", "nonpast")
        copula = ("でした" if tense == "past" else "です") if polite else ("だった" if tense == "past" else "だ")
        self._emit(copula, _POS["aux"], [])

    # ---- 接続・終助詞・その他 ---- #
    def _slot_adverb(self) -> None:
        if not self.lex.adverbs:
            return
        adv = self.rng.choice(self.lex.adverbs)
        self._emit(adv["s"], _POS["adv"], [])

    def _slot_reason_conj(self) -> None:
        self._emit("ので", _POS["part"], [])

    def _slot_contrast_conj(self) -> None:
        self._emit("が", _POS["part"], [])

    def _slot_conjunction(self) -> None:
        conj = self.rng.choice(self.lex.conjunctions) if self.lex.conjunctions else None
        if conj:
            self._emit(conj["s"], _POS["conj"], [])
            self._emit("、", _POS["reading"], [])

    def _slot_question_mark(self) -> None:
        self._emit("か", _POS["part"], [])
        self._emit("。", _POS["punct"], [])

    def _slot_conjecture_aux(self) -> None:
        self._emit("でしょう", _POS["aux"], [])
        self._emit("。", _POS["punct"], [])

    def _slot_sentence_end(self) -> None:
        end = self.rng.choice(["。", "。", "ね。", "よ。"])
        for tok, pos in (("ね", _POS["part"]), ("よ", _POS["part"]), ("。", _POS["punct"])):
            if tok in end:
                self._emit(tok, pos, [])
        if not any(t["surface"] in ("。",) for t in self.ctx["tokens"][-2:]):
            self._emit("。", _POS["punct"], [])

    def _slot_exclamation_end(self) -> None:
        self._emit("ね", _POS["part"], [])
        self._emit("。", _POS["punct"], [])

    def _slot_clause(self) -> None:
        """複文の節: 「名詞を動詞」の簡易節。"""
        noun = self._pick_noun("を")
        if noun:
            self._emit(noun["s"], _POS["noun"], noun.get("t", []))
            self._emit("を", _POS["part"], [])
        verb = self._pick_verb()
        if verb:
            self._emit(self._verb_surface(verb), _POS["verb"], verb.get("t", []))

    # ================================================================== #
    # 出力ヘルパ
    # ================================================================== #
    def _prev_token(self) -> dict | None:
        return self.ctx["tokens"][-1] if self.ctx["tokens"] else None

    def _prev_surface(self) -> str | None:
        t = self._prev_token()
        return t["surface"] if t else None

    def _prev_pos(self) -> str | None:
        t = self._prev_token()
        return t["pos"] if t else None

    def _emit(self, surface: str, pos: str, tags: list[str]) -> None:
        self.ctx["tokens"].append({"surface": surface, "pos": pos, "tags": tags})

    def _ensure_sentence_end(self) -> None:
        """文末が終止記号でなければ「。」を補う。"""
        last = self._prev_token()
        if last is None:
            return
        if last["surface"] not in ("。", "！", "？", "か"):
            self._emit("。", _POS["punct"], [])

    def _render(self) -> str:
        return "".join(t["surface"] for t in self.ctx["tokens"])
