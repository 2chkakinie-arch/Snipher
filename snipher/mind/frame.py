"""思考の材料になるデータ構造（発話枠・主張・証拠・規則・手番）。

Snipher の中核は「定型文を引く」ことではありません。発話を **意味の構造**に分解し、
証拠（知識ベース／実辞書／計算／ウェブ）を揃え、**主張（Claim）として組み立てて**から、
日本語に書き下ろします。このモジュールはその受け皿だけを持っています。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Rule:
    """発話から読み取った「守るべき条件」。生成後は必ず検証する。"""

    kind: str                      # len / morae / start / end / charset / style / form / forbid / require / count / chain / lang / line
    value: object = None
    raw: str = ""                  # どの表現から取ったか（UI の根拠表示用）
    hard: bool = True

    def describe(self) -> str:
        return {
            "len": f"{self.value}文字",
            "morae": f"{self.value}音",
            "start": f"「{self.value}」で始める",
            "end": f"「{self.value}」で終わる",
            "charset": f"{self.value}のみ",
            "style": f"{self.value}調",
            "form": str(self.value),
            "forbid": f"「{self.value}」は使わない",
            "require": f"「{self.value}」を含める",
            "count": f"{self.value}件",
            "chain": "前の語の最後の音から始める",
            "lang": str(self.value),
            "line": f"{self.value}行",
        }.get(self.kind, self.kind)


@dataclass
class Entity:
    """発話の中に現れた「対象」。辞書にあるか／知識ベースにあるか／未知語かを覚える。"""

    surface: str
    kind: str = "noun"             # noun / ascii / kana / digits / mixed
    reading: str = ""
    pos: str = ""
    morae: int = 0
    known_dict: bool = False       # 実辞書に見出しがある
    known_kb: str = ""             # 知識ベースの話題名（当たれば採用）
    web_needed: bool = False       # 手元に無い語なので、調べないと答えられない

    def as_dict(self) -> dict:
        return {"surface": self.surface, "kind": self.kind, "reading": self.reading,
                "pos": self.pos, "morae": self.morae, "known_dict": self.known_dict,
                "kb": self.known_kb or None, "web_needed": self.web_needed}


@dataclass
class Claim:
    """応答に載せる 1 つの「こと」。文の形ではなく中身として持つのが要点。"""

    kind: str                       # definition / fact / reason / step / result / compare / time /
                                    # lexical / evidence / example / format / answer / ask / note
    content: str = ""
    subject: str = ""
    slot: str = ""
    source: str = "local"           # local:kb / lex / tool:math / tool:code / web / clock
    urls: list[str] = field(default_factory=list)
    weight: float = 0.7
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "content": self.content, "subject": self.subject,
                "source": self.source, "weight": round(self.weight, 3)}


@dataclass
class Activity:
    """会話の途中で始まった「共同作業」の状態（手番・規則・直前の語）。"""

    name: str = ""                  # 例: 語の連鎖（しりとり系）
    kind: str = ""                  # chain / quiz / role / count / translate-loop
    turn: int = 0
    last_word: str = ""             # 直前に発話された語（相手の語）
    our_word: str = ""              # 私が出した語
    used: list[str] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    open: bool = False
    source: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "turn": self.turn,
                "last_word": self.last_word, "our_word": self.our_word,
                "used": self.used[-8:], "open": self.open, "source": self.source,
                "rules": [r.describe() for r in self.rules]}


@dataclass
class Frame:
    """1 発話の意味構造。"""

    raw: str = ""
    norm: str = ""
    act: str = "declare"            # ask / command / invite / declare / ack / greet / thanks /
                                    # apology / agree / disagree / farewell / play / meta
    ask: str = ""                    # definition / reason / procedure / comparison / count / price /
                                    # when / where / who / manner / yesno / list / transform /
                                    # compute / code / write / translate / capability / now / ...
    entities: list[Entity] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    mood: str = "neutral"
    register: str = "polite"         # polite / plain
    language: str = "ja"
    topic: str = ""                  # 検索に渡す話題文字列
    needs_web: bool = False
    is_followup: bool = False        # 会話を続ける短い発話（「続き」「なんで」）
    activity: Activity | None = None
    flags: dict = field(default_factory=dict)

    @property
    def main_entity(self) -> Entity | None:
        return self.entities[0] if self.entities else None

    def has_rule(self, kind: str) -> bool:
        return any(r.kind == kind for r in self.rules)

    def rule(self, kind: str) -> Rule | None:
        for r in self.rules:
            if r.kind == kind:
                return r
        return None

    def as_dict(self) -> dict:
        return {"act": self.act, "ask": self.ask, "topic": self.topic, "mood": self.mood,
                "register": self.register, "entities": [e.as_dict() for e in self.entities[:5]],
                "rules": [r.describe() for r in self.rules], "needs_web": self.needs_web,
                "is_followup": self.is_followup, "flags": {k: v for k, v in self.flags.items()
                                                            if isinstance(v, (int, float, str, bool))}}


@dataclass
class Evidence:
    """証拠 1 件。引用元と、そこから取った文。"""

    text: str
    url: str = ""
    title: str = ""
    origin: str = "local"            # local / web / kb / lex / tool
    score: float = 0.0
    span: tuple[int, int] = (0, 0)

    def as_dict(self) -> dict:
        return {"text": self.text[:160], "url": self.url, "title": self.title,
                "origin": self.origin, "score": round(self.score, 3)}


_SENT_SPLIT = re.compile(r"(?<=[。！？!?])\s*")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(str(text or "")) if s.strip()]


__all__ = ["Rule", "Entity", "Claim", "Activity", "Frame", "Evidence", "split_sentences"]
