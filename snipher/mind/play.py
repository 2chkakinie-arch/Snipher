"""手番の取り合い — 発話から読み取った *規則* に従って次の一手を作る。

しりとり専用の分岐はここにありません。あるのは次だけ:

* 規則コンパイラが「語の連鎖（前の語の最後の拍を、次の語の頭にする）」を読んだ
* 語彙バンク（13 万語）に、その条件に合う語を問い合わせた
* 会話状態が「既に使った語」を覚えていたので、重複を避けた
* ルール違反（「ん」で終わる等）は、違反したことを *理由付き* で指摘する

なので「語尾を揃える」「5 音以内にする」「A で始める」のような指定も同じ通路で動きます。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..lang import lex
from ..lang.phonetics import (
    chain_key,
    char_count,
    is_all_kana,
    mora_count,
    normalize,
    to_hiragana,
)
from .frame import Claim, Rule
from .rules import chain_ok, check


@dataclass
class Move:
    """1 手分の結果（出す語と、相手に返す指摘）。"""

    word: str = ""
    reading: str = ""
    from_: str = ""                # "lex" / "corpus" / "user"
    legal: bool = True
    violation: str = ""           # 相手が壊した規則（理由つき）
    reason: str = ""              # 私がその語を選んだ根拠
    candidates: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"word": self.word, "reading": self.reading, "legal": self.legal,
                "violation": self.violation, "reason": self.reason,
                "candidates": self.candidates[:5], "from": self.from_, "notes": self.notes[:3]}


_EVERYDAY: frozenset[str] | None = None


def _everyday_nouns() -> frozenset[str]:
    """同梱の人手語彙テーブル（nouns/verbs/adjectives）にある語。手遊びで優遇する。"""
    global _EVERYDAY
    if _EVERYDAY is None:
        import json
        from pathlib import Path

        data = Path(__file__).resolve().parents[1] / "data"
        out: set[str] = set()
        keys = ("s", "base", "b", "word", "surface", "adj", "v", "n")

        def walk(obj) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and (k in keys or not isinstance(k, str)) \
                            and 1 <= len(v) <= 8:
                        out.add(v)
                    else:
                        walk(v)
            elif isinstance(obj, list):
                for x in obj:
                    walk(x)

        for name in ("nouns.json", "verbs.json", "adjectives.json", "other.json", "corpus.json"):
            fp = data / name
            if fp.exists():
                try:
                    walk(json.loads(fp.read_text(encoding="utf-8")))
                except Exception:  # noqa: BLE001
                    continue
        _EVERYDAY = frozenset(out)
    return _EVERYDAY


def rule_list(rules: list[Rule]) -> list[Rule]:
    return [r for r in rules if r.kind in ("chain", "len", "morae", "start", "end", "forbid",
                                           "require", "charset")]


def word_from_turn(text: str) -> str:
    """発話から「そのターンで出された語」を拾う（助詞・挨拶・記号を落とす）。"""
    t = normalize(text)
    if not t:
        return ""
    m = re.search(r"[「『]([^」』\n]{1,14})[」』]", t)
    if m:
        return m.group(1).strip("。、.!！?？ ")
    t = re.sub(r"^(?:じゃあ|それじゃ|それでは|では|よし|うん|はい|えーと|さて|次は|じゃ|ほら)[、, ]*", "", t)
    t = re.sub(r"[、,。.!！?？〜~…]+$", "", t)
    t = re.sub(r"(?:で|からだ|が先|で始める|から始めて|で終わる|で終わります|ですね|だね|だよ|だわ|かな|ね|よ|さ|わ|んか|の|で|を|が|は)$", "", t)
    t = re.sub(r"(?:でした|です|であります|っす|だよ|だもん|だよな)$", "", t)
    t = re.sub(r"[。！!？?]+$", "", t).strip()
    if not t:
        return ""
    # 語そのもの（1 語）ならそのまま、文なら最長の既知語を拾う
    if lex.bank().has(t) and 1 <= len(t) <= 12:
        return t
    words = [w for w, pos in lex.bank().segment(t)
             if pos.split("/")[0] in {"名詞", "動詞", "形容詞"} and 2 <= len(w) <= 12]
    if words:
        return max(words, key=len)
    return t if 1 <= len(t) <= 12 else ""


def judge(prev: str, word: str, rules: list[Rule], *, used: list[str] | None = None) -> Move:
    """相手の手を規則に照らして判定する（壊れていれば理由を返す）。"""
    used = used or []
    mv = Move(word=word, reading=to_hiragana(word) if word else "", from_="user")
    if not word:
        mv.legal = False
        mv.violation = "語が読み取れませんでした"
        return mv
    for r in rules:
        if r.kind == "chain" and prev and not chain_ok(prev, word):
            mv.legal = False
            mv.violation = (f"前の語「{prev}」の最後は「{chain_key(prev)}」なので、"
                            f"「{word}」は始まりの音が合いません")
            return mv
        if r.kind == "forbid" and to_hiragana(word).endswith(to_hiragana(str(r.value))):
            mv.legal = False
            mv.violation = f"「{r.value}」で終わる語は禁止されている規則でした"
            return mv
        if r.kind == "morae" and abs(mora_count(word) - int(r.value or 0)) > 0:
            mv.legal = False
            mv.violation = f"{r.value} 音の指定に対して「{word}」は {mora_count(word)} 音です"
            return mv
        if r.kind == "len":
            n = int(r.value or 0)
            if n and char_count(word) > n + 1:
                mv.legal = False
                mv.violation = f"{n} 文字までの指定に「{word}」は {char_count(word)} 文字です"
                return mv
        if r.kind == "start":
            head = to_hiragana(str(r.value))
            if head and not to_hiragana(word).startswith(head):
                mv.legal = False
                mv.violation = f"「{r.value}」で始めるルールでした"
                return mv
        if r.kind == "end":
            tail = to_hiragana(str(r.value))
            if tail and not to_hiragana(word).endswith(tail):
                mv.legal = False
                mv.violation = f"語尾を「{r.value}」にするルールでした"
                return mv
    if word in used:
        mv.legal = False
        mv.violation = f"「{word}」はこの会話でもう出ています"
    return mv


def pick(prev: str, rules: list[Rule], *, used: list[str] | None = None,
         pos: tuple[str, ...] = ("名詞",), max_len: int = 6) -> Move:
    """規則を満たす語を語彙バンクから選ぶ。"""
    used = used or []
    forbid_ends = ["ん"]
    want_len = 0
    want_morae = 0
    head = ""
    for r in rules:
        if r.kind == "forbid":
            forbid_ends.append(to_hiragana(str(r.value)))
        elif r.kind == "len":
            want_len = int(r.value or 0)
        elif r.kind == "morae":
            want_morae = int(r.value or 0)
        elif r.kind == "start":
            head = to_hiragana(str(r.value))
    key = chain_key(prev) if prev else ""
    start = head or key
    mv = Move(from_="lex")
    if not start:
        cands = lex.bank().by_morae(want_morae or 3, pos=pos, limit=12)
    else:
        cands = lex.bank().chain_candidates(start, exclude=tuple(used),
                                             forbid_ends=tuple(forbid_ends), pos=pos,
                                             limit=28, max_len=max(want_len or 0, max_len))
    if not cands:
        cands = lex.bank().by_reading_prefix(start, limit=16, pos=pos, max_len=max_len + 2)
    if not cands:
        mv.legal = False
        mv.reason = "語彙バンクに見合う語が見つかりませんでした"
        return mv
    everyday = _everyday_nouns()
    scored: list[tuple[tuple[int, int, int, int], str, str]] = []
    for w in cands:
        rd = w.kana()
        if rd in used or w.surface in used:
            continue
        penalty = 0
        if any(rd.endswith(x) for x in forbid_ends):
            penalty += 100
        if want_len and abs(len(w.surface) - want_len) > 1:
            penalty += 4
        if want_morae and mora_count(rd) != want_morae:
            penalty += 3
        if not is_all_kana(rd):
            penalty += 1                     # 読みが不安定な語は後回し
        if re.search(r"[ぁぃぅぇぉ]", rd):
            penalty += 1
        if rd == rd.lower() and all("ぁ" <= c <= "ん" for c in rd):
            penalty -= 1                     # ひらがなで書ける語は分かりやすい
        if any(ch.isdigit() for ch in w.surface):
            penalty += 50
        if len(w.surface) > max_len:
            penalty += 3
        if w.surface in everyday:
            penalty -= 30                    # 手元の日常語リストにある語を優先（打ちやすい）
        if 2 <= len(rd) <= 5:
            penalty -= 1
        scored.append(((penalty, w.cost, len(w.surface), -len(rd)), w.surface, rd))
    if not scored:
        mv.legal = False
        mv.reason = "既に使った語ばかりで、規則を満たす語が残りませんでした"
        return mv
    scored.sort()
    _pen, surface, reading = scored[0]
    mv.word = surface
    mv.reading = reading
    mv.candidates = [s for _p, s, _r in scored[:6]]
    if key:
        why = [f"前の語「{prev}」の「{key}」から始められる語を実辞書から探しました"]
    else:
        why = ["先手なので、よく使う名詞から始めました"]
    if forbid_ends:
        why.append("「" + "」や「".join(forbid_ends) + "」で終わる語は除きました")
    if want_len:
        why.append(f"{want_len} 文字前後を好み")
    mv.reason = "、".join(why)
    ok, _why = check(mv.word, rules)
    mv.legal = ok
    if not ok:
        mv.notes.append("選定後に規則検査で落ちたため、別の語に替えます: " + _why)
        alt = [s for p, s, _r in scored if p[0] == 0 and s != mv.word]
        mv.word = alt[0] if alt else mv.word
    return mv


def claims_for_move(mv: Move, *, prev: str, our_turn: bool) -> list[Claim]:
    out: list[Claim] = []
    if mv.word:
        out.append(Claim(kind="answer", content=mv.word, subject="turn", source="lex",
                         weight=0.94, extra={"reading": mv.reading, "prev": prev,
                                             "candidates": mv.candidates[:5]}))
    if mv.reason:
        out.append(Claim(kind="note", content=mv.reason, source="lex", weight=0.6))
    if mv.violation:
        out.append(Claim(kind="correction", content=mv.violation, source="lex", weight=0.9))
    return out


__all__ = ["Move", "pick", "judge", "word_from_turn", "rule_list", "claims_for_move"]
