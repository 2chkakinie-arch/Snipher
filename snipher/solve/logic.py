"""論理の仕事 — 全称命題の適用と、規則（条件文）の適用。

ここで扱うのは *材料に書かれている文だけ* を使う 2 種類の推論です。

    1. 三段論法（全称命題）
       「人間は必ず息をします。」（= すべての人間は息をする）
       「太郎は人間です。」（= 太郎は人間に属する）
       → 太郎は息をするか？ → はい

    2. 規則の適用（条件文）
       「赤信号では止まれ、青信号では進め。」（= 条件 → 行動）
       「いま信号は赤です。」（= いまの条件）
       → 赤を含む条件の行動を採る → 止まる

どちらも、答えを書くのに使った文を `steps` にそのまま残します。材料の中で条件が
見つからないときは None を返します（推測で埋めない）。
"""

from __future__ import annotations

import re

from ..lang.phonetics import normalize
from .math import Solution

# --------------------------------------------------------------------------- #
# 全称命題（〜は必ず〜する / 〜は必ず〜だ）
# --------------------------------------------------------------------------- #
_UNIVERSAL = re.compile(
    r"^(?P<subject>[^、。]{1,24}?)は(?P<mid>必ず|すべて|みな|いつも|常に)?"
    r"(?P<predicate>[^。]{1,60}?)(?:です|である|ます|だ|する|します)$")
_INSTANCE = re.compile(r"^(?P<name>[^、。]{1,24}?)は(?P<klass>[^、。]{1,24}?)です$")
_YN_ASK = re.compile(r"(?P<name>[^、。]{1,24}?)は(?P<predicate>[^、。]{1,60}?)(?:ます)?か[？?]?$")


def _sentences(text: str) -> list[str]:
    body = normalize(str(text or ""))
    body = re.sub(r"[。！!]+\s*$", "", body)
    return [s.strip(" 　、") for s in re.split(r"[。\n]", body) if s.strip()]


def _predicate_key(predicate: str) -> str:
    """述語の核（「息をします」→「息」）を取り出す。"""
    p = normalize(predicate).strip(" 　、。")
    p = re.sub(r"(?:し?ます|し?た|する|です|である|だ|している)$", "", p).strip()
    p = re.sub(r"^(?:必ず|きっと|かならず)", "", p).strip()
    p = re.sub(r"(?:を|が|は)$", "", p).strip()
    return p


def syllogism(text: str, *, want: str = "") -> Solution | None:
    """全称命題 + 個体から、はい／いいえ を決める。"""
    sentences = _sentences(text)
    if len(sentences) < 2:
        return None
    universal: tuple[str, str] | None = None
    universal_raw = ""
    instances: list[tuple[str, str]] = []
    asks: list[str] = []
    for sent in sentences:
        if re.search(r"(?:か|ですか|ますか)\s*[？?]?\s*$", sent) and "？" in sent + "?":
            asks.append(sent)
            continue
        if re.search(r"(?:かを|ください|答えて|下さい)", sent):
            asks.append(sent)
            continue
        m_u = _UNIVERSAL.match(sent)
        if m_u and m_u.group("mid"):
            universal = (m_u.group("subject"), _predicate_key(m_u.group("predicate")))
            universal_raw = sent
            continue
        m_i = _INSTANCE.match(sent)
        if m_i:
            instances.append((m_i.group("name"), m_i.group("klass")))
    if not universal or not instances:
        return None
    subj, pred = universal
    # 個体が全称命題の主語に属するか（表記ゆれは含む／含まれるで見る）
    holder: tuple[str, str] | None = None
    for name, klass in instances:
        if klass == subj or klass in subj or subj in klass:
            holder = (name, klass)
            break
    if holder is None:
        return None
    name, klass = holder
    # 問いの述語が全称命題の述語と一致するか
    asked_pred = ""
    for sent in asks:
        m = _YN_ASK.search(strip_question(sent))
        if m and (m.group("name") == name or name in m.group("name")):
            asked_pred = _predicate_key(m.group("predicate"))
            break
    if not asked_pred:
        asked_pred = pred
    hit = bool(asked_pred) and (asked_pred in pred or pred in asked_pred)
    negation = bool(re.search(r"(?:し?ない|ません|無い|ない)", asked_pred))
    answer = "いいえ" if negation else ("はい" if hit else "")
    if not answer:
        return None
    steps = [f"全称命題: 「{universal_raw}」（材料の文そのまま）",
             f"個体: 「{name}は{klass}です」→ {name} は {subj} に属する",
             f"{name} に全称命題を当てると「{name}は{pred}する」", f"問いの述語「{asked_pred}」と一致"]
    return Solution(answer=answer, steps=steps, kind="syllogism", verified=True,
                    detail={"universal": {"subject": subj, "predicate": pred},
                            "instance": {"name": name, "klass": klass},
                            "asked": asked_pred, "answer": answer})


def strip_question(text: str) -> str:
    return re.sub(r"^(?:質問|問い|問)\s*[：:]\s*", "", str(text or "")).strip()


# --------------------------------------------------------------------------- #
# 規則の適用（条件文 → 行動）
# --------------------------------------------------------------------------- #
_IMPERATIVE_TAIL = re.compile(
    r"(?:しろ|せよ|せい|しなさい|して下さい|してください|ください|下さい|するな|してはいけない|"
    r"すべし|べし|だめ|禁止|可|よし|よい|OK)$")

#: 活用表（実辞書 16,319 語）から「命令形 → 辞書形」の索引を作る。命令文の動詞を
#: 辞書の活用で戻すために使います（語尾の文字差し替えに頼らない）。
_IMPERATIVE_INDEX: dict[str, str] | None = None
_VERB_LEMMAS: set[str] | None = None


def _imperative_index() -> dict[str, str]:
    global _IMPERATIVE_INDEX
    if _IMPERATIVE_INDEX is None:
        table: dict[str, str] = {}
        try:
            from ..lang import morph

            for lemma, forms in morph.table().items():
                imp = forms.get("命令ｅ") or forms.get("命令ｒｏ") or forms.get("命令ｉ")
                if imp and imp not in table:
                    table[imp] = lemma
        except Exception:  # noqa: BLE001
            table = {}
        _IMPERATIVE_INDEX = table
    return _IMPERATIVE_INDEX


def is_command(action: str) -> bool:
    """行動の文が *命令* かどうか（規則の右辺をここで見分ける）。"""
    tail = str(action or "").strip().strip("。！!").split("、")[-1].strip(" 　")
    if _IMPERATIVE_TAIL.search(tail):
        return True
    if _imperative_index().get(tail, ""):
        return True
    # 「〜するな」の禁止も命令（動詞の辞書形 + な）
    if tail.endswith("な") and tail[:-1] in _dictionary_verbs():
        return True
    return False


def _dictionary_verbs() -> set[str]:
    """実辞書の動詞（辞書形）の集合。禁止形を命令として見分けるために使います。"""
    global _VERB_LEMMAS
    if _VERB_LEMMAS is None:
        try:
            from ..lang import morph

            _VERB_LEMMAS = set(morph.table())
        except Exception:  # noqa: BLE001
            _VERB_LEMMAS = set()
    return _VERB_LEMMAS
_CLAUSE = re.compile(r"(?P<cond>[^、。]{1,30}?)(?:では|なら|ならば|のときは|時は|の場合は)"
                     r"(?P<action>[^、。]{1,40})")
_STATE = re.compile(r"(?:いま|今|現在)?\s*(?P<obj>[^、。]{1,20}?)は(?P<value>[^、。]{1,20}?)(?:です|だ|である)")


def apply_rule(text: str, *, want: str = "") -> Solution | None:
    """規則（〜では〜せよ）と、いまの状態から、採る行動を決める。"""
    body = normalize(str(text or ""))
    rules: list[tuple[str, str, str]] = []
    for m in _CLAUSE.finditer(body):
        cond = m.group("cond").strip(" 　、")
        action = m.group("action").strip(" 　、。")
        if not cond or not action:
            continue
        if not is_command(action):
            continue
        rules.append((cond, action, m.group(0)))
    if not rules:
        return None
    states: list[tuple[str, str]] = []
    for m in _STATE.finditer(body):
        obj = m.group("obj").strip(" 　、。")
        val = m.group("value").strip(" 　、。")
        # 規則文の中の条件（「赤信号では…」）は状態ではない
        if any(obj and obj in r[2] for r in rules):
            continue
        states.append((obj, val))
    if not states:
        return None
    for obj, val in states:
        for cond, action, raw in rules:
            key = re.sub(r"信号|では|のとき|なら", "", cond)
            if val and (val in cond or cond in f"{val}{obj}" or key and val in key):
                verb = _action_verb(action)
                steps = [f"規則（材料の文）: 「{raw}」",
                         f"いまの状態: 「{obj}は{val}です」（材料の文）",
                         f"「{val}」は条件「{cond}」に当たる → 行動は「{action}」"]
                return Solution(answer=verb, steps=steps, kind="rule", verified=True,
                                detail={"rule": raw, "state": {"obj": obj, "value": val},
                                        "action": action, "answer": verb})
    return None


def _action_verb(action: str) -> str:
    """命令形を、答えとして読める形（止まれ → 止まる）に直す。

    直し方は *実辞書の活用表* です（命令形 → 基本形）。表に無い語はそのまま返し、
    勝手に語尾を削りません。
    """
    a = re.sub(r"(?:してください|して下さい|ください|下さい|しなさい)$", "する", action)
    a = re.sub(r"(?:するな)$", "しない", a)
    a = a.rstrip("。！!")
    head, sep, tail = a.rpartition("、")
    lemma = _imperative_index().get(tail.strip(" 　"))
    if lemma:
        return (head + sep if sep else "") + lemma
    return a


__all__ = ["syllogism", "apply_rule", "strip_question"]
