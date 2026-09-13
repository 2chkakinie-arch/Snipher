"""会話状態 — 「今この会話が何をしている最中か」を復元する。

定型文的な見え方のいちばんの原因は、**前ターンを覚えていない**ことです。
ここでは履歴から (1) 進行中の共同作業 (2) 直前に発話された語 (3) ユーザーの文体・言語 (4) 指す対象
を再構成する。ルール・手番・語の在庫まで覚えるので、遊びも相談も「続き」として成立する。
"""

from __future__ import annotations

import re

from ..lang.lex import bank
from ..lang.phonetics import chain_key, normalize, to_hiragana
from .frame import Activity, Rule
from .rules import compile_rules, from_definition

_QUOTED = re.compile(r"[「『]([^」』\n]{1,12})[」』]")
_BULLET = re.compile(r"^\s*[・\-]|^\s*\d+[.)、]")
_DONE_MARKS = ("やめる", "終わり", "おわり", "もういい", "やめとく", "終了", "止める", "やめた")
_CONT_MARKS = ("続き", "もう一回", "もう1回", "もう一問", "もっと", "つづけて", "続けて")


def _assistant_texts(history: list[dict]) -> list[str]:
    return [str(m.get("content") or "") for m in history or [] if m.get("role") == "assistant"]


def _user_texts(history: list[dict]) -> list[str]:
    return [str(m.get("content") or "") for m in history or [] if m.get("role") == "user"]


def mentions(text: str) -> list[str]:
    """文中に「言及」として出てくる語（引用符・見出し・行頭）。

    語の連鎖の手番復元や「その語をもう一度」の指代に使う、汎用の抽出。
    """
    t = str(text or "")
    out: list[str] = [m.group(1).strip() for m in _QUOTED.finditer(t) if m.group(1).strip()]
    return out


def turn_word(text: str) -> str:
    """1 ターン分の発話から「出した語」を取り出す（引用 > 行頭 > 最長の既知語）。"""
    t = normalize(text).strip()
    if not t:
        return ""
    for cand in mentions(t):
        if 1 <= len(cand) <= 10:
            got = cand.strip("。、,.!?！？ ")
            return re.sub(r"(?:でした|です|だね|ですね|だよ|かな|っす)$", "", got) or got
    first = re.split(r"[\s。、,!?！？]+", t)[0]
    first = first.strip("「」『』()（）「」【】・")
    first = re.sub(r"(?:でした|です|ですね|だね|だよ|かな|っす|だわ|だよな)$", "", first) or first
    if 1 <= len(first) <= 12:
        return first
    return ""



_NAME_PAT = re.compile(r"(?:私は|僕は|俺は|ボクは|わたしは|自分は|名前は|私の名前は)"
                       r"([^。、,]{1,16}?)(?:です|だよ|だぞ|って呼んで|と呼んで|と申します|といいます)"
                       r"|([^。、,・\s]{1,12}?)(?:と申します|と申す|といいます|と言う者です)")
_NAME_INNER = re.compile(r"[がのをにでとものやへ]")


def _looks_like_name(x: str) -> bool:
    """名乗りとして不自然でないか（助詞・数量・記号を含まない 1〜12 文字）。"""
    if not x or len(x) > 12 or _NAME_INNER.search(x) or re.search(r"[0-9０-９]", x):
        return False
    return not re.search(r"(歳|才|年|月|日|人|回|番|円|個|種類)", x)


def _user_facts(user_texts: list[str]) -> dict:
    """会話の中で相手が自分で明した事柄を拾う（名前・年齢・好き嫌い）。

    記憶として持つのはこの 1 箇所だけに限る。「先に言っていたこと」を根拠に答えるための
    材料で、引き当てた文とターン数を覚えるので、答えるときに出所を示せる。
    """
    facts: dict = {}
    for idx, raw in enumerate(user_texts):
        t = normalize(raw)
        # 名乗り: 明示的な主語（私は／名前は）が有るときだけ拾う。
        # 「猫が好きです」を名前に誤るのを避けるため、内容語の途中に助詞が入ったら捨て、
        # 数量表現（36 歳など）も名前として数えない。
        for m in _NAME_PAT.finditer(t):
            cand = (m.group(1) or m.group(2) or "").strip("、。.・ ")
            if _looks_like_name(cand):
                facts["name"] = cand
                facts["name_at"] = idx + 1
                break
        m = re.search(r"(?:私は|僕は|俺は|ボクは|自分は)?[^。、,]{0,6}?(\d{1,3})\s*(?:歳|さい)", t)
        if m:
            facts["age"] = int(m.group(1))
            facts["age_at"] = idx + 1
        m = re.search(r"(?:私は|僕は|俺は|ボクは|自分は)([^。、,]{1,20}?)(?:が好き|が苦手|が嫌い|大好き|が欲しい)", t)
        if m:
            key = "dislikes" if ("嫌い" in m.group(0) or "苦手" in m.group(0)) else "likes"
            want = m.group(1).strip("、。 ")
            if want and not _NAME_INNER.search(want):
                facts[key] = want
                facts[key + "_at"] = idx + 1
    return facts



class ConversationState:
    """履歴から作る会話の横断状態（軽量・同期・副作用なし）。"""

    def __init__(self, history: list[dict] | None = None, *, kb=None):
        self.history = history or []
        self.kb = kb
        self.users = _user_texts(self.history)
        self.assistants = _assistant_texts(self.history)
        self.turn = len(self.users)
        self.register = "polite"
        self.language = "ja"
        self.last_user = self.users[-1] if self.users else ""
        self.last_assistant = self.assistants[-1] if self.assistants else ""
        self.activity: Activity | None = None
        self.recent_words: list[str] = []
        self.user_facts = _user_facts(self.users)
        self._detect_style()
        self._restore_activity()

    # ------------------------------------------------------------------ #
    def _detect_style(self) -> None:
        body = " ".join(self.users[-3:])
        # 敬体/常体は発話ごとに確かめる。遊びの最中は 1 文だけ丁寧語が混ざることが
        #あるので、「直近のどこかでくだけていて、どこにも敬語が無ければ」常体とみる。
        casual_pat = re.compile(r"(しよ$|しよう$|しような|しようぜ|だよ|だろ|だな|だね|やろ|"
                                r"遊ぼ|あそぼ|してくれる|教えてよ|教えてよ|くれない)")
        polite_pat = re.compile(r"(です|ます|でしょう|ください|ですね|ますね)")
        window = self.users[-8:]      # 一度定めた文体は会話を離れないので、後ろまで見る
        casual = any(casual_pat.search(normalize(u)) for u in window)
        polite = any(polite_pat.search(normalize(u)) for u in window)
        if (casual or re.search(r"(である|タメ口|くだけた)", body)) and not polite:
            self.register = "plain"
        ascii_ratio = sum(c.isascii() and c.isalpha() for c in body) / max(1, len(body))
        if ascii_ratio > 0.72:
            self.language = "en"

    def _restore_activity(self) -> None:
        """直近のターンから進行中の作業（語の連鎖・クイズ等）を復元する。"""
        if not self.users:
            return
        activity: Activity | None = None
        used: list[str] = []
        created = False
        for u, a in zip(self.users, self.assistants):
            lu, la = normalize(u), normalize(a)
            created = False
            if activity is None:
                if re.search(r"(やめ|終わり|もういい|結構)", lu) and any(x in lu for x in _DONE_MARKS):
                    continue
                name = _activity_name(lu)
                if not name:
                    continue
                rules = compile_rules(lu)
                if not any(r.kind == "chain" for r in rules):
                    rules = rules + _rules_from_knowledge(name, kb=self.kb)
                activity = Activity(name=name, kind="chain" if any(r.kind == "chain" for r in rules)
                                    else "open", rules=rules, open=True)
                created = True
            if activity is None:
                continue
            if any(x in lu for x in _DONE_MARKS):
                activity.open = False
                break
            uw, aw = turn_word(lu), turn_word(la)
            # 遊びを始めたターンは、発話そのもの（「しりとりしよ」）が語ではない。
            # 手の打ち手（assistant 側）だけを登録する。
            if uw and not created:
                used.append(uw)
                activity.last_word = uw
            if aw:
                used.append(aw)
                activity.our_word = aw
            activity.turn += 1
        if activity is not None:
            activity.used = used
            self.activity = activity
            self.recent_words = used[-10:]

    # ------------------------------------------------------------------ #
    def expects_continuation(self, text: str) -> bool:
        """発話自体は意味をなさないが、進行中の作業の「手番」として読めるか。

        例: しりとり中に「りんご」と打たれた / クイズ中に答えが打たれた。
        """
        if self.activity is None or not self.activity.open:
            return False
        t = normalize(text)
        if not t or len(t) > 24:
            return False
        if re.search(r"(とは|なぜ|どうして|教えて|って何|方法|調べ)", t):
            return False
        return True

    def previous_token(self) -> str:
        """连锁の相手側（=私たちが最後に言った語）。"""
        if self.activity is None:
            return ""
        return self.activity.our_word or self.activity.last_word

    def forbid(self) -> list[str]:
        return list(self.recent_words)

    def as_dict(self) -> dict:
        return {"turn": self.turn, "register": self.register, "language": self.language,
                "activity": self.activity.as_dict() if self.activity else None,
                "recent_words": self.recent_words[-6:]}


_ACTIVITY_WORDS = {"しりとり": "chain", "けとり": "chain", "語の連鎖": "chain",
                  "last letter": "chain", "word chain": "chain"}


def _activity_name(text: str) -> str:
    t = normalize(text).lower()
    if not t:
        return ""
    if not re.search(r"(しよう|しよ|やろう|やろ|遊ぼ|あそぼ|しませんか|したい|つきあって|付き合って|play|let'?s)", t):
        return ""
    for name in _ACTIVITY_WORDS:
        if name in t:
            return name
    return ""


def _rules_from_knowledge(name: str, *, kb=None) -> list[Rule]:
    """話題名に対して手元の知識が「遊び方」を持っていれば、そこから規則を読む。

    特別なゲーム処理は書かない。知識ベースの説明文に規則が書いてあれば誰でも従う。
    """
    out: list[Rule] = [Rule(kind="chain", value="", raw=f"語の連鎖（{name}）")]
    if kb is None:
        return out
    try:
        from ..ground.evidence import exact_definition

        text = exact_definition(name, kb=kb)        # 部分一致で引いた別話題は使わない
        if text:
            derived = from_definition(text, topic=name)
            if derived:
                return derived + [r for r in out if r.kind == "chain"][:1]
    except Exception:  # noqa: BLE001
        pass
    return out


def known_words(words: list[str]) -> list[str]:
    b = bank()
    return [w for w in words if b.has(w)]


def chain_target(prev: str) -> str:
    return chain_key(to_hiragana(normalize(prev)))


__all__ = ["ConversationState", "mentions", "turn_word", "known_words", "chain_target"]
