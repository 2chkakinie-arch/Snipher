"""決める仕事 — 数の大小・オン/オフの反転・選択肢の解決。

ここに来るのは「材料（数・状態・選択肢）だけで答えが決まる」仕事です。
答えは *裸*（飾りなし）で返します。「数字だけで」「「オン」か「オフ」で」
「2 文字以内」のような形の指定に応えるのが役目です。

    数の比較 : 「10 と 5 はどちらが大きい」→ 10
    反転     : 「オフになっています。1 回押すと」→ オン（偶奇で決める）
    選択肢   : 「「東京」か「大阪」のどちらか」+ 問い → 知識ベースの QA・事実で
               裏取りし、当たる選択肢だけを返す。当たらなければ None
"""

from __future__ import annotations

import re

from ..lang.phonetics import normalize
from .math import Solution

# --------------------------------------------------------------------------- #
# 1) 数の比較
# --------------------------------------------------------------------------- #
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_COMPARE = re.compile(
    r"(?:どちら|どっち|どれ)(?:が|は|の|です|だ)?.{0,8}"
    r"(?P<adj>大きい|小さい|多い|少ない|長い|短い|高い|低い|広い|狭い|"
    r"早い|遅い|速い|大きい方|小さい方)|"
    r"(?P<adj2>大きい|小さい|多い|少ない|長い|短い|高い|低い|広い|狭い|"
    r"早い|遅い|速い)(?:方|ほう)(?:の|は|を)?.{0,8}(?:数字|数|だけ|答え|教え|は)")
_MAX_ADJ = ("大きい", "多い", "長い", "高い", "広い", "速い")
_MIN_ADJ = ("小さい", "少ない", "短い", "低い", "狭い", "遅い", "早い")


def compare_numbers(text: str) -> Solution | None:
    """「10 と 5 はどちらが大きいですか」→「10」（数字だけ）。

    数が 2 つ以上あり、大小・多少の比較を求めているときにだけ答える。
    """
    body = normalize(str(text or ""))
    m = _COMPARE.search(body)
    if not m:
        return None
    adj = (m.group("adj") or m.group("adj2") or "").replace("方", "")
    nums = _NUMBER.findall(body)
    if len(nums) < 2:
        return None
    # 字数指定（2 文字以内）のような数は比較の対象ではない
    nums = nums[:6]
    try:
        vals = [(float(x), x) for x in nums]
    except ValueError:
        return None
    want_max = any(a in adj for a in _MAX_ADJ)
    want_min = any(a in adj for a in _MIN_ADJ)
    if want_max == want_min:
        return None
    picked = max(vals, key=lambda v: v[0]) if want_max else min(vals, key=lambda v: v[0])
    others = [x for _, x in vals if x != picked[1]]
    steps = [f"材料の数（{'／'.join(x for _, x in vals)}）を比べた",
             f"「{adj}」を求めている → {'最大' if want_max else '最小'}の {picked[1]}"]
    return Solution(answer=picked[1], steps=steps, kind="compare", verified=True,
                    detail={"numbers": [x for _, x in vals], "adj": adj,
                            "answer": picked[1], "others": others})


# --------------------------------------------------------------------------- #
# 2) オン/オフの反転
# --------------------------------------------------------------------------- #
_STATE_WORD = re.compile(r"(?P<state>オン|オフ|on|off)")
_PRESS = re.compile(r"(?P<n>\d+)\s*回.{0,8}(?:押し|押す|押した|切替|切り替え|トグル|操作|たた)")
_PRESS_ONCE = re.compile(r"(?:1\s*回|一回|ひと押し|押すと|押したら|押して|スイッチを|トグル)")
_ON_OFF_ASK = re.compile(r"「オン」か「オフ」|オン.?か.?オフ|オン.?と.?オフのどちら|どうなりますか")


def toggle_switch(text: str) -> Solution | None:
    """「オフになっています。スイッチを 1 回押すと」→「オン」。

    いまの状態と押す回数（偶奇）だけで決める。状態が読めなければ None。
    """
    body = normalize(str(text or ""))
    states = [m.group("state") for m in _STATE_WORD.finditer(body)]
    if not states:
        return None
    m_n = _PRESS.search(body)
    if m_n:
        n = int(m_n.group("n"))
    elif _PRESS_ONCE.search(body):
        n = 1
    else:
        return None
    # いまの状態は、選択肢（「オン」か「オフ」）ではない方の記述から読む
    bare = re.sub(r"[「『].*?[」』]", "", body)
    m_state = _STATE_WORD.search(bare)
    current = ""
    if m_state:
        current = m_state.group("state")
    if not current:
        # 「オフになっています」の形を直接読む
        m2 = re.search(r"(オン|オフ)(?:になっています|になって|です|だ|の状態)", body)
        current = m2.group(1) if m2 else ""
    if not current:
        return None
    current = "オン" if current.lower() in ("オン", "on") else "オフ"
    answer = current if n % 2 == 0 else ("オフ" if current == "オン" else "オン")
    steps = [f"いまの状態（材料）: {current}",
             f"押す回数（材料）: {n} 回（{'偶数→変わらない' if n % 2 == 0 else '奇数→反転する'}）",
             f"→ {answer}"]
    return Solution(answer=answer, steps=steps, kind="toggle", verified=True,
                    detail={"current": current, "presses": n, "answer": answer})


# --------------------------------------------------------------------------- #
# 3) 選択肢の解決 — 「「A」か「B」のどちらか」+ 問い
# --------------------------------------------------------------------------- #
_QUOTED_PAIR = re.compile(r"[「『\"](?P<a>[^「」『』\"']{1,12})[」』\"]\s*か\s*"
                          r"[「『\"](?P<b>[^「」『』\"']{1,12})[」』\"]")
_OR_WORDS = re.compile(r"(?P<a>[^、。？?「」]{1,12}?)(?:か|または|あるいは|若しくは|もしくは)"
                       r"(?P<b>[^、。？?「」]{1,12}?)(?:の|のうち)?(?:どちら|どっち|どれ)")
_WHICH = re.compile(r"どちら|どっち|どれ|いずれ")


def choice_options(text: str) -> list[str]:
    """「「東京」か「大阪」」のような選択肢を抜く（無いときは空）。"""
    body = normalize(str(text or ""))
    opts: list[str] = []
    for m in _QUOTED_PAIR.finditer(body):
        for g in (m.group("a"), m.group("b")):
            g = g.strip(" 　、。")
            if g and g not in opts and len(g) <= 12:
                opts.append(g)
    if not opts:
        m = _OR_WORDS.search(body)
        if m:
            for g in (m.group("a"), m.group("b")):
                g = re.sub(r"^(?:は|が|を|に|で|と|から|まで|より)", "", g.strip(" 　、。"))
                g = g.strip(" 　、。「」『』")
                # 選択肢は裸の名詞（助詞を含む「が好きな動物は」は選択肢ではない）
                if re.search(r"[がはをにでとへのもや]$|[がはをにでと]..", g):
                    continue
                if g and g not in opts and 1 <= len(g) <= 12:
                    opts.append(g)
            if len(opts) < 2:
                opts = []
    return opts[:4]


def _kb_evidence(query: str, kb) -> list[tuple[str, str]]:
    """問いに対する知識ベースの根拠文（(文, 出どころ)）を集める。"""
    out: list[tuple[str, str]] = []
    if kb is None:
        return out
    try:
        hit = kb.match_qa(query)
        if hit:
            _i, question, ans, _score = hit
            if ans:
                out.append((str(ans), f"qa:{question}"))
    except Exception:  # noqa: BLE001
        pass
    try:
        for item in kb.search(query, top_k=2) or []:
            topic = item.get("topic", "")
            for key in ("def", "facts"):
                vals = item.get(key)
                if isinstance(vals, str):
                    vals = [vals]
                for v in vals or []:
                    if v:
                        out.append((str(v), f"{topic}:{key}"))
    except Exception:  # noqa: BLE001
        pass
    return out[:12]


def resolve_choice(text: str, *, kb=None) -> Solution | None:
    """「「東京」か「大阪」のどちらか」+「日本の首都はどこ」→「東京」。

    知識ベースの QA・事実に当たる選択肢だけを返す。複数の選択肢が当たる・
    どれも当たらないときは None（当てずっぽうで選ばない）。
    """
    body = normalize(str(text or ""))
    opts = choice_options(body)
    if len(opts) < 2 or not _WHICH.search(body):
        return None
    # 問い文（選択肢の指定を除いた部分）
    query = re.sub(r"[「『][^「」『』]*[」』]\s*か\s*[「『][^「」『』]*[」』]"
                   r"(?:のどちら|のどっち|のどれ)?.{0,16}(?:答えて|ください|ですか|ますか)?", "", body)
    query = re.sub(r"\d+\s*文字(?:以上|以内|以下|まで)?", "", query)
    query = re.sub(r"(?:で|に)?(?:答えて|ください|下さい|お答え|おしえて|教えて)(?:ください)?$", "", query)
    query = query.strip(" 　、。？?")
    if kb is None:
        try:
            from ..knowledge import KnowledgeBase

            kb = KnowledgeBase.shared()
        except Exception:  # noqa: BLE001
            kb = None
    evidence = _kb_evidence(query or body, kb)
    if not evidence:
        return None
    hits: list[tuple[str, str]] = []
    for opt in opts:
        for sent, src in evidence:
            if opt and opt in sent:
                # 否定文（「大阪ではありません」）の中の語は答えにしない
                neg = re.search(re.escape(opt) + r".{0,8}(?:ではない|ではありません|じゃない|でない|とは限らない)", sent)
                if neg:
                    continue
                hits.append((opt, f"{src}「{sent[:60]}」"))
                break
    # ちょうど 1 つの選択肢だけが当たったときだけ答える
    winners = sorted({opt for opt, _ in hits})
    if len(winners) != 1:
        return None
    answer = winners[0]
    src = next(s for o, s in hits if o == answer)
    steps = [f"選択肢（材料）: {'／'.join(opts)}",
             f"根拠（知識ベース）: {src}",
             f"当たるのは「{answer}」だけ → {answer}"]
    return Solution(answer=answer, steps=steps, kind="choice", verified=True,
                    detail={"options": opts, "answer": answer, "evidence": src})


__all__ = ["compare_numbers", "toggle_switch", "choice_options", "resolve_choice"]
