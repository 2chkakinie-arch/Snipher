"""要約 — 指定された *件数* と *短さ* で、材料の中身だけを使って要点を作る。

生成モデルに要約させると「それっぽい嘘の一文」が混ざります。ここは抽出型です:

    1. 材料を文に割る（「〜であり、」のような *述語の無い切れ目* では割らない）
    2. 内容語（名詞・動詞・形容詞・副詞）の出現頻度から文の重みを出す
    3. 指定件数になるまで、隣の文と **節を繋いで** 1 つの要点にする（情報を落とさない）
    4. 「短く」なら節（、）の単位で詰める。述語を失う詰め方は使わない
    5. 指示された口調（だ・である／です・ます／〜だよ）に組み替える

材料に無い語を足さないので、要約結果は必ず原文の語で検証できます
（`coverage` がその証拠で、run.verify が閾値未満を弾きます）。
"""

from __future__ import annotations

import re
from collections import Counter

from ..lang import lex
from ..lang.phonetics import normalize
from .style import clause_trim, count_chars, fit_length, is_predicate_end, restyle

#: 文末（。！？）と *述語を持った接続*（〜であり、／〜が、）だけで割る。
#: 「〜を行い、」のような連用止めでは割らない（割ると非文ができる）。
_SPLIT = re.compile(r"(?<=[。！？!?；;\n])|(?<=であり、)|(?<=でなく、)|(?<=ではなく、)")
_NOISE = re.compile(
    r"^(?:なお|ちなみに|また|そして|さらに|加えて|ただし|一方で|つまり|すなわち|まず|次に|"
    r"最後に|例えば|たとえば|そこで|しかし|だが|しかも|なお)[、,]?\s*")

_POS_KEEP = ("名詞", "動詞", "形容詞", "副詞")
_STOP = {"もの", "こと", "ため", "場合", "とき", "よう", "なら", "これ", "それ", "あれ", "この",
         "その", "あの", "どこ", "何", "ある", "いる", "する", "なる", "できる", "ない", "の",
         "に", "は", "を", "が", "と", "で", "も", "な", "よ", "ね", "など", "さらに"}


def split_units(text: str) -> list[str]:
    """材料を「要点の単位」に割る。"""
    src = str(text or "")
    if re.search(r"[ぁ-んァ-ヶ一-龯]", src):
        src = re.sub(r"[ \t]+", "", src)
    out: list[str] = []
    for piece in _SPLIT.split(src):
        s = (piece or "").strip()
        if not s:
            continue
        s = _NOISE.sub("", s).strip()
        if len(s) < 4:
            if out:
                out[-1] = out[-1] + s
            continue
        out.append(s)
    return out


def content_words(text: str) -> list[str]:
    try:
        pairs = lex.bank().segment(str(text or ""))
    except Exception:  # noqa: BLE001
        pairs = []
    words = [w for w, pos in pairs
             if str(pos).split("/")[0] in _POS_KEEP and len(w) >= 2 and w not in _STOP]
    if not words:
        words = re.findall(r"[一-龯ァ-ヶー]{2,}|[A-Za-z][A-Za-z0-9_]{2,}", str(text or ""))
    return words


def _weights(units: list[str], freq: Counter) -> list[float]:
    total = sum(freq.values()) or 1
    out: list[float] = []
    for i, u in enumerate(units):
        words = content_words(u)
        score = sum(freq[w] / total for w in words)
        score *= 1.0 + 0.12 * len([w for w in words if freq[w] >= 2])
        n = max(1, count_chars(u))
        score = score / (n ** 0.35)
        if i == 0:
            score *= 1.18
        if i == len(units) - 1 and len(units) > 2:
            score *= 1.06
        if re.search(r"(?:つまり|すなわち|要するに|結論|ポイントは|重要|必要|不可欠)", u):
            score *= 1.08
        out.append(round(score, 6))
    return out


def _end_period(text: str) -> str:
    t = str(text or "").rstrip().rstrip("、，")
    return t if re.search(r"[。！？!?…]$", t) else t + "。"


def _compress(unit: str, budget: int, freq: Counter) -> str:
    """1 文を budget に収める（節単位。述語を失う詰め方はしない）。"""
    unit = unit.strip()
    if count_chars(unit) <= budget:
        return _end_period(unit)
    trimmed = clause_trim(unit, budget)
    if trimmed and is_predicate_end(trimmed.rstrip("。！？!?")):
        return _end_period(trimmed)
    parts = [p.strip() for p in re.split(r"[、，]", unit) if p.strip()]
    if len(parts) >= 2:
        # 述語を持つ *後ろ* の節を残す（前の節だけ残すと非文になる）
        for i in range(len(parts) - 1, 0, -1):
            cand = "、".join(parts[i:])
            if count_chars(cand) <= budget and is_predicate_end(parts[-1]):
                return _end_period(cand)
        best = max(parts[:-1], key=lambda p: sum(freq[w] for w in content_words(p)))
        cand = f"{best}、{parts[-1]}"
        if count_chars(cand) <= budget and is_predicate_end(parts[-1]):
            return _end_period(cand)
        if is_predicate_end(best) and count_chars(best) <= budget:
            return _end_period(best)
    return _end_period(unit)


def _join_units(units: list[str], budget: int, freq: Counter) -> str:
    """隣り合う文を 1 つの要点にまとめる（予算の限り情報を落とさない）。

    文と文は「、」で繋ぎません（繋ぐと非文になります）。*述語で終わっている文* の
    後ろには次の文をそのまま置き、入りきらない文は述語を持つ末尾の節だけを残します。
    """
    if not units:
        return ""
    units = [u.strip() for u in units if u and u.strip()]
    if not units:
        return ""
    if len(units) == 1:
        return _compress(units[0], budget, freq)
    out = _end_period(_NOISE.sub("", units[0]))
    for u in units[1:]:
        body = _NOISE.sub("", u).strip()
        if not body:
            continue
        cand = _end_period(body)
        if count_chars(out) + count_chars(cand) <= budget:
            out += cand                                  # 文 + 文（、では繋がない）
            continue
        room = max(8, budget - count_chars(out))
        tail = _fit_suffix(body.rstrip("。！？!? "), room, freq)
        if tail:
            head = out.rstrip("。")
            sep = "" if re.search(r"(?:であり|でなく|ではなく)$", head) else "、"
            out = _end_period(head + sep + tail)
            continue
        short = _compress(body, max(16, int(budget * 0.6)), freq)
        if short and count_chars(out) + count_chars(short) <= budget * 1.25:
            out += short                                 # 独立した 1 文として足す
    if count_chars(out) > budget * 1.3:
        trimmed = clause_trim(out, int(budget * 1.3))
        if trimmed and is_predicate_end(trimmed.rstrip("。！？!?")):
            out = _end_period(trimmed)
    return out


def _suffix_options(body: str, room: int, freq: Counter) -> list[str]:
    """文の末尾から作れる *述語を持つ* 候補を「情報を保つ順」に並べる。

    末尾の節（述語を含む）を必ず先頭にするので、「〜であり。」のような
    途中で切れた形だけが返ることはありません。
    """
    clauses = [c.strip("、， ") for c in re.split(r"(?<=[、，])", body) if c.strip()]
    if not clauses:
        return []
    heavy = max(clauses[:-1], key=lambda c: sum(freq[w] for w in content_words(c))) \
        if len(clauses) >= 2 else ""
    opts: list[str] = []
    for i in range(len(clauses) - 1, -1, -1):
        cand = "、".join(clauses[i:])
        if is_predicate_end(clauses[-1]) and cand not in opts:
            opts.append(cand)
    if heavy and clauses:
        cand = heavy + "、" + clauses[-1]
        if is_predicate_end(clauses[-1]) and cand not in opts:
            opts.append(cand)
    for i in range(len(clauses) - 1, -1, -1):
        cand = "、".join(clauses[i:])
        if cand not in opts and is_predicate_end(cand):
            opts.append(cand)
    scored = [c for c in opts if count_chars(c) <= room]
    scored.sort(key=lambda c: (count_chars(c), sum(freq[w] for w in content_words(c))),
                reverse=True)
    return scored


def _fit_suffix(body: str, room: int, freq: Counter) -> str:
    """文の末尾から *述語を持つ* 部分列を取り出す（予算 room の限り長く・重く）。"""
    got = _suffix_options(body, room, freq)
    return got[0] if got else ""


def _fit_point(point: str, room: int, freq: Counter) -> str:
    """1 つの要点を room に収める。述語を失う形にはしない。"""
    if count_chars(point) <= room:
        return point
    # まず *文の単位* で収まるぶんだけ残す（文を途中で切らない）
    sents = [x for x in re.split(r"(?<=[。！？!?])", point) if x.strip()]
    if len(sents) > 1:
        keep: list[str] = []
        used = 0
        for s_ in sents:
            if used + count_chars(s_) <= room or not keep:
                keep.append(s_)
                used += count_chars(s_)
        if len(keep) < len(sents):
            return "".join(keep).strip()
    trimmed = clause_trim(point, room)
    if trimmed and is_predicate_end(trimmed.rstrip("。！？!?")):
        return _end_period(trimmed)
    body = point.rstrip("。！？!? ")
    for cand in _suffix_options(body, room, freq):
        return _end_period(cand)
    parts = [c.strip("、， ") for c in re.split(r"(?<=[、，])", body) if c.strip()]
    if parts and is_predicate_end(parts[-1]) and count_chars(parts[-1]) <= room:
        return _end_period(parts[-1])
    return point


def _fit_suffix_old(body: str, room: int, freq: Counter) -> str:
    """(旧実装)"""
    clauses = [c for c in re.split(r"(?<=[、，])", body) if c.strip()]
    if not clauses:
        return ""
    fits = []
    for i in range(len(clauses)):
        cand = "".join(clauses[i:]).strip("、， ")
        if cand and count_chars(cand) <= room and is_predicate_end(cand):
            fits.append(cand)
    if fits:
        # 長いほど情報を保つ。同程度なら内容語の重いほう。
        fits.sort(key=lambda c: (count_chars(c), sum(freq[w] for w in content_words(c))),
                  reverse=True)
        best = fits[0]
        if count_chars(best) < room * 0.5 and len(clauses) >= 2:
            head = max(clauses[:-1], key=lambda c: sum(freq[w] for w in content_words(c)))
            cand = head.strip("、， ") + "、" + best
            if count_chars(cand) <= room and is_predicate_end(cand):
                return cand
        return best
    tail = clauses[-1].strip("、， ")
    if tail and is_predicate_end(tail) and count_chars(tail) <= room:
        return tail
    return ""


def summarize(text: str, *, bullets: int = 3, brief: bool = False, tone: str = "",
              register: str = "", numbered: bool = False, max_chars: int = 0,
              bullet_char: str = "・", lead: str = "") -> dict:
    """材料を要点にする。返るのは {text, points, coverage, units, notes}。"""
    src = str(text or "").strip()
    notes: list[str] = []
    units = split_units(src)
    if not units:
        return {"text": "", "points": [], "coverage": 0.0, "units": 0, "notes": ["材料が空"]}
    freq: Counter = Counter()
    for u in units:
        freq.update(set(content_words(u)))
    weights = _weights(units, freq)
    n = max(1, int(bullets or 3))
    budget = 62 if brief else 92
    if max_chars:
        budget = max(28, min(budget, int(max_chars / max(1, n)) - 2))

    # ---- 隣接する文をまとめて n 個のグループにする -------------------------- #
    groups: list[list[int]] = []
    if len(units) <= n:
        groups = [[i] for i in range(len(units))]
    else:
        order = sorted(range(len(units)), key=lambda i: -weights[i])
        seed = sorted(order[:n])
        # 選ばれた文を *各要点の末尾* にして、その手前の文を同じ要点にまとめる
        start = 0
        for end in seed[1:] + [len(units)]:
            groups.append(list(range(start, end)))
            start = end
    points = [_join_units([units[i] for i in g], budget, freq) for g in groups]
    points = [p for p in points if p]
    # 1 項目の長さは「短く」と言われたとき *実際に* 短くなっている必要がある
    room = budget + (24 if brief else 40)
    if any(count_chars(p) > room for p in points):
        points = [_fit_point(p, room, freq) for p in points]
        notes.append(f"per_point:{room}字")

    if tone or register:
        styled, fixes = restyle("\n".join(points), tone=tone, register=register, bullets=True)
        points = [x for x in styled.split("\n") if x.strip()]
        notes.extend(fixes)

    if numbered:
        lines = [f"{i}. {p}" for i, p in enumerate(points, 1)]
    else:
        lines = [f"{bullet_char}{p}" for p in points]
    body = "\n".join(lines)
    if lead:
        body = f"{lead}\n{body}"
    cap = n * (budget + 12)
    if count_chars(body) > cap:
        trimmed, fixes = fit_length(lines, hard_max=cap)
        body = ("\n".join(trimmed)) if not lead else lead + "\n" + "\n".join(trimmed)
        notes.extend(fixes)
    if max_chars and count_chars(body) > max_chars:
        kept, fixes = fit_length(lines, hard_max=max(1, max_chars - count_chars(lead or "")))
        body = "\n".join(kept)
        notes.extend(fixes)

    return {"text": body, "points": points, "coverage": round(_coverage(points, src), 3),
            "units": len(units), "notes": notes, "groups": [[units[i] for i in g] for g in groups]}


def _overlap(a: str, b: str, n: int = 5) -> float:
    ga = {a[i:i + n] for i in range(max(0, len(a) - n + 1))}
    gb = {b[i:i + n] for i in range(max(0, len(b) - n + 1))}
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


def _coverage(points: list[str], src: str) -> float:
    """要点の語がどれだけ材料に出ているか（＝ねつ造していないかの指標）。"""
    src_words = set(content_words(src))
    if not src_words:
        return 1.0
    used: set[str] = set()
    total = 0
    for p in points:
        words = content_words(p)
        total += len(words)
        used.update(w for w in words if w in src_words)
    if not total:
        return 1.0
    return len(used) / max(1, min(total, len(src_words)))


__all__ = ["summarize", "split_units", "content_words"]
