"""日本語そのものの仕事 — 空欄を埋める・かなに直す・仲間を選ぶ・語の関係を引く・短く書く。

ここに来るのは「知識」ではなく「言葉の扱い」を頼まれた仕事です。だから答は
*その場で確かめられる材料* だけで作ります。材料が無ければ、作らずに None を返します。

    空欄補充 : 候補（助詞）を 1 つずつ入れて、5-gram の審判（`snipher/lm.py`）が
               文全体の流暢さを比べる。差（margin）が根拠として残る
    かな書き : 実辞書（`snipher/lang/lex.py`）の *読み* をそのまま使う（山羊 → やぎ）
    選択    : 知識ベースの話題・別名と、語の分類（`nouns.json` の `c` / `t`）だけで
               仲間かどうかを決める。決められない語は「決められない」と書く
    語の関係 : 語義（英語グロス）が一致する語＝類義語。対義のグロス対応表
               （`data/gloss_opposites.json`）に一致する語＝対義語
    短文    : 知識ベースの定義文を、字数に合わせて節の切れ目で切る（述語の形は検査する）

「たぶんこうだろう」で語を作らないのがこのモジュールの役目です。作れないときは
`None` を返し、呼び手（指示層・mind）が「材料が無い」ことを正直に言えるようにします。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..lang import lex
from ..lang.phonetics import normalize, to_hiragana, to_katakana
from .math import Solution

_DATA = Path(__file__).resolve().parents[1] / "data"

#: 空欄の書き方（かっこ・下線・「空欄」の語）
GAP = re.compile(r"[（(]\s*[）)]|\[\s*空欄\s*\]|＿{2,}|_{2,}|【\s*】")
#: 引用符で括られた塊（材料を差し出す書き方）
_QUOTED = re.compile(r"[「『\"']([^「」『』\"']{2,120}?)[」』\"']")
_ASK_KANA = re.compile(r"カタカナ|かたかな|ひらがな|平仮名|かな書き|かなに|読み(?:方)?を?")


def _clean(text: str) -> str:
    return normalize(str(text or "")).strip()


def _quoted(text: str) -> list[str]:
    return [m.group(1).strip() for m in _QUOTED.finditer(_clean(text)) if m.group(1).strip()]


# --------------------------------------------------------------------------- #
# 1) 空欄補充 — 候補を入れて、5-gram に審判させる
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def candidate_particles() -> tuple[str, ...]:
    """答えの候補になる 1 文字の助詞（実データ `particles.json` から引く）。"""
    chars: list[str] = []
    try:
        rows = json.loads((_DATA / "particles.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        rows = []
    for row in rows:
        s = str(row.get("s") or "")
        if len(s) == 1 and s not in chars:
            chars.append(s)
    for s in ("は", "が", "を", "に", "へ", "で", "と", "も", "の"):
        if s not in chars:
            chars.append(s)
    return tuple(chars)


@lru_cache(maxsize=1)
def particle_roles() -> dict[str, str]:
    """助詞 → 役割（「は」＝主題、「を」＝対格 …）。説明の根拠に使う。"""
    try:
        rows = json.loads((_DATA / "particles.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {str(r.get("s")): str(r.get("role") or r.get("f") or "") for r in rows if r.get("s")}


def gap_sentence(text: str) -> str:
    """空欄を含む *文そのもの* を取り出す（指示文ではなく材料のほう）。"""
    t = _clean(text)
    for q in _quoted(text):
        if GAP.search(q):
            return q
    m = re.search(r"[^。\n]*" + GAP.pattern + r"[^。\n]*。[」』]?", t)
    return m.group(0).strip("「」『』 ") if m else ""


def fill_particle(text: str, *, lm=None) -> Solution | None:
    """空欄に入る助詞を 1 つ選ぶ。候補ごとの流暢さを 5-gram が比べた結果を根拠にします。"""
    sentence = gap_sentence(text)
    if not sentence:
        return None
    if lm is None:
        try:
            from ..lm import shared as _shared_lm

            lm = _shared_lm()
        except Exception:  # noqa: BLE001
            lm = None
    if lm is None:
        return None
    scored: list[tuple[str, float, dict]] = []
    for ch in candidate_particles():
        filled = GAP.sub(ch, sentence, count=1)
        try:
            got = lm.score(filled)
        except Exception:  # noqa: BLE001
            continue
        scored.append((ch, float(got.get("logprob", 0.0)), got))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[1])
    best, best_lp, best_got = scored[0]
    second_lp = scored[1][1] if len(scored) > 1 else best_lp - 1.0
    margin = best_lp - second_lp
    roles = particle_roles()
    gap = GAP.search(sentence)
    gap_text = gap.group(0) if gap else "空欄"
    top = "、".join(f"「{c}」{lp:.2f}" for c, lp, _ in scored[:3])
    steps = [f"{gap_text} に候補を 1 つずつ入れて 5-gram で比べた（{top}）",
             f"最上位「{best}」と 2 位の差は {margin:.2f}（大きいほど確か）"]
    if roles.get(best):
        steps.append(f"「{best}」の役割は {roles[best]}（particles.json）")
    steps.append(f"選んだ文: {GAP.sub(best, sentence, count=1)}")
    detail = {"gap": sentence, "answer": best, "margin": round(margin, 3),
              "candidates": [{"particle": c, "logprob": round(lp, 3),
                              "perplexity": round(float(g.get("perplexity", 0.0)), 1)}
                             for c, lp, g in scored[:5]],
              "role": roles.get(best) or "", "lm_confidence": round(float(best_got.get("confidence", 0.0)), 3)}
    return Solution(answer=best, steps=steps, kind="fill",
                    verified=margin >= 0.25, detail=detail)


# --------------------------------------------------------------------------- #
# 2) かな書き — 実辞書の読みを使う
# --------------------------------------------------------------------------- #
def _want_kana(text: str) -> str:
    t = _clean(text)
    if "カタカナ" in t or "かたかな" in t:
        return "katakana"
    if "ひらがな" in t or "平仮名" in t:
        return "hiragana"
    return "hiragana"


def kana_write(text: str, *, kind: str = "") -> Solution | None:
    """語をかなで書く（山羊 → やぎ／ヤギ、りんご → リンゴ）。読みは実辞書から引きます。"""
    terms = _quoted(text) or []
    target = ""
    for cand in terms:
        if not _ASK_KANA.search(cand):
            target = cand
            break
    if not target:
        target = _subject_word(text)
    if not target:
        return None
    kind = kind or _want_kana(text)
    word = lex.lookup(target)
    if not word:
        return None
    reading = str(word.get("reading") or "")
    if not reading:
        return None
    out = to_katakana(reading) if kind == "katakana" else to_hiragana(reading)
    steps = [f"実辞書（{word.get('pos') or '語'}）の読みは「{reading}」（{word.get('morae')} 拍）",
             f"{target} → {out}"]
    return Solution(answer=out, steps=steps, kind="kana",
                    verified=bool(lex.bank().has(target)),
                    detail={"surface": target, "reading": reading, "script": kind,
                            "pos": word.get("pos"), "morae": word.get("morae")})


_KANA_TAIL = re.compile(
    r"(?:を|は|が|に|へ|で|と|も|の)?\s*(?:カタカナ|かたかな|ひらがな|平仮名|かな|ローマ字)?\s*"
    r"(?:に|へ)?\s*(?:変換して|直して|して|に)?.*$")


def _subject_word(text: str) -> str:
    """指示文から *操作する語* を取り出す（引用符 → 「〜を」の前 → 全体）。"""
    t = _clean(text)
    q = _quoted(text)
    if q:
        return q[0]
    m = re.match(r"^(.{1,24}?)(?:を|は|が|の)?\s*(?:カタカナ|ひらがな|かな|ローマ字)", t)
    if m:
        return m.group(1).strip("「」『』 　")
    m2 = re.match(r"^([^\s、。「」]{1,24}?)(?:を|は|が)", t)
    if m2:
        return m2.group(1).strip("「」『』 　")
    return text.strip("「」『』 　。？！?")


# --------------------------------------------------------------------------- #
# 3) 選択 — 仲間かどうかを、材料の分類だけで決める
# --------------------------------------------------------------------------- #
_LIST_BLOCK = re.compile(r"[\[［]([^\]］]{2,200})[\]］]")
_LIST_SPLIT = re.compile(r"\s*[,、，]\s*|\s+、?\s*")


@lru_cache(maxsize=1)
def _noun_index() -> dict[str, dict]:
    try:
        rows = json.loads((_DATA / "nouns.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, dict] = {}
    for row in rows:
        s = str(row.get("s") or "")
        if s:
            out[s] = row
    return out


def select_items(text: str, *, kb=None) -> Solution | None:
    """「〜の中から〈カテゴリ〉だけを選ぶ」。

    仲間かどうかは (1) 知識ベースの話題・別名 (2) 語の分類（nouns.json の c / t）で決めます。
    どちらでも決められない語は *仲間でない* とは言わず、「決められない」として残します。
    """
    t = _clean(text)
    m = _LIST_BLOCK.search(t)
    items: list[str] = []
    if m:
        items = [x.strip(" 　「」『』") for x in _LIST_SPLIT.split(m.group(1))]
    if not items:
        return None
    items = [x for x in items if x and len(x) <= 12][:8]
    if len(items) < 2:
        return None
    # カテゴリは「〜だけ」「〜の中から」の直前に置かれた語（引用符があればそれが印）
    cat = ""
    m_cat = re.search(r"[「『]([^」』]{1,12})[」』]\s*(?:だけ|のみ|を|に)", t)
    if m_cat:
        cat = m_cat.group(1).strip()
    if not cat:
        m_cat2 = re.search(r"([一-龯ァ-ヶーぁ-ん]{2,10})\s*(?:だけ|のみ|を選|に属|の仲間)", t)
        cat = m_cat2.group(1) if m_cat2 else ""
    if not cat:
        return None
    from ..knowledge import KnowledgeBase

    kb = kb or KnowledgeBase.shared()
    nouns = _noun_index()

    def evidence(word: str) -> tuple[bool | None, str]:
        """(仲間か, 根拠)。None は「決められない」。"""
        w = _clean(word)
        entry = nouns.get(w)
        if entry:
            cat_row = str(entry.get("c") or "")
            tags = [str(x) for x in entry.get("t") or []]
            if cat in (cat_row, *tags):
                return True, f"語の分類（nouns.json）が「{cat_row}」/ {tags}"
        if kb is not None:
            try:
                hit = kb.search(w, top_k=1)
            except Exception:  # noqa: BLE001
                hit = []
            if hit:
                item = hit[0].get("item") or {}
                topic = str(item.get("topic") or "")
                cat_kind = str(item.get("cat") or "")
                aliases = [str(x) for x in item.get("aliases") or []]
                if cat in (topic, cat_kind) or w in [x for x in aliases if x == w] and cat == topic:
                    return True, f"知識ベースの話題「{topic}」（分類 {cat_kind}）"
                if topic and topic != cat:
                    return False, f"知識ベースでは話題「{topic}」（分類 {cat_kind}）"
        if entry and str(entry.get("c") or ""):
            return False, f"語の分類（nouns.json）は「{entry.get('c')}」"
        return None, "手元の材料に分類が無い"

    chosen: list[str] = []
    why: list[dict] = []
    for word in items:
        is_member, basis = evidence(word)
        why.append({"item": word, "in_category": is_member, "basis": basis})
        if is_member:
            chosen.append(word)
    if not chosen:
        return None
    answer = "、".join(chosen)
    steps = [f"「{cat}」に当てはまるかを 1 語ずつ材料で確かめた"]
    for row in why:
        mark = "○" if row["in_category"] else ("×" if row["in_category"] is False else "？")
        steps.append(f"{mark} {row['item']}: {row['basis']}")
    detail = {"category": cat, "items": items, "chosen": chosen, "why": why}
    unknown = [r["item"] for r in why if r["in_category"] is None]
    return Solution(answer=answer, steps=steps, kind="select",
                    verified=not unknown, detail=detail)


# --------------------------------------------------------------------------- #
# 4) 語の関係 — 語義（英語グロス）で引く
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _gloss_entries() -> tuple[tuple[str, str, str], ...]:
    """(語, 読み, 語義グロス) の一覧を実データから作る。"""
    rows: list[tuple[str, str, str]] = []
    for name in ("adjectives.json", "verbs.json"):
        try:
            data = json.loads((_DATA / name).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for row in data:
            s = str(row.get("s") or "")
            g = str(row.get("g") or "").strip()
            if s and g:
                rows.append((s, str(row.get("r") or ""), g))
    return tuple(rows)


@lru_cache(maxsize=1)
def _gloss_opposites() -> dict[str, str]:
    try:
        data = json.loads((_DATA / "gloss_opposites.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    pairs = data.get("pairs") if isinstance(data, dict) else None
    out: dict[str, str] = {}
    for row in pairs or []:
        if isinstance(row, (list, tuple)) and len(row) == 2:
            a, b = str(row[0]).strip().lower(), str(row[1]).strip().lower()
            if a and b:
                out[a] = b
                out[b] = a
    return out


def _gloss_atoms(gloss: str) -> list[str]:
    return [x.strip().lower() for x in re.split(r"[/,;]", gloss) if x.strip()]


_REL_WORD = re.compile(r"[「『\"']([^「」『』\"']{1,16})[」』\"']")


def word_relation(text: str, *, kind: str = "") -> Solution | None:
    """類義語（同じ語義）／対義語（対義の語義）を、実データの語義グロスで引く。"""
    t = _clean(text)
    m = _REL_WORD.search(t)
    target = m.group(1).strip() if m else _subject_word(t)
    if not target:
        return None
    k = kind or ("antonym" if re.search(r"対義語|反対語|逆の意味|反意語|オポジット", t) else "synonym")
    entries = _gloss_entries()
    mine = [g for s, _r, g in entries if s == target]
    if not mine:
        try:
            word = lex.lookup(target)
        except Exception:  # noqa: BLE001
            word = None
        if not word:
            return None
        return None
    atoms = [a for g in mine for a in _gloss_atoms(g)]
    found: list[tuple[str, str, str]] = []
    basis: list[str] = []
    if k == "synonym":
        for s, r, g in entries:
            if s == target:
                continue
            shared = [a for a in _gloss_atoms(g) if a in atoms]
            if shared:
                found.append((s, r, g))
                basis.append(f"{s}: 語義「{g}」が {target} の「{', '.join(shared)}」と一致")
    else:
        opposites = _gloss_opposites()
        want = {opposites[a] for a in atoms if a in opposites}
        if not want:
            return None
        basis.append(f"{target} の語義「{', '.join(atoms)}」の対義は "
                     f"{', '.join(sorted(want))}（gloss_opposites.json）")
        for s, r, g in entries:
            if s == target:
                continue
            hit = [a for a in _gloss_atoms(g) if a in want]
            if hit:
                found.append((s, r, g))
                basis.append(f"{s}: 語義「{g}」が対義語側の「{', '.join(hit)}」に一致")
    if not found:
        return None
    answer = found[0][0]
    steps = [f"{target} の語義は「{'/'.join(atoms)}」（実データの語義グロス）", *basis,
             f"答え: {answer}"]
    return Solution(answer=answer, steps=steps, kind="relation",
                    verified=True,
                    detail={"target": target, "relation": k,
                            "candidates": [{"word": s, "reading": r, "gloss": g} for s, r, g in found[:5]],
                            "gloss": atoms})


# --------------------------------------------------------------------------- #
# 5) 短文 — 知識ベースの定義を、字数に合わせて切る
# --------------------------------------------------------------------------- #
_CLAUSE_SPLIT = re.compile(r"[、,]")
_END_PUNCT = "。！？!?"


def short_text(topic: str, *, max_chars: int, kb=None, lm=None) -> Solution | None:
    """「〜についての短文を N 文字以内で」。材料は知識ベースの定義・事実だけです。"""
    topic = _clean(topic).strip("「」『』 　")
    if not topic or max_chars <= 0:
        return None
    if kb is None:
        try:
            from ..knowledge import KnowledgeBase

            kb = KnowledgeBase.shared()
        except Exception:  # noqa: BLE001
            kb = None
    if kb is None:
        return None
    try:
        hit = kb.search(topic, top_k=1)
    except Exception:  # noqa: BLE001
        hit = []
    if not hit:
        return None
    item = hit[0].get("item") or {}
    # 引けた話題が *頼まれた語* か（太陽 で 月 を引いてしまう取り違えを防ぐ）
    names = [str(item.get("topic") or "")] + [str(x) for x in item.get("aliases") or []]
    if not any(n == topic or n in topic or topic in n for n in names if n):
        return None
    sources: list[str] = []
    for key in ("def", "facts", "answers"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            sources.append(val.strip())
        elif isinstance(val, list):
            sources.extend(str(x).strip() for x in val if str(x).strip())
    if not sources:
        return None
    if lm is None:
        try:
            from ..lm import shared as _shared_lm

            lm = _shared_lm()
        except Exception:  # noqa: BLE001
            lm = None
    best: tuple[tuple, str, dict] | None = None
    # 定義（def）を最優先。事実（facts）は定義で字数が足りないときだけ使う。
    ordered = list(sources)
    if isinstance(item.get("def"), str) and item.get("def").strip() in ordered:
        ordered.remove(item["def"].strip())
        ordered.insert(0, item["def"].strip())
    for src_i, src in enumerate(ordered[:6]):
        body = src
        if not body.startswith(topic):
            body = f"{topic}は{body}" if not body.startswith("「") else body
        is_def = 1 if (isinstance(item.get("def"), str)
                       and src.strip() == str(item.get("def")).strip()) else 0
        for cand in _clip_candidates(body, max_chars, topic):
            if not _valid_end(cand):
                continue
            score = {}
            if lm is not None:
                try:
                    score = lm.score(cand)
                except Exception:  # noqa: BLE001
                    score = {}
            # 5-gram は *壊れた連接* を弾く審判として使う（手書きの定義文は語彙が
            # 豊かで confidence が低めに出るため、足切りには使わない）
            if lm is not None and float(score.get("bad_ratio", 0.0) or 0.0) > 0.15:
                continue
            if lm is not None and not score.get("ok", True):
                continue
            # 定義（def）由来 → 中身のある長さ → 話題の語入り → 長さ → 文らしさ
            rank = (is_def, 1 if len(cand) >= 5 else 0, 1 if topic in cand else 0,
                    len(cand), _end_kind(cand), cand)
            if best is None or rank > best[0]:
                best = (rank, cand, score)
    if best is None:
        return None
    _rank, text, score = best
    steps = [f"知識ベースの「{item.get('topic')}」の記述を材料にした",
             f"{max_chars} 文字以内に収めるため、節の切れ目で切った: 「{text}」（{len(text)} 文字）"]
    if score:
        steps.append(f"5-gram の審判: confidence {float(score.get('confidence', 0.0)):.2f} / "
                     f"悪い連接 {int(score.get('bad', 0))}")
    return Solution(answer=text, steps=steps, kind="short",
                    verified=len(text) <= max_chars,
                    detail={"topic": item.get("topic"), "chars": len(text),
                            "max_chars": max_chars, "source": "kb",
                            "lm": {k: (round(v, 3) if isinstance(v, float) else v)
                                   for k, v in score.items()} if score else None})


def _clip_candidates(body: str, max_chars: int, topic: str = "") -> list[str]:
    """字数以内に収まる候補を、長い順に返す（述語の形は壊さない）。

    節に割り、節が長ければ *核の名詞句*（実辞書で割った最後の名詞とその修飾部）だけを
    残します。話題で始まらない節は「話題は」を補い、材料の主題が消えないようにします。
    """
    out: list[str] = []

    def add(cand: str) -> None:
        c = _tidy(cand)
        if c and len(c) <= max_chars and c not in out:
            out.append(c)

    for piece in [body] + _CLAUSE_SPLIT.split(body):
        piece = _tidy(piece)
        if not piece:
            continue
        add(piece)
        lead = piece if _topic_initial(piece, topic) else (f"{topic}は{piece}" if topic else piece)
        add(lead)
        tail = _tidy(re.split(r"(?:と|や|、)", piece)[-1])
        add(tail)
        add(re.sub(r"(?:で|に|を|が|は)$", "", tail))
        core = _head_phrase(re.sub(r"(?:で|に|を|が|は|と)$", "", piece) or piece)
        if core and len(core) >= 3:
            # すでに述語で終わっている核に「です」を足すと「〜ますです」になる
            core = re.sub(r"^[てでにをがはのや、]", "", core)
            pred = _end_kind(core) == 2 or bool(
                re.search(r"(?:ます|です|た|る|う|く|す|つ|む|ぐ|ぶ|い|て|で|り|し)$", core))
            for suffix in ("です。", "です", "。", ""):
                if pred and suffix.startswith("です"):
                    continue
                lead = "" if _topic_initial(core, topic) else (f"{topic}は" if topic else "")
                add(lead + core + suffix)
    return sorted(out, key=len, reverse=True)


#: 核の名詞句に含めてよい語（名詞・形容詞・連体詞・「の」）
_HEAD_OK = ("名詞", "形容詞", "連体詞", "接頭辞")
_HEAD_STOP = ("動詞", "助動詞", "副詞", "接続詞", "感動詞")


def _head_phrase(piece: str) -> str:
    """句の *核*（最後の名詞とその修飾部）を取り出す。辞書で語に割れないときは空。"""
    try:
        words = lex.bank().segment(_tidy(piece))
    except Exception:  # noqa: BLE001
        return ""
    if len(words) < 2:
        return ""
    idx = None
    for i in range(len(words) - 1, -1, -1):
        pos = str(words[i][1] or "")
        if pos.startswith("名詞"):
            idx = i
            break
    if idx is None:
        return ""
    # 連体の連なり（名詞・形容詞・「の」・連体形の動詞）は核に含める。
    # 「太陽系の中心にある恒星」のように、動詞の連体形が修飾に入ることがある。
    _lb = ("ある", "いる", "する", "した", "ない", "なる", "たる", "おる", "できる")
    _link = ("の", "な", "に", "で", "と", "へ", "から", "まで", "や")
    start = idx
    while start - 1 >= 0:
        prev_word, prev = words[start - 1]
        prev_word, prev = str(prev_word), str(prev or "")
        if prev.startswith(_HEAD_OK) or prev_word in _link or prev_word in _lb \
                or prev.startswith("未知語"):
            start -= 1
            continue
        break
    return "".join(w for w, _p in words[start:idx + 1])


def _tidy(text: str) -> str:
    t = normalize(str(text or "")).strip().strip("「」『』 ").strip(" 　、,。")
    return re.sub(r"\s+", "", t)


def _end_kind(text: str) -> int:
    """文の終わり方（2=述語で終わる文、1=体言止め、0=その他）。"""
    t = str(text or "").strip()
    if t.endswith("です") or t.endswith("。"):
        return 2
    try:
        last = str(lex.bank().segment(t)[-1][1] or "")
    except Exception:  # noqa: BLE001
        return 1
    if last.startswith(("動詞", "形容詞", "助動詞")):
        return 2
    return 1 if last.startswith("名詞") else 0


def _topic_initial(core: str, topic: str) -> bool:
    """核がすでに話題で始まっているか（「猫は…」なら前置きしない）。"""
    if not topic or not core.startswith(topic):
        return False
    rest = core[len(topic):len(topic) + 1]
    return rest in ("", "は", "が", "を", "に", "へ", "で", "と", "も", "の", "、")


#: 文末に来てはいけない語（助詞で切れた断片を弾く）
_BAD_TAIL = ("の", "は", "が", "を", "に", "で", "と", "も", "や", "へ", "から", "まで", "より", "って")


def _valid_end(text: str) -> bool:
    """短文として終われる形か（体言止め・終止形・「です」だけを許す）。"""
    t = str(text or "").strip()
    if not t:
        return False
    if t.endswith("。"):
        t = t[:-1]
    if not t or t.endswith(_BAD_TAIL):
        return False
    if t.endswith("です"):
        return True
    try:
        words = lex.bank().segment(t)
    except Exception:  # noqa: BLE001
        return True
    if not words:
        return False
    last = str(words[-1][1] or "")
    return last.startswith(("名詞", "動詞", "形容詞", "助動詞"))


def head_phrase(text: str) -> str:
    """句の *核*（最後の名詞とその修飾部）を取り出す公開ヘルパ（例: 太陽系の中心にある恒星）。"""
    return _head_phrase(text)


def definition_core(body: str) -> str:
    """定義文から *一言の核* を取り出す（「太陽は、太陽系の中心にある恒星で…」→ その恒星句）。

    特定の話題の答えをコードに書かず、文法（「Xは…」の切り分けと助詞での分割）だけで
    取り出します。取り出せなければ空文字を返します（呼び手が元の文を残せるように）。
    """
    first = re.split(r"[。！？!?\n]", str(body or "").strip())[0].strip()
    if not first:
        return ""
    m = re.match(r"^(.{1,24}?)(?:は|とは|って)[、,]?(.+)$", first)
    if not m:
        return ""
    tail = m.group(2).strip()
    clause = re.split(r"[、,]", tail)[0].strip()
    clause = re.sub(r"(?:です|である|だ|で|ます)$", "", clause).strip()
    if len(clause) < 3:
        return ""
    core = _head_phrase(clause) or clause
    if len(core) < 3 or len(core) > 30:
        return ""
    # 助詞で始まる核は *断片*（「のうち甘みや酸味」）なので使わない
    if core[0] in "のをにがはでもとやへからまで":
        return ""
    return f"{core}です。"


# --------------------------------------------------------------------------- #
# 2b) 文全体のかな書き — 実辞書の読みで 1 語ずつ直す（v8）
# --------------------------------------------------------------------------- #
_HIRAGANA_ONLY = re.compile(r"[ぁ-ん]+$")
_KANJI = re.compile(r"[一-龯々〆ヶ]")
_KANA_PIECE = re.compile(r"[ぁ-んァ-ヶー]+$")


def _reading_of(surface: str) -> str:
    """表層形の読みを実辞書から引く（活用の送りがなは残して戻す）。"""
    if not surface:
        return ""
    w = lex.lookup(surface)
    if w and w.get("reading"):
        return str(w["reading"])
    # 活用形（食べました／青かった）は、辞書にある最長の前部 + 残りの送りがな
    bank = lex.bank()
    for j in range(len(surface) - 1, 0, -1):
        head, tail = surface[:j], surface[j:]
        if not _KANA_PIECE.match(tail):
            continue
        e = bank.entry(head)
        if e is not None and e.reading:
            return e.reading + tail
    # 漢字 1 字ずつ（熟語に見出しが無いときの最後の手段）
    if _KANJI.search(surface) and len(surface) <= 6:
        pieces: list[str] = []
        for ch in surface:
            if _KANA_PIECE.match(ch):
                pieces.append(ch)
                continue
            e = bank.entry(ch)
            if e is not None and e.reading:
                pieces.append(e.reading)
            else:
                return ""
        return "".join(pieces)
    return ""


def kana_sentence(text: str, *, kind: str = "hiragana", strict: bool = False) -> Solution | None:
    """文全体をかなで書く（「空が青い」→「そらがあおい」）。

    対象は引用符で差し出された文。1 語ずつ実辞書の読みに直し、かなの語は
    そのまま残す。`strict` のときは指定の文字種 *以外を全部落とす*
    （「ひらがなだけで。他の文字や記号は含めないで」の形）。
    """
    kind = kind or _want_kana(text)
    terms = _quoted(text) or []
    target = ""
    for cand in terms:
        if _ASK_KANA.search(cand):
            continue
        target = cand
        break
    if not target:
        target = _subject_word(text)
    target = target.strip().strip("。？！?!")
    if not target or _KANJI.search(target) is None and _KANA_PIECE.fullmatch(target):
        # かなだけの語は kana_write（1 語の仕事）に譲る
        return None
    if not target:
        return None
    bank = lex.bank()
    pieces: list[str] = []
    used: list[str] = []
    failed: list[str] = []
    for surface, _pos in bank.segment(target):
        if not surface.strip():
            continue
        if _KANA_PIECE.fullmatch(surface):
            pieces.append(surface)
            continue
        if re.fullmatch(r"[0-9０-９]+", surface):
            if strict:
                failed.append(surface)
            else:
                pieces.append(surface)
            continue
        if _KANJI.search(surface):
            reading = _reading_of(surface)
            if reading:
                pieces.append(reading)
                used.append(f"{surface}→{reading}")
            else:
                failed.append(surface)
            continue
        # 記号・空白
        if strict:
            continue
        pieces.append(surface)
    if failed:
        return None
    out = "".join(pieces)
    out = to_katakana(out) if kind == "katakana" else to_hiragana(out)
    if strict:
        if kind == "katakana":
            out = re.sub(r"[^ァ-ヶー]", "", out)
        else:
            out = re.sub(r"[^ぁ-ん]", "", out)
    if not out:
        return None
    steps = [f"材料の文「{target}」を分かち書きして 1 語ずつ実辞書の読みに直した",
             *used[:8]]
    if strict:
        steps.append(f"{'カタカナ' if kind == 'katakana' else 'ひらがな'}以外の字・記号は全部落とした")
    steps.append(f"{target} → {out}")
    return Solution(answer=out, steps=steps, kind="kana",
                    verified=not failed, detail={"surface": target, "reading": out,
                                                 "script": kind, "strict": strict})


# --------------------------------------------------------------------------- #
# 2c) 並びのつなぎ直し — 区切り文字の指定どおりに並べる（v8）
# --------------------------------------------------------------------------- #
_SEP_WORDS = (
    ("カンマ", ","),
    ("comma", ","),
    ("コンマ", ","),
    ("読点", "、"),
    ("てん", "、"),
    ("中点", "・"),
    ("なかぐろ", "・"),
    ("スペース", " "),
    ("空白", " "),
    ("縦棒", "|"),
    ("パイプ", "|"),
    ("スラッシュ", "/"),
)


def _join_items_from(text: str) -> list[str]:
    t = _clean(text)
    quoted = _quoted(text)
    material = [q for q in quoted if not re.search(r"カンマ|区切り|リスト|改行|説明|不要|変換|出力", q)]
    if len(material) >= 2 and all(len(q) <= 12 for q in material):
        return material[:12]
    if material:
        first = material[0]
        parts = [x.strip(" 　") for x in re.split(r"[、，,・|/\s]+", first) if x.strip(" 　")]
        if len(parts) >= 2:
            return parts[:12]
    m = re.search(r"[\[［]([^\]］]{2,120})[\]］]", t)
    if m:
        parts = [x.strip(" 　「」『』") for x in re.split(r"\s*[,、，]\s*", m.group(1))]
        parts = [x for x in parts if x]
        if len(parts) >= 2:
            return parts[:12]
    return []


def join_items(text: str, *, sep: str = "") -> Solution | None:
    """「りんご、ゴリラ、ラッパ」をカンマ区切りのリストに（語は 1 字も変えない）。

    区切り文字は指示の語（カンマ／読点／中点…）から決める。改行・説明は付けない。
    """
    items = _join_items_from(text)
    if len(items) < 2:
        return None
    if not sep:
        t = _clean(text)
        sep = ","
        for word, mark in _SEP_WORDS:
            if word in t:
                sep = mark
                break
        else:
            if "、" in t and "カンマ" not in t and "comma" not in t.lower():
                sep = "、"
    out = sep.join(items)
    steps = [f"材料の {len(items)} 語（{'／'.join(items)}）を区切り「{sep}」でつないだ",
             "改行・説明は付けていない（指示どおり）"]
    return Solution(answer=out, steps=steps, kind="join", verified=True,
                    detail={"items": items, "sep": sep})


# --------------------------------------------------------------------------- #
# 2d) 好みの解決 — 極性（好き／嫌い）で指している語を決める（v8）
# --------------------------------------------------------------------------- #
_LIKE = re.compile(r"(?P<ent>[^、。？?]{1,14}?)(?:が|は|って|というのは)?"
                   r"(?P<pol>好き|きらい|嫌い|大好き|大嫌い)(?P<neg>じゃない|ではない|じゃありません|ではありません)?")
_WANT_POL = re.compile(r"(?:好き|きらい|嫌い|大好き|大嫌い)(?:な|の)(?:方|ほう|動物|人|もの|物|どれ|方か)"
                       r"|どちら(?:です|だ|なの)?か|どれ(?:です|だ|なの)?か|何が(?:好き|嫌い)|どっち")


def resolve_preference(text: str) -> Solution | None:
    """「私は犬が好きで、猫は嫌いです」+「好きな動物はどちら」→「犬」。

    材料の文から〈対象・極性〉を抜き、問いが求める極性の語だけを返す。
    問いが無い・極性が読めないときは None（当てずっぽうで語を選ばない）。
    """
    body = _clean(text)
    if not _WANT_POL.search(body):
        return None
    likes: list[str] = []
    dislikes: list[str] = []
    for m in _LIKE.finditer(body):
        ent = m.group("ent").strip(" 　、私はあなたは彼かれ彼女かの")
        ent = re.sub(r"^(?:私は|あなたは|彼は|彼女は|それは|これは|あれは)", "", ent).strip()
        ent = ent.strip("がはをにでと、。")
        if not ent or len(ent) > 10 or re.search(r"どちら|どれ|何が|動物|好き|嫌い|きらい", ent):
            continue
        pol = m.group("pol")
        neg = bool(m.group("neg"))
        is_like = ("好き" in pol) != neg
        # 「私は犬が好きで」のように主語つきの文は、最後の名詞が対象
        ent = re.sub(r"^.*?(?:は|が)", "", ent).strip() or ent
        (likes if is_like else dislikes).append(ent)
    if not likes and not dislikes:
        return None
    # 問いが求める極性（「好きな…どちら」「嫌いな…どれ」）。
    # 「どちら」の *直前* の極性語で決める（材料の文の極性を拾わない）。
    want_like = True
    m_w = re.search(r"(?:どちら|どれ|どっち)", body)
    if m_w:
        window = body[max(0, m_w.start() - 14):m_w.start()]
        nearest: str = ""
        for mm in re.finditer(r"大好き|大嫌い|好き|きらい|嫌い", window):
            nearest = mm.group(0)
        if nearest:
            want_like = nearest in ("大好き", "好き")
        else:
            # 「どちらですか」だけで極性が無い → 好きの方（肯定の問いが普通）
            want_like = True
    picked = (likes or dislikes) if want_like else (dislikes or likes)
    if want_like and not likes:
        return None
    if not want_like and not dislikes:
        return None
    answer = picked[0]
    steps = [f"材料から〈好き: {'／'.join(likes) or 'なし'}〉〈嫌い: {'／'.join(dislikes) or 'なし'}〉を読んだ",
             f"問いは{'「好きな方」' if want_like else '「嫌いな方」'}を求めている",
             f"→ {answer}"]
    return Solution(answer=answer, steps=steps, kind="prefer", verified=True,
                    detail={"likes": likes, "dislikes": dislikes, "want_like": want_like})


__all__ = ["GAP", "gap_sentence", "fill_particle", "kana_write", "kana_sentence", "join_items",
           "resolve_preference", "select_items", "word_relation",
           "short_text", "candidate_particles", "particle_roles", "head_phrase", "definition_core"]


# ``Solution`` を再輸出する（呼び手が 1 か所から取れるように）
JpSolution = Solution


@dataclass
class _Meta:
    """このモジュールが使う材料の在庫（UI の説明に使えるようにする）。"""

    particles: int
    glosses: int
    opposites: int

    def as_dict(self) -> dict:
        return {"particles": self.particles, "glosses": self.glosses, "opposites": self.opposites}


def material_stats() -> dict:
    return _Meta(particles=len(candidate_particles()), glosses=len(_gloss_entries()),
                 opposites=len(_gloss_opposites())).as_dict()
