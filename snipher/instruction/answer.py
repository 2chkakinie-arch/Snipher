"""問いへの答え — 役割・口調・文字数・形式の指定を守って、証拠から文を書く。

指示層の `answer` タスクは「質問に答える」を *1 つの仕事* として扱います。
v3 の会話経路との違いは、指示に書かれた **出力仕様を契約として守る** ことです:

    役割（あなたは〜です）      → 応答の立ち位置として 1 文目に置く
    口調（〜だよ、専門的）      → 文末の述語を辞書形に引き直して組み替える
    長さ（200文字程度）         → 文を切らずに「入る文を選ぶ」＋節単位で詰める
    形式（箇条書き3つ／JSON）   → その形にして、機械的に検証する

材料は既存の証拠層（知識ベース → 実辞書 → ウェブ裏取り）から集めます。
集まらなかったときは、推測で埋める代わりに *指示文に書かれていた材料*（payload）を
根拠として使い、それでも足りなければ **問われている語そのものについて数えられる事実**
（語彙にあるか・読み・拍・検索が通ったか）を返します。話題と無関係な語の辞書情報で
埋め尽くすことはしません（v3 で「テキスト」の読みが返っていた事故の再発防止）。
どの道でも「できません」とは言いません。
"""

from __future__ import annotations

import re

from ..lang import lex
from ..lang.phonetics import char_count, mora_count, normalize, to_hiragana
from ..mind.frame import Claim, split_sentences
from .style import clause_trim, count_chars, fit_length, is_predicate_end, restyle

_ORDER = ("result", "definition", "answer", "fact", "reason", "step", "advice", "example",
          "opinion", "list", "evidence", "lexical", "note")
_NO_ECHO = ("参考までに",)
_NO_ANSWER = ("のあたりを話せます", "を話せます", "に近い話題", "近い話題は")
_PARTICLES = {"を", "は", "が", "に", "で", "と", "も", "の", "から", "まで", "より", "へ", "や",
              "か", "な", "ね", "よ", "わ", "し"}


# --------------------------------------------------------------------------- #
# 問いの「主体」を読む（frame.topic が語彙の断片を拾ったときの補正）
# --------------------------------------------------------------------------- #
def question_subject(question: str) -> str:
    """「質問：WebAssembly（Wasm）をブラウザで…」→ WebAssembly のように主体を取る。"""
    q = str(question or "").strip()
    body = re.sub(r"^(?:質問|問い|問|設問)\s*[：:]\s*", "", q)
    m = re.match(r"^[「『\"']?([^「」『』\"'？?をはがにでと]{2,28}?)[」』\"']?\s*"
                 r"(?:とは|って何|とは何|を|は|が|に|で|の|って|についての|について)", body)
    if m:
        cand = m.group(1).strip(" 　、,")
        if cand and cand not in _PARTICLES:
            return cand
    m_what = re.search(r"([^\n「」『』？?]{2,24}?)\s*(?:とは|って)\s*(?:何|なん|どういう)", body)
    if m_what:
        cand = m_what.group(1).strip(" 　、,。をはがにでとも")
        if len(cand) >= 2:
            return cand
    m2 = re.search(r"([A-Za-z][A-Za-z0-9.+#_\-]{2,})", body)
    if m2:
        return m2.group(1)
    m3 = re.search(r"([ァ-ヶー一-龯]{2,10})", body)
    return m3.group(1) if m3 else body[:12]


def _frame_for(question: str, *, kb=None, history=None):
    from ..mind.parse import build_frame
    from ..mind.state import ConversationState

    state = ConversationState(history or [], kb=kb)
    frame = build_frame(question, history=history or [], kb=kb, state=state)
    subject = question_subject(question)
    # frame.topic が語彙の断片（「ブラ」）を拾ったときは、問いの主体に差し替える。
    ent = next((e for e in frame.entities if e.surface == subject), None)
    if ent is None and subject:
        from ..mind.frame import Entity

        ent = Entity(surface=subject, kind="ascii" if subject.isascii() else "mixed",
                     web_needed=True)
        frame.entities.insert(0, ent)
    if subject and (not frame.topic or frame.topic != subject):
        weak = frame.topic and not any(e.surface == frame.topic and e.web_needed
                                       for e in frame.entities)
        if weak or not frame.topic:
            frame.topic = subject
    if ent is not None and ent.web_needed:
        frame.needs_web = True
    return frame


def gather_claims(d: "Directive", *, kb=None, web=None, history=None, lm=None,
                  core=None) -> tuple[list[Claim], list[dict], list[str], object]:
    """答えの材料を集める。返るのは (主張, 出典, メモ, frame)。"""
    from ..ground.evidence import gather

    question = d.question or d.subject
    notes: list[str] = []
    frame = _frame_for(question, kb=kb, history=history)
    subject = question_subject(question)
    # 証拠集めには *問いの 1 文だけ* を渡す。「100文字程度で答えてください」のような
    # 指示の文を混ぜると、話題の一致判定（alias / qa 類似度）がずれて知識を引けません。
    try:
        frame.raw = question
    except Exception:  # noqa: BLE001
        pass
    dossier = None
    try:
        dossier = gather(frame, kb=kb, web=web, tool_claims=[], history_text=question)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"証拠集め: {type(exc).__name__}")
    if subject and (dossier is None or float(getattr(dossier, "coverage", 0.0) or 0.0) < 0.5):
        # 問いの文（「…とは何ですか？」）で話題が引けなかったときは、*語そのもの* で引き直す
        try:
            alt = _frame_for(subject, kb=kb, history=history)
            alt.raw = subject
            got2 = gather(alt, kb=kb, web=web, tool_claims=[], history_text=subject)
            if got2 is not None and float(getattr(got2, "coverage", 0.0) or 0.0) > float(
                    getattr(dossier, "coverage", 0.0) or 0.0):
                dossier, frame = got2, alt
                notes.append(f"証拠: 語「{subject}」で引き直した")
        except Exception:  # noqa: BLE001
            pass
    claims: list[Claim] = list(getattr(dossier, "claims", []) or [])
    sources: list[dict] = list(getattr(dossier, "sources", []) or [])
    via = str(getattr(dossier, "via", "") or "")
    coverage = float(getattr(dossier, "coverage", 0.0) or 0.0)
    notes.append(f"証拠: via={via or 'none'} coverage={coverage:.2f} claims={len(claims)}")
    notes.extend(str(x) for x in (getattr(dossier, "notes", []) or [])[:3])
    web_tried = any(("ウェブ" in str(x) or "web" in str(x))
                    for x in (getattr(dossier, "notes", []) or []))

    # 問いの語と重なりの無い *語彙の断片*（「ブラウザ」の中の「ブラ」の読みなど）は
    # 答えにしない。知識ベース・検索・指示文の材料は話題判定を通っているので残す。
    claims = [c for c in claims
              if str(c.source) != "lex" or _relevant(c, question, subject)]
    solid = [c for c in claims
             if str(c.source).startswith(("tool", "web")) or "kb" in str(c.source)]
    if not solid and d.payload and len(d.payload) >= 12:
        # 指示文が *自分で材料を差し出している*（定義や前提）なら、それを根拠にする
        for sent in split_sentences(d.payload)[:4]:
            solid.append(Claim(kind="fact", content=sent, subject=subject,
                               source="prompt", weight=0.68))
        notes.append("材料: 指示文に書かれていた記述を使った")
    need = int(d.fmt.target_chars or 0) or int(d.fmt.max_chars or 0)
    have = sum(len(str(c.content or "")) for c in solid)
    if not solid or (need and have < need * 0.8):
        # 材料が足りないぶんは、*確かめられること*（語彙・検索の状態・近い話題）で埋める。
        # 推測で語義を作ることはしないので、ここで足せるのは数えられる事実だけです。
        solid.extend(fallback_claims(question, frame, subject=subject, web=web,
                                     kb=kb, notes=notes))
    return solid, sources, notes, frame


def _stands_alone(word: str, whole: str) -> bool:
    """その語が *材料の中で独立して立っているか*（別の語の断片でないか）。"""
    w = normalize(word or "")
    src = normalize(whole or "")
    if not w or w not in src:
        return False
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9.+#_\-]*", word or ""):
        return True                        # 欧文の語はそのまま扱う
    for m in re.finditer(r"[ァ-ヶー一-龯]{2,}", src):
        tok = m.group(0)
        if tok != w and w in tok:
            return False                   # より長い語の部分列
    return True


def _relevant(claim: Claim, question: str, subject: str) -> bool:
    content = str(claim.content or "")
    if any(content.startswith(x) for x in _NO_ECHO):
        return False
    if any(x in content for x in _NO_ANSWER):
        return False            # 別の話題の案内は、この問いの答えではない
    if claim.source == "prompt" or str(claim.source).startswith(("tool", "web")):
        return True
    low_q = normalize(question).lower()
    low_c = normalize(content).lower()
    if subject and normalize(subject).lower() in low_c:
        return True
    for w in re.findall(r"[A-Za-z][A-Za-z0-9.+#_\-]{2,}|[ァ-ヶー一-龯]{2,6}", question):
        if len(w) >= 2 and normalize(w).lower() in low_c:
            return True
    for i in range(len(subject) - 1):
        if len(subject[i:i + 2]) >= 2 and normalize(subject[i:i + 2]).lower() in low_c:
            return True
    return claim.kind in ("note",) and "kb" not in str(claim.source)


# --------------------------------------------------------------------------- #
# 材料が薄いときの手（数えられる事実だけ）
# --------------------------------------------------------------------------- #
def fallback_claims(question: str, frame, *, subject: str, web=None, kb=None,
                    notes: list[str] | None = None, tried: bool = False,
                    dossier=None) -> list[Claim]:
    """問いの語そのものについて、*確かめられること* だけを並べる。"""
    notes = notes if notes is not None else []
    b = lex.bank()
    words: list[str] = []
    for ent in getattr(frame, "entities", [])[:6]:
        w = str(ent.surface or "").strip()
        if not w or w in _PARTICLES or len(w) < 2 or w in words:
            continue
        words.append(w)
    if subject and subject not in words:
        words.insert(0, subject)
    known: list[str] = []
    unknown: list[str] = []
    whole = normalize(question or "")
    for w in words[:6]:
        if any(w != o and w in o for o in words):
            continue                       # 別の語の断片（「ブラウザ」の中の「ブラ」）は材料にしない
        if w and not _stands_alone(w, whole):
            continue                       # 長い語の一部（「ブラウザ」の「ブラ」）は語として立たない
        if re.fullmatch(r"[ぁ-ん]{1,3}", w):
            continue                       # かな 1〜3 文字は語として立たない
        if b.has(w):
            pos = str(b.pos(w) or "")
            if pos.split("/")[0] in ("動詞", "形容詞", "副詞", "助詞", "助動詞", "接続詞"):
                continue                   # 問いの中の動詞の辞書情報は答えにならない
            known.append(w)
        else:
            unknown.append(w)

    out: list[Claim] = []
    if unknown:
        bits = "・".join(unknown[:3]).replace(" ", "")
        out.append(Claim(kind="note",
                         content=f"問われている {bits} は、手元の語彙バンク "
                                 f"{b.stats()['words']:,} 語の見出しには無い語です。",
                         subject=subject, source="lex", weight=0.6))
    if known:
        w = known[0]
        read = b.reading(w) or to_hiragana(w)
        out.append(Claim(kind="lexical",
                         content=f"「{w}」は見出しにあって、読みは「{read}」・"
                                 f"{mora_count(read or w)} 拍・品詞 {b.pos(w) or '不明'}です。",
                         subject=w, source="lex", weight=0.55))
    tried = bool(tried or getattr(frame, "flags", {}).get("web_tried"))
    available = False
    if web is not None:
        try:
            available = bool(web.available())
        except Exception:  # noqa: BLE001
            available = False
    if available and not tried:
        out.append(Claim(kind="note",
                         content="検索は使える設定なので、裏取りに出れば出典つきの記述を"
                                 "持ってこられます。",
                         source="local:meta", weight=0.56))
    elif tried:
        out.append(Claim(kind="note",
                         content="検索には出ましたが、証拠にできる文は取れませんでした。",
                         source="local:meta", weight=0.56))
    if tried and getattr(dossier, "sources", None):
        out.append(Claim(kind="note",
                         content=f"取れた出典は {len(dossier.sources)} 件なので、"
                                 "そこを起点に本文を読み直します。",
                         source="local:meta", weight=0.54))
    else:
        out.append(Claim(kind="note",
                         content="この設定ではウェブ検索に出られないので、推測で語義は埋めません。",
                         source="local:meta", weight=0.56))
    if kb is not None:
        try:
            near = [x for x in kb.suggest(question, top_k=3) if x]
        except Exception:  # noqa: BLE001
            near = []
        if near:
            out.append(Claim(kind="note",
                             content=f"手元の知識ベースで近い話題は {'、'.join('「' + x + '」' for x in near[:3])} "
                                     f"なので、そこなら定義も理由も本文から引けます。",
                             source="local:kb", weight=0.52))
    out.append(Claim(kind="ask",
                     content=f"{subject or 'この語'}について、どの切り口（意味・利点・手順・比較・値段）が"
                             "必要かを一言もらえれば、その欄を狙って組み立てます。",
                     subject=subject, source="local:meta", weight=0.5))
    notes.append("材料が薄い: 問いの語について数えられる事実を出した")
    return out


# --------------------------------------------------------------------------- #
# 文の組み立て
# --------------------------------------------------------------------------- #
def _sentences_of(claims: list[Claim], *, limit: int = 8) -> list[str]:
    ranked = sorted(claims, key=lambda c: (_ORDER.index(c.kind) if c.kind in _ORDER else 99,
                                           -float(c.weight or 0)))
    out: list[str] = []
    for c in ranked:
        body = str(c.content or "").strip()
        if not body or any(body.startswith(x) for x in _NO_ECHO):
            continue
        if any(x in body for x in _NO_ANSWER):
            continue
        for sent in split_sentences(body):
            s = sent.strip()
            if not s or len(s) < 6:
                continue
            if any(_overlap(s, prev) > 0.72 for prev in out):
                continue
            out.append(s)
            if len(out) >= limit:
                return out
    return out


def _overlap(a: str, b: str, n: int = 6) -> float:
    ga = {a[i:i + n] for i in range(max(0, len(a) - n + 1))}
    gb = {b[i:i + n] for i in range(max(0, len(b) - n + 1))}
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


def opening_line(d: "Directive", subject: str, *, tone: str, register: str) -> str:
    """役割を与えられたときの 1 文目（指示の語から作り、口調に合わせて組み替える）。"""
    if d.fmt.no_greeting or d.fmt.strict:
        return ""
    role = (d.role or "").strip()
    subject = (subject or "").strip(" 　。、？?！!")
    if not subject or len(subject) < 2:
        subject = ""
    if not role and not subject:
        return ""
    if role and subject:
        base = f"{role}の立場から、{subject}についてです"
    elif role:
        base = f"{role}の立場から答えます"
    elif subject:
        base = f"{subject}について答えます"
    else:
        return ""
    got, _fixes = restyle(base + "。", tone=tone, register=register)
    return got.strip()


def _shorten(sentence: str, room: int) -> str:
    if count_chars(sentence) <= room:
        return sentence
    trimmed = clause_trim(sentence, room)
    if trimmed and is_predicate_end(trimmed.rstrip("。！？!?")):
        return trimmed
    return sentence


def _expand_to_target(sentences: list[str], target: int, *, hard_max: int = 0) -> tuple[list[str], list[str]]:
    """目標の文字数に届くように文を足す（ねつ造はしない・文を途中で切らない）。

    節を切り出して 1 文に見せると「WebAssembly についてだよ」のような述語の壊れた
    文になるので、*文は文のまま* 足します。足りないぶんは呼び出し側が材料を
    集め直す（`gather_claims` の fallback）ことで埋めます。
    """
    notes: list[str] = []
    budget = max(1, (hard_max or int(target * 1.35)))
    out: list[str] = []
    used = 0
    for s in sentences:
        if used + count_chars(s) <= budget:
            out.append(s)
            used += count_chars(s)
            continue
        room = max(20, budget - used)
        trimmed = clause_trim(s, room)
        if trimmed and is_predicate_end(trimmed.rstrip("。！？!?")) \
                and count_chars(trimmed) >= 20:
            out.append(trimmed)
            used += count_chars(trimmed)
        break
    if used < target * 0.55:
        notes.append(f"short_of_target:{used}字/{target}字")
    return out, notes


def answer(d: "Directive", *, kb=None, web=None, history=None, lm=None, core=None,
           turn: int = 0) -> dict:
    """問いに答える。返るのは {text, sentences, sources, notes, confidence, meta}。"""
    fmt = d.fmt
    tone = fmt.tone or ("plain" if fmt.register == "plain" else "")
    register = fmt.register or ("polite" if not tone else "")
    question = d.question or d.subject
    subject = question_subject(question)
    claims, sources, notes, frame = gather_claims(d, kb=kb, web=web, history=history,
                                                 lm=lm, core=core)
    sentences = _sentences_of(claims)
    if not sentences:
        sentences = [f"{subject}について、いま手元の材料で言えることを並べます。"]
        notes.append("文: 材料なし")

    bullets = int(fmt.bullets or 0)
    target = int(fmt.target_chars or 0)
    hard = int(fmt.max_chars or 0)

    opening = opening_line(d, subject, tone=tone, register=register)
    if opening:
        sentences = [opening] + sentences

    if bullets:
        pick = sentences[:bullets]
        if len(pick) < bullets:
            pick = (pick + sentences[len(pick):])[:bullets]
        room = max(24, int((target or hard or 240) / bullets) + 12)
        body = [_shorten(s, room) for s in pick]
    else:
        body = list(sentences)
        if target or hard:
            cap = hard or int(target * 1.4) + 40
            keep, f2 = fit_length(body, target=target, hard_max=cap)
            notes.extend(f2)
            body, f3 = _expand_to_target(keep or body, target, hard_max=cap)
            notes.extend(f3)
        else:
            body = body

    styled, fixes = restyle("\n".join(body), tone=tone, register=register)
    notes.extend(fixes)
    body = [x for x in styled.split("\n") if x.strip()]

    if bullets:
        if fmt.numbered:
            lines = [f"{i}. {s}" for i, s in enumerate(body, 1)]
        else:
            lines = [f"{fmt.bullet_char or '・'}{s}" for s in body]
        text = "\n".join(lines)
    else:
        text = "\n".join(body) if len(body) > 1 else "".join(body)

    if sources and not fmt.strict:
        cite = "\n".join(f"[{i}] {s.get('title') or s.get('url')} — {s.get('url')}"
                         for i, s in enumerate(sources[:4], 1))
        text = f"{text}\n出典:\n{cite}"

    conf = 0.6
    if any(str(c.source).startswith("tool") for c in claims):
        conf = 0.9
    elif any(str(c.source) == "web" for c in claims):
        conf = 0.86
    elif any("kb" in str(c.source) for c in claims):
        conf = 0.8
    elif any(str(c.source) == "prompt" for c in claims):
        conf = 0.68
    if target and abs(count_chars(text) - target) > target * 0.6:
        conf -= 0.08
    return {"text": text.strip(), "sentences": body, "sources": sources,
            "notes": notes, "confidence": round(max(0.3, min(0.95, conf)), 3),
            "meta": {"claims": [c.as_dict() for c in claims[:6]], "tone": tone or None,
                     "role": d.role or None, "subject": subject,
                     "chars": count_chars(text), "target_chars": target or None,
                     "bullets": bullets or None, "topic": getattr(frame, "topic", "")}}


__all__ = ["answer", "gather_claims", "question_subject", "fallback_claims", "opening_line"]
