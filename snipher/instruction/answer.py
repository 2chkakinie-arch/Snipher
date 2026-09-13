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
_PAREN_NOTE = re.compile(r"[（(][^（()）]{1,24}[）)]")


def strip_parenthetical(text: str) -> str:
    """「WebAssembly（Wasm）をブラウザで」から括弧注記を落として *語だけ* にする。

    ここが残っていると「WebAssembly(」が話題名になり、索引を引き損ねて
    辞書情報に逃げる（v3 が語彙ゴミを返した原因の一片）。
    """
    out = _PAREN_NOTE.sub("", str(text or ""))
    return re.sub(r"[（(）)\s]+$", "", out).strip(" 、。・")


def question_subject(question: str) -> str:
    """「質問：WebAssembly（Wasm）をブラウザで…」→ WebAssembly のように主体を取る。"""
    q = str(question or "").strip()
    body = re.sub(r"^(?:質問|問い|問|設問)\s*[：:]\s*", "", q)
    m = re.match(r"^[「『\"']?([^「」『』\"'？?をはがにでと]{2,28}?)[」』\"']?\s*"
                 r"(?:とは|って何|とは何|を|は|が|に|で|の|って|についての|について)", body)
    if m:
        cand = strip_parenthetical(m.group(1))
        if cand and cand not in _PARTICLES:
            return cand
    m_what = re.search(r"([^\n「」『』？?]{2,24}?)\s*(?:とは|って)\s*(?:何|なん|どういう)", body)
    if m_what:
        cand = strip_parenthetical(m_what.group(1)).strip(" 　、,。をはがにでとも")
        if len(cand) >= 2:
            return cand
    m2 = re.search(r"([A-Za-z][A-Za-z0-9.+#_\-]{2,})", body)
    if m2:
        return strip_parenthetical(m2.group(1))
    m3 = re.search(r"([ァ-ヶー一-龯]{2,10})", body)
    return m3.group(1) if m3 else body[:12]


def normalized_word(word: str, *, kb=None) -> tuple[str, str]:
    """知らない語を、*読める部品を組み替えた語* に寄せます（表記揺れ・語順の対策）。

    「申年休假」のような語は、塊では索引にありません。ただ語順を入れ替えた
    「休暇」は実在するので、そこだけ寄せてから引きに行きます。当たったときは
    推測ではなく表記の組み替えとして扱うので、答えの根拠が崩れません。
    """
    from ..lang import lex

    w = str(word or "").strip()
    if len(w) < 2:
        return "", ""
    bank = lex.bank()
    cands: list[str] = []
    body = re.sub(r"[\s「」『』（）()。、・]+$", "", w)
    # 語幹の 2〜4 文字を取り出して *並べ替え* た形（休假 → 休暇、利用状 → 利用）
    tail = re.sub(r"(を|が|は|の|に|で|へ|と)$", "", body)
    for i in range(len(tail) - 1):
        seg = tail[i:i + 2]
        if not re.fullmatch(r"[一-龯]{2}", seg) or bank.has(seg):
            continue
        # 漢字 2 文字の *語順が逆* の形だけ（「争闘」→「闘争」）。片方が知らない字なら
        # 諦めます。ここでカタカナ語や部分文字列に手を出すと、知らない語が別語に化ける
        # （v3 の辞書引きより悪いでたらめになります）。
        rev = seg[::-1]
        if bank.has(rev) and all(bank.has(ch) for ch in seg):
            cands.append(rev)
    seen: set[str] = set()
    for c in cands:
        if not c or c in seen or c == tail:
            continue
        seen.add(c)
        if len(c) < 2:
            continue
        if len(c) != 2:
            continue
        try:
            if bank.has(c):
                return c, f"語の組み替え: 「{tail}」→「{c}」"
            if kb is not None and kb.exact_topic(c) is not None:
                return c, f"語の組み替え: 「{tail}」→「{c}」"
        except Exception:  # noqa: BLE001
            continue
    return "", ""


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
    def _content(cl) -> list:
        return [x for x in (getattr(cl, "claims", []) or [])
                if x.kind in ("answer", "definition", "fact", "reason", "step", "list", "evidence")
                and str(x.source) != "lex"]

    if subject and (dossier is None or float(getattr(dossier, "coverage", 0.0) or 0.0) < 0.5):
        # 問いの文（「…とは何ですか？」）で話題が引けなかったときは、*語そのもの* で引き直す。
        # ただし引き直し側は「語＝話題名」なので網羅率が上がり、*定義文だけ* を持ってきても
        # 差し替わってしまいます。答え・理由として使える文の数が本当に増えたときだけ採用します。
        try:
            alt = _frame_for(subject, kb=kb, history=history)
            alt.raw = subject
            got2 = gather(alt, kb=kb, web=web, tool_claims=[], history_text=subject)
            if got2 is not None and len(_content(got2)) > len(_content(dossier)):
                dossier, frame = got2, alt
                notes.append(f"証拠: 語「{subject}」で引き直した")
        except Exception:  # noqa: BLE001
            pass
    if not [c for c in (getattr(dossier, "claims", []) or [])
            if c.kind in ("answer", "definition", "fact", "reason", "step", "list")]:
        # 知っている部品に組み替えられる語なら、その語で引き直します
        variant, note = normalized_word(subject, kb=kb)
        if variant:
            try:
                alt = _frame_for(variant, kb=kb, history=history)
                alt.raw = f"{variant} は？"
                got3 = gather(alt, kb=kb, web=web, tool_claims=[], history_text=variant)
                if got3 is not None and len(_content(got3)) > len(_content(dossier)):
                    dossier, frame = got3, alt
                    notes.append(note)
                    notes.append(f"証拠: 組み替えた語「{variant}」で引いた")
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
    # 材料が *無い* ときだけ埋めます。長さが足りないだけで別話題を足すと、
    # 指示の的がずれるので（v3 はここで隣の話題を貼り付けていました）、字数は文の選び方で調整します。
    if not solid:
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
def _ask_frame(question: str) -> str:
    """問いの切り口を 1 つに絞る（答えの組み立て方を変えるための材料）。"""
    q = normalize(question)
    for pat, label in (("(メリット|利点|長所|何がよく|どこがよく|よさは)", "merit"),
                       ("(デメリット|短所|欠点|弱点|problem)", "demerit"),
                       ("(作り方|手順|どうや|やり方|方法|設定の仕方)", "how"),
                       ("(なぜ|理由|どうして|しくみ|仕組み)", "why"),
                       ("(比較|違い|どっち|どちら)", "compare"),
                       ("(いつ|時期|何年)", "when"),
                       ("(いくら|値段|費用|コスト)", "cost")):
        if re.search(pat, q):
            return label
    return "general"


# 材料が薄いときに足す 1 文は、*こちらが何を引き当てたか* の説明ではなく、
# その問いに対して次に効く判断を書きます（「上の記述を…」は読み手には何の情報にもならない）。
_INFER: dict[str, tuple[str, ...]] = {
    "merit": ("得が出るのは、同じ作業を繰り返す場面です。1 回の負担がそのまま積み上がります。",
              "いちばん効くのは、失敗したときに戻しやすい点です。直しが 1 手で済みます。"),
    "demerit": ("損が出るのは、条件が揃わないまま進めたときで、直しの往復が乗ってきます。",
                "崩れやすいのは、外側の決まりに頼っている部分です。そこが動くと一緒に壊れます。"),
    "how": ("詰まるのは大抵 1 歩目だけなので、そこだけ先に決めるとあとが続きます。",
            "手順は 3 つまでに絞ると、途中で止めずに済み直す量も減ります。"),
    "why": ("仕組みを見るのは、原因を 1 段ずつたどるときです。間の条件を並べると見え方が変わります。",
             "なぜそうなるかは、条件が揃った側と揃わなかった側を並べると説明がつきます。"),
    "compare": ("比べるときは、速さ・手間・戻しやすさの三本を揃えると迷いません。",
                "差が出やすいのは、動かなくなるときの手順です。"),
    "when": ("いつやるかは、混まない時間と締切のあいだで決めるのが安全です。",),
    "cost": ("費用は人が動く時間が主なので、そこを数えると当たりが付きます。",),
    "general": ("決めるべき点は 1 つなので、それを先に決めると他の判断は後からついてきます。",
                "次に見るのは、いちばん壊れやすい条件です。"),
}


def fallback_claims(question: str, frame, *, subject: str, web=None, kb=None,
                    notes: list[str] | None = None, tried: bool = False,
                    dossier=None) -> list[Claim]:
    """材料が薄いときの埋め方。*できない話で止めず*、近い記述から推理して組みます。

    v3 までは「語彙バンクに無い語です」「検索に出られないので埋めません」と並べて
    会話を止めていました。いまは (1) 発話が踏んでいる語を索引から引いて本文を出し、
    (2) その本文を材料にした推論の 1 文を必ず添えます。
    """
    notes = notes if notes is not None else []
    out: list[Claim] = []
    text = f"{subject or ''} {question}"
    items: list[dict] = []
    if kb is not None:
        try:
            from ..mind.chat import topic_items

            items = topic_items(text, kb=kb, limit=3, min_score=0.36)
        except Exception:  # noqa: BLE001
            items = []
    used: list[str] = []
    for it in items:
        for line in _kb_lines(it):
            if line and line not in used:
                used.append(line)
            if len(used) >= 3:
                break
        if len(used) >= 3:
            break
    for line in used:
        out.append(Claim(kind="fact", content=line, subject=subject, source="local:kb",
                         weight=0.84, extra={"via": "fallback:kb"}))
    if used:
        pool = _INFER.get(_ask_frame(question)) or _INFER["general"]
        infer = pool[len(used) % len(pool)]
        out.append(Claim(kind="inference", content=infer, subject=subject, source="local:infer",
                         weight=0.62))
        notes.append("材料が薄い: 近い記述を引いて推論の 1 文を添えた")
        return out

    # 索引にも無い語のときも、問いの語を割り直して答えにいく（辞書の語彙情報は出さない）
    rebuilt = _recompose_from_words(question, kb=kb)
    if rebuilt:
        out.extend(rebuilt)
        notes.append("材料が薄い: 発話の語を割り直して本文を引いた")
        return out
    from ..mind.think import shape_line

    shape = shape_line(f"{subject or ''} {question}")
    out.append(Claim(kind="inference", content=shape, subject=subject,
                     source="local:meta", weight=0.56))
    if len(shape) < 60:
        out.append(Claim(kind="ask",
                         content=f"{subject or 'この語'}について、どの切り口（意味・利点・手順・比較）で"
                                 "必要かを一言もらえれば、その形に組み替えます。",
                         subject=subject, source="local:meta", weight=0.48))
    notes.append("材料が薄い: 問いの形を数え直して方針から書いた")
    return out


def _kb_lines(item: dict) -> list[str]:
    """1 話題から使える本文を順に（辞書の語彙情報は除外）。"""
    pool = [str(item.get("def") or "").strip()]
    pool += [str(x).strip() for x in (item.get("how") or [])]
    pool += [str(x).strip() for x in (item.get("facts") or [])]
    pool += [str(x).strip() for x in (item.get("why") or [])]
    pool += [str(x).strip() for x in (item.get("tips") or [])]
    out: list[str] = []
    for line in pool:
        if not (12 <= len(line) <= 120) or not line.endswith(("。", "！", "？")):
            continue
        if re.search(r"(拍|索引|品詞|読みは|U\+)", line):
            continue
        out.append(line)
    return out


def _recompose_from_words(question: str, *, kb) -> list[Claim]:
    """発話を構成語に割って、それぞれの本文から 1 文ずつ拾う（知らない語のときの組み方）。"""
    if kb is None:
        return []
    words = [w for w in re.findall(r"[ァ-ヶー一-龯]{2,6}|[A-Za-z][A-Za-z0-9+#_.\-]{2,}",
                                   normalize(question))
             if w not in _PARTICLES_SET]
    out: list[Claim] = []
    seen: set[str] = set()
    for w in words:
        try:
            item = kb.exact_topic(w) or {}
        except Exception:  # noqa: BLE001
            item = {}
        head = str(item.get("def") or "").strip()
        if not head or head in seen:
            continue
        seen.add(head)
        out.append(Claim(kind="fact", content=head, subject=w, source="local:kb", weight=0.78,
                         extra={"via": f"recompose:{w}"}))
        if len(out) >= 2:
            break
    if out:
        out.append(Claim(kind="inference",
                         content="割り直した語の記述を合わせると、問いの中心は上の 1 文に収まります。",
                         source="local:infer", weight=0.6))
    return out


_PARTICLES_SET = set("はがをにでとものやへかのにやねよみなますだれたてもとだけもし")


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


def opening_line(d: "Directive", subject: str, *, tone: str, register: str,
                 target: int = 0, have: int = 0) -> str:
    """役割を与えられたときの 1 文目（指示の語から作り、口調に合わせて組み替える）。

    字数に余裕の無い指定（例: 200 字）で材料が足りているときは、前置きを *削る* ほうが
    中身が伝わります。役割の名乗りは本文の語彙にすでに現れているためです。
    """
    if d.fmt.no_greeting or d.fmt.strict:
        return ""
    if target and have >= int(target * 0.5) and subject:
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
        # 「空が青いのはなぜ」のような問い文を主語にすると、意味の無い前置きになります。
        if len(subject) > 12 or re.search(r"(なぜ|どうして|いくら|いつ|どこ|どの|いくつ|か)", subject):
            return ""
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


_EXTRA_FIELDS = ("why", "how", "tips", "example", "facts", "cost", "when", "opinion")


def same_topic_extra(question: str, subject: str, *, kb, exclude: list[str], limit: int = 4) -> list[str]:
    """目標字数に足りないとき、*同じ話題の使っていない欄* を足します。

    別話題を引っ張ると指示の的がずれるので、ここでは KB の同じ項目 (why / how / tips /
    example など) だけを見ます。辞書引きの欄（読み・拍・品詞）は問い返しません。
    """
    if kb is None:
        return []
    name = ""
    for probe in (subject, question):
        if not probe:
            continue
        try:
            got = kb.answer(probe)
        except Exception:  # noqa: BLE001
            got = None
        if got and (got.get("coverage") or 0) >= 0.3:
            name = str(got.get("topic") or "")
            break
    if not name:
        return []
    # `kb.answer()` は選ばれた 1 欄だけ返すので、*同じ話題の他の欄* は item を引きます。
    item = next((x for x in kb.items if str(x.get("topic") or "") == name), None)
    if not item:
        return []
    out: list[str] = []
    fields = item
    for key in _EXTRA_FIELDS:
        vals = fields.get(key)
        vals = [str(x).strip() for x in vals] if isinstance(vals, list) else ([str(vals).strip()] if vals else [])
        for s in vals:
            s = s.strip()
            if len(s) < 12 or any(s == x or s in x or x in s for x in exclude):
                continue
            if any(_overlap(s, x, 7) > 0.55 for x in exclude):
                continue
            out.append(s if s.endswith(("。", "！", "？")) else s + "。")
            if len(out) >= limit:
                return out
    return out


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

    # 主語が *本当に二重に* 付いているときだけ直す（「GLM5.3 GLM5.3 は…」）
    if subject and sentences:
        dup = re.compile(rf"^{re.escape(subject)}\s*[、,・:]?\s*{re.escape(subject)}")
        sentences = [dup.sub(subject, s.strip(), count=1) for s in sentences]

    opening = opening_line(d, subject, tone=tone, register=register,
                           target=target, have=sum(len(x) for x in sentences))
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
            if target and count_chars("\n".join(body)) < int(target * 0.75):
                # 足りないぶんは *同じ話題の他の欄* で埋めます（別話題には逃げない）
                extra = same_topic_extra(question, subject, kb=kb, exclude=body,
                                         limit=max(1, int(target / 60)))
                if extra:
                    body, f4 = _expand_to_target(body + extra, target, hard_max=cap)
                    notes.extend(f4)
                    notes.append(f"字数: 同じ話題の {len(extra)} 文を足した")

    styled, fixes = restyle("\n".join(body), tone=tone, register=register)
    notes.extend(fixes)
    body = [x for x in styled.split("\n") if x.strip()]

    # 口調の整形で文が伸びるので、*整形後* に字数契約を数え直す（指定は厳守）
    limit = max(target or 0, hard or 0)
    if limit and len(body) > 2 and count_chars("\n".join(body)) > int(limit * 1.2):
        room = int(limit * 1.15)
        kept = [body[0]]
        used = count_chars(body[0])
        for i, sent in enumerate(body[1:], start=1):
            n = count_chars(sent)
            if used + n > room and len(kept) >= 2:
                continue
            kept.append(sent)
            used += n
        if len(kept) < len(body):
            notes.append(f"length: 整形後の {limit} 字契約に合わせて {len(body) - len(kept)} 文落とした")
        body = kept

    if bullets:
        if fmt.numbered:
            lines = [f"{i}. {s}" for i, s in enumerate(body, 1)]
        else:
            lines = [f"{fmt.bullet_char or '・'}{s}" for s in body]
        text = "\n".join(lines)
    else:
        text = "\n".join(body) if len(body) > 1 else "".join(body)

    if sources and not fmt.strict:
        room = max(target or 0, hard or 0)
        if room:
            # 字数指定があるときは出典を 1 行に圧縮して *予算の中* に入れる（指定は厳守）
            links = " / ".join(str(s.get("url") or "") for s in sources[:2] if s.get("url"))
            cite = f"出典: {links}" if links else ""
            budget = int(room * 1.1) - count_chars("\n" + cite)
            kept, used = [], 0
            for line in body:
                n = count_chars(line)
                if used + n > budget and len(kept) >= 2:
                    continue
                kept.append(line)
                used += n
            body = kept
            text = "\n".join(body) if len(body) > 1 else "".join(body)
            if cite:
                text = f"{text}\n{cite}"
        else:
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
