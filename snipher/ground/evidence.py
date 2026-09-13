"""証拠の調達 — 手元の知識・実辞書・ウェブ検索を「主張の材料」にそろえる。

    local  … 知識ベース v2（BM25 + 問いの型）から、求められている欄を引く
    lex    … 語彙バンクから表記・読み・拍・品詞・活用という *検証できる事実* を引く
    web    … 手元に無い語・現在性の必要な語だけを、Edge 検索と html-fetch で裏取る
    tool   … 計算・コード・暦の結果（検算済み）

同じ話題でも「何が足りていないか」で選ぶ欄が変わるので、定型の答えは出ません。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..lang import lex, morph
from ..lang.phonetics import mora_count, normalize, to_hiragana
from ..mind.frame import Claim, Evidence
from .web import WebGrounding


@dataclass
class Dossier:
    """1 ターン分の証拠束。voice はこれを文章にします。"""

    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    coverage: float = 0.0                # 手元知識でどれだけ満たせたか 0..1
    sources: list[dict] = field(default_factory=list)
    topic: str = ""
    via: str = "none"                    # kb / lex / web / tool / none
    web_used: bool = False
    notes: list[str] = field(default_factory=list)

    def by_kind(self, *kinds: str) -> list[Claim]:
        return [c for c in self.claims if c.kind in kinds]

    def as_dict(self) -> dict:
        return {"coverage": round(self.coverage, 3), "topic": self.topic, "via": self.via,
                "web_used": self.web_used, "claims": [c.as_dict() for c in self.claims[:8]],
                "sources": self.sources[:6], "notes": self.notes[:4]}


def lexical_claims(frame, *, limit: int = 2) -> list[Claim]:
    """発話の主語になった語について、辞書から確認できる事実を並べる。"""
    out: list[Claim] = []
    bank = lex.bank()
    for ent in frame.entities[:limit]:
        w = bank.entry(ent.surface)
        if w is None:
            continue
        bits = [f"表記「{w.surface}」、読み「{w.reading or to_hiragana(w.surface)}」",
                f"{mora_count(w.reading or w.surface)} 拍", f"品詞 {w.pos or '不明'}"]
        out.append(Claim(kind="lexical", content="、".join(bits) + "です。",
                         subject=w.surface, source="lex", weight=0.86,
                         extra={"reading": w.reading, "pos": w.pos,
                                "dictform": w.dictform}))
        if w.pos.startswith(("動詞", "形容詞")):
            forms: dict[str, str] = {}
            for f in ("ます", "て", "た", "ない"):
                try:
                    forms[f] = morph.inflect(w.dictform or w.surface, f)
                except Exception:  # noqa: BLE001
                    continue
            if forms:
                out.append(Claim(kind="lexical",
                                 content="活用は " + "、".join(f"{k}が「{v}」" for k, v in forms.items()) + "。",
                                 subject=w.surface, source="lex", weight=0.8,
                                 extra={"forms": forms}))
    return out


def _is_wordish(chunk: str) -> bool:
    """複合語の修飾部分が、実辞書で語として立つかわかる最小検査。"""
    if not chunk or len(chunk) < 2:
        return False
    try:
        b = lex.bank()
        if b.has(chunk):
            return True
        toks = [t for t, pos in b.segment(chunk) if pos.split("/")[0] in
                {"名詞", "動詞", "形容詞", "副詞"} and len(t) >= 2]
        return bool(toks)
    except Exception:  # noqa: BLE001
        return False


_PARTICLES = set("がをはにでのともしなますだましたれよねわぞぜばからまでよりだけつもやへ")


def _topic_tokens(query: str) -> list[str]:
    """発話を実辞書で分かち書きした語（誤一致検査の基準）。

    辞書に無い語（お腹・スマホ 等）で切れてしまうので、分かち書きに加えて
    2〜5 文字の続きの窓も候補として並べます（判定は *語境界* の側で行う）。
    """
    out: list[str] = []
    try:
        out += [t for t, _pos in lex.bank().segment(normalize(query)) if len(t) >= 1]
    except Exception:  # noqa: BLE001
        pass
    t = normalize(query)
    body = re.sub(r"[\s、。！？!?・…「」『』()（）]+", " ", t)
    for chunk in body.split(" "):
        if len(chunk) < 2:
            continue
        for ln in range(2, min(6, len(chunk)) + 1):
            for i in range(0, len(chunk) - ln + 1):
                out.append(chunk[i:i + ln])
    try:
        out += re.split(r"[\s、。！？!?・…「」『()（）]+", normalize(query))
    except Exception:  # noqa: BLE001
        pass
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _inside_longer_word(query: str, name: str) -> bool:
    """別名が発話の中で *より長い辞書見出しの一部* になっているか。

    しりとり ⊃ とり のように、語の途中に埋もれている一致は誤りです。
    分かち書きで出てくる語そのもの（お腹、タピオカ）なら一致として認めます。
    """
    n = normalize(name)
    if not n:
        return False
    try:
        toks = [tok for tok, _pos in lex.bank().segment(normalize(query))]
    except Exception:  # noqa: BLE001
        return False
    for tok in toks:
        tn = normalize(tok)
        if n in tn and tn != n and len(tn) > len(n) and lex.bank().has(tn):
            return True
    return False


def _qa_similarity(query: str, item: dict) -> float:
    """KB の「よくある問い」と発話の文字 2-gram 類似度（最大値）。"""
    try:
        from ..knowledge import dice
    except Exception:  # noqa: BLE001
        return 0.0
    q = normalize(query)
    best = 0.0
    for question in (item.get("questions") or []):
        qq = normalize(str(question or ""))
        if not qq:
            continue
        if qq in q or q in qq:
            return 1.0
        best = max(best, dice(q, qq))
    return best


def _at_word_boundary(query: str, name: str) -> bool:
    """別名が発話の中で「語として」現れたか（語の途中に埋もれていないか）。

    しりとり ⊃ とり は語の途中なので棄却、お腹が痛い ⊃ お腹 は語頭+助詞なので採用。
    """
    q = normalize(query)
    n = normalize(name)
    if not n:
        return False
    start = 0
    while True:
        i = q.find(n, start)
        if i < 0:
            return False
        before = q[i - 1] if i > 0 else ""
        j = i + len(n)
        after = q[j] if j < len(q) else ""
        ok_before = (not before) or before in "、。，『「(（ " or before in _PARTICLES \
            or not re.match(r"[ぁ-ん]", before)
        ok_after = (not after) or after in "、。，!！?？『」)( 　" or after in _PARTICLES \
            or not re.match(r"[ぁ-んァ-ヶ]", after)
        if ok_before and ok_after:
            return True
        start = i + 1


def _alias_match(kb, topic: str, tokens: list[str], query: str) -> tuple[str, str]:
    """KB が引けた話題が *本当に発話の語と一致するか* を確かめる。

    「しりとり」を発話から分詞して得た「とり」が鳥の別名に当たり、
    的外れな知識が coverage 1.0 で返る — v2 のいちばん酷い誤りです。
    そこで (1) 発話の語そのものが話題名・別名に等しいか、(2) 語が別名を **語として**
    含む（三毛猫 ⊃ 猫）かだけを通します。文字列の部分一致だけでは通しません。
    """
    norm = normalize(query)
    item = None
    try:
        item = next((it for it in kb.items if str(it.get("topic")) == topic), None)
    except Exception:  # noqa: BLE001
        item = None
    names: set[str] = set()
    if item is not None:
        names |= {normalize(str(x)) for x in (item.get("aliases") or []) if str(x)}
        names.add(normalize(topic))
    kana_names = {to_hiragana(x) for x in names if x}
    topic_norm, topic_kana = normalize(topic), to_hiragana(normalize(topic))
    for tok in tokens:
        t = normalize(tok)
        tk = to_hiragana(t)
        # 1 文字語（猫・犬・雪）も「発話の語＝話題名」なら確実な一致として通す。
        # 別名側の 1 文字一致は「しりとり ⊃ り」型の誤りになるので使わない。
        single_ok = (t == topic_norm or tk == topic_kana)
        if (len(t) >= 2 or single_ok) and (t in names or tk in kana_names):
            # 語の途中に埋もれている一致（しりとり ⊃ とり）は棄てる
            if not _inside_longer_word(query, t):
                return "exact", tok
            continue
        if len(tk) >= 3:
            for name in kana_names:
                if len(name) >= 1 and tk.endswith(name) and len(tk) - len(name) >= 2:
                    # 複合語の受け方は「修飾側が実際に語として立つ」ことを条件にする
                    rest = tk[: len(tk) - len(name)]
                    if _is_wordish(rest):
                        return "hypernym", tok
    # 和名の部分一致は「しりとり → とり(鳥)」のような誤りになるので使わない。
    # 英字の略語（GLM / LFM など）だけを、単語境界で一致を見る。
    for name in names:
        n = str(name)
        if len(n) >= 2 and n.isascii() and re.search(r"(?<![A-Za-z0-9])" + re.escape(n.lower())
                                                      + r"(?![A-Za-z0-9])", norm.lower()):
            return "exact", n
    return "none", ""


def kb_claims(query: str, *, kb, frame) -> tuple[list[Claim], dict]:
    """知識ベースから「問いの型に合う欄」を引き、足りない切り口を 1 つ足す。

    ただし **発話と話題が本当に合っているか** を先に確かめます（v2 までの
    「しりとり → 鳥」型の誤りは、ここで止めます）。許すのは次の 3 つだけ:

      exact     … 発話の中の語が、話題名または別名そのもの
      hypernym  … 発話の語が別名を含む複合語（三毛猫 ⊃ 猫）で、修飾側も語として立つ
      qa_text   … KB の「よくある問い」と発話がほぼ同じ文
    """
    if kb is None:
        return [], {}
    hits: list[Claim] = []
    meta: dict = {}
    try:
        material = kb.answer(query) or {}
    except Exception:  # noqa: BLE001
        material = {}
    kind, tok = "none", ""
    topic = str(material.get("topic") or "")
    if material:
        kind, tok = _alias_match(kb, topic, _topic_tokens(query), query)
        if kind == "none":
            item = None
            try:
                item = next((it for it in kb.items if str(it.get("topic") or "") == topic), None)
            except Exception:  # noqa: BLE001
                item = None
            if item is not None and _qa_similarity(query, item) >= 0.62:
                material = dict(material)
                material["via"] = "qa_text"
                kind = "qa"
            else:
                meta = {"rejected": f"KB の「{topic}」は発話の語と一致しません", "coverage": 0.0}
                material = {}
        elif kind == "hypernym":
            meta["hypernym"] = {"surface": tok, "topic": topic}

    if not material:
        # 複合語（三毛猫・量子もつれなど）は、末尾 / 先頭の構成語で引き直す。
        # 引き当てたら「構成語の知識です」と明かすので、ずれた話が見えません。
        head = _strip_ask_words(query)
        cands, cmeta = _compound_claims(head, kb=kb, frame=frame)
        if cands:
            return cands, cmeta

    text = str(material.get("text") or "").strip()
    if not text:
        return hits, meta
    field_name = str(material.get("field") or material.get("usage") or "answer")
    # 定義を求められているのに qa 欄を引いたなら、定義文を先に添える
    if field_name == "qa" and frame.ask == "definition":
        try:
            item = next((it for it in kb.items if str(it.get("topic")) == topic), None)
            defin = str((item or {}).get("def") or "").strip()
            if defin and defin not in text:
                hits.append(Claim(kind="definition",
                                  content=defin if defin.endswith("。") else defin + "。",
                                  subject=topic, source="local:kb", slot="def", weight=0.7))
                text = defin + " " + text
        except Exception:  # noqa: BLE001
            pass
    hits.append(Claim(kind=_kind_for_field(field_name), content=text, subject=topic,
                      source="local:kb", slot=field_name,
                      weight=float(material.get("confidence") or 0.62),
                      extra={"coverage": material.get("coverage"), "score": material.get("score"),
                             "qtype": material.get("qtype"), "via": material.get("via")}))
    if kind == "exact" and tok and normalize(topic) != normalize(tok) \
            and normalize(tok) not in normalize(text) and len(tok) <= 6 \
            and not re.search(r"[?？!！]", tok) and lex.bank().has(tok):
        hits.append(Claim(kind="note",
                          content=f"「{tok}」は単独の項目に無いので、近い話題「{topic}」の知識として出します。",
                          subject=topic, source="local:kb", weight=0.5, extra={"related": True}))
    hy = meta.get("hypernym") or {}
    if hy.get("surface"):
        hits.insert(0, Claim(kind="note",
                             content=f"「{hy['surface']}」は単独の項目を持っていないので、"
                                     f"上位の語「{hy.get('topic') or topic}」の知識で答えます。",
                             subject=hy["surface"], source="local:kb", weight=0.55,
                             extra={"hypernym": True}))
    follow = str(material.get("followup") or "").strip()
    if follow and not re.search(r"(か|かな|でしょう)[。！？?]?$", text):
        hits.append(Claim(kind="ask", content=follow if follow.endswith(("。", "？", "！"))
                         else follow.rstrip() + "。", subject=topic, source="local:kb",
                         slot="followup", weight=0.48))
    meta = {"topic": topic, "coverage": material.get("coverage"), "score": material.get("score"),
            "field": field_name, "via": material.get("via") or "kb",
            "hypernym": meta.get("hypernym") or {}}
    if frame.ask in ("definition", "describe", "open", ""):
        hits.extend(_extra_angles(query, kb=kb, topic=topic, have=text))
    return hits, meta


_FIELD_KIND = {"def": "definition", "why": "reason", "how": "step", "tips": "advice",
               "qa": "answer", "fact": "fact", "when": "fact", "where": "fact",
               "who": "fact", "cost": "fact", "opinion": "opinion", "followup": "ask"}


def _kind_for_field(field_name: str) -> str:
    return _FIELD_KIND.get(str(field_name or ""), "fact")


def _extra_angles(query: str, *, kb, topic: str, have: str) -> list[Claim]:
    out: list[Claim] = []
    if not topic:
        return out
    try:
        item = next((it for it in kb.items if str(it.get("topic")) == topic), None)
    except Exception:  # noqa: BLE001
        item = None
    if not item:
        return out
    for key in ("why", "tips", "how", "facts"):
        vals = item.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        for v in vals[:2]:
            sent = str(v or "").strip()
            if not sent or sent in have or _overlap(sent, have) > 0.5:
                continue
            out.append(Claim(kind=_kind_for_field(key), content=sent if sent.endswith(("。", "！", "？"))
                             else sent + "。", subject=topic, source="local:kb", slot=key,
                             weight=0.6))
            if len(out) >= 1:
                return out
    return out


def _overlap(a: str, b: str, n: int = 6) -> float:
    ga = {a[i:i + n] for i in range(max(0, len(a) - n + 1))}
    gb = {b[i:i + n] for i in range(max(0, len(b) - n + 1))}
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga)


_ASK_TAIL = re.compile(
    r"(?:とは|って何|ってなに|どういうもの|どんなもの|とは何|何ですか|なんですか|教えてください|"
    r"教えて|について|の意味|の定義|使い方|読み方)[^。！？?]*$"
)


def _strip_ask_words(query: str) -> str:
    """発話から「聞き方の定型」を落として、話題として引き直す語だけを出す。"""
    t = normalize(query)
    t = _ASK_TAIL.sub("", t)
    t = re.sub(r"[？！?!。、・…]+$", "", t).strip()
    t = re.sub(r"^(?:私は|僕らは|自分は)", "", t).strip()
    return t[:12]


def suggestion_claims(text: str, *, kb) -> list[Claim]:
    """手元に近い話題があれば、*名前を挙げて* 提案する（索引が言う事実で、推測ではない）。"""
    out: list[Claim] = []
    if kb is None:
        return out
    try:
        names = [str(x) for x in (kb.suggest(text, top_k=3) or []) if x]
    except Exception:  # noqa: BLE001
        names = []
    q_chars = {c for c in normalize(text) if "一" <= c <= "龯" or c.isascii() and c.isalnum()}
    if q_chars:
        names = [n for n in names if len({c for c in n if c in q_chars}) >= 2]
    if not names:
        return out
    # 話題名の羅列（「〜のあたりを話せます」）は *何も答えていない* ので、
    # 近い話題の本文を 1 文引いて返します。本文が無いときだけ名前を添えます。
    for name in names[:3]:
        try:
            item = kb.exact_topic(name) or {}
        except Exception:  # noqa: BLE001
            item = {}
        body = str(item.get("def") or "").strip()
        if len(body) < 10:
            continue
        out.append(Claim(kind="fact", content=body if body.endswith("。") else body + "。",
                         subject=name, source="local:kb", weight=0.6,
                         extra={"suggest": name, "coverage": 0.5}))
        break
    if not out:
        out.append(Claim(kind="note",
                         content=f"手元の索引には {'・'.join(names[:3])} の記述があります。",
                         source="local:kb", weight=0.5, extra={"suggest": names[:3]}))
    return out


def _compound_claims(head: str, *, kb, frame) -> tuple[list[Claim], dict]:
    """上位・構成語の知識を *正直な注記つき* で使うための検索（三毛猫 → 猫）。"""
    out: list[Claim] = []
    meta: dict = {}
    if kb is None or not head or len(head) < 2 or re.search(r"[?？\s]", head):
        return out, meta
    if head[-1] in "がをはにでとものよねか":
        return out, meta
    if any(ch.isspace() for ch in head) and len(head.split()) > 2:
        return out, meta
    tries: list[str] = []
    for ln in range(len(head) - 1, 0, -1):
        tries.append(head[-ln:])            # 末尾（三毛猫 → 猫）
        tries.append(head[:ln])            # 先頭（量子もつれ → 量子）
    item = None
    matched = ""
    for cand in dict.fromkeys(tries):
        if len(cand) < 1 or cand == head:
            continue
        try:
            got = kb.exact_topic(cand)
        except Exception:  # noqa: BLE001
            got = None
        if got:
            item, matched = got, cand
            break
    if not item:
        return out, meta
    topic = str(item.get("topic") or matched)
    field = _field_for_ask(frame.ask, item)
    body = str(item.get(field) or "").strip()
    if not body and item.get("def"):
        field, body = "def", str(item.get("def")).strip()
    if not body:
        return out, meta
    if head == topic:
        return out, meta
    _is_japanese_word = bool(re.search(r"[ぁ-ん一-龯]", head))
    if _is_japanese_word:      # 複合語を借りた事実だけを明示する（欧文語では発火させない）
        out.append(Claim(kind="note",
                         content=f"「{head}」は単独の項目を持っていないので、構成語「{topic}」の知識で答えます。",
                         subject=head, source="local:kb", weight=0.52,
                         extra={"hypernym": topic, "surface": head}))
    out.append(Claim(kind=_kind_for_field(field), content=body if body.endswith("。") else body + "。",
                     subject=topic, source="local:kb", slot=field, weight=0.66,
                     extra={"coverage": 0.5, "via": f"compound:{matched}"}))
    for extra_field in ("why", "tips", "facts"):
        vals = item.get(extra_field) or []
        if isinstance(vals, str):
            vals = [vals]
        if vals:
            sent = str(vals[0] or "").strip()
            if sent and _overlap(sent, body) < 0.5:
                out.append(Claim(kind=_kind_for_field(extra_field),
                                 content=sent if sent.endswith("。") else sent + "。",
                                 subject=topic, source="local:kb", slot=extra_field, weight=0.56))
                break
    meta = {"topic": topic, "coverage": 0.52, "field": field, "via": f"compound:{matched}",
            "hypernym": {"surface": head, "topic": topic}}
    return out, meta


def _field_for_ask(ask: str, item: dict) -> str:
    table = {"definition": "def", "reason": "why", "procedure": "how", "when": "when",
             "where": "where", "who": "who", "price": "cost", "opinion": "opinion",
             "recommend": "tips", "trouble": "tips", "count": "facts"}
    field = table.get(str(ask or ""), "def")
    if not item.get(field):
        for alt in ("def", "why", "how", "tips", "facts"):
            if item.get(alt):
                return alt
    return field


def exact_definition(name: str, *, kb) -> str:
    """話題名そのもの（または別名）に *完全一致* する定義だけ返す。

    発話の部分一致で引いた別話題（「しりとり」→「鳥」）を遊びの规则に流用しないため、
    ここでは曖昧一致を一切許しません。
    """
    if kb is None or not name:
        return ""
    try:
        item = kb.exact_topic(normalize(name)) or kb.exact_topic(to_hiragana(normalize(name)))
    except Exception:  # noqa: BLE001
        item = None
    if not item:
        return ""
    if str(item.get("topic") or "") and normalize(str(item.get("topic"))) != normalize(name) \
            and normalize(name) not in [normalize(x) for x in (item.get("aliases") or [])]:
        return ""
    return str(item.get("def") or "").strip()


def web_claims(query: str, *, frame, web: WebGrounding | None,
               extra_queries: tuple[str, ...] = ()) -> tuple[list[Claim], list[dict], list[Evidence]]:
    """ウェブ裏取り → 証拠文を「根拠のある主張」に変える（貼らない・出典つき）。"""
    if web is None or not web.available():
        return [], [], []
    topic = frame.topic or query
    terms = [e.surface for e in frame.entities if e.surface][:3] or [topic]
    q = " ".join(dict.fromkeys(terms + [topic]))
    want = 6 if frame.ask in ("list", "comparison", "recommend") else 5
    try:
        # ここまで来ている時点で「手元に無いので調べる」と判定済みなので、
        # 検索エンジン側の「調べる価値があるか」の推測は挟まない（明示的に依頼する）。
        g = web.gather(q, explicit=True, want=want,
                       extra_queries=tuple(x for x in extra_queries if x))
    except Exception:  # noqa: BLE001
        return [], [], []
    claims: list[Claim] = []
    for i, ev in enumerate(g.evidence):
        claims.append(Claim(kind="evidence", content=ev.text, subject=ev.title or topic,
                            source="web", urls=[ev.url] if ev.url else [],
                            weight=max(0.42, min(0.9, ev.score / 12.0)) * (1.0 - i * 0.06),
                            extra={"title": ev.title, "url": ev.url}))
    return claims, list(g.sources), list(g.evidence)


def gather(frame, *, kb=None, web: WebGrounding | None = None,
           tool_claims: list[Claim] | None = None, history_text: str = "") -> Dossier:
    """1 ターン分の証拠を束ねる（ここが「調べてから答える」の合流点）。"""
    d = Dossier()
    q = frame.raw or frame.topic
    tool_claims = tool_claims or []

    d.claims.extend(tool_claims)
    if tool_claims:
        d.via = "tool"
        d.coverage = 1.0

    local, meta = kb_claims(q, kb=kb, frame=frame)
    if local:
        d.claims.extend(local)
        d.coverage = max(d.coverage, float(meta.get("coverage") or 0.0))
        d.topic = str(meta.get("topic") or "")
        d.via = "kb" if d.via == "none" else d.via
        d.notes.append(f"知識ベース: {d.topic or '該当なし'} / 欄 {meta.get('field')} / "
                       f"網羅 {round(float(meta.get('coverage') or 0), 2)}")

    if d.coverage < 0.5:
        for c in suggestion_claims(q, kb=kb):
            if not any(k.content == c.content for k in d.claims):
                d.claims.append(c)

    # 辞書情報（表記・読み・拍・品詞）は、語そのものを尋ねる発話にだけ使います。
    # 話題の質問に辞書引きで答えるのが v3 の最大の欠点だったので、ここで門番を通します。
    from ..mind.parse import wants_word_info

    lex_claims = [] if d.coverage >= 0.5 or not wants_word_info(q, frame) else lexical_claims(frame)
    if lex_claims:
        d.claims.extend(lex_claims)
        if d.via == "none":
            d.via = "lex"
            d.coverage = max(d.coverage, 0.34)

    need_web = bool(frame.needs_web) or (d.coverage < 0.42 and frame.ask in (
        "definition", "reason", "procedure", "comparison", "recommend", "list", "when",
        "price", "who", "where", "count", "estimate")) or frame.flags.get("current")

    if need_web and web is not None and web.available():
        extra = ()
        if frame.ask == "definition":
            extra = (f"{d.topic or frame.topic} 概要",)
        elif frame.ask in ("when", "price", "count"):
            extra = (f"{frame.topic} 最新",)
        wclaims, sources, evidence = web_claims(q, frame=frame, web=web, extra_queries=extra)
        if wclaims:
            d.claims.extend(wclaims)
            d.evidence.extend(evidence)
            d.sources = sources
            d.web_used = True
            d.via = "web" if d.via in ("none", "lex") else d.via
            d.coverage = max(d.coverage, 0.72)
            d.notes.append(f"ウェブ裏取り: {len(evidence)} 文 / 出典 {len(sources)} 件")
        else:
            d.notes.append("ウェブは繋がったが使える文が無かった（または未接続）")
    elif need_web:
        d.notes.append("web 無効")
    return d


__all__ = ["Dossier", "gather", "lexical_claims", "kb_claims", "web_claims"]
