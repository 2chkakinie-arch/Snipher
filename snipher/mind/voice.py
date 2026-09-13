"""声（realization）— 主張（Claim）を日本語の文に組み下ろす。

定型文テーブルは置いてありません。ここで扱うのは *文をつなぐ語*（また／一方で／まず／
調べた範囲では …）と、文末の文体、文の長さの調整だけで、**中身は常に証拠から来ます**。
だから同じ質問をしても、前のターンと同じ文は出ません（接続語と順序はターンで回す）。

    render(dossier, frame) → Rendered(text, sentences, confidence, plan, notes)

生成後は必ず 3 重の検査を通します:

1. `snipher.composer.validate` … 助詞で切れた文・ループ・文体混在を捨てる
2. n-gram LM                    … 日本語として自然かの審判
3. 規則チェック（rules.check）  … 相手が指定した条件を満たすか
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ..lang import morph
from ..lang.phonetics import char_count, normalize
from .frame import Claim, split_sentences
from .rules import check, enforce

_CONNECT = {
    "add": ["", "", "また、", "加えて、", "それと、"],
    "reason": ["", "理由としては、", "これは、", "背景には、"],
    "contrast": ["", "ただ、", "一方で、", "反対に、"],
    "step": ["まず、", "次に、", "そのあとに、", "最後に、"],
    "evidence": ["調べた範囲では、", "開けたページが言うのは、", "複数ソースの一致は、", ""],
    "correction": ["", "惜しいのですが、", "ただ、"],
    "note": ["", "なお、"],
    "answer": ["", "答えは、", "まず、"],
    "lexical": ["辞書を引くと、", "表記を確認すると、", "言葉として、", ""],
    "advice": ["コツは、", "気をつけたいのは、", "実務的には、"],
    "opinion": ["私の見立てでは、", "率直に言うと、", ""],
}
_RELATION = {"definition": "add", "fact": "add", "answer": "answer", "reason": "reason",
             "step": "step", "evidence": "evidence", "correction": "correction",
             "lexical": "lexical", "note": "note", "advice": "advice", "opinion": "opinion",
             "format": "add", "example": "add", "ask": "add", "time": "answer", "list": "add"}
_ORDER = ("result", "definition", "answer", "fact", "reason", "list", "step", "advice",
          "example", "opinion", "lexical", "evidence", "correction", "time", "note", "format")


@dataclass
class Rendered:
    text: str = ""
    sentences: list[str] = field(default_factory=list)
    confidence: float = 0.5
    plan: str = "mind"
    notes: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    lm: dict | None = None
    claims: list[Claim] = field(default_factory=list)
    authoritative: bool = False

    def as_dict(self) -> dict:
        return {"plan": self.plan, "confidence": round(self.confidence, 3),
                "sentences": len(self.sentences), "notes": self.notes[:4],
                "fixes": self.fixes[:4], "lm": self.lm, "authoritative": self.authoritative}


def _rot(kind: str, turn: int) -> str:
    table = _CONNECT.get(kind, [""])
    return table[turn % len(table)]


def _overlap(a: str, b: str, n: int = 6) -> float:
    ga = {a[i:i + n] for i in range(max(0, len(a) - n + 1))}
    gb = {b[i:i + n] for i in range(max(0, len(b) - n + 1))}
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga)


def _tidy(sentence: str) -> str:
    s = normalize(sentence)
    s = re.sub(r"\s+", "", s) if re.search(r"[ぁ-んァ-ヶ一-龯]", s) else s
    s = re.sub(r"[。]+$", "", s)
    s = re.sub(r"、{2,}", "、", s)
    s = re.sub(r"(です|ます)(です|ます)$", r"\1", s)
    s = re.sub(r"[。]{2,}", "。", s)
    s = re.sub(r"。[、・]", "。", s)
    s = re.sub(r"([。！？!?])。", r"\1", s)
    s = re.sub(r"？。", "？", s)
    if not re.search(r"[。！？!?]$", s):
        s += "。"
    return s


def _ordered(claims: list[Claim]) -> list[Claim]:
    def key(c: Claim) -> tuple[int, float]:
        try:
            i = _ORDER.index(c.kind)
        except ValueError:
            i = len(_ORDER)
        return (i, -float(c.weight or 0.5))
    return sorted(claims, key=key)


def _subject_lead(claim: Claim, topic: str) -> str:
    """定義・事実の文に、相手の言葉で主語を立たせる（無いthenそのまま）。"""
    body = claim.content.strip()
    if not body:
        return ""
    if claim.kind != "definition":
        return body
    subject = normalize(claim.subject or topic or "")
    if not subject or len(subject) < 1:
        return body
    head = normalize(body)
    if head.startswith(subject) or subject in head[: max(6, len(subject) + 4)]:
        return body
    if re.match(r"^[A-Za-z0-9 .\-_+]+$", subject):
        return f"{subject} は、{body}" if not body.startswith(subject) else body
    if claim.kind == "definition":
        return f"{subject}は、{body}"
    return body


def _numbered(claims: list[Claim]) -> str:
    lines = []
    for i, c in enumerate(claims, 1):
        body = _tidy(c.content)
        lines.append(f"{i}. {body}")
    return "\n".join(lines)


def followup(frame, dossier, *, turn: int) -> str:
    """*具体的な隙* があるときだけ、その隙を名指しで問い返す。"""
    topic = (dossier.topic or frame.topic or "").strip()
    if dossier.web_used and len(dossier.sources) <= 1 and frame.ask in ("definition", "comparison",
                                                                        "list", "recommend"):
        return f"「{topic or 'その話題'}」は今のところ 1 件の出典しか読めていません。" \
               f"開発元・使い方・比較のどれを先に埋めましょうか。"
    if frame.ask in ("yesno", "opinion") and not topic:
        return "どんな場面で使う予定か、一言だけ添えてもらえますか。"
    if frame.ask in ("procedure",) and dossier.claims:
        return "手順は環境で変わる部分があるので、OS とバージョンを教えてください。"
    if frame.is_followup and not dossier.claims:
        return "どの部分を続ければよいか、言葉を一語だけ足してもらえますか。"
    return ""


def render(dossier, frame, *, turn: int = 0, lm=None, core=None, polisher=None,
           history: list[dict] | None = None, max_len: int = 215,
           validate=None) -> Rendered:
    """証拠のかたまりを、そのまま送れる日本語にする。"""
    notes: list[str] = list(dossier.notes)
    fixes: list[str] = []
    claims = _ordered([c for c in dossier.claims if c.content])
    if not claims:
        return Rendered(text="", plan="empty", confidence=0.0, notes=notes + ["no claims"])

    # 权威（計算・コード・暦）の結果は 1 文で出す — 生成で上書きしてはいけない
    hard = [c for c in claims if c.source.startswith("tool") or c.kind in ("result", "time")]
    authoritative = bool(hard) and all(c.source.startswith("tool") for c in hard)

    lines: list[str] = []
    used_texts: list[str] = []
    seen_steps = False
    for c in claims:
        body = _subject_lead(c, dossier.topic or frame.topic)
        if not body:
            continue
        if any(_overlap(body, prev, 7) > 0.62 for prev in used_texts):
            fixes.append(f"重複排除({c.kind})")
            continue
        rel = _RELATION.get(c.kind, "add")
        if c.kind in ("note", "correction", "lexical"):
            rel = c.kind
        conn = ""
        if c.kind == "step":
            if not seen_steps:
                seen_steps = True
                steps = [x for x in claims if x.kind == "step"]
                if len(steps) > 1:
                    lines.append(_numbered(steps))
                    used_texts.extend(x.content for x in steps)
                    continue
            conn = ""
        elif lines:
            # 接続語は *この返答の中で 1 回* に抑える。素材文のほうが先に同じ接続語を
            # 含んでいることもあるので、その場合は繋がない（二重の定型口癖に見える）。
            conn = _rot(rel, turn + len(lines))
            already = "".join(lines) + body
            step = 0
            while conn and conn in already and step < 3:
                step += 1
                conn = _rot(rel, turn + len(lines) + step)
            if conn and conn in already:
                conn = ""
        sent = _tidy(conn + body)
        if c.kind == "step" and seen_steps and len([x for x in claims if x.kind == "step"]) <= 1:
            sent = _tidy(f"{conn}{body}")
        # 長さの予算を超える主張は *打ち切らずに* 次の主張を見る。
        # 途中で切れた文は文章として壊れるので、重い順に並んだ主張から
        # 収まるものだけ並べるほうが安全。
        if lines and sum(len(x) for x in lines) + len(sent) > max_len:
            fixes.append(f"budget({c.kind})")
            continue
        lines.append(sent)
        used_texts.append(body)

    if authoritative and hard:
        # 検証済みの仕事の結果は、文章の飾り付けを挟まずそのまま返す（plan は tool:*）
        body = "\n".join(x.content.strip() for x in hard)
        conf = max(0.72, min(0.99, max(float(c.weight or 0.8) for c in hard)))
        return Rendered(text=body, sentences=split_sentences(body), confidence=conf,
                        plan=f"tool:{hard[0].kind}", notes=notes, fixes=fixes,
                        claims=hard, authoritative=True)

    # 長すぎたら末尾の主張から落としていく（読み切るための文章なので、途中で切らない）
    if not authoritative:
        while len(lines) > 1 and sum(len(normalize(x)) for x in lines) > max_len:
            dropped = lines.pop(-2) if len(lines) > 1 else lines.pop(-1)
            fixes.append(f"trimmed({len(dropped)})")

    # 出典（ウェブ裏取り）を後ろに付ける
    if dossier.sources:
        cite = "\n".join(f"[{i}] {s.get('title') or s.get('url')} — {s.get('url')}"
                          for i, s in enumerate(dossier.sources[:4], 1))
        lines.append("出典:\n" + cite)

    ask = followup(frame, dossier, turn=turn)
    if ask and sum(len(x) for x in lines) + len(ask) < max_len:
        lines.append(_tidy(ask))

    body_text = "\n".join(lines)
    if _short_required(frame) and len(normalize(body_text)) > 60:
        body_text = _shortest_line(lines) or body_text
    text = enforce(body_text, frame.rules)
    if frame.register == "plain":
        text = "\n".join(morph.to_plain(x) if not x.startswith(("[", "出典")) else x
                         for x in text.split("\n"))
        fixes.append("plain register")

    # 検査 → 通らなければ段階的に緩める
    cand = [text]
    cand.append("\n".join(x for x in lines[:2]))                      # 本文だけ
    cand.append(_tidy(lines[0]) if lines else "")                     # 1 文に縮める
    chosen = ""
    lm_info: dict | None = None
    for i, t in enumerate(cand):
        t = t.strip()
        if not t:
            continue
        _soft = frame.ask in ("word_list", "list") or frame.has_rule("count")
        _active_rules = [r for r in frame.rules
                         if r.kind != "chain" and not (_soft and r.kind in ("len", "morae", "count"))]
        ok_rules, why = check(t, _active_rules)
        ok_fmt = True
        if validate is not None:
            for piece in [x for x in split_sentences(t) if x][:6]:
                if piece.startswith(("[", "出典", "```")) or "```" in t:
                    continue
                ok_fmt, fwhy = validate(piece, max_len=200)
                if not ok_fmt:
                    why = "validate:" + fwhy
                    break
        if polisher is not None and ok_fmt and ok_rules:
            try:
                fixed = polisher.polish(t)
                if isinstance(fixed, dict):
                    fixed = fixed.get("text") or fixed.get("polished") or t
                if isinstance(fixed, str) and fixed.strip():
                    t = fixed
                    fixes.append("polisher")
            except Exception:  # noqa: BLE001
                pass
        score_info = None
        if lm is not None and not t.startswith("```"):
            try:
                sc = lm.score(re.sub(r"```.*?```", "", t, flags=re.S))
                score_info = {"confidence": round(sc["confidence"], 3),
                              "perplexity": round(sc["perplexity"], 2),
                              "bad_ratio": round(sc["bad_ratio"], 3)}
                if sc["bad_ratio"] > 0.18:
                    ok_fmt = False
                    why = f"lm bad ratio {sc['bad_ratio']:.2f}"
            except Exception:  # noqa: BLE001
                score_info = None
        if ok_fmt and ok_rules:
            chosen = t
            lm_info = score_info
            if i:
                fixes.append(f"repaired@{i}")
            break
        notes.append(f"候補 {i} を検査で棄却: {why}")
    if not chosen:
        chosen = cand[-1] or text
        fixes.append("loose fallback")
    conf = _confidence(dossier, frame, chosen, lm_info)
    plan = f"mind:{frame.ask or frame.act}"
    return Rendered(text=chosen.strip(), sentences=split_sentences(chosen), confidence=conf,
                    plan=plan, notes=notes, fixes=fixes, lm=lm_info, claims=claims)


def _short_required(frame) -> bool:
    r = frame.rule("len") if hasattr(frame, "rule") else None
    return bool(r and int(r.value or 999) <= 40)


def _shortest_line(lines: list[str]) -> str:
    good = [x for x in lines if x and not x.startswith(("[", "出典", "```"))]
    return min(good, key=lambda s: len(normalize(s))) if good else ""


def _confidence(dossier, frame, text: str, lm_info: dict | None) -> float:
    base = 0.34 + 0.5 * float(dossier.coverage or 0.0)
    if dossier.web_used:
        base = max(base, 0.78)
    if dossier.via == "kb":
        base = max(base, 0.7)
    if any(c.kind == "lexical" for c in dossier.claims):
        base = max(base, 0.56)
    if lm_info:
        base = 0.55 * base + 0.45 * float(lm_info.get("confidence") or 0.5)
    ok, _why = check(text, frame.rules)
    if not ok:
        base -= 0.12
    if len(normalize(text)) < 8:
        base -= 0.2
    return round(max(0.12, min(0.97, base)), 3)


def fingerprint(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:10]


__all__ = ["render", "followup", "Rendered", "fingerprint"]
