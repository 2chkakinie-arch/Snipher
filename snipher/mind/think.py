"""Snipher の思考ループ — 見る(parse) → 集める(ground) → 決める(claims) → 書く(voice) → 検べる。

    ① 発話を Frame に分解する（語気・問いの型・制約・未知語）
    ② 会話状態を見て、進行中の手遊び・指示対象を復元する
    ③ 厳密に解ける仕事（計算・コード・暦・文字操作）はここで片付ける
    ④ 足りない知識は証拠として集める（知識ベース → 実辞書 → ウェブ裏取り）
    ⑤ 主張（Claim）を組み立てて voice が日本語に書く
    ⑥ validate + n-gram LM + 規則チェック を通さないと出さない

どの一手も「決まった文を引き当てる」ことはしません。素材が足りないなら、足りないと
分かる *具体的な理由* と、いま手元にある材料（語の分析・計算・出典）を並べます。
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field

from ..lang import lex, morph
from ..lang.phonetics import char_count, kana_to_ro, mora_count, normalize, to_hiragana
from .chat import claims_for as _chat_claims
from .frame import Claim, split_sentences
from .parse import build_frame, opaque_reason
from .play import claims_for_move, judge, pick, word_from_turn
from .rules import chain_ok, check
from .state import ConversationState
from .voice import Rendered, render

log = logging.getLogger(__name__)


@dataclass
class Thought:
    """1 ターン分の思考の記録（UI の「根拠」表示にも使う）。"""

    frame: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    dossier: dict = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)
    move: dict | None = None
    knowledge: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {"frame": self.frame, "state": self.state, "evidence": self.dossier,
               "steps": self.steps[:6]}
        if self.move:
            out["move"] = self.move
        return out


# --------------------------------------------------------------------------- #
# 特別な输入への「中身のある」応答（定型ではなく、入力を実際に解析した結果を出す）
# --------------------------------------------------------------------------- #
def opaque_claims(text: str, turn: int = 0) -> list[Claim]:
    """数字だけ・1 文字だけ・文字化けの入力に対し、**読める範囲を確定して**返す。"""
    t = normalize(text)
    claims: list[Claim] = []
    reason = opaque_reason(t)
    body = re.sub(r"[\s。、！？!?・…「」『』()（）]+", "", t)
    if reason in ("digits_only", "") and re.fullmatch(r"[0-9０-９.,%]+", body or "x") and body != "x":
        raw = body.replace(",", "").replace("．", ".")
        nums = [int(x) for x in re.findall(r"\d+", raw)][:2]
        facts: list[str] = []
        if len(nums) == 1:
            n = nums[0]
            from ..solve.math import _factorize, is_prime  # 局所 import（循環回避）

            facts.append(f"{n} は {char_count(str(n))} 桁の整数で、"
                         + ("素数です" if is_prime(n) else f"素因数分解すると {' × '.join(str(p) for p, e in _factorize(n) for _ in range(e))} です"))
            if n % 2 == 0:
                facts.append("2 で割れるので偶数です")
            else:
                facts.append("2 で割れないので奇数です")
            if 0 <= n <= 3000:
                facts.append(f"西暦 {n} 年という読み方もできます")
        else:
            facts.append(f"数字が {len(nums)} つ（{', '.join(map(str, nums))}）見えます。"
                         f"足すと {sum(nums)}、引くと {nums[0] - (nums[1] if len(nums) > 1 else 0)} です")
        tails = ("この数字が何を指すのか（年齢・金額・数量・年）が分かれば、そこに絞って答えます",
                 "どんな数の話ですか（金額・個数・年・割合）。一文くれれば計算に落とします",
                 "数字として読み取れました。何の数量かを一言もらえれば、そこから組み立てます")
        facts.append(tails[int(turn) % len(tails)])
        claims.append(Claim(kind="result", content="。".join(facts) + "。", source="tool:math",
                           weight=0.8, extra={"numbers": nums}))
        return claims
    if reason == "single_char" and body:
        # 「は？」「え？」は *聞き返しの記号*。字形の説明を返すと会話がちぐはぐになります。
        if re.fullmatch(r"[はへえあうぉん]{1,3}", body) and re.search(r"[?？]", t):
            return []
        ch = body[0]
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = ""
        if re.fullmatch(r"[ぁ-ん]", ch):
            name = f"ひらがなの「{ch}」（ローマ字 {kana_to_ro(ch)}）"
        elif re.fullmatch(r"[ァ-ヶ]", ch):
            name = f"カタカナの「{ch}」（ローマ字 {kana_to_ro(ch)}）"
        ent = lex.bank().entry(ch)
        bits = []
        if name:
            bits.append(f"{name} です（U+{ord(ch):04X}）")
        if ent is not None:
            bits.append(f"見出し語としても立っていて、読みは「{ent.reading or ch}」、品詞 {ent.pos}")
        else:
            bits.append("それ単独では品詞が付きにくい文字です")
        tails1 = ("前後の語をもう 1 語足してもらえれば、そこから意味を組み立てます",
                  "この一文字は独立した語として立たないので、続く語をください",
                  "単独の文字としては読めました。どんな文の中の一文字ですか")
        bits.append(tails1[int(turn) % len(tails1)])
        claims.append(Claim(kind="lexical", content=_join_bits(bits), subject=ch,
                            source="lex", weight=0.72))
        return claims
    if reason == "mojibake":
        codes = " ".join(f"U+{ord(c):04X}" for c in t[:8])
        claims.append(Claim(kind="result",
                            content=f"文字化けした列として読めました（{codes}）。UTF-8 / Shift_JIS "
                                    f"のどちらかで保存し直すと、こちらの解析は正常に通じます。",
                            source="tool:text", weight=0.7))
        return claims
    if reason == "latin_noise":
        letters = re.findall(r"[A-Za-z]+", t)
        claims.append(Claim(kind="result",
                            content=(f"欧文の断片 {len(letters)} 個（{' / '.join(letters[:4])}）と読めました。"
                                     + ("英単語なら綴り、コードなら言語を一言添えてください。組み立て直します。"
                                        if int(turn) % 2 == 0 else
                                        "日本語で何について聞きたいか一語だけください。そこから組み直します。")),
                            source="tool:text", weight=0.66))
        return claims
    claims.append(Claim(kind="note",
                        content=("送られた文字列には記号と空白しか見当たりません。語を 1 つ足してもらえれば、"
                                 "その語から組み立てます。" if int(turn) % 2 == 0 else
                                 "読める語がありませんでした。調べたい語を一語だけ送ってもらえれば、そこから始めます。"),
                        source="lex", weight=0.5))
    return claims


def capability_claims(*, kb=None, lm=None, core=None, web=None) -> list[Claim]:
    """「何ができる？」に、実測の規模と実モジュールで答える（自己紹介の定型ではない）。"""
    # 「何ができる」に対する答えは、*相手の仕事は何が進むか* で書きます。
    # 内部の索引語数や品詞・拍のような道具立てを並べるのは自己紹介ではありません。
    bits = ["質問にその場で答える、渡された文を指定の形に整える（JSON・箇条書き・表・文字数）、"
            "コードを書いて実行結果まで確認する、会話の途中で調べ物を引き継ぐ、の四つができます"]
    if kb is not None:
        try:
            s = kb.stats()
            bits.append(f"手元には {s.get('topics', 0)} 話題・{s.get('qa', 0)} 問答の材料があるので、"
                        "定義・理由・手順・いつ・値段のように問いの形で引き分けられます")
        except Exception:  # noqa: BLE001
            pass
    if core is not None:
        try:
            bits.append(f"内蔵の小型モデル（{core.n_params():,} パラメータ）で文の続きとその場の言い換えもします")
        except Exception:  # noqa: BLE001
            pass
    if web is not None:
        try:
            prov = ", ".join(web.status().get("providers") or [])
            bits.append(f"知らない語は {prov or '検索'} で本文まで開いてから書きます")
        except Exception:  # noqa: BLE001
            pass
    bits.append("数え物は分数・平方根まで厳密に検算します")
    # 1 文に繋げると *検証の長さ上限* でまとめて弾かれ、その隙に古い定型棚が
    # 顔を覗かせます。項目ごとに短い主張として渡し、長さの予算で落ちるようにします。
    kinds = ["fact", "fact", "opinion", "fact", "fact", "fact"]
    return [Claim(kind=kinds[i] if i < len(kinds) else "fact", content=b, subject="Snipher",
                  source="local:meta", weight=0.82 - 0.02 * i)
            for i, b in enumerate(bits)][:5]


def intro_claims(*, kb=None, core=None) -> list[Claim]:
    """自己紹介は 3 点だけ: 名前・役割・相手に何が起きるか。道具立ての数えません。"""
    bits = [Claim(kind="definition",
                  content="私は Snipher。日本語で質問に答えて、渡された文を指定の形に整えて、"
                          "コードまでその場で書くアシスタントです。",
                  subject="Snipher", source="local:meta", weight=0.84)]
    can = "問いには検索まで裏を取って答え、形式の指定（JSON・箇条書き・文字数・口調）は守るように組みます"
    if kb is not None:
        try:
            can += f"。手元の見出しは {int(kb.stats().get('topics') or 0)} 話題まであります"
        except Exception:  # noqa: BLE001
            pass
    bits.append(Claim(kind="fact", content=can + "。", subject="Snipher",
                      source="local:meta", weight=0.78))
    return bits


def identity_claims(*, kb=None) -> list[Claim]:
    out = [Claim(kind="definition",
                 content="私は Snipher という日本語の会話 AI で、答えるときは知識ベース・実辞書・"
                         "計算・ウェブ検索の順に材料を集めて、その場で文を組み立てています。",
                 subject="Snipher", source="local:meta", weight=0.8)]
    return out


# --------------------------------------------------------------------------- #
# 手遊び（規則に基づく手番）
# --------------------------------------------------------------------------- #
def play_turn(frame, state: ConversationState, *, turn: int) -> tuple[list[Claim], dict]:
    activity = frame.activity or state.activity
    if activity is None:
        return [], {}
    rules = activity.rules or frame.rules
    claims: list[Claim] = []
    info: dict = {"name": activity.name, "kind": activity.kind, "turn": activity.turn + 1}
    user_word = word_from_turn(frame.raw)
    prev = activity.our_word or ""
    if user_word and (activity.turn >= 1 or frame.ask != "definition"):
        mv = judge(prev, user_word, rules) if prev else \
            Move_stub(user_word)
        info["user_word"] = user_word
        info["legal"] = mv.legal
        if not mv.legal and prev:
            claims.append(Claim(kind="correction", content=mv.violation, source="lex", weight=0.9,
                                extra={"word": user_word}))
            # 違反しても会話は止めない。続きを手伝う（語彙バンクから助ける）
            helper = pick(prev, rules, used=activity.used)
            if helper.word:
                claims.append(Claim(kind="note",
                                    content=f"「{prev}」の次なら「{helper.word}」のような手があります。",
                                    source="lex", weight=0.7,
                                    extra={"word": helper.word, "candidates": helper.candidates[:5]}))
            return claims, info
        prev = user_word
    if not prev:
        # 先手はこちら。語彙バンクから一般的な名詞で始める（語尾が ん ではない語）
        starter = pick("", rules, used=activity.used)
        if starter.word:
            claims.extend(claims_for_move(starter, prev="", our_turn=True))
            info["word"] = starter.word
            return claims, info
        return [], info
    mv = pick(prev, rules, used=activity.used)
    if mv.word:
        claims.extend(claims_for_move(mv, prev=prev, our_turn=True))
        info["word"] = mv.word
        info["prev"] = prev
    return claims, info


class Move_stub:
    """相手が先に打った語の最小表現（判定をスキップする用途）。"""

    def __init__(self, word: str):
        self.word = word
        self.reading = to_hiragana(word)
        self.legal = True
        self.violation = ""
        self.reason = ""
        self.candidates: list[str] = []
        self.notes: list[str] = []
        self.from_ = "user"

    def as_dict(self) -> dict:
        return {"word": self.word, "legal": True}


def propose_activity_claims(text: str, *, kb=None) -> tuple[list[Claim], dict] | None:
    """「〜しよう」と誘われたら、その活動の *ルール* を知識から読んで始める。

    しりとり専用の分岐はありません。「語の連鎖」という規則を定義文から読み、
    語彙バンクで成立する手を選びます。
    """
    from .state import _activity_name      # 内部ヘルパ（同じ判定を再利用）

    name = _activity_name(normalize(text))
    if not name:
        return None
    rules: list = []
    def_text = ""
    if kb is not None:
        try:
            from ..ground.evidence import exact_definition

            def_text = exact_definition(name, kb=kb)
        except Exception:  # noqa: BLE001
            def_text = ""
    from .rules import from_definition

    rules = from_definition(def_text, topic=name) or [_chain_rule()]
    mv = pick("", rules, used=[])
    claims: list[Claim] = [Claim(kind="answer", content=mv.word or "", source="lex", weight=0.9,
                                 extra={"reading": mv.reading, "start": True})]
    note_bits = []
    if def_text:
        note_bits.append(f"ルールは知識にある通り「{_short(def_text)}」という進め方なので、")
    note_bits.append(f"私は「{mv.word}」から始めます。{mv.reason}")
    claims.append(Claim(kind="note", content="".join(note_bits).strip(), source="lex", weight=0.7))
    info = {"name": name, "kind": "chain", "word": mv.word, "rules": [r.describe() for r in rules]}
    return claims, info


def _chain_rule():
    from .frame import Rule

    return Rule(kind="chain", value="", raw="語の連鎖")


def _short(text: str, n: int = 34) -> str:
    t = normalize(text).strip("。 ")
    return t if len(t) <= n else t[:n] + "…"


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def _kb_has_actionable(text: str, *, kb) -> bool:
    """発話そのものが KB の問いと *ほぼ同じ* ときだけ、感情より対処を先に返す。

    「お腹が痛い」→ 一致高（QA 文が同じ）→ 対処を先にする。
    「疲れた」→ 弱い語彙一致で別話題を引っ張らない → まずは気持ちを受け止める。
    """
    if kb is None:
        return False
    try:
        hit = kb.answer(text) or {}
    except Exception:  # noqa: BLE001
        return False
    via = str(hit.get("via") or "")
    cov = float(hit.get("coverage") or 0)
    strong = via in ("qa_strong", "exact_topic", "exact", "alias") or via.startswith("qa:")
    if not strong:
        return False
    if via.startswith("qa:") and cov < 0.55:
        return False
    return str(hit.get("usage") or hit.get("field") or "") in {"qa", "tips", "how", "why"}


def plan_name(frame, dossier_obj, claims: list[Claim]) -> str:
    """どの知能が応答を作ったかを、既存の経路名に写す（UI・統計・テストの契約）。"""
    via = str((dossier_obj or {}).get("via") or "")
    if via == "tool":
        kind = claims[0].source.split(":")[-1] if claims else "task"
        if kind in ("clock", "facts"):
            return "tool:facts"
        return f"tool:{kind}"
    if via == "web":
        return "research:web"
    fields = [c.slot for c in claims if c.slot]
    if via == "kb":
        return f"knowledge:{fields[0] if fields else 'answer'}"
    if via == "lex":
        return f"lex:{frame.ask or 'word'}"
    if frame.act in ("greet", "thanks", "apology", "farewell", "agree", "disagree", "praise"):
        return frame.act
    if frame.ask in ("identity", "capability", "self_intro"):
        return "self"
    if frame.ask == "identity_user":
        return "session:profile"
    if frame.flags.get("opaque"):
        return "opaque_input"
    if frame.act in ("declare", "wish") and not claims:
        return "statement"
    return f"mind:{frame.ask or frame.act}"


def knowledge_meta(frame, dossier_obj, claims: list[Claim], thought: Thought) -> dict:
    meta: dict = {}
    raw = (dossier_obj or {})
    src = raw.get("claims") or []
    via = raw.get("via") or "none"
    topic = raw.get("topic") or ""
    cov = raw.get("coverage") or 0.0
    for c in src:
        if c.get("source", "").startswith("local:kb"):
            meta = {"topic": topic, "coverage": cov, "via": via, "field": None}
            break
    if not meta:
        meta = {"topic": topic or (frame.topic or None), "coverage": cov, "via": via}
    # KB 由来の主张には 欄名（field / usage）を付ける（既存契約）
    kb_claim = next((c for c in claims if c.source.startswith("local:kb")
                     and (not topic or str((c.extra or {}).get("topic") or "") in (topic, frame.topic or "")
                          or str((c.extra or {}).get("topic") or "") in (frame.topic or ""))), None)
    if kb_claim is not None:
        extra = kb_claim.extra or {}
        # 欄名は *実際に引けた欄* だけ名乗ります。「answer」とでっち上げると、
        # 知識を引き当てたふりになったので（v3 の契約検査がここを見ています）。
        slot = str(kb_claim.slot or extra.get("slot") or "")
        meta.update({"field": slot or None, "usage": slot or None,
                     "score": extra.get("score"), "qtype": extra.get("qtype"),
                     "coverage": extra.get("coverage", cov), "via": extra.get("via", via)})
    if via == "web":
        meta["sources"] = (raw.get("sources") or [])[:4]
    if thought.move:
        meta["move"] = thought.move
    return {k: v for k, v in meta.items() if v is not None}


def _finalize(out: Rendered, thought: Thought, frame, dossier_obj, claims: list[Claim],
              *, plan_hint: str = "") -> tuple[Rendered, Thought]:
    """応答の「経路名」と「根拠メタ」をここで確定させる（composer は写すだけ）。"""
    raw = dossier_obj.as_dict() if hasattr(dossier_obj, "as_dict") else dict(dossier_obj or {})
    via = str(raw.get("via") or "")
    # 検証済みの仕事（計算・暦・コード・TaskRouter）が含まれていれば、経路はその場で
    # tool に確定させる（証拠集めの via に引きずられない）。
    tool_claims = [c for c in claims if c.source.startswith("tool")]
    if tool_claims:
        via = "tool"
    _LEGACY_PLAN = {"greeting_name": "greeting:name", "word_problem": "math:word_problem",
                    "unit": "unit:convert", "equation": "math:equation",
                    "arithmetic": "math:arithmetic", "compare": "compare:answer"}
    kb_claim = next((c for c in claims if c.source.startswith("local:kb")), None)
    substantive = [c for c in claims
                   if c.kind not in ("note", "lexical", "ask")
                   or c.slot in ("def", "why", "how", "tips", "qa", "facts", "when", "where",
                                 "who", "cost", "opinion")]
    if plan_hint:
        plan = plan_hint
    elif frame.flags.get("opaque") and not any(c.source.startswith("tool") for c in claims):
        plan = "opaque_input"
    elif claims and not substantive:
        plan = "unknown_topic"
    elif via == "tool":
        kind = ((tool_claims or claims)[0].source.split(":")[-1] if claims else "task")
        if kind in _LEGACY_PLAN:
            plan = _LEGACY_PLAN[kind]
        else:
            plan = "tool:facts" if kind in ("clock", "facts") else f"tool:{kind}"
    elif via == "web":
        plan = "research:web"
    elif via == "kb":
        plan = f"knowledge:{(kb_claim.slot if kb_claim and kb_claim.slot else 'answer')}"
    elif via == "lex":
        plan = f"lex:{frame.ask or 'word'}"
    elif via == "local":
        plan = "statement" if frame.act in ("declare", "wish") else (frame.act or "statement")
    elif frame.act in ("greet", "thanks", "apology", "farewell", "agree", "disagree", "praise"):
        plan = frame.act
    elif frame.ask in ("identity", "capability", "self_intro"):
        plan = "self"
    elif frame.ask == "identity_user":
        plan = "session:profile"
    else:
        plan = f"mind:{frame.ask or frame.act}"
    out.authoritative = bool(tool_claims) or out.authoritative
    out.plan = plan
    meta: dict = {"via": via or None, "topic": raw.get("topic") or (frame.topic or None),
                  "coverage": raw.get("coverage"), "claims": raw.get("claims")}
    if kb_claim is not None:
        extra = kb_claim.extra or {}
        # `field` は「知識ベースのどの欄を引いたか」を名乗る欄です。会話は *話題の断片* を
        # 素材に使うだけなので、欄を引いていないときに "answer" と書くと、引き当てを
        # 装ったことになり、v3 の契約検査が弾きます（実际、知らない語の応答で出ていました）。
        slot = str(kb_claim.slot or extra.get("slot") or "")
        if via == "kb" and slot:
            meta.update({"field": slot, "usage": slot})
        else:
            meta.update({"field": None, "usage": None})
        meta.update({"score": extra.get("score"), "qtype": extra.get("qtype"),
                     "coverage": extra.get("coverage", meta.get("coverage")),
                     "via": extra.get("via", via)})
    if via == "web":
        meta["sources"] = (raw.get("sources") or [])[:4]
    if thought.move:
        meta["move"] = thought.move
    if out.lm:
        meta["lm"] = out.lm
    if plan in ("statement",) and not any(c.source.startswith(("local:kb", "web", "tool"))
                                          for c in claims):
        out.confidence = min(float(out.confidence), 0.52)
    # 相手の主語が *辞書にも索引にも無い語* のとき、周りの話題から話を組んではいますが、
    # その語について分かったふりはしません。確信度は上げておかないのが契約です。
    unknown_head = any(bool(getattr(e, "web_needed", False)) and not bool(getattr(e, "known_dict", True))
                       for e in (getattr(frame, "entities", None) or []))
    if unknown_head and via != "kb" and plan in ("statement", "unknown_topic"):
        out.confidence = min(float(out.confidence), 0.54)
    if plan == "opaque_input":
        # 入力が読めないターンは「答えられた」と見せない（確信度を盛らない）
        thought.knowledge = {}
        out.confidence = min(float(out.confidence), 0.36)
    else:
        thought.knowledge = {k: v for k, v in meta.items() if v is not None}
    if not out.text:
        thought.steps.append("応答を組めませんでした（材料なし）")
    return out, thought


def _run_instruction(text: str, *, kb=None, web=None, lm=None, core=None,
                     history: list[dict] | None = None, turn: int = 0):
    """指示層を 1 回だけ試す。指示でなければ None（会話の組み立てへ進む）。

    指示層は *例外で会話を止めない* 設計です。落ちたら None を返して、
    いつもの経路（社会的発話 → 証拠集め → 合成）がそのまま引き継ぎます。
    """
    try:
        from ..instruction import parse as _parse_instr
        from ..instruction import run_directive as _run_instr

        d = _parse_instr(text)
        if d is None:
            return None
        res = _run_instr(d, kb=kb, web=web, history=history, lm=lm, core=core, turn=turn)
        if res is None or not str(res.text or "").strip():
            return None
        body = str(res.text)
        notes = [f"task:{res.task}"]
        notes += [f"check:{c['name']}={'ok' if c['ok'] else 'ng'}" for c in res.checks[:4]]
        out = Rendered(
            text=body,
            sentences=[x for x in re.split(r"(?<=[。！？!?])|(?<=\n)", body) if x.strip()][:8],
            confidence=float(res.confidence), plan=res.plan, notes=notes[:4],
            fixes=["instruction"], lm=None, claims=[], authoritative=bool(res.authoritative))
        return out, notes
    except Exception:  # noqa: BLE001
        log.debug("指示層が失敗（会話経路へ続行）", exc_info=True)
        return None


def think(text: str, *, history: list[dict] | None = None, kb=None, web=None, lm=None,
          core=None, polisher=None, tasks=None, web_flag: bool | None = None,
          turn: int | None = None) -> tuple[Rendered, Thought]:
    """1 発話を受けて応答を組み立てる。返るのは (文章, 思考の記録)。"""
    history = history or []
    state = ConversationState(history, kb=kb)
    frame = build_frame(text, history=history, kb=kb, state=state)
    turn_no = int(turn if turn is not None else len(state.users) + 1)
    thought = Thought(frame=frame.as_dict(), state=state.as_dict(),
                      steps=[f"語気={frame.act} 問い={frame.ask or '-'} 話題={frame.topic or '-'}"])
    claims: list[Claim] = []

    # ---- 0-) 指示（プロンプト）: 頼まれた仕事を実行して、その結果をそのまま返す -- #
    # ここは会話の組み立てより前に置きます。理由: 指示文には「テキスト」「文章」
    # 「JSON」といった *材料を指す語* が入っているので、単語を見て反応する経路に
    # 先に触られると「テキストを読みました」が返ってしまいます（v3 の事故）。
    # 指示部と材料部を分けて読み、出力仕様（JSON スキーマ・件数・文字数・口調）まで
    # 契約として受け取り、実行 → 検証まで済ませてから返します。
    instr = _run_instruction(text, kb=kb, web=web, lm=lm, core=core, history=history,
                             turn=turn_no)
    if instr is not None:
        out, notes = instr
        thought.steps.append("指示: 指示部と材料を分けて読み、仕事を 1 つ実行した")
        thought.steps.extend(notes[:3])
        thought.knowledge = {"via": "instruction", "task": out.plan.split(":", 1)[-1],
                             "topic": frame.topic or None, "authoritative": True}
        thought.dossier = {"via": "instruction", "claims": [], "notes": notes[:6]}
        return out, thought

    # ---- 0) 挨拶・礼・感情の受け取り: 相手の語を返して続ける（先に決める） -- #
    if frame.act in ("greet", "thanks", "apology", "farewell", "agree", "disagree", "praise") \
            and not re.search(r"(とは|なぜ|どうして|教えて|何ですか|いくら|いつ|どこ)", frame.norm):
        claims.extend(_social_claims(frame, state))
        thought.steps.append("社会的発話: 相手の語を戻して受け取る")
        dossier_obj = _DossierLite(claims=claims, coverage=0.6, topic=frame.topic, via="local",
                                   sources=[], web_used=False, notes=[], evidence=[])
        out = render(dossier_obj, frame, turn=turn_no, lm=lm, polisher=polisher)
        thought.dossier = dossier_obj.as_dict()
        return _finalize(out, thought, frame, dossier_obj, claims)

    # ---- 0a) 検証可能な計算（2+2 / 3.5*2 / 2+2= …）はソルバで確定 --------- #
    # 「読めない入力」より前に置く: 式は不透明では無い。解けるときだけ確定させ、
    # 解けないときはそのまま次の段階へ落とす。
    _raw = str(text or "").strip()
    if re.fullmatch(r"[0-9０-９+\-×÷*/^=.%?\s]+", _raw) and re.search(r"[+\-×÷*/^=]", _raw) \
            and len(_raw) <= 60 and tasks is not None:
        try:
            _kind = tasks.classify(_raw, web=False)
        except Exception:  # noqa: BLE001
            _kind = None
        if _kind in ("arithmetic", "equation", "word_problem"):
            try:
                _ans = tasks.answer(_raw, web=False)
            except Exception:  # noqa: BLE001
                _ans = None
            if _ans is not None and str(getattr(_ans, "text", "") or "").strip():
                claims.extend([Claim(kind="result", content=str(_ans.text).strip(),
                                     subject=_kind, source=f"tool:{_kind}",
                                     weight=float(_ans.confidence or 0.9),
                                     extra={"task": _ans.as_dict()})])
                thought.steps.append(f"task={_kind}")
                dossier_obj = _DossierLite(claims=claims, coverage=0.9, topic=frame.topic,
                                           via="tool", sources=[], web_used=False,
                                           notes=["arithmetic"], evidence=[])
                out = render(dossier_obj, frame, turn=turn_no, lm=lm, polisher=polisher)
                thought.dossier = {"via": "tool", "claims": [c.as_dict() for c in claims]}
                return _finalize(out, thought, frame, dossier_obj, claims)

    # ---- 0b) 入力が読めないとき: 読めた部分を確定させる -------------------- #
    if frame.flags.get("opaque") in ("digits_only", "single_char", "mojibake", "latin_noise"):
        claims.extend(opaque_claims(text, turn_no))
        if not claims:
            # 「は？」「え？」は聞き返し。文字の説明ではなく、何を読み直せばよいかを数える
            claims.extend(_chat_claims(text, frame=frame, kb=kb, history=history, turn=turn_no))
        if claims:
            via = "tool" if claims[0].source.startswith("tool") else "local"
            dossier_obj = _DossierLite(claims=claims, coverage=0.5, topic=frame.topic, via=via,
                                       sources=[], web_used=False, notes=["opaque input"],
                                       evidence=[])
            out = render(dossier_obj, frame, turn=turn_no, lm=lm, polisher=polisher)
            thought.dossier = {"via": via, "claims": [c.as_dict() for c in claims]}
            return _finalize(out, thought, frame, dossier_obj, claims, plan_hint="opaque_input")

    # ---- 1) 進行中の手遊び / 遊びの提案 ------------------------------------- #
    move_info: dict = {}
    if state.activity is not None and (frame.flags.get("continuation") or frame.activity):
        pclaims, move_info = play_turn(frame, state, turn=turn_no)
        if pclaims:
            claims.extend(pclaims)
            thought.move = move_info
            thought.steps.append("進行中の規則で手番を打った")
    if not claims:
        prop = propose_activity_claims(text, kb=kb)
        if prop is not None:
            pclaims, move_info = prop
            claims.extend(pclaims)
            thought.move = move_info
            thought.steps.append("遊びの提案: 定義から規則を読んで開始した")

    # ---- 2) 自己紹介・能力（実測値で答える） -------------------------------- #
    if not claims and frame.ask == "identity_user":
        # 「私の名前は？」は自己紹介の質問ではない。会話の履歴を読む。
        facts = dict(getattr(state, "user_facts", {}) or {})
        name = str(facts.get("name") or "")
        if name:
            claims.append(Claim(kind="note",
                                content=f"{name} さんですね。{int(facts.get('name_at') or 1)} ターン目に"
                                        f"そう名乗っていました。",
                                source="session", weight=0.86,
                                extra={"name": name, "at": facts.get("name_at")}))
        else:
            claims.append(Claim(kind="note",
                                content="この会話ではまだ名乗ってもらえていないので、"
                                        "そちらの呼び方は手元に記録がありません。",
                                source="session", weight=0.8, extra={"name": None}))
            claims.append(Claim(kind="question",
                                content="名前を一言もらえれば、この先はその呼び方で通します。",
                                source="session", weight=0.72))
        if facts.get("age"):
            claims.append(Claim(kind="note",
                                content=f"年齢は {int(facts['age'])} 歳（"
                                        f"{int(facts.get('age_at') or 1)} ターン目）と覚えています。",
                                source="session", weight=0.7))
        thought.steps.append("相手の記憶: 自己紹介ではなく会話履歴を読んだ")
    if not claims and frame.ask == "self_intro":
        claims.extend(intro_claims(kb=kb, core=core))
        thought.steps.append("自己紹介: 名前・役割・相手benefit の 3 点だけ")
    if not claims and frame.ask in ("identity",):
        claims.extend(identity_claims(kb=kb))
        thought.steps.append("自己言及: 実構成で答えた")
    if not claims and frame.ask == "capability":
        claims.extend(capability_claims(kb=kb, lm=lm, core=core, web=web))
        thought.steps.append("能力は実測の規模で答えた")


    # ---- 2a2) 翻訳の依頼: 確認できる形の話 + 取れた用例 -------------------------------- #
    if frame.ask == "translation" and not any(str(c.source) == "web" for c in claims):
        word, lang = _translation_target(text)
        moved, romaji = "", ""
        if word:
            try:
                from ..instruction.translate import translate as _tr

                got, _nt = _tr(word, lang or "en")
                moved = str(got or "").strip()
            except Exception:  # noqa: BLE001
                moved = ""
            # 一語の依頼では "It is a cat." のように枠が足されるので、核だけ拾います。
            moved = re.sub(r"^(?:It is|This is|That is|It's|This is it)\s+(?:(?:a|an|the)\s+)?",
                           "", moved)
            moved = moved.strip().rstrip(".。").strip()
            try:
                romaji = kana_to_ro(to_hiragana(word))
            except Exception:  # noqa: BLE001
                romaji = ""
            want_ja = "日本" in lang
            if want_ja:
                ok = bool(re.search(r"[ぁ-んァ-ヶ一-龯]", moved)) and moved != word
            else:
                ok = (bool(re.search(r"[A-Za-z]", moved)) and moved != word
                      and moved.lower() not in (romaji.lower(), word.lower()))
            moved = moved if ok else ""
        if word and moved:
            body = f"「{word}」は {lang} で {moved} です。"
            if len(word) <= 4:
                body += "文ごとなら、その文をどうぞ。"
        elif word and romaji:
            # 対応語が引けない語を「訳せた」とは言いません。音写にとどめます。
            body = (f"「{word}」は音写して {romaji} と書きます。{lang} 側の対応語はまだ届いていないので、"
                    f"文をもらえれば文脈から訳を組みます。")
        elif word:
            body = f"「{word}」を {lang} に写すには、もう一言まわりの文をください。"
        else:
            body = "変換する語が特定できませんでした。"
        claims.append(Claim(kind="answer", content=body, subject=word, source="lex", weight=0.64,
                            extra={"word": word, "target": lang}))
        thought.steps.append("翻訳要求: 確認できた対応語で答えた")

    # ---- 2b) 日常の平叙・報告・こぼれ言 → chat 層（中身を見て組み立てる） ----- #
    # v3 までは「相手の文をそのまま引用 → 語を一語ください」だけでした。chat 層は
    # 発話行為・気分・内容語を読み、知識ベースから *その話題について言える事実* を引いて
    # 組み立てます。長文（今日○に行く予定、等）もここで処理するので、40 字で打ち切りません。
    if not claims and not _kb_has_actionable(text, kb=kb):
        _is_plain = (frame.act in ("declare", "wish", "invite")
                     and not re.search(r"(とは|なぜ|どうして|何ですか|いくら|教えて|方法|手順|使い方)",
                                       frame.norm))
        # 「X って何」のような定義問いで、X が未登録の *既知部品からなる語* なら、
        # 頭語（猿）の知識で答えるより *全体を分解して推測* するほうが正直です。
        _is_unknown_defq = (frame.ask in ("definition", "what")
                            and not re.search(r"(なぜ|どうして|いくら|いつ|どこ|方法|手順|使い方)",
                                              frame.norm))
        if _is_plain or _is_unknown_defq:
            compound = _unknown_compound(text)
            if compound:
                inferred = _infer_unknown_term(compound, turn=turn_no, text=text, kb=kb)
                if inferred:
                    claims.append(Claim(kind="note",
                                        content=_join_bits([inferred,
                                                            _inference_tail(compound, turn_no, text=text)]),
                                        subject=compound, source="lex", weight=0.62,
                                        extra={"inferred": True, "term": compound}))
                    thought.steps.append(f"未知語: 「{compound}」を部品分解して推測した")
        if _is_plain and not claims:
            claims.extend(_chat_claims(text, frame=frame, kb=kb, history=history, turn=turn_no))
        if _is_plain and not claims and not frame.flags.get("opaque"):
            #  unknown 語を含む発話は「〜なんですね」の言い換えでは終わらせません。
            claims.extend(_feeling_claims(frame, turn_no) if frame.mood != "neutral"
                          else _statement_claims(frame, turn_no))
        if claims:
            thought.steps.append("平叙の受け取り: 発話行為と内容語から組み立てた")
            from ..ground.evidence import Dossier

            dossier_obj = Dossier(claims=claims, coverage=0.55, topic=frame.topic, via="local",
                                  notes=["statement"], evidence=[])
            thought.dossier = dossier_obj.as_dict()
            out = render(dossier_obj, frame, turn=turn_no, lm=lm, polisher=polisher)
            if out.text:
                return _finalize(out, thought, frame, dossier_obj, claims)
            claims = []

    # ---- 3) 厳密に解ける仕事（計算・暦・文字・コード） ---------------------- #
    # 先に TaskRouter の確定分野（ここは元々厳密に解いている）を見る
    if not claims and tasks is not None:
        try:
            pre = tasks.classify(text, web=False)
        except Exception:  # noqa: BLE001
            pre = None
        if pre in ("greeting_name", "word_problem", "unit", "equation", "compare"):
            try:
                ans = tasks.answer(text, web=False)
            except Exception:  # noqa: BLE001
                ans = None
            if ans is not None and getattr(ans, "text", ""):
                claims.append(Claim(kind="result", content=ans.text.strip(), subject=pre,
                                    source=f"tool:{pre}", weight=float(ans.confidence or 0.9),
                                    extra={"task": ans.as_dict()}))
                thought.steps.append(f"task={pre}（先）")
    tool_claims: list[Claim] = []
    if not claims:
        if re.search(r"(今は|いまは|西暦何年|今日は何日|明日は何日|何曜日|今何時|UNIX| unix )", frame.norm):
            from ..solve import facts as _facts

            sol = _facts.handle(text)
            if sol is not None:
                tool_claims = [Claim(kind="time", content=" → ".join(sol.steps) + "。答えは "
                                     + sol.answer + "。", source="tool:clock", weight=0.96,
                                     extra=sol.as_dict())]
                thought.steps.append("clock verified")
    if not claims and not tool_claims:
        tool_claims = _solve_claims(frame, text, tasks=tasks, thought=thought)
        if tool_claims:
            claims.extend(tool_claims)

    if tool_claims and not claims:
        claims.extend(tool_claims)

    # ---- 4) 証拠（KB → 辞書 → web） ---------------------------------------- #
    dossier_obj = None
    from ..ground.evidence import gather

    # 手元に solid な材料があるときだけ検索を省く。「solid」= 道具层の答え、
    # または話題そのものを踏んだ知識ベースの記述。薄いつかみ（単語の読みなど）は
    # solid と数えないので、未知語の定義要求では必ずインターネットに出る。
    topic_key = str(frame.topic or "")

    def _solid(c) -> bool:
        src = str(getattr(c, "source", "") or "")
        if src.startswith("tool") or src == "web":
            return True
        # 翻訳の対応語が決まっていれば、それが答え。別話題（英語などの言語名）の
        # KB 記述を足すと「リスニングは…」「どのくらいやってきましたか」が増えるだけ。
        if frame.ask == "translation" and c.kind == "answer" and str(c.source) == "lex":
            return True
        if "kb" in src and topic_key and topic_key in str(getattr(c, "content", "")):
            return True
        return False

    solid = any(_solid(c) for c in claims)
    wants_web = (web is not None and web_flag is not False
                 and bool(frame.needs_web or frame.flags.get("current")))
    need_web = (bool(frame.needs_web) or bool(frame.flags.get("current"))) and web_flag is not False
    if not claims or (wants_web and not solid):
        dossier_obj = gather(frame, kb=kb,
                             web=web if need_web else None,
                             tool_claims=[c for c in claims if str(c.source).startswith("tool")]
                             or tool_claims, history_text=text)
        if web is not None and need_web:
            frame.flags["web_tried"] = True
        if dossier_obj.claims:
            merged: list = []
            seen: set[str] = set()
            for c in list(dossier_obj.claims) + list(claims):
                key = str(c.content)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(c)
            claims = merged
            dossier_obj.claims = merged
        thought.dossier = dossier_obj.as_dict()
        thought.steps.append(f"証拠: via={dossier_obj.via} claims={len(claims)} "
                             f"coverage={dossier_obj.coverage:.2f} web={dossier_obj.web_used}")
    else:
        from ..ground.evidence import Dossier

        dossier_obj = Dossier(claims=claims, coverage=0.8 if tool_claims else 0.7,
                              topic=frame.topic or state.previous_token(), via="lex",
                              notes=["手遊び/自己言及は証拠集合を省略"], evidence=[])
        thought.dossier = dossier_obj.as_dict()

    # ---- 5) 何も引けなかったときも「知らない」で止めない -------------------- #
    if not claims and frame.act == "declare" and frame.mood != "neutral":
        claims.extend(_feeling_claims(frame, turn_no))
        thought.steps.append("感情の受け取り: 述語を組み替り返す")
    if not claims:
        # 辞書の情報（読み・拍・品詞）は、*語そのものを尋ねる質問* の場合にだけ使います。
        # 「今日のニュースは？」に拍数を返すのが v3 の事故だったので、ここでは門番を通す。
        from ..ground.evidence import exact_definition, lexical_claims
        from ..mind.parse import wants_word_info

        if wants_word_info(text, frame):
            claims.extend(lexical_claims(frame, limit=3))
        have = {str(c.content) for c in claims}
        # *問いかけではない発話* に近い話題を被せると、報告が百科事典の一文に
        # 化けます（「深夜になった」→「夜は、体温と覚醒が…」）。ここでは問いだけ。
        if _is_question_text(text):
            extra = [x for x in suggestion_claims(text, kb=kb) if str(x.content) not in have]
            claims.extend(extra)
        if not claims:
            claims.extend(_decompose_claims(frame, text, turn_no, kb=kb))
            thought.steps.append("材料が薄いため、発話そのものの分析を返す")

    # 検索すべき語だったのに裏が取れなかったときは、*何を確かめて何が無いのか* を先に言う。
    disclosed = False
    has_substance = any(str(c.source).startswith(("tool", "web", "session"))
                        or ("kb" in str(c.source) and c.kind != "note")
                        or (frame.ask == "translation" and c.kind == "answer"
                            and str(c.source) == "lex")
                        for c in claims)
    if frame.needs_web and not has_substance:
        extra = _decompose_claims(frame, text, turn_no, kb=kb)
        if extra:
            claims.extend(extra)
            disclosed = True
            thought.steps.append("未知語の開示: 手元に無い理由と取り方を添えた")

    # 話題そのものを踏んだ記述（道具层の答え・検索・KB の本文）が 1 つも無いなら、
    # 索引から数えられる事実で一手足す。ターンごとに手が変わるので同じ相槌を返さない。
    from ..mind.parse import wants_word_info as _wants_word_info

    _lex_ok = _wants_word_info(text, frame) or frame.ask in ("reading", "word_property",
                                                              "synonym", "antonym", "word_list")
    # 「しりとりしよ」のような *遊びの提案* は語を尋ねていません。ここで索引の手を
    # 足すと、ゲームを始めた直後に「その語はどんな場面ですか」と聞き返します。
    _lex_ok = _lex_ok and not frame.flags.get("propose_activity")
    if not disclosed and _lex_ok and frame.ask not in ("translation", "code", "compute") \
            and not any(str(c.source).startswith(("tool", "web", "session")) or
                        ("kb" in str(c.source) and topic_key and topic_key in str(c.content))
                        for c in claims):
        menu = _topic_moves(frame, text, kb=kb, turn=turn_no, history=history or [])
        if menu:
            # 同じ語を名指しで掘る手を選んだなら、ただの羅列は冗長なので引く
            named = {str(c.extra.get("topic") or "") for c in menu if c.extra.get("topic")}
            if named:
                claims = [c for c in claims
                          if c in menu or not (c.extra.get("suggest")
                                               and any(f"「{t}」" in str(c.content) for t in named))]
            claims.extend(menu)
            thought.steps.append("手の選択: 索引から数えられる事実で一手足す")

    # 翻訳・計算・コードのように *実行して返した* 回では、「〜の話ですね」の相槌を
    # 前に足すと冗長です。実語の結果が既にあるなら、受け取りの一文だけを引きます。
    if any(c.kind == "answer" and str(c.source) in ("lex", "tool") for c in claims):
        def _is_ack(c) -> bool:
            s = str(c.content).strip()
            if c.kind not in ("note", "answer"):
                return False
            if c.kind == "answer" and str(c.source) != "local:chat":
                return False
            return bool(re.search(r"(の話ですね。|の話でした。|話をもらえました。"
                                  r"|の話を受け取りました。|件、受け取りました。)$", s))

        claims = [c for c in claims if not _is_ack(c)]
        if dossier_obj is not None:
            dossier_obj.claims = [c for c in dossier_obj.claims if not _is_ack(c)]

    if dossier_obj is not None and claims and not dossier_obj.claims:
        dossier_obj.claims = list(claims)
    if dossier_obj is None:
        from ..ground.evidence import Dossier as _D

        via = "tool" if any(c.source.startswith("tool") for c in claims) else (
            "web" if any(c.source == "web" for c in claims) else "local")
        dossier_obj = _D(claims=list(claims), coverage=1.0 if via == "tool" else 0.6,
                         topic=frame.topic, via=via, notes=["直接回答"], evidence=[])
        thought.dossier = dossier_obj.as_dict()
    out = render(dossier_obj, frame, turn=turn_no, lm=lm, polisher=polisher)
    # 翻訳の依頼は対応語が決まれば *検証済みの実行結果*。小型モデルのフォローアップを
    # 足さない（「どのくらいやってきましたか」のような無関係な一文が増えるだけ）。
    if frame.ask == "translation" and out.text and any(
            c.kind == "answer" and str(c.source) == "lex" for c in claims):
        out.authoritative = True
    if not out.text:
        from ..ground.evidence import Dossier

        out = render(Dossier(claims=claims, coverage=0.3, topic=frame.topic, via="lex",
                             notes=[], evidence=[]), frame, turn=turn_no, lm=lm,
                     polisher=polisher)
    if move_info:
        out.notes.append(f"move: {move_info}")
    return _finalize(out, thought, frame, dossier_obj, claims)


def _DossierLite(**kw):
    from ..ground.evidence import Dossier

    return Dossier(**kw)



def _topic_moves(frame, text: str, *, kb=None, turn: int = 0,
                 history: list[dict] | None = None) -> list[Claim]:
    """語彙にしか無い話題について、「いま事実として言える別々のこと」から一手選ぶ。

    話題ごとの定型文は持たない。すべて (1) 実辞書の索引 (2) 音の数 (3) 知識ベースの
    隣接 —— という *いつだって検証できる材料* から作るので、どの名詞が来ても同じ手順で
    返せる。ターンが進むと別の手に移るので、同じ相槌が繰り返されない。
    """
    b = lex.bank()
    w = str(frame.topic or "").strip()
    if not w:
        return []
    # 話題名に助詞が残っていたら落とす（「今日のニュース」型の取り込み防止）
    w = re.sub(r"^(私の|僕の|俺の|ボクの)?", "", w)
    # 指示文の札がそのまま語になった形（「質問:Redis」）は、問い返しで引用すると
    # 画面が内部語彙を見せてしまうので、語だけに残します。
    w = re.sub(r"^(?:質問|問い|設問|テーマ|topic|文章|テキスト|本文)\s*[:：]\s*", "", w, flags=re.I)
    w = w.strip("「」『』“”‘’\"'。、・:： 　")
    if len(w) > 16 or re.search(r"[。！？\n]", w):
        w = w[:16].strip("、。・ ")
    said = " ".join(str(m.get("content") or "") for m in (history or [])
                    if m.get("role") == "assistant")
    menu: list[list[Claim]] = []

    # v3 は語彙の数を *どんな質問にも* 混ぜて「辞書引きボット」に見えていました。
    # 索引・音・拍の話は、発話がそれそのものを尋ねているときだけ通します。
    ask_lex = bool(re.search(r"(拍|モーラ|音数|読み|ふりがな|を含む語|同类|品詞|活用|何文字|画数)",
                             normalize(text)))
    if not ask_lex:
        # (ア)(イ)(ウ) は作らず、(エ) の隣接話題だけ数えます。
        lexical_only = False
    else:
        lexical_only = True

    # (ア) 複合語の在庫 ── 語彙索引が実際に持っている数を言う
    try:
        comps = [x.surface for x in b.containing(w, limit=12) if x.surface != w]
    except Exception:  # noqa: BLE001
        comps = []
    if comps and lexical_only:
        menu.append([Claim(
            kind="note",
            content=f"「{w}」を含む語は手元の語彙に {len(comps)} 語見えて、"
                    f"例は {'、'.join(comps[:3])} です。",
            source="lex", weight=0.6, extra={"words": comps[:6]})])

    # (イ) 音で連なる語 ── 読みを引けるので漢字表記でも壊れない
    read = to_hiragana(lex.bank().reading(w) or "") or to_hiragana(w)
    try:
        kin = [x.surface for x in b.by_reading_prefix(read, limit=6, min_len=len(read) + 1)
               if x.surface != w]
    except Exception:  # noqa: BLE001
        kin = []
    if len(read) >= 2 and kin and lexical_only:
        menu.append([Claim(
            kind="note",
            content=f"読み「{read}」で始まる語なら {'、'.join(kin[:3])} があります。",
            source="lex", weight=0.58, extra={"words": kin[:6]})])

    # (ウ) 拍の数 ── 音拍索引で数えた実数
    n = mora_count(w)
    if n >= 1 and lexical_only:
        try:
            same = len(b.by_morae(int(n), limit=2000))
        except Exception:  # noqa: BLE001
            same = 0
        # 索引は打ち切りがあるので、上限に当たった数は「以上」としか言えない。
        tail = (f"同じ {int(n)} 拍の名詞は索引に {same} 語以上あります。"
                if same >= 2000 else (f"同じ {int(n)} 拍の名詞は索引に {same} 語あります。" if same else ""))
        menu.append([Claim(
            kind="note",
            content=f"「{w}」は {int(n)} 拍の語です。{tail}",
            source="lex", weight=0.56, extra={"morae": int(n)})])

    # (エ) 手元の隣接話題を 1 つ、名指しで説明する（羅列だけしない）
    if kb is not None:
        try:
            near = [str(x) for x in (kb.suggest(text, top_k=5) or []) if x]
        except Exception:  # noqa: BLE001
            near = []
        for cand in near[:5]:
            try:
                item = kb.exact_topic(cand) or {}
            except Exception:  # noqa: BLE001
                item = {}
            line = str(item.get("def") or "").strip()
            if line and line not in said:
                menu.append([Claim(
                    kind="note",
                    content=f"手元に「{cand}」の記述もあるので、そちらも話せます。{line}",
                    source="local:kb", weight=0.54, extra={"topic": cand})])
                break

    # (オ) 話題の続きを尋ねる（会話を止めない）
    menu.append([Claim(
        kind="question",
        content=f"「{w}」はどんな場面で使う語ですか。用途を一言もらうと、そちらの言い方に合わせて組みます。",
        source="lex", weight=0.5)])

    live = [m for m in menu if not all(c.content in said for c in m)]
    if not live:
        live = menu
    return list(live[int(turn) % len(live)])

_SHAPES: tuple[tuple[str, str], ...] = (
    (r"(手順|申請|手続き|やり方|使い方|方法|how\s*to)",
     "手続きを尋ねる形として組みます。決めるのは、いつまでに・誰に出すか・どの形で残すか、の三つです。"),
    (r"(値段|いくらか|費用|コスト|料金)",
     "費用の話として組みます。材料代と手間時間のどちらを先に押さえるかで答えの形が変わります。"),
    (r"(違い|比較|vs| versus|どっち)",
     "比較の形にします。速さ・手間・あとから拡張できるかの三本で揃えて比べるのが安全です。"),
    (r"(おすすめ|選び方|向いて|どれがいい|価値|コスパ|使いやすい)",
     "向きの話として組みます。扱う量と、壊れたときに立て直す時間を基準にすると迷いません。"),
    (r"(エラー|動かない|失敗|直したい|bug)",
     "詰まっている話として受け取ります。まず *最後に変わった一点* を切り分けると早く減ります。"),
)


def shape_line(text: str) -> str:
    """発話が *どの形の問い* かを読んで、当面の組み立て方針を 1 文で返します。

    語彙に無い語が来たとき、知らない語の説明をする代わりに、この方針で進めます。
    """
    t = normalize(str(text or ""))
    for pat, msg in _SHAPES:
        if re.search(pat, t, re.IGNORECASE):
            return msg
    if re.search(r"(いくつ|何個|数量|どれくらい|量)", t):
        return "量の話を聞いている形にします。単位と、数える対象が決まれば答えられます。"
    return "どんな場面で使う語かを一言もらえれば、その場で同じ形に組みます。"


def _is_question_text(text: str) -> bool:
    """問いの形か。「〜とは」「〜って」は疑問記号が無くても語を尋ねています。"""
    t = normalize(str(text or ""))
    if re.search(r"[？?]|教えて|知りたい|何ですか|どう|なぜ|いくら|いつ|どこ|できますか|ですか|"
                 r"ありますか|いますか|って何|とは何|とは何だ|何$|なん$|なぜか|なんで", t):
        return True
    if re.search(r"(?:か|かな)\s*[?？!！。]*$", t):
        return True
    return bool(re.search(r"(?:とは|って(?:は)?|という(?:意味)?は)\s*[。.]?\s*$", t))


# --------------------------------------------------------------------------- #
# 未知語の推論（inference_templates.json = 部品分解型の思考パターン集）
# --------------------------------------------------------------------------- #
# 知識ベースに無い語が来ても「一語ください」で止めないためのエンジン。
# 語を *既知の部品*（テンプレート辞書の語・実辞書の語・漢字の意味）に分解し、
# 部品の意味を足し合わせて読み进行す。コーパスは snipher/data/inference_templates.json
# （tools/build_inference.py で再生成）。読み込みは 1 回だけ・プロセス内で共有。

_TEMPLATE_INDEX: dict[str, dict] | None = None
_KANJI_MEANINGS: dict[str, str] = {
    "黄": "黄色く輝く", "金": "金のように貴重で輝く", "銀": "銀のように光る金属",
    "銅": "金属の一つ", "鉄": "硬い金属", "電": "電気", "光": "光・輝き",
    "闇": "暗さ", "炎": "火・熱", "火": "火・熱", "水": "水・流れる", "氷": "冷たい氷",
    "雪": "雪・白い", "雨": "雨", "風": "風", "雲": "雲", "霧": "霧・かすみ",
    "日": "太陽・日", "月": "月", "星": "星", "天": "空・天", "空": "空",
    "山": "山", "川": "川・流れる水", "海": "海", "森": "森",
    "木": "木・植物", "花": "花", "草": "草", "葉": "葉", "果": "実・果実",
    "土": "土・大地", "砂": "砂", "石": "石", "岩": "岩", "壁": "壁",
    "心": "心・気持ち", "意": "意志・意図", "思": "思う", "知": "知る・知識",
    "学": "学ぶ・学問", "文": "文章", "字": "文字", "言": "言葉", "語": "言葉",
    "名": "名前", "声": "声", "音": "音", "歌": "歌", "話": "話す",
    "比": "割合・比率", "率": "割合", "数": "数・量", "量": "量", "計": "はかる・計画",
    "算": "計算", "理": "理・ことわり", "論": "論じる", "法": "法則・方法",
    "則": "決まり", "規": "規範", "道": "道・教え", "術": "術・技術",
    "技": "技術", "工": "工作・工学", "器": "器具", "械": "機械", "機": "機械・仕組み",
    "球": "丸い形", "円": "円", "角": "角", "形": "形・かたち", "状": "状態",
    "体": "体・からだ", "身": "身", "頭": "頭", "手": "手", "足": "足・歩く",
    "目": "目", "耳": "耳", "口": "口・話す", "舌": "舌", "唇": "唇",
    "人": "人・ひと", "民": "民・人々", "王": "王", "将": "将・大将", "士": "士",
    "家": "家", "村": "村", "町": "町", "国": "国", "都": "都・首都",
    "世": "世", "代": "代", "時": "時", "期": "期間", "年": "年", "紀": "紀元",
    "朝": "朝", "夕": "夕", "夜": "夜", "昼": "昼",
    "色": "色", "赤": "赤", "青": "青", "白": "白", "黒": "黒", "紫": "紫",
    "緑": "緑", "藍": "藍", "朱": "朱", "灰": "灰色",
    "力": "力・ちから", "気": "気・空気", "魂": "魂", "霊": "霊", "神": "神",
    "鬼": "鬼", "獣": "獣", "鳥": "鳥", "魚": "魚", "虫": "虫", "馬": "馬",
    "牛": "牛", "犬": "犬", "猫": "猫", "猿": "猿", "狼": "狼", "虎": "虎",
    "龍": "龍", "竜": "竜", "蛇": "蛇", "亀": "亀", "貝": "貝",
    "食": "食べる", "飲": "飲む", "味": "味", "甘": "甘い", "辛": "辛い",
    "塩": "塩", "糖": "砂糖", "飯": "飯", "茶": "茶",
    "服": "服", "衣": "衣", "布": "布", "紙": "紙", "筆": "筆", "墨": "墨",
    "書": "書く", "画": "描く", "作": "作る", "生": "生まれる・生",
    "死": "死", "命": "命", "老": "老いる", "幼": "幼い",
    "愛": "愛", "情": "情", "恋": "恋", "友": "友", "親": "親", "子": "子",
    "父": "父", "母": "母", "男": "男", "女": "女",
    "走": "走る", "飛": "飛ぶ", "泳": "泳ぐ", "登": "登る", "降": "降る",
    "動": "動く", "止": "止まる", "転": "転がる", "跳": "跳ぶ",
    "追": "追う", "捕": "捕まえる", "逃": "逃げる",
    "見": "見る", "聞": "聞く", "触": "触れる", "嗅": "嗅ぐ",
    "考": "考える", "判": "分かる", "覚": "覚える", "忘": "忘れる",
    "教": "教える", "習": "習う", "研": "研究", "究": "究める", "索": "探す",
    "求": "求める", "探": "探す", "調": "調べる", "査": "調べる",
    "開": "開く", "閉": "閉じる", "入": "入る", "出": "出る", "帰": "帰る",
    "来": "来る", "往": "往く", "行": "行く", "届": "届く", "送": "送る",
    "取": "取る", "持": "持つ", "放": "放す", "置": "置く", "積": "積む",
    "並": "並ぶ", "続": "続く", "連": "連なる", "接": "接する", "結": "結ぶ",
    "切": "切る", "割": "割る", "分": "分ける", "合": "合わせる", "融": "溶ける・融合",
    "混": "混ざる", "排": "排する", "除": "除く", "選": "選ぶ", "抜": "抜く",
    "補": "補う", "増": "増える", "減": "減る", "倍": "倍", "半": "半",
    "全": "すべて", "一": "一つ", "二": "二つ", "三": "三つ", "四": "四つ",
    "五": "五つ", "六": "六つ", "七": "七つ", "八": "八つ", "九": "九つ",
    "十": "十", "百": "百", "千": "千", "万": "万", "億": "億",
    "大": "大きい", "小": "小さい", "長": "長い", "短": "短い", "高": "高い",
    "低": "低い", "深": "深い", "浅": "浅い", "広": "広い", "狭": "狭い",
    "厚": "厚い", "薄": "薄い", "重": "重い", "軽": "軽い", "速": "速い",
    "遅": "遅い", "早": "早い", "新": "新しい", "古": "古い",
    "強": "強い", "弱": "弱い", "硬": "硬い", "軟": "軟らかい", "滑": "滑る",
    "粗": "粗い", "細": "細かい", "多": "多い", "少": "少ない", "有": "ある",
    "無": "ない", "非": "非・反", "反": "反対", "対": "対する",
    "較": "比べる", "争": "争う", "戦": "戦う", "勝": "勝つ",
    "敗": "負ける", "攻": "攻める", "守": "守る", "防": "防ぐ", "護": "護る",
    "治": "治める", "統": "統べる", "領": "領する", "導": "導く",
    "使": "使う", "用": "使う", "働": "働く", "勤": "働く", "職": "職業",
    "業": "業・仕事", "商": "商う", "売": "売る", "買": "買う",
    "価": "価格", "額": "額", "費": "費用", "利": "利益",
    "損": "損", "債": "債", "税": "税", "給": "給与", "賞": "賞",
    "医": "医学", "薬": "薬", "病": "病", "康": "健康", "健": "健康",
    "美": "美しい", "麗": "麗しい", "妙": "妙", "奇": "奇妙", "偉": "偉い",
    "壮": "壮", "豪": "豪", "優": "優れる", "良": "良い", "善": "善い",
    "悪": "悪い", "毒": "毒", "危": "危ない", "険": "険しい",
    "安": "安い・安全", "静": "静か", "穏": "穏やか", "激": "激しい",
    "熱": "熱い", "寒": "寒い", "温": "温かい", "涼": "涼しい", "冷": "冷たい",
    "明": "明るい", "暗": "暗い", "濃": "濃い", "淡": "淡い",
    "鮮": "鮮やか", "艶": "艶", "輝": "輝く", "照": "照らす", "映": "映す",
    "閃": "閃く", "燃": "燃える", "焼": "焼く",
    "爆": "爆発", "炸": "炸裂", "蒸": "蒸す", "煮": "煮る", "炒": "炒める",
    "砕": "砕く", "破": "破る", "壊": "壊す", "裂": "裂ける",
    "亡": "亡くなる", "滅": "滅ぶ", "消": "消える", "尽": "尽きる",
    "満": "満つ", "溢": "溢れる", "流": "流れる", "注": "注ぐ",
    "滴": "滴る", "波": "波", "潮": "潮",
    "湧": "湧く", "噴": "噴く", "沸": "沸く", "湯": "湯", "泉": "泉",
    "源": "源", "派": "派", "系": "系",
    "類": "類", "種": "種", "族": "族", "型": "型",
    "態": "態", "式": "式", "様": "様子",
    "容": "容", "姿": "姿", "貌": "容貌",
    "面": "面・顔", "皮": "皮", "毛": "毛", "髪": "髪", "爪": "爪", "骨": "骨",
    "筋": "筋", "血": "血", "肉": "肉", "脂": "脂",
    "肺": "肺", "肝": "肝", "胃": "胃", "腸": "腸",
    "腎": "腎", "脳": "脳",
    "管": "管", "路": "路", "径": "径", "線": "線", "糸": "糸", "縄": "縄",
    "網": "網", "編": "編む", "織": "織る", "縫": "縫う",
    "裁": "裁く", "断": "断つ", "絶": "絶つ",
    "剣": "剣", "刀": "刀", "槍": "槍", "矢": "矢", "弓": "弓", "弾": "弾",
    "砲": "砲", "銃": "銃",
    "軍": "軍", "兵": "兵", "隊": "隊",
    "営": "営む", "陣": "陣", "幕": "幕", "旗": "旗",
    "印": "印", "章": "章", "紋": "紋",
    "絵": "絵", "図": "図", "像": "像", "写": "写す", "真": "真", "影": "影",
    "灯": "灯り", "灰": "灰",
    "燐": "燐", "磁": "磁気・磁石",
}

# 推論の言い回し（ターンと語で回して、毎回同じ文にしない）
_INFER_OPEN_2 = (
    "「{t}」は「{a}」と「{b}」を合わせた語と推測します。つまり、{m}だと考えられます。",
    "「{t}」を分解すると「{a}」＋「{b}」です。{m}と読めます。",
    "「{t}」は「{a}」の部分と「{b}」の部分から成ると考えます。{m}でしょう。",
)
_INFER_OPEN_N = (
    "「{t}」は{ps}から成ると推測します。全体として、{m}だと考えられます。",
    "「{t}」を分解すると{ps}です。{m}という読みが自然です。",
    "「{t}」は{ps}の組み合わせと考えられます。{m}に近いものだと推測します。",
)
_INFER_ONE = (
    "「{t}」は「{a}」に関わる語と推測します。文脈から、{a}の性質を持つものとして読みます。",
    "「{t}」の核は「{a}」だと考えられます。{a}にまつわるものとして組みます。",
)
_TAIL_ASK = (
    "この読みで組み立てます。",
    "まずはこの推測で進め、文脈が違えばその場で読み替えます。",
    "断定は控えますが、文字から見る限りこの読みが最も自然です。",
    "別の意味で使われていれば、同じ形に組み替えて答えます。",
    "文脈をもらえれば、この推測を裏取りします。",
)
_TAIL_TALK = (
    "この読みを軸に、会話を組み立てます。",
    "名前の由来なのか、特徴なのか、どちらからでも続けてください。",
    "どんな話をしたいのか、そのままの言葉で聞かせてください。",
    "この読みで続きを組むので、思うままどうぞ。",
    "知っていることがあれば、この上に足していく形で話せます。",
)


def _stable_hash(s: str) -> int:
    return sum(ord(c) for c in str(s)) % 1000


def _template_index() -> dict[str, dict]:
    """inference_templates.json を読み、term → 項目 の索引を作る（1 回だけ）。"""
    global _TEMPLATE_INDEX
    if _TEMPLATE_INDEX is None:
        idx: dict[str, dict] = {}
        try:
            import json
            import pathlib

            path = pathlib.Path(__file__).resolve().parents[1] / "data" / "inference_templates.json"
            if path.exists():
                for item in json.loads(path.read_text(encoding="utf-8")):
                    term = str(item.get("term") or "").strip()
                    if term:
                        idx[term.lower()] = item
        except Exception:  # noqa: BLE001
            pass
        _TEMPLATE_INDEX = idx
    return _TEMPLATE_INDEX


def _decompose_term_parts(term: str) -> list[tuple[str, str]]:
    """語を既知の部品に最長一致で分解する。→ [(surface, meaning)]"""
    idx = _template_index()
    bank = lex.bank()
    best: list[tuple[str, str]] = []
    i = 0
    n = len(term)
    while i < n:
        found = ""
        meaning = ""
        # 1) 推論テンプレート辞書の既知語（2〜6 文字、長い順）
        for L in sorted({min(6, n - i), 5, 4, 3, 2}, reverse=True):
            if L < 2:
                break
            cand = term[i:i + L].lower()
            entry = idx.get(cand)
            if entry is not None:
                found = term[i:i + L]
                first_morph = (entry.get("morphemes") or [{}])[0]
                meaning = str(first_morph.get("meaning") or "").strip() or found
                break
            # 2) 実辞書の語（2〜4 文字。語幹の連続「ぬるぬる」もここで拾う）
            if L in (4, 3, 2):
                w = term[i:i + L]
                if bank.has(w) or bank.entry(w) is not None:
                    found = w
                    meaning = w
                    break
        if not found:
            # 3) カタカナの塊（3 文字以上）は分解不能なので 1 部品として残す
            run = 0
            while i + run < n and "ァ" <= term[i + run] <= "ヶ" or (i + run < n and term[i + run] == "ー"):
                run += 1
            if 3 <= run <= n - i:
                found = term[i:i + run]
                meaning = found
            else:
                # 4) 漢字の意味表（1 文字）
                ch = term[i]
                found = ch
                meaning = _KANJI_MEANINGS.get(ch, ch)
        best.append((found, meaning))
        i += len(found)
    # 同一部品の連続（「ぬる」+「ぬる」）は畳んで 1 語にする
    merged: list[tuple[str, str]] = []
    for s, m in best:
        if merged and merged[-1][0] == s and len(merged[-1][0]) * 2 <= 6:
            prev, pm = merged.pop()
            merged.append((prev + s, m))
        else:
            merged.append((s, m))
    return merged


def _infer_unknown_term(term: str, *, turn: int = 0, text: str = "", kb=None) -> str:
    """未知の語を部品分解して *賢く推測する*（ユーザーの「黄金比」型の思考）。

    1) 推論テンプレート辞書に完全一致（黄金比・電球・GLM …）
    2) バージョン番号を落とした一致（glm5.3 → GLM）
    3) 既知の部品（テンプレート語・実辞書の語・漢字の意味）に分解して意味を足し合わせる
    4) 欧文・カタカナの語は固有名・略語としての読み
    """
    t = str(term or "").strip()
    if not t:
        return ""
    idx = _template_index()
    h = _stable_hash(t)

    # 1) 完全一致
    entry = idx.get(t.lower())
    if entry is not None and str(entry.get("inference") or "").strip():
        return str(entry["inference"]).strip().rstrip("。")

    # 2) バージョン番号を落とした一致（GLM5.3 → GLM、X2.0 → X）
    base = re.sub(r"[\d.]+$", "", t.lower())
    if 1 < len(base) < len(t):
        entry = idx.get(base)
        if entry is not None and str(entry.get("inference") or "").strip():
            body = str(entry["inference"]).strip().rstrip("。")
            if t.lower() not in body.lower():
                # 項目が語全体に触れていなければ、数字の扱いだけ足す（二重にはしない）
                body += f"。{t} の数字はバージョンや改訂版を示すと推測します"
            return body

    is_kana = bool(re.fullmatch(r"[ァ-ヶー]+", t))
    is_latin = bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9._\-]*", t))
    # カタカナ・欧文の語は 1 文字ずつの分解が無意味なので、固有名としての読みに行く
    if is_kana or is_latin:
        if is_kana:
            return (f"「{t}」はカタカナの語で、外来の固有名（製品・作品・人物・現象）か造語と推測します。"
                    f"文字は音の書き起こしなので、意味は周りの文脈から最も自然に読みます")
        return (f"「{t}」は欧文の語で、固有名（製品・人物・作品）か略語と推測します。"
                f"アルファベットの並びから、{t} という名で呼ばれるものだと読みます")

    # 3) 部品分解（テンプレート語 → 実辞書の語 → 漢字の意味、この順で最長一致）
    parts = _decompose_term_parts(t)
    kb_note = ""
    if kb is not None:
        try:
            for s, m in parts:
                if len(s) >= 2 and kb.index.topics_of(s):
                    kb_note = f"「{s}」の部分は手元の知識とつながるので、そこから補います。"
                    break
        except Exception:  # noqa: BLE001
            pass
    if len(parts) >= 3 or (len(parts) == 2 and all(len(s) == 1 for s, _ in parts)):
        named = "、".join(f"「{s}（{m}）」" if (m and m != s) else f"「{s}」"
                         for s, m in parts[:4])
        first_m = parts[0][1] if parts[0][1] != parts[0][0] else ""
        last_m = parts[-1][1] if parts[-1][1] != parts[-1][0] else ""
        if first_m and last_m:
            m = f"{first_m}の性質を持ちつつ{last_m}に関わるもの"
        elif last_m:
            m = f"{last_m}に関わるもの"
        else:
            m = f"部品の性質（{'、'.join(s for s, _ in parts[:3])}）を合わせたもの"
        tpl = _INFER_OPEN_N[(turn + h) % len(_INFER_OPEN_N)]
        out = tpl.format(t=t, ps=named, m=m)
    elif len(parts) == 2:
        a, ma = parts[0]
        b, mb = parts[1]
        if ma and ma != a and mb and mb != b:
            m = f"{ma}で{mb}に関わるもの"
        elif mb and mb != b:
            m = f"{a}の性質を持ちつつ{mb}に関わるもの"
        elif ma and ma != a:
            m = f"{ma}が際立つ{b}に関わるもの"
        else:
            m = f"{a}と{b}の性質を合わせたもの"
        tpl = _INFER_OPEN_2[(turn + h) % len(_INFER_OPEN_2)]
        out = tpl.format(t=t, a=f"{a}（{ma}）" if ma and ma != a else a,
                         b=f"{b}（{mb}）" if mb and mb != b else b, m=m)
    elif len(parts) == 1:
        a, ma = parts[0]
        tpl = _INFER_ONE[(turn + h) % len(_INFER_ONE)]
        out = tpl.format(t=t, a=a)
    else:
        out = (f"「{t}」は手元の知識に無い語ですが、文字の成り立ちから "
               f"{t} という性質を持つものと推測します")
    if kb_note and kb_note not in out:
        out += "。" + kb_note.rstrip("。")
    return out


def _inference_tail(term: str, turn: int, *, text: str = "") -> str:
    """推論の後の軽い受け止め（ターンと語で回す。「一語ください」型の強要はしない）。"""
    h = _stable_hash(term)
    if _is_question_text(text):
        return _TAIL_ASK[(turn + h) % len(_TAIL_ASK)]
    return _TAIL_TALK[(turn * 2 + h) % len(_TAIL_TALK)]


def _decompose_claims(frame, text: str, turn: int = 0, *, kb=None) -> list[Claim]:
    """語彙に無い語が来たとき、*読める断片と問いの形* から当たりを付けて先に進みます。

    索引の語数・読み・拍は答えとして出さない（v3 はここで辞書を引き返していました）。
    返すのは、問いの形に対する当面の組み立て方針だけです。
    """
    t = normalize(text)
    pieces = [p for p, _ in lex.bank().segment(t) if len(p) >= 1][:8]
    known = [p for p in pieces if len(p) >= 2 and lex.bank().has(p)]
    bits = []
    # 数量の問い（何個 / 何億 / いくら …）に既知の名詞が入っているとき、
    # その名詞で KB に当たる。定義・事実に *数* が書かれていることが多いので、
    # 「何個の島？」→「島」→ 日本の「4 つの大きな島…」で答えになります。
    if frame is not None and str(getattr(frame, "ask", "") or "") in ("count", "price"):
        for n in [p for p in known if len(p) >= 2][:2]:
            try:
                mat = kb.answer(n, min_score=0.30) if kb is not None else None
            except Exception:  # noqa: BLE001
                mat = None
            if mat and str(mat.get("text") or "").strip():
                _topic = str(mat.get("topic") or "")
                return [Claim(kind="answer",
                              content=(f"数えものとしての「{n}」は「{_topic}」につながります。"
                                       + str(mat["text"]).strip()),
                              subject=n, source="local:kb", weight=0.68,
                              extra={"topic": _topic})]
    # *平叙（報告・独り言）* に対して語の分解を返すと、「どんな場面で使う語ですか」で
    # 会話を止める形になります。まず会話の受け取りを試して、それが組めるときはそこに任せます。
    if not _is_question_text(text):
        try:
            got = _chat_claims(text, frame=frame, kb=kb, turn=turn)
        except Exception:  # noqa: BLE001
            got = []
        if got:
            return list(got)
    # 欧文・数字・カタカナの塊で、まだ語彙に無いもの。「それを聞かれている」ことを
    # 最初に名のると、答えが相手の発話から離れて見えません（無関係の定型文に見える）。
    opaque = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9.＿_-]*|\d[\d.]*|[ァ-ヶー]{3,}", t)
              if w not in ("って", "とは") and not lex.bank().has(w) and len(w) >= 2]
    if not opaque:
        # 和語・混成の語も「無い語は無い」と先に伝えると、答えが相手の発話から離れません。
        # 助詞で割った塊のうち、語彙にも索引にも無い 3〜8 文字だけを語として数えます。
        for run in re.split(r"[がをにはへと]", t):
            run = re.sub(r"(につい|につ|たい|たく|する|して|します|です|ます|である|れる|られ)+$", "", run)
            run = run.strip("。、!?！？「」『』 ・")
            if 3 <= len(run) <= 8 and not lex.bank().has(run) and lex.bank().entry(run) is None:
                opaque.append(run)
                break
    inferred = ""
    if opaque:
        term = opaque[0]
        inferred = _infer_unknown_term(term, turn=turn, text=text)
        if inferred:
            bits.append(inferred)
        else:
            bits.append(f"「{term}」については手元に記録がありませんが、文字の成り立ちから推測して組みます")
    # 形状の案内は、推論できなかったときだけ足す（推論できたのに形の話を重ねると二重になる）
    if not inferred:
        bits.append(shape_line(t))
    if known and sum(len(p) for p in known) >= max(4, int(len(t) * 0.35)) and not inferred:
        bits.append("読める部品は " + "、".join(f"「{p}」" for p in known[:3]) + " なので、そこを軸に組みます")
    # 推論できたときは *確認を強要しない*（「もう一語ください」型の聞き返しばかりでは会話が止まる）
    if inferred:
        bits.append(_inference_tail(term, turn, text=text))
    return [Claim(kind="note", content=_join_bits(bits), source="lex", weight=0.46)]


def suggestion_claims(text: str, *, kb) -> list[Claim]:
    from ..ground.evidence import suggestion_claims as _s

    return _s(text, kb=kb)


_FOLLOWUP_ASKS = {
    "negative": ("どうしてそうなったか、語を一語で教えてもらえますか。",
                 "いま一番しんどいのはどこですか。"),
    "positive": ("いちばん良かったのはどこでしたか。",
                 "次に同じことがあれば、何が足したいですか。"),
    "neutral": ("どんな状況だったか、語を一語だけ教えてください。",
                "そこから何を変えたかったですか。"),
}


def ask_pos(turn: int) -> str:
    pool = ("いちばん良かった部分をどこに置きましたか。", "次はどんな風に進めたいですか。")
    return pool[int(turn) % len(pool)]


def _unknown_compound(text: str) -> str:
    """*既知の修飾語 + 既知の名詞* でできていて、全体が未登録の語を出す。

    「ぬるぬる猿」は ぬるぬる（副詞）＋ 猿（名詞）に割れるが、その全体は語彙に
    無い *新しい語* です。こういう発話では頭語（猿）の知識で答えると造語が
    消えるので、全体を未知語として推論に渡します。助詞・動詞を含む普通の
    文（「猿を見た」等）は修飾語ではありません。
    """
    t = normalize(str(text or "")).strip(" 。、！？!?…・")
    # 質問の尾（〜って何 / 〜とは / 〜の意味 …）を落として *語そのもの* を判定する
    core = re.sub(r"(って何ですか|って何|とは何ですか|とは何|とは|の意味は|の意味|は何か|"
                  r"はなんですか|はなん|って何\?|何かな|なんだろう)$", "", t).strip("、。 ・")
    if core and len(core) >= 2:
        t = core
    if not (4 <= len(t) <= 12) or re.search(r"[\s「」『』:：]", t):
        return ""
    try:
        bank = lex.bank()
    except Exception:  # noqa: BLE001
        return ""
    if bank.has(t) or bank.entry(t) is not None:
        return ""
    try:
        segs = list(bank.segment(t))
    except Exception:  # noqa: BLE001
        return ""
    if not (2 <= len(segs) <= 3):
        return ""
    for w, pos in segs:
        p = str(pos).split("/")[0]
        if p not in ("名詞", "形容詞", "副詞", "接頭詞"):
            return ""
    if not any(str(pos).startswith("名詞") for _w, pos in segs):
        return ""
    return t


def _statement_claims(frame, turn: int = 0) -> list[Claim]:
    """質問ではない平叙・報告。述語を受け取り、続きを問う（相手の語を使う）。"""
    t = normalize(frame.raw).strip("。！？!? ")
    if not t or len(t) > 40:
        return []
    from ..lang.phonetics import kana_ratio

    if kana_ratio(t) < 0.35 or len(t) < 2:
        return []          # 欧文の羅列・1 文字は「読めない入力」側に回す
    pred = re.sub(r"(ですね|だよ|だわ|かな|のだった|のだ)$", "", t)
    pred = re.sub(r"(について|の話を|について語って|語って|話して|話してよ|教えて|教えてよ|見せて|"
                  r"してよ|してくれ|してほしい)+$", "", pred).strip("、。 ・")
    if not pred or pred[-1] in "にでをがはのとつ":
        pred = frame.topic or pred
    lemma = morph.lemma_of(pred) or pred
    # 相手の言い方をそのまま受け取る（活用を組み替えて「寒かっただった」のような
    # 非文法を作るより、自然で安全）
    ta = pred
    if frame.mood == "neutral" and not lex.bank().entry(lemma) and len(pred) < 3:
        return []
    pool = _FOLLOWUP_ASKS.get(frame.mood, _FOLLOWUP_ASKS["neutral"])
    ask = pool[int(turn) % len(pool)]
    quoted = pred if re.search(r"[いうたでる]", pred[-1:]) or len(pred) > 8 else f"「{pred}」"
    tail = "んですね" if quoted.endswith("い") else "なんですね"
    body = f"{quoted}{tail}。{ask}"
    return [Claim(kind="answer", content=body, subject=lemma, source="lex", weight=0.58,
                  extra={"predicate": pred, "lemma": lemma})]


_LANG_WORDS = ("英語", "日本語", "中国語", "中文", "韓国語", "朝鮮語", "フランス語", "ドイツ語",
               "スペイン語", "イタリア語", "ポルトガル語", "ロシア語", "外国語", "ローマ字",
               "タイ語", "インドネシア語", "越南語", "ベトナム語")
_TRANSLATE_RE = re.compile(
    r"[「『]?([^」』\n。]{1,24}?)[」』]?\s*(?:を|は)\s*(?:" + "|".join(_LANG_WORDS) + r")\s*(?:に|で)")


def _join_bits(bits: list[str]) -> str:
    """断片を 1 文に綴じる。敬体で言い切ったところでは読点でなく句点で切る。"""
    out = ""
    for raw in bits:
        x = str(raw).strip().rstrip("。！？!?")
        if not x:
            continue
        if not out:
            out = x
        elif out.endswith(("です", "ます", "ません", "しました", "でした", "でしたら",
                           "！", "？", "!", "?")):
            out += "。" + x
        else:
            out += "、" + x
    return (out + "。") if out else ""


def _translation_target(text: str) -> tuple[str, str]:
    """「X を英語にして」型の依頼から、変換対象 X と目標言語を取り出す。"""
    t = normalize(text)
    lang = next((x for x in _LANG_WORDS if x in t), "外国語")
    m = _TRANSLATE_RE.search(t)
    src = (m.group(1) if m else "").strip(" 、。「」『』")
    if not src:
        m2 = re.search(r"([A-Za-z][A-Za-z0-9'\\-]{1,24})", t)
        src = m2.group(1) if m2 else ""
    if not src:
        # 「A はどう英語にする？」型は動詞の型が掛からないことがあるので、*名詞 1 語* を
        # 探します。文をそのまま渡すと "The weather do to dou English" のような
        # 誤訳になるので、品詞で絞るのが安全です。
        skip = set(_LANG_WORDS) | {"どう", "何", "いう", "言い方", "翻訳", "訳", "訳し", "訳して",
                                   "訳す", "にして", "にしたい", "教えて", "ください", "して",
                                   "したい", "欲しい", "方法", "仕方", "場合", "場面", "語", "文"}
        try:
            for w, pos in lex.bank().segment(re.sub(r"[「」『』\s]", "", t)):
                w = re.sub(r"(って|とは|は|が|を|の|で|に|も|と|や|へ|から|まで|より|して)+$", "", w)
                if str(pos).split("/")[0] != "名詞" or not w or w in skip:
                    continue
                if any(k in w for k in ("訳", "翻訳", "英語", "日本語", "口調")):
                    continue
                src = w
                break
        except Exception:  # noqa: BLE001
            src = ""
    if not src and lang != "外国語":
        # 「猫って英語？」は言語名の *手前* が対象です。
        head = re.split(lang, t, 1)[0].strip()
        head = re.sub(r"(って|とは|は|が|を|の|で|に|も|と)+$", "", head)
        head = head.strip(" 、。「」『』？?")
        if 1 <= len(head) <= 12 and not re.search(r"(どう|何|いう|訳|翻訳|教えて|欲しい)", head):
            src = head
    return src, lang


def _feeling_claims(frame, turn: int = 0) -> list[Claim]:
    """「疲れた」「忙しい」など述語だけの発話への受け取り方（相手の語を組み替える）。"""
    t = normalize(frame.raw)
    # 依頼・命令・長い文は「こぼれ言」ではない。語をそのまま返すだけの応答になる。
    if len(t) > 12 or re.search(r"(して|してよ|くれ|ください|教えて|訳し|翻訳|書いて|作って|"
                                r"やりたい|したい|ほしい|方法|手順|いくら|何時|何年|何日)", t):
        return []
    pred = re.sub(r"[。！？!?ねよさわよ]+$", "", t)
    if not pred:
        return []
    lemma = morph.lemma_of(pred)
    ta = pred if re.search(r"(い|た|る|だ)$", pred) else pred + "の"
    if frame.mood == "negative":
        pool = (f"{ta}のは、今日だけのことですか。それとも続いていますか。"
                f"続けられる範囲で、今日できることを一つだけ一緒に決めましょう。",
                f"{ta}のはつらいですね。いちばんしんどいのはどこですか。休める時間はありますか。",
                f"{ta}んですね。いつ頃から続いていますか。短く教えてもらえれば、そこから組みます。")
        body = pool[int(turn) % len(pool)]
    elif frame.mood == "positive":
        pool = (f"{pred}、いいですね。きっかけを一言もらえますか。",
                f"{pred}とのこと。一度きりでしたか、続いていますか。",
                f"{pred}のはいいですね。{ask_pos(turn)}")
        body = pool[int(turn) % len(pool)]
    else:
        body = f"{ta}んですね。どういう状況だったのか、語を一語でいいので教えてください。"
    return [Claim(kind="answer", content=body, subject=lemma, source="lex", weight=0.6,
                  extra={"predicate": pred, "lemma": lemma, "mood": frame.mood})]


def _social_claims(frame, state: ConversationState) -> list[Claim]:
    """挨拶・礼・謝罪は *相手の語を返して* 続ける（定型の一文を出さない）。"""
    t = normalize(frame.raw)
    words = [w for w, pos in lex.bank().segment(t)
             if pos.split("/")[0] in {"名詞", "動詞", "形容詞"} and len(w) >= 2][:2]
    if frame.act == "greet":
        when = _when_phrase()
        return [Claim(kind="answer",
                      content=f"{_greeting_word(frame.norm)}。{when}、何を扱いますか。"
                              f"語彙も計算もコードも、この場で組み立てます。",
                      source="local", weight=0.7)]
    if frame.act == "thanks":
        return [Claim(kind="answer", content="いえいえ。続きがあれば、その語だけ送ってください。",
                      source="local", weight=0.66)]
    if frame.act == "apology":
        return [Claim(kind="answer", content="問題ありません。言葉を足してもらえれば、そこから組み直します。",
                      source="local", weight=0.6)]
    if frame.act == "farewell":
        return [Claim(kind="answer", content="では、また。次の会話でも分解から組み立てます。",
                      source="local", weight=0.6)]
    if frame.act in ("praise", "agree"):
        echo = words[0] if words else "その反応"
        return [Claim(kind="answer", content=f"{echo} の件、こちらもうまく通ってよかったです。",
                      source="local", weight=0.62)]
    if frame.act == "disagree":
        echo = words[0] if words else "そこ"
        return [Claim(kind="correction",
                      content=f"{echo} の部分は私の側で根拠が弱かったです。どの点が違うのか、語を一語だけ教えてください。",
                      source="local", weight=0.6)]
    return []


def _greeting_word(t: str) -> str:
    n = normalize(t)
    if "おはよう" in n:
        return "おはようございます"
    if "こんばんは" in n or "夜" in n:
        return "こんばんは"
    if "はじめまして" in n or "初めまして" in n:
        return "はじめまして"
    if "もしもし" in n:
        return "もしもし"
    return "こんにちは"


def _when_phrase() -> str:
    try:
        from ..solve.facts import now

        dt = now()
        return f"{dt.month} 月 {dt.day} 日の {dt.strftime('%H:%M')}"
    except Exception:  # noqa: BLE001
        return ""


def _solve_claims(frame, text: str, *, tasks=None, thought: Thought | None = None) -> list[Claim]:
    """計算・暦・文字操作・コード。ToolRouter より先に置かない（誤読を避ける）。"""
    out: list[Claim] = []
    from .. import solve as solver

    # a) 厳密計算・暦・言葉の操作（frame の型で絞る）
    if frame.ask in ("compute", "now", "when", "transform", "word_property", "reading",
                    "translation", "example", "word_list", "count", "summarize", "extract",
                    "synonym", "antonym", ""):
        try:
            got = solver.handle(text, hint="code" if frame.ask == "code" else "")
        except Exception:  # noqa: BLE001
            log.debug("solver.handle 失敗", exc_info=True)
            got = None
        if got is not None and got.solution is not None:
            sol = got.solution
            body = sol.answer
            if sol.steps:
                body = " → ".join(sol.steps) + "。答えは " + sol.answer + "。"
            out.append(Claim(kind="result", content=body, subject=frame.topic,
                             source=f"tool:{got.kind}", weight=0.95 if sol.verified else 0.8,
                             extra=sol.as_dict()))
            if thought is not None:
                thought.steps.append(f"solver={got.kind} verified={sol.verified}")
            return out
    # b) コード（生成 → 実行 → 検証）
    if frame.ask == "code":
        try:
            pair = solver.solve_code(text)
        except Exception:  # noqa: BLE001
            pair = None
        if pair:
            got, res = pair
            sol = got.solution
            out.append(Claim(kind="result", content=sol.answer, subject="code",
                             source="tool:code", weight=0.95 if res.ok else 0.72,
                             extra={"language": res.language, "ran": res.ran,
                                    "stdout": res.stdout[:200], "notes": res.notes[:3]}))
            if thought is not None:
                thought.steps.append(f"code verified={res.ok} ran={res.ran} lang={res.language}")
            return out
    # c) 方程式・文章題・創作・比較は TaskRouter の厳密solversが詳しい
    if tasks is not None:
        try:
            kind = tasks.classify(text, web=False)
        except Exception:  # noqa: BLE001
            kind = None
        if kind in ("arithmetic", "equation", "word_problem", "unit", "compare", "greeting_name"):
            try:
                ans = tasks.answer(text, web=False)
            except Exception:  # noqa: BLE001
                ans = None
            if ans is not None and getattr(ans, "text", ""):
                out.append(Claim(kind="result", content=ans.text.strip(), subject=kind,
                                 source=f"tool:{kind}", weight=float(ans.confidence or 0.9),
                                 extra={"task": ans.as_dict()}))
                if thought is not None:
                    thought.steps.append(f"task={kind}")
    return out


__all__ = ["think", "Thought", "capability_claims", "identity_claims", "intro_claims",
           "shape_line", "opaque_claims",
           "play_turn", "propose_activity_claims"]
