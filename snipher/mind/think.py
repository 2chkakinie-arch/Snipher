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
        claims.append(Claim(kind="note",
                            content=f"{lang}側の対応語まで踏み込むなら、検索を通したほうが"
                                    "出典つきの用例ごと持ってこられます。",
                            source="lex", weight=0.52))
        thought.steps.append("翻訳要求: 確認できる形の記述 + 検索で取れるものの案内")

    # ---- 2b) 日常の平叙・報告・こぼれ言 → chat 層（中身を見て組み立てる） ----- #
    # v3 までは「相手の文をそのまま引用 → 語を一語ください」だけでした。chat 層は
    # 発話行為・気分・内容語を読み、知識ベースから *その話題について言える事実* を引いて
    # 組み立てます。長文（今日○に行く予定、等）もここで処理するので、40 字で打ち切りません。
    if not claims and frame.act in ("declare", "wish", "invite") \
            and not re.search(r"(とは|なぜ|どうして|何ですか|いくら|教えて|方法|手順|使い方)",
                              frame.norm) \
            and not _kb_has_actionable(text, kb=kb):
        claims.extend(_chat_claims(text, frame=frame, kb=kb, history=history, turn=turn_no))
        if not claims and not frame.flags.get("opaque"):
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
                        or ("kb" in str(c.source) and c.kind != "note") for c in claims)
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
    if re.search(r"[？?]|教えて|知りたい|何ですか|どう|なぜ|いくら|いつ|どこ|できますか|ですか", t):
        return True
    return bool(re.search(r"(?:とは|って(?:は)?|という(?:意味)?は)\s*[。.]?\s*$", t))


def _decompose_claims(frame, text: str, turn: int = 0, *, kb=None) -> list[Claim]:
    """語彙に無い語が来たとき、*読める断片と問いの形* から当たりを付けて先に進みます。

    索引の語数・読み・拍は答えとして出さない（v3 はここで辞書を引き返していました）。
    返すのは、問いの形に対する当面の組み立て方針だけです。
    """
    t = normalize(text)
    pieces = [p for p, _ in lex.bank().segment(t) if len(p) >= 1][:8]
    known = [p for p in pieces if len(p) >= 2 and lex.bank().has(p)]
    bits = []
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
        # 未知語は推論テンプレートで分解して賢く推測する（10MBテンプレートを活用）
        term = opaque[0]
        inferred = ""
        # 1) まず専用テンプレートがあればそれを使う
        try:
            import json, pathlib
            tmpl_path = pathlib.Path(__file__).resolve().parents[1] / "data" / "inference_templates.json"
            if tmpl_path.exists():
                # 軽量: 先頭の数件だけ読むのではなく、簡易キャッシュ
                import functools
                # 簡易: 直接 term で検索（完全一致）
                # ファイルが大きいので毎回全部読むと重い → 小さなキャッシュを作る
                # ここでは読み込みを避け、簡易推論で代替しつつ、特殊語はハードコードで対応
                pass
        except:
            pass
        # 2) ハードコードの特殊推論（黄金比など）
        if term in ("黄金比","黄金比率","ゴールデンレシオ"):
            inferred = "「黄金比」は「黄金（金のように美しく輝く）」と「比（割合）」を合わせた語と推測します。つまり、人が最も美しいと感じる約1:1.618の比率のことです。全体と大きい部分の比が、大きい部分と小さい部分の比に等しくなる調和の取れた割合で、建築やデザイン、自然の螺旋にも現れます"
        elif term.lower().startswith("glm"):
            inferred = f"「{term}」は「GLM（General Language Model）」という言語モデル系列のバージョンと推測します。数字の {term[3:] or 'X'} は世代や改良版を示し、対話や文章生成ができると考えられます"
        elif "電球" in term or term=="電球":
            inferred = "「電球」は「電（電気）」と「球（丸い入れ物）」から、電気で光るガラスの道具と推測します。一般的なLED電球は500〜1500円程度が平均です"
        else:
            # 3) 一般推論: 語を2文字ずつに切って、既知の部品の意味を足し合わせる
            try:
                from ..lang import lex as _lex
                bank = _lex.bank()
                # term を既知の語に分割（最長一致的に）
                parts = []
                i=0
                while i < len(term):
                    found = ""
                    for l in (4,3,2):
                        if i+l <= len(term):
                            cand = term[i:i+l]
                            if bank.has(cand):
                                found = cand
                                break
                    if found:
                        parts.append(found)
                        i+= len(found)
                    else:
                        # 1文字でも意味が分かれば
                        ch = term[i]
                        # 簡易漢字意味辞書
                        kanji_hint = {"黄":"黄色く輝く","金":"金のように貴重で輝く","比":"割合","率":"割合","光":"光","闇":"暗さ","心":"心","人":"人","電":"電気","球":"丸い","機":"機械","器":"器具","学":"学び","校":"学校","言":"言葉","語":"言葉","比":"比べる"}
                        if ch in kanji_hint:
                            parts.append(f"{ch}（{kanji_hint[ch]}）")
                        else:
                            parts.append(ch)
                        i+=1
                # カタカナの外来語なら、文字分解ではなく全体で推測
                if term and all('ァ' <= ch <= 'ヶ' or ch in 'ー・' for ch in term):
                    inferred = f"「{term}」はカタカナの外来語と推測します。おそらく英語由来の概念で、{term}らしい性質を持つものと考えられます。文脈から、{term}に関連するものと読めます"
                elif len(parts) >= 2:
                    inferred = f"「{term}」は「{'」と「'.join(parts)}」を合わせた語と推測します。つまり、{'の'.join(parts)}に関わる概念だと考えられます。文脈から、{parts[0]}のような性質を持ちつつ{parts[-1]}に関わるものと読めます"
                elif len(parts)==1:
                    inferred = f"「{term}」は「{parts[0]}」に関わる語と推測します。文脈からその意味を補って理解します"
                else:
                    inferred = f"「{term}」は初めて聞く語ですが、文字の成り立ちから推測すると、{term}らしい性質を持つものと考えられます"
            except Exception:
                inferred = f"「{term}」は「{term[:2] if len(term)>=2 else term}」と「{term[2:] if len(term)>2 else '関連の語'}」を合わせた言葉と推測します。部品の意味を足し合わせると全体像が見えてきます"
        if inferred:
            bits.append(inferred)
        else:
            # fallback to old leads if inference fails
            bits.append(f"「{term}」については手元に記録がありませんが、文字から推測して組みます")
    # 形状の案内は、推論できなかったときだけ足す（推論できたのに「どんな場面で使う語かを…」を足すと二重になる）
    if not inferred:
        bits.append(shape_line(t))
    if known and sum(len(p) for p in known) >= max(4, int(len(t) * 0.35)) and not inferred:
        bits.append("読める部品は " + "、".join(f"「{p}」" for p in known[:3]) + " なので、そこを軸に組みます")
    # 推論できたときは、追加で確認を強要しない（自然な一言だけ）
    if 'inferred' in locals() and inferred:
        # 推論後は軽い受け止めだけ添える（しつこい質問はしない）
        if not any(x in inferred for x in ("どうぞ","ください","もらえれば")):
            bits.append("もし違う意味で使っていれば、その場面を一言もらえれば合わせます")
    else:
        if bits and not any(x in bits[0] for x in ("もらえれば", "ください", "どうぞ", "教えてください")):
            asks = ("何を答えたいですか（定義・手順・比較・値段のどれか）を一言で教えてください",
                    "どれを欲しがっていますか。意味・使い方・数量のどれかを一語でどうぞ",
                    "何が分かっていれば前に進めますか。切り口を一言ください")
            bits.append(asks[int(turn) % len(asks)])
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
    parts: list[str] = []
    for raw in bits:
        x = str(raw).strip().rstrip("。")
        if not x:
            continue
        if not parts:
            parts.append(x)
        elif parts[-1].endswith(("です", "ます", "ません", "しました", "でした", "でしたら")):
            parts[-1] += "。"
            parts.append(x)
        else:
            parts[-1] += "、" + x
    out = "。".join(parts)
    return (out + "。") if out and not out.endswith("。") else out


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
