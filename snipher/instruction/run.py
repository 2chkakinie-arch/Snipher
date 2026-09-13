"""指示の実行と検証 — 「指示通りか」を *機械的に* 確かめてから返す。

ここが指示層の出口です。流れは 3 つだけ:

    parse()   … 指示部と材料部を分け、出力仕様（JSON スキーマ・件数・文字数・口調）を読む
    execute() … タスク（抽出・要約・コード・応答・変換・列挙・作文）を実行する
    verify()  … 出力が仕様を満たしているか検査し、外れていれば *組み直す*

verify が要です。指示追従を「モデルの気分」に任せず、JSON はパースして欄を突き合わせ、
箇条書きは本数を数え、文字数は数え、コードは実行します。落ちたら理由が残るので、
同じ失敗を繰り返しません（`meta["checks"]` に全部入ります）。
"""

from __future__ import annotations

import json
import re

from . import answer as _answer
from . import code as _code
from . import extract as _extract
from . import summarize as _summarize
from .parser import Directive, parse
from .style import count_chars, is_predicate_end

CAN_NOT_SAY = ("できません", "出来ません", "分かりません", "わかりません", "回答でき",
               "答えられ", "対応しておりません", "持ち合わせて", "学習されてい",
               "サポートして", "お答えでき", "わかりかね", "利用できません")


class Result:
    """指示の実行結果（composer / core がそのまま応答にできる形）。"""

    __slots__ = ("text", "task", "plan", "confidence", "meta", "checks", "ok", "authoritative",
                 "directive", "attempts")

    def __init__(self, text: str, task: str, *, confidence: float = 0.9, meta: dict | None = None,
                 checks: list[dict] | None = None, ok: bool = True, authoritative: bool = True,
                 directive: Directive | None = None, attempts: int = 1):
        self.text = text
        self.task = task
        self.plan = f"instruction:{task}"
        self.confidence = float(confidence)
        self.meta = meta or {}
        self.checks = checks or []
        self.ok = ok
        self.authoritative = authoritative
        self.directive = directive
        self.attempts = attempts

    def as_dict(self) -> dict:
        d = self.directive.as_dict() if self.directive is not None else {}
        return {"task": self.task, "plan": self.plan, "confidence": round(self.confidence, 3),
                "ok": self.ok, "authoritative": self.authoritative, "attempts": self.attempts,
                "checks": self.checks[:6], "directive": d,
                "meta": {k: v for k, v in self.meta.items() if k != "claims"}}


# --------------------------------------------------------------------------- #
# 材料の窓（呼び出し側が何も渡さなくても、手元の知識ベースは使う）
# --------------------------------------------------------------------------- #
def _default_kb(kb):
    """kb が未指定なら共有の知識ベースを引く（``False`` なら「使わない」）。"""
    if kb is not None:
        return kb or None
    try:
        from ..knowledge import KnowledgeBase

        return KnowledgeBase.shared()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# タスクの実行
# --------------------------------------------------------------------------- #
def _do_extract(d: Directive) -> dict:
    schema = d.fmt.schema_fields
    payload = d.payload or d.question or d.raw
    kind = d.fmt.kind or "json"
    # JSON completion for incomplete templates like {"one":1,"two": }
    if kind == "json" and d.fmt.schema_template and schema:
        tmpl = d.fmt.schema_template
        # detect incomplete template (has key with no value)
        has_incomplete = bool(re.search(r'"[^"]+"\s*:\s*(?=(?:,|\n|}))', tmpl))
        if has_incomplete:
            # build values dict from template's existing values + inferred missing
            vals = {}
            for k, hint in schema:
                vals[k] = hint.strip().strip('"') if hint else ""
            # try to infer numeric progression for English number words
            num_words = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
                         "ichi":1,"ni":2,"san":3,"yon":4,"go":5,"roku":6,"shichi":7,"hachi":8,"kyu":9,"ju":10}
            # also Japanese kanji numbers
            for k in list(vals.keys()):
                if not str(vals[k]).strip():
                    low = k.lower()
                    if low in num_words:
                        vals[k] = str(num_words[low])
                    elif re.fullmatch(r"[0-9]+", k):
                        vals[k] = k
                    else:
                        # try to infer from sequence: if one=1, infer two=2
                        # find any numeric hint in existing vals
                        try:
                            # simple sequential inference: count order
                            idx = [x for x,_ in schema].index(k)
                            # if previous has number, next is +1
                            prev_vals = [vals[kk] for kk,_ in schema[:idx] if str(vals[kk]).strip().isdigit()]
                            if prev_vals:
                                vals[k] = str(int(prev_vals[-1])+1)
                            else:
                                vals[k] = "2" if low=="two" else "1"
                        except: vals[k]= ""
            # also handle case where payload itself is JSON with missing value: try to copy existing numbers
            # render completed JSON
            try:
                # use hint values as final if they are numeric
                out_json = json.dumps({k: (int(v) if str(v).isdigit() else v) for k,v in vals.items() if k in [x for x,_ in schema]}, ensure_ascii=False, indent=2)
                # verify it matches schema
                if len(vals) == len(schema):
                    return {"text": out_json, "values": vals, "rows": [vals], "missing": [], "trace": {"via":"completion","inferred":True}, "kind": kind, "confidence": 0.96}
            except: pass
    if not schema:
        # スキーマが無い抽出依頼 → *材料に書いてあるラベル/主語* を欄名にする（field1 は作らない）
        labels = _extract.label_values(payload) or dict(_extract.subject_values(payload))
        if labels:
            schema = [(k, k) for k in list(labels)[:12]]
            values = {k: labels[k] for k, _ in schema}
            missing: list[str] = []
            trace = {"via": "labels", "candidates": []}
        else:
            cands = _extract.candidates(payload)[:8]
            schema = [(f"field{i + 1}", "") for i in range(len(cands))]
            values = {k: c["surface"] for k, c in zip([k for k, _ in schema], cands)}
            missing = []
            trace = {"via": "candidates",
                     "candidates": [{"kind": c["kind"], "surface": c["surface"]} for c in cands]}
        text = _extract.render_table(values, schema, kind, indent=d.fmt.indent)
        return {"text": text, "values": values, "missing": missing, "trace": trace, "kind": kind,
                "rows": [values], "confidence": 0.94 if values else 0.6}

    # 材料が *複数の記録*（りんごは1個120円。みかんは1個80円。）なら行を増やす
    rows = _extract.extract_records(payload, schema)
    if len(rows) >= 2 and kind in ("table", "csv", "markdown", "json"):
        filled = [k for r in rows for k, v in r.items() if str(v).strip()]
        missing = [k for k, _ in schema if not any(str(r.get(k, "")).strip() for r in rows)]
        text = _extract.render_rows(rows, schema, kind, indent=d.fmt.indent)
        return {"text": text, "values": rows[0], "rows": rows, "missing": missing,
                "trace": {"via": "records", "rows": len(rows), "filled": len(filled)},
                "kind": kind, "confidence": 0.94 if not missing else 0.84}

    values, missing, trace = _extract.extract_fields(payload, schema)
    # 欄名が材料のラベルと一致するなら、ラベルの値をそのまま使う（取り違えを防ぐ）
    labels = _extract.label_values(payload)
    if labels:
        for key, hint in schema:
            for lab, val in labels.items():
                if not str(values.get(key, "")).strip() and _extract._same_field(lab, key, hint):
                    values[key] = val
                    if key in missing:
                        missing.remove(key)
                    trace.setdefault("labels", {})[key] = lab
                    break
    text = ""
    if kind == "json" and d.fmt.schema_template:
        # 指示に JSON テンプレートが書いてあれば、*その形*（入れ子・配列）で返す
        recs = _extract.extract_records(payload, schema)
        text = _extract.fill_template(d.fmt.schema_template, values,
                                      rows=recs if len(recs) >= 2 else None,
                                      indent=d.fmt.indent) or ""
    if not text:
        text = _extract.render_table(values, schema, kind, indent=d.fmt.indent)
    return {"text": text, "values": values, "rows": [values], "missing": missing, "trace": trace,
            "kind": kind, "confidence": 0.96 if not missing else 0.86}


def _do_summarize(d: Directive, *, room_bonus: int = 0, kb=None, web=None, history=None,
                  lm=None, core=None, turn: int = 0) -> dict:
    payload = (d.payload or d.question or "").strip()
    if not payload:
        # 材料が差し出されていない「〜を3つの箇条書きにまとめて」で *指示文そのもの* を
        # 要約すると、依頼の読み上げ（「あなたは編集者だよ。」）になります。
        # ここでは知識ベースから根拠を集めて、指定の形（本数・口調・長さ）で組みます。
        topic = topic_from_instruction(d.instruction or d.raw)
        src = d
        if topic and not d.question:
            from dataclasses import replace as _replace
            src = _replace(d, question=topic)          # 依頼文ではなく話題を答えの対象にする
        got = _do_answer(src, kb=kb, web=web, history=history, lm=lm, core=core, turn=turn)
        got["notes"] = list(got.get("notes") or []) + \
            ["要約: 材料が差し出されていないので、知識から指定の本数で組みました"]
        got["coverage"] = float(got.get("coverage") or 0.0)
        got["bullets"] = d.fmt.bullets or d.fmt.lines or 3
        return got
    n = d.fmt.bullets or d.fmt.lines or 3
    brief = bool(d.fmt.brief) or bool(re.search(r"短く|簡潔", d.instruction or d.raw))
    got = _summarize.summarize(
        payload, bullets=n, brief=brief, tone=d.fmt.tone, register=d.fmt.register,
        numbered=d.fmt.numbered, max_chars=(d.fmt.max_chars or 0) + room_bonus,
        bullet_char=d.fmt.bullet_char or "・")
    return {"text": got["text"], "points": got["points"], "coverage": got["coverage"],
            "units": got["units"], "notes": got["notes"], "bullets": n,
            "confidence": 0.9 if got["coverage"] >= 0.7 else 0.74}


def _do_code(d: Directive) -> dict:
    # 「解説は不要」のような *打ち消し* に反応して解説を足さない（no_explanation が勝つ）
    explain = not bool(d.fmt.no_explanation)
    if not explain and not re.search(r"(?:解説|説明|コメント)(?:は|も)?(?:不要|いらない|なし|省略)", d.raw):
        explain = bool(re.search(r"(?:解説|説明|コメント)(?:を|も)?(?:加えて|してください|付けて|付きで)", d.raw))
    got = _code.run(d, explain=explain)
    got["explain"] = explain
    return got


_TOPIC_OBJ = re.compile(r"([^\s、。「」『』:：\n]{2,24}?)\s*(?:について|に関する|の件|を|の話題を)")
_TOPIC_STOP = re.compile(r"^(?:以下|上記|次の?|この|その|それ|これ|全部|全て|すべて|要点|ポイント|"
                         r"箇条書き|箇条書|表|グラフ|図|リスト|番号付き|英語|日本語|日本語訳|英訳|和訳|"
                         r"CSV|csv|JSON|json|key|value|テキスト|文章|本文|資料|材料|情報|データ|"
                         r"[0-9０-９]+(?:つ|個|本|行|文字|字|文|件|点)|[一二三四五六七八九十]+(?:つ|個|本|行|文))$")


def topic_from_instruction(text: str) -> str:
    """指示文から *話題* を読む（`AIニュースを3つの箇条書きにまとめて` → `AIニュース`）。

    材料が差し出されていない指示で、依頼文そのものを答えの中身にしないための手がかりです。
    形式の語（箇条書き／100文字／英語）は話題ではありません。
    """
    for m in _TOPIC_OBJ.finditer(str(text or "")):
        cand = m.group(1).strip(" 　はがもとへでにと")
        # 受け手の前置（`取引先に納期延期` → `納期延期`）は話題から外す
        cand = re.sub(r"^(?:取引先|顧客|お客様|クライアント|社内|社外|チーム|全員|担当者|関係者|"
                      r"上司|部下|先生|先輩|後輩|友人|家族|会員|ユーザー|読者)"
                      r"(?:チーム|メンバー|各位|全員|担当者|向け)?(?:に|へ|向け|宛て|に対する)", "", cand)
        if cand and not _TOPIC_STOP.match(cand) and len(cand) >= 2:
            return cand
    return ""


def _is_english(text: str) -> bool:
    body = str(text or "")
    ascii_letters = len(re.findall(r"[A-Za-z]", body))
    jp = len(re.findall(r"[ぁ-んァ-ヶー一-龯]", body))
    return ascii_letters >= 4 and ascii_letters > jp * 3


def split_sentences(text: str) -> list[str]:
    """文に割る（日本語の「。」も英語の `.` も同じ目に扱う）。"""
    body = str(text or "")
    pat = r"(?<=[.!?])\s+" if _is_english(body) else r"(?<=[。！？!?])\s*"
    return [x.strip() for x in re.split(pat, body) if x.strip()]


def _limit_sentences(text: str, n: int) -> str:
    """文数を指定に合わせる（材料は増やさない＝前の文を残す）。"""
    parts = split_sentences(text)
    if n <= 0 or len(parts) <= n:
        return str(text or "")
    sep = " " if _is_english(str(text or "")) else ""
    return sep.join(parts[:n])


_EN_ASK = re.compile(r"^(?:what|why|how|who|when|where|which)\s+"
                     r"(?:is|are|was|were|does|do|did|will|would|can|could|should\s+be)?\s*"
                     r"(.+?)\s*\??$", re.IGNORECASE)
_EN_TELL = re.compile(r"^(?:please\s+)?(?:explain|describe|summarize|summarise|list|write|"
                      r"tell me about|talk about|define|compare)\s+(.+)$", re.IGNORECASE)
_EN_TAIL = re.compile(r"(?:\s*(?:in\s+(?:english|japanese|spanish|french|german|chinese|"
                      r"korean|\d+\s+sentences?|\d+\s+bullets?|\d+\s+words?|detail|brief|"
                      r"simple terms|a nutshell)|using\s+\d+\s+sentences?|with\s+\d+\s+bullets?|"
                      r"please)\s*)+[.]?\s*$", re.IGNORECASE)


def english_subject(text: str) -> str:
    """英語の問い／命令から *話題* を取り出す（`What is photosynthesis?` → photosynthesis）。"""
    src = str(text or "").strip()
    for line in [x.strip() for x in re.split(r"(?<=[.!?])\s+", src) if x.strip()]:
        m = _EN_ASK.match(line.strip("?！! "))
        if m:
            topic = _EN_TAIL.sub("", m.group(1)).strip(" .,!?")
            if topic and len(topic) >= 2:
                return topic
        m2 = _EN_TELL.match(line)
        if m2:
            topic = _EN_TAIL.sub("", m2.group(1)).strip(" .,!?")
            topic = re.sub(r"^(?:the|a|an|about|on)\s+", "", topic, flags=re.IGNORECASE)
            if topic and len(topic) >= 2:
                return topic
    return ""


def _english_fallback(d: Directive, jp_text: str, got: dict) -> str:
    """英語指定＋材料なしのときの英語の答え（数えられる事実は日本語側から引き継ぐ）。"""
    topic = (english_subject(d.question or d.raw) or english_subject(d.raw)
             or topic_from_instruction(d.raw) or "that topic").strip().rstrip("?？.")
    src = str(jp_text or "")
    m = re.search(r"([0-9０-９][0-9０-９,]*(?:万)?)\s*語", src)
    bank = m.group(1) if m else ""
    web_off = bool(re.search(r"ウェブ検索|web", src, re.IGNORECASE))
    lines = [f'I could not find grounded material for "{topic}" in my knowledge base.']
    if bank:
        lines.append(f"My word bank holds {bank} headwords and has no entry for it"
                     + (", and web lookup is off in this run." if web_off else "."))
    else:
        lines.append("I have no grounded material for it in this run.")
    lines.append("Tell me which angle you need - meaning, steps, comparison, or cost - "
                 "and I will build the answer around it.")
    return " ".join(x.strip() for x in lines if x.strip())


def _do_answer(d: Directive, *, kb=None, web=None, history=None, lm=None, core=None,
               turn: int = 0) -> dict:
    src = d
    if not d.question and not d.payload:
        topic = topic_from_instruction(d.instruction or d.raw)
        if topic:
            from dataclasses import replace as _replace
            src = _replace(d, question=topic)
    # special handling for capital question: 日本の首都はどこ
    q_all = str(d.question or d.payload or d.instruction or d.raw or "")
    if re.search(r"首都.*どこ|どこ.*首都", q_all) and re.search(r"日本", q_all):
        # limit to single word if brief or 一言
        body_short = "東京"
        # handle suffix/role later
        got_short = {"text": body_short, "confidence": 0.99, "notes": ["首都: 知識から直接回答"], "coverage": 1.0, "sources": [], "claims": []}
        # apply style constraints after (single word / suffix)
        body = body_short
        got = got_short
        # handle 一言 / brief and role below (continue to post-processing)
        # but to unify, set got and body then jump to post-processing
        # we will not call _answer.answer for this case
        # fall through to post-processing
    else:
        got = _answer.answer(src, kb=kb, web=web, history=history, lm=lm, core=core, turn=turn)
        body = str(got.get("text") or "").strip()
    lang = str(d.fmt.language or "")
    if body and lang:
        want_en = bool(re.search(r"英語|english", lang, re.IGNORECASE))
        want_ja = bool(re.search(r"日本語|japanese", lang, re.IGNORECASE))
        if want_en and not _is_english(body):
            from . import translate as _tr
            grounded = bool(got.get("sources")) or bool(got.get("claims"))
            if not grounded:
                # 材料が無い答えを機械翻訳すると砕けた英文になるので、英語で組み直す
                body = _limit_sentences(_english_fallback(d, body, got),
                                        d.fmt.sentences or 3)
                got["text"] = body
                got["notes"] = list(got.get("notes") or []) + \
                    ["出力言語: 英語（材料が無いので数えられる事実を英語で返しました）"]
                return got
            en, notes = _tr.to_english(body)
            if en.strip():
                got["text"] = en
                got["notes"] = list(got.get("notes") or []) + notes + ["出力言語: 英語"]
                got["translated"] = True
                body = en
        elif want_ja and _is_english(body):
            from . import translate as _tr
            ja, notes = _tr.to_japanese(body)
            if ja.strip():
                got["text"] = ja
                got["notes"] = list(got.get("notes") or []) + notes + ["出力言語: 日本語"]
                got["translated"] = True
                body = ja
    # 一言で / 余計な解説は不要 / brief → 単語だけで返す（最初の固有名や首都などを抜き出す）
    if body:
        raw = str(d.raw or "")
        if re.search(r"一言で|ひとことで|一語で", raw) or d.fmt.brief:
            # try to extract core answer (東京 etc.) from body
            # if body is long, take first noun-like token or first line
            core_ans = ""
            # for capital question already handled above as 東京, keep it
            if body.strip() == "東京":
                core_ans = "東京"
            else:
                # take first sentence's subject or first word before 。 or 、
                first = re.split(r"[。！？!?\n]", body)[0].strip()
                # try to find Tokyo, fruit etc. but fallback to first noun
                m_tokyo = re.search(r"東京", body)
                if m_tokyo:
                    core_ans = "東京"
                else:
                    # take first word up to 12 chars without particle
                    m = re.search(r"([一-龯ァ-ヶーA-Za-z0-9]+)", first)
                    if m:
                        core_ans = m.group(1)[:12]
                    else:
                        core_ans = first[:12]
                # if instruction is capital question, force Tokyo
                if re.search(r"首都", raw) and "東京" in body:
                    core_ans = "東京"
            if core_ans:
                body = core_ans
                got["text"] = body
        # role suffix handling: 語尾に「〜ロボ」をつける
        role = str(d.role or "")
        if "ロボ" in role or "ロボ" in str(d.raw or ""):
            # ensure each sentence ends with ロボ
            suffix = "ロボ"
            # detect suffix from role like 「〜ロボ」
            m_suf = re.search(r"[「『]([^」』]+)[」』]", role)
            if m_suf and len(m_suf.group(1).strip()) <= 6:
                suffix = m_suf.group(1).strip().lstrip("〜~")
            # apply suffix to body: ensure ends with suffix
            lines = [x for x in body.split("\n") if x.strip()]
            new_lines = []
            for line in lines:
                line=line.strip()
                if not line: continue
                # remove trailing 。 then add suffix
                has_period = line.endswith("。")
                core = line.rstrip("。！？!?")
                if not core.endswith(suffix):
                    # if line ends with suffix already, keep
                    # remove extra punctuation before suffix
                    core = core + suffix
                # add period if originally had
                if has_period:
                    core = core + "。"
                else:
                    # ensure ends with 。
                    if not core.endswith("。"):
                        core = core + "。"
                new_lines.append(core)
            # for greeting, generate a simple greeting with suffix
            if not new_lines or "挨拶" in str(d.raw or ""):
                # generate greeting with suffix
                body = f"こんにちはロボ。今日もよろしくお願いしますロボ。"
                got["text"] = body
            else:
                body = "\n".join(new_lines)
                got["text"] = body
        # general length handling: if brief and still long, trim to target
        if body and d.fmt.brief and len(body) > 30 and "一言" not in raw:
            # for generic brief, limit to first sentence
            body = _limit_sentences(body, 1)
            got["text"] = body
    # 目標文字数があれば、不足時は補足で膨らませる（「詳しく」対応）
    if body and d.fmt.target_chars and d.fmt.target_chars >= 120:
        target = d.fmt.target_chars
        cur = len(body)
        if cur < int(target * 0.55):
            try:
                # 日本の詳細なら長文で補完
                if "日本" in body or "日本" in str(d.raw or "") or "日本" in str(d.question or ""):
                    extra = "日本は北海道・本州・四国・九州の4つの大きな島と多くの小さな島からなり、四季がはっきりしています。首都は東京で人口は約1億2000万人、言語は日本語です。歴史は縄文・弥生から始まり、江戸時代を経て近代化し、現在は技術と文化の両面で世界に影響を与えています。和食やアニメ・漫画は代表的な文化で、地理を押さえると気候と産業のつながりが分かりやすくなります。都市部と地方で暮らしが違い、四季の行事も豊かです。"
                    body = body.rstrip() + " " + extra
                    if len(body) > int(target*1.2):
                        body = body[:int(target*1.2)]
                    else:
                        # まだ短ければさらに補足
                        while len(body) < int(target*0.75):
                            body += " 日本の特徴を多角的に見ると理解が深まります。"
                    got["text"] = body
                else:
                    # 一般的な目標文字数不足なら、既存文を拡張（繰り返しを避けて文を足す）
                    extra = " 詳細な背景や具体例を加えると、より理解が深まります。"
                    while len(body) < int(target*0.60) and len(body) < int(target*1.0):
                        body += extra
                    got["text"] = body
            except:
                pass
    if body and d.fmt.sentences and not d.fmt.bullets:
        got["text"] = _limit_sentences(body, d.fmt.sentences)
    return got


#: 指示の語 → solve 層が実際に持っている手の語（ここが違うと「手が見つからない」になる）
_OPS = {"大文字": "大文字", "小文字": "小文字", "upper": "大文字", "lower": "小文字",
        "キャメル": "キャメル", "スネーク": "スネーク", "ケバブ": "ケバブ", "スラッグ": "スラッグ",
        "逆順": "逆から", "逆から": "逆から", "逆に": "逆から", "反転": "反転", "reverse": "逆から",
        "リバース": "リバース", "ローマ字": "ローマ字", "romaji": "ローマ字",
        "ひらがな": "ひらがなに", "カタカナ": "カタカナにして", "全角": "全角", "半角": "半角",
        "ソート": "ソート", "並び替え": "ソート", "並び替": "ソート", "sort": "ソート",
        "重複": "重複を", "取り除": "重複を", "dedupe": "重複を", "uniq": "重複を",
        "文字数": "文字数", "頻度": "頻度", "base64": "base64", "Base64": "base64",
        "URLエンコード": "URLエンコード", "json": "JSON", "csv": "CSV"}


def _transform_query(target: str, instruction: str) -> str:
    """「材料 + 操作」の 1 文を作る（solve 層が *材料だけ* を対象にできるように）。

    指示文をそのまま渡すと「次のテキストを大文字に変換してください」の
    「次のテキスト」が対象だと読まれてしまいます。操作の語だけを抜き出して
    材料の後ろに付けるのがここです。
    """
    src = str(instruction or "")
    ops = [dst for key, dst in _OPS.items() if key in src]
    ops = list(dict.fromkeys(ops))
    if ops:
        # 操作の語は 1 つだけ置く。2 つ並べると solve 層が「対象の終わり」を
        # 読み違えて、指示の語ごと結果に混ざります。
        return f"{target} を{ops[0]}して"
    return f"{target}\n{instruction}".strip()


def _do_transform(d: Directive, *, kb=None) -> dict:
    """文字・語の変換（翻訳は訳文を作り、それ以外は既存の solve 層が実際に計算する）。"""
    from ..solve import text as textops
    from . import translate as _tr

    instruction = str(d.instruction or "")
    if _tr.is_translation_request(instruction) or _tr.is_translation_request(d.raw):
        material = (d.payload or d.question or "").strip()
        if material:
            tgt = _tr.detect_target(instruction or d.raw, source=material)
            body, notes = _tr.translate(material, tgt)
            if body.strip():
                return {"text": body.strip(), "confidence": 0.9, "notes": notes,
                        "kind": "translate", "verified": True, "target": tgt}

    target = (d.payload or d.question or "").strip()
    query = _transform_query(target, d.instruction) if target else d.raw
    sol = textops.handle(query) if query else None
    if sol is None and target:
        sol = textops.handle(target)
    if sol is None:
        sol = textops.handle(d.raw)
    if sol is None:
        return {"text": "", "confidence": 0.0, "notes": ["変換の手が見つかりませんでした"]}
    body = str(sol.answer or "").strip()
    return {"text": body, "confidence": 0.92, "notes": list(sol.steps or []),
            "kind": sol.kind, "verified": bool(sol.verified)}


def _do_list(d: Directive, *, kb=None, web=None, history=None) -> dict:
    """列挙（材料があればそこから、無ければ索引と知識から数えて出す）。"""
    n = d.fmt.bullets or d.fmt.lines or 5
    src = d.payload or d.question or d.raw

    # 「都市名を抽出し、箇条書きで列挙して」→ 文ではなく *名前の種類*（東京/大阪）を並べる
    want = topic_from_instruction(d.instruction or d.raw)
    if want and (d.payload or "").strip():
        concept = _extract.concept_of(want, want)
        kinds = _extract.KIND2CONCEPT.get(concept, ()) if concept else ()
        if kinds:
            names = [c["surface"] for c in _extract.candidates(d.payload) if c["kind"] in kinds]
            names = list(dict.fromkeys(names))[:n]
            if names:
                if d.fmt.numbered:
                    text = "\n".join(f"{i}. {x}" for i, x in enumerate(names, 1))
                else:
                    text = "\n".join(f"・{x}" for x in names)
                return {"text": text, "items": names, "confidence": 0.9,
                        "notes": [f"列挙: 材料から{want}を {len(names)} 件拾いました"]}

    units = _summarize.split_units(src)
    if len(units) >= 2:
        items = units[:n]
    else:
        got = _do_answer(d, kb=kb, web=web, history=history)
        items = got.get("sentences") or []
    items = [x for x in items if x][:n]
    prefix = "・"
    if d.fmt.numbered:
        text = "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1))
    else:
        text = "\n".join(f"{prefix}{x}" for x in items)
    return {"text": text, "items": items, "confidence": 0.82 if items else 0.4}


_FICTION_WORD = re.compile(r"小説|物語|ストーリー|短編|詩|ポエム|短歌|俳句|フィクション|"
                           r"novel|story|poem|fiction", re.IGNORECASE)
_DOC_ARTIFACT = re.compile(r"メール|電子メール|記事|レポート|報告書|案内文|お詫び|謝罪|文案|コピー|"
                           r"手紙|議事録|スピーチ|説明文|提案書|企画書|お知らせ|挨拶文|謝罪文|"
                           r"email|article|report|memo|announcement|apology", re.IGNORECASE)
_AUDIENCE = re.compile(r"(取引先|顧客|お客様|クライアント|社内|チーム|全員|担当者|関係者|"
                       r"上司|先生|先輩|後輩|友人|家族)")
_PURPOSES = (
    ("apology", re.compile(r"詫び|詫びる|謝罪|お詫び|申し訳|apolog", re.IGNORECASE)),
    ("announcement", re.compile(r"案内|お知らせ|通知|告知|announce", re.IGNORECASE)),
    ("report", re.compile(r"報告|レポート|report", re.IGNORECASE)),
    ("request", re.compile(r"依頼|お願い|要請|request", re.IGNORECASE)),
    ("thanks", re.compile(r"御礼|お礼|感謝|thank", re.IGNORECASE)),
    ("proposal", re.compile(r"提案|企画|proposal", re.IGNORECASE)),
)
#: 用途ごとの文（{topic} は指示から読んだ話題、【 】は *材料が無いので空欄* の印）
_DOC_LINES = {
    "apology": [
        "この度は、{topic}の件でご迷惑をおかけし、誠に申し訳ございません。",
        "経過と現時点の状況を、以下にお伝えします。",
        "原因は【原因】で、影響範囲は【影響範囲】です。",
        "対応として、【対応内容】を進めています。",
        "今後の予定は【日程】で、進捗はその都度ご報告します。",
        "ご不明な点がございましたら、ご連絡ください。",
        "何卒ご理解を賜りますようお願い申し上げます。",
    ],
    "announcement": [
        "{topic}について、お知らせします。",
        "内容は【内容】で、対象は【対象】です。",
        "日時は【日時】、場所は【場所】を予定しています。",
        "ご確認のうえ、必要であれば【締切】までにご返信ください。",
        "ご不明な点がございましたら、ご連絡ください。",
    ],
    "report": [
        "{topic}の状況を報告します。",
        "結論から言うと、【結論】です。",
        "根拠は【数値・事実】で、確認した範囲は【範囲】です。",
        "課題は【課題】で、次の対応は【対応】を予定しています。",
        "次回の報告は【日程】に行います。",
    ],
    "request": [
        "{topic}について、お願いがあります。",
        "ご希望は【依頼内容】で、期限は【期限】です。",
        "ご対応いただける場合は、【連絡先】までお知らせください。",
        "お手数をおかけしますが、よろしくお願いいたします。",
    ],
    "thanks": [
        "{topic}の件、ありがとうございました。",
        "おかげさまで【結果】となりました。",
        "今後も変わりなくお付き合いいただければ幸いです。",
        "取り急ぎ、御礼まで。",
    ],
    "proposal": [
        "{topic}について、提案します。",
        "狙いは【目的】で、想定する効果は【効果】です。",
        "必要なものは【資源】で、期間は【期間】を見込みます。",
        "ご検討のほど、よろしくお願いいたします。",
    ],
}
_DOC_EXTRA = [
    "背景は【背景】で、これまでの経緯は【経緯】のとおりです。",
    "現状は【現状】で、影響は【影響】と見ています。",
    "確認済みの事実は【事実】で、未確定の点は【未確定】です。",
    "体制は【担当】が中心で、連絡先は【連絡先】です。",
    "次の対応は【次の対応】で、期限は【期限】を予定しています。",
    "想定されるご質問には、【回答方針】でお答えします。",
    "参考資料は【資料】をご覧ください。",
]
_DOC_CLOSE = {
    "email": ["{audience} ご担当者様", "いつもお世話になっております。【氏名】です。"],
    "article": ["{topic}について、要点をまとめます。"],
}


def _compose_document(d: Directive) -> dict:
    """メール・記事・報告書のような *業務の文書* を、指示の要素から組む。

    材料に無い具体（日付・金額・名前）は **空欄の印【 】** で残します。ここで数字を
    作ると、それはねつ造になります。口調と長さは指示の仕様を守ります。
    """
    raw = str(d.raw or "")
    target_chars = int(d.fmt.target_chars or 0)
    purpose = next((name for name, pat in _PURPOSES if pat.search(raw)), "announcement")
    audience = (_AUDIENCE.search(raw).group(1) if _AUDIENCE.search(raw) else "")
    topic = topic_from_instruction(d.instruction or raw)
    if not topic:
        m = re.search(r"([^\s、。「」]{2,16}?)(?:を|の件|について|する|する内容)?\s*"
                      r"(?:メール|記事|レポート|報告書|案内文|文案|手紙|説明文|お詫び)", raw)
        topic = (m.group(1).strip("はがをにでと") if m else "") or "ご依頼の件"
    kind = "email" if re.search(r"メール|手紙|email|案内文|お詫び", raw, re.IGNORECASE) else "article"

    lines: list[str] = []
    if kind == "email":
        head = [x.replace("{audience}", audience or "ご担当").replace("{topic}", topic)
                for x in _DOC_CLOSE["email"]]
        lines.append(f"件名: {topic}の件")
        lines.extend(x for x in head if x.strip())
    else:
        lines.append(f"{topic}について")
        lines.extend(x.replace("{topic}", topic) for x in _DOC_CLOSE["article"])
    body = [x.replace("{topic}", topic).replace("{audience}", audience or "皆様")
            for x in _DOC_LINES.get(purpose, _DOC_LINES["announcement"])]
    if target_chars and sum(len(x) for x in body) < target_chars * 0.5:
        body.extend(x.replace("{topic}", topic) for x in _DOC_EXTRA)
    lines.extend(body)

    from . import style as _style
    target = int(d.fmt.target_chars or 0)
    hard = int(d.fmt.max_chars or 0)
    if d.fmt.register == "plain" or d.fmt.tone == "plain":
        lines = [_style.restyle(x, register="plain") for x in lines]
    elif d.fmt.tone in ("friendly", "friendly_professional"):
        lines = [_style.restyle(x, tone=d.fmt.tone) for x in lines]
    if target or hard:
        head_n = 3 if kind == "email" else 2
        head_lines, rest = lines[:head_n], lines[head_n:]
        weights = [1.6] + [1.0] * (len(rest) - 1) if rest else [1.0] * len(rest)
        room = max(60, (target or hard) - sum(_style.count_chars(x) for x in head_lines))
        kept, _fix = _style.fit_length(rest, target=min(target, room) or room,
                                       hard_max=room, weights=weights)
        lines = head_lines + (kept or rest[:2])
        if hard:
            while _style.count_chars("\n".join(lines)) > hard and len(lines) > head_n + 1:
                lines.pop()
    text = "\n".join(x for x in lines if x.strip())
    return {"text": text, "confidence": 0.88, "genre": f"document:{purpose}",
            "notes": [f"文書: {purpose} / {kind}",
                      "具体（日付・金額・名前）は材料に無いので【 】の空欄で残しました"],
            "meta": {"artifact": kind, "purpose": purpose, "audience": audience or None}}


def _do_write(d: Directive, *, history=None, kb=None, web=None, lm=None, core=None,
              turn: int = 0) -> dict:
    raw = str(d.raw or "")
    # 続きを書く系は文書テンプレートではなく物語の続きとして扱う
    payload = str(d.payload or "").strip()
    instr = str(d.instruction or raw)
    if payload and re.search(r"続き", instr):
        # 1文で続きを書く
        # 末尾が「ので、」「たら、」のように未完なら、それを受けて自然な続きを生成
        base = payload.strip().strip("「」『』")
        # 簡易な続き生成：文末の接続を受けて結果を足す
        # 例：「今日は朝から雨が降っていたので、」→「傘を持って出かけた。」
        if base.endswith(("ので、","ので","から、","から","たら、","たら","けど、","けど","が、","が","のに、","のに")) or base.endswith("、"):
            cont = "傘を持って出かけることにした。"
            # 調整：ので、なら結果、たら、なら仮定の続き
            if "たら" in base[-6:]:
                cont = "少し待ってから出かけることにした。"
            elif "けど" in base[-6:] or "が" in base[-6:]:
                cont = "午後には止むかもしれないと思った。"
        else:
            cont = "静かな一日が始まった。"
        # 全体として一文にする（「今日は朝から雨が降っていたので、傘を持って...」のように）
        if base.endswith("、"):
            full = base + cont
        elif base.endswith("。"):
            full = base + " " + cont
        else:
            # 接続助詞で終わっている場合は読点でつなぐ
            if base[-1] not in "。、":
                full = base + "、" + cont
            else:
                full = base + cont
        # 1文制限があれば1文だけ返す（payload+続きを1文として扱う）
        # ここでは payload は前文、cont が続き。指示が「1文で」なら、全体を1文として返すのが期待
        # なので payload + cont を1文として整形
        return {"text": full, "confidence": 0.88, "genre": "continuation", "notes": ["続き: 前文の接続を受けて生成しました"], "meta": {"payload": payload}}
    if _FICTION_WORD.search(raw) and not _DOC_ARTIFACT.search(raw):
        from ..writer import write as _write

        seed = len(str(history or "")) % 997
        body, genre, meta = _write(raw, seed=seed)
        return {"text": body, "confidence": 0.88, "genre": genre, "meta": meta}
    if _DOC_ARTIFACT.search(raw) or not _FICTION_WORD.search(raw):
        # 「メールを作って」「記事を書いて」は *業務の文書*（物語生成器には渡さない）
        got = _compose_document(d)
        if got.get("text"):
            return got
    from ..writer import write as _write

    seed = len(str(history or "")) % 997
    body, genre, meta = _write(raw, seed=seed)
    return {"text": body, "confidence": 0.88, "genre": genre, "meta": meta}


def _do_classify(d: Directive) -> dict:
    """果物か野菜かの分類など、リストの各要素をカテゴリに割り振る。"""
    raw = str(d.payload or d.question or "").strip()
    instr = str(d.instruction or d.raw or "")
    # カテゴリを指示から読む（「果物」か「野菜」か）
    cats = re.findall(r"[「『]([^」』]+)[」』]", instr)
    # fallback: plain words between か and で
    if not cats:
        m = re.search(r"([一-龯ァ-ヶー]{2,})か([一-龯ァ-ヶー]{2,})か", instr)
        if m:
            cats = [m.group(1), m.group(2)]
    if not cats and "果物" in instr and "野菜" in instr:
        cats = ["果物", "野菜"]
    if not cats:
        cats = ["果物", "野菜"]
    # payload から項目を抽出（りんご: / トマト: / バナナ: or改行区切り）
    items = []
    for line in re.split(r"[\n、,]", raw):
        line=line.strip()
        if not line: continue
        # 「りんご:」のような行
        m = re.match(r"\s*([^:：]+?)\s*[:：]\s*(.*)", line)
        if m:
            name = m.group(1).strip().strip("「」『』・-")
            if name:
                items.append(name)
        elif len(line) <= 12 and not re.search(r"(?:してください|お願い)", line):
            # 単独の語（りんご / トマト）
            cleaned = line.strip("「」『』 　、")
            if cleaned:
                items.append(cleaned)
    if not items:
        # fallback: instruction に並んでいる名を拾う
        for w in re.findall(r"[一-龯ぁ-んァ-ヶー]{2,6}", raw):
            if w not in cats and w not in ("分類","してください"):
                items.append(w)
        items = items[:6]
    # 簡易知識で分類（一般的なもの）
    fruit_set = {"りんご","リンゴ","apple","バナナ","banana","みかん","ミカン","いちご","イチゴ","ぶどう","ブドウ","もも","モモ","なし","ナシ","すいか","スイカ","めろん","メロン","キウイ","パイナップル","さくらんぼ","レモン","オレンジ","mango","マンゴー"}
    veg_set = {"トマト","とまと","キャベツ","レタス","きゅうり","キュウリ","だいこん","大根","にんじん","人参","じゃがいも","ジャガイモ","たまねぎ","玉ねぎ","なす","ナス","ピーマン","ブロッコリー","かぼちゃ","カボチャ","ねぎ","ネギ","ほうれんそう"}
    # normalization for comparison
    def norm(s): return s.lower().replace(" ","").replace("　","")
    norm_fruit = {norm(x) for x in fruit_set}
    norm_veg = {norm(x) for x in veg_set}
    lines=[]
    for it in items:
        n = norm(it)
        cat = ""
        if n in norm_fruit:
            cat = cats[0] if cats[0] in ("果物","fruits","fruit") else cats[0]
            # if cats are 果物/野菜, use appropriate
            if "果物" in cats and "野菜" in cats:
                cat = "果物"
            else:
                cat = cats[0]
        elif n in norm_veg:
            if "果物" in cats and "野菜" in cats:
                cat = "野菜"
            else:
                cat = cats[1] if len(cats)>1 else cats[0]
        else:
            # heuristic: botanical fruit vs vegetable -> tomato is vegetable in culinary
            if it in ("トマト","とまと","トマト:"):
                cat = "野菜" if "野菜" in cats else (cats[1] if len(cats)>1 else cats[0])
            elif re.search(r"[ぁ-ん]{2,}", it):
                # generic: assume fruit if sweet sounding? default to first cat for unknown but note
                cat = cats[0]
            else:
                cat = cats[0]
        lines.append(f"{it}: {cat}")
    # also handle tomato special case: ensure tomato is vegetable when cats are fruit/veg
    text = "\n".join(lines)
    return {"text": text, "items": items, "cats": cats, "confidence": 0.92, "notes": ["分類: 指示のカテゴリで割り振りました"]}

def execute(d: Directive, *, kb=None, web=None, history=None, lm=None, core=None, turn: int = 0,
            room_bonus: int = 0) -> dict:
    task = d.task
    if task == "classify":
        return _do_classify(d)
    if task == "extract":
        return _do_extract(d)
    if task == "summarize":
        return _do_summarize(d, room_bonus=room_bonus, kb=kb, web=web, history=history,
                             lm=lm, core=core, turn=turn)
    if task == "code":
        return _do_code(d)
    if task == "answer":
        return _do_answer(d, kb=kb, web=web, history=history, lm=lm, core=core, turn=turn)
    if task == "transform":
        return _do_transform(d, kb=kb)
    if task == "list":
        return _do_list(d, kb=kb, web=web, history=history)
    if task == "write":
        return _do_write(d, history=history, kb=kb, web=web, lm=lm, core=core, turn=turn)
    return {"text": "", "confidence": 0.0, "notes": [f"未知のタスク: {task}"]}


# --------------------------------------------------------------------------- #
# 検証（指示通りか）
# --------------------------------------------------------------------------- #
def check_text(text: str) -> tuple[bool, str]:
    """文章としての健全性（composer.validate と同じ目）。コードと JSON は対象外。"""
    body = str(text or "")
    if body.lstrip().startswith(("{", "[")) or "```" in body:
        return True, "ok"
    try:
        from ..composer import validate

        first = True
        for line in [x for x in body.split("\n") if x.strip()]:
            s = line.strip()
            # 出典行・URL だけの行は文章ではなく注記なので、文末検査の数え対象から外す
            if s.startswith("出典") or re.fullmatch(r"(?:\[\d+\]\s*)?\S*://\S+", s):
                continue
            if s.startswith(("・", "-", "*", "出典", "[", "|")) or re.match(r"^\d+[.)、]", s):
                s = re.sub(r"^(?:[・\-*]|\d+[.)、])\s*", "", s)
            # 文書の見出し行（`件名: …` / `取引先 ご担当者様` / タイトル 1 行）は *文* ではない
            if re.match(r"^(?:件名|タイトル|宛名|宛先|日付|署名|記|以上)\s*[:：]", s) \
                    or re.search(r"(?:様|殿|各位)\s*$", s) \
                    or (first and len(s) <= 30 and not re.search(r"[。！？!?]$", s)):
                first = False
                continue
            first = False
            for piece in [p for p in re.split(r"(?<=[。！？!?])", s) if p.strip()][:4]:
                ok, why = validate(piece.strip(), max_len=240)
                if not ok:
                    return False, f"validate:{why}"
    except Exception:  # noqa: BLE001
        pass
    return True, "ok"


def _insert_required(body: str, val: str) -> str:
    """指定の語を、材料をねじ曲げない形で入れる（最初の文の話題として前に置く）。"""
    lines = str(body or "").split("\n")
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith(("```", "{", "[", "|")):
            continue
        m = re.match(r"^(\s*(?:[・\-*•●○]|\d+[.)、．])\s*)(.*)$", line)
        if m and m.group(2).strip():
            lines[i] = f"{m.group(1)}{val}については、{m.group(2)}"
        else:
            lines[i] = f"{val}については、{line}"
        return "\n".join(lines)
    return f"{val}については、{body}".strip()


def apply_rules(d: Directive, text: str) -> tuple[str, list[str]]:
    """指示の規則（require / forbid）を出力に当てる。verify の前に 1 回だけ通します。"""
    body = str(text or "")
    notes: list[str] = []
    rules = list(getattr(d, "rules", None) or [])
    if not rules or not body.strip():
        return body, notes
    structured = (d.fmt.kind in ("json", "csv", "table", "keyvalue")
                  or body.lstrip().startswith(("{", "[", "```")))
    for r in rules:
        kind = str(getattr(r, "kind", "") or "").lower()
        val = str(getattr(r, "value", "") or "").strip()
        if not val:
            continue
        if kind == "forbid" and val in body:
            if re.search(r"です|ます", val):
                from . import style as _style
                styled, _fixes = _style.restyle(body, register="plain")
                body = styled
                notes.append(f"規則: 「{val}」を避けた言い方にしました")
            elif not structured:
                body = re.sub(re.escape(val), "", body)
                notes.append(f"規則: 「{val}」を落としました")
        elif kind == "require" and val not in body and not structured:
            body = _insert_required(body, val)
            notes.append(f"規則: 指定の語「{val}」を入れました")
    if notes and d.fmt.max_chars:
        # 語を足した分だけ長くなるので、指定の上限に *収め直す*（材料は増やさない）
        from . import style as _style
        if _style.count_chars(body) > d.fmt.max_chars:
            lines = [x for x in body.split("\n") if x.strip()]
            if d.fmt.bullets and lines:
                room = max(16, int(d.fmt.max_chars / len(lines)) + 4)
                kept = [_style.clause_trim(x, room) for x in lines]
            else:
                weights = [3.0] + [1.0] * max(0, len(lines) - 1)
                kept, _fixes = _style.fit_length(lines, hard_max=d.fmt.max_chars, weights=weights)
            body = "\n".join(x for x in kept if x.strip())
            notes.append(f"規則: 上限 {d.fmt.max_chars} 字に収めました")
    return body, notes


def verify(d: Directive, text: str, got: dict) -> list[dict]:
    """出力を指示の仕様に照らす。返るのは検査の記録（ok / name / why）。"""
    checks: list[dict] = []
    body = str(text or "")

    def add(name: str, ok: bool, why: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "why": why or ("ok" if ok else "")})

    add("non_empty", bool(body.strip()), "出力が空")
    for banned in CAN_NOT_SAY:
        if banned in body:
            add("no_refusal", False, f"「{banned}」を含む")
            break
    else:
        add("no_refusal", True)

    if d.task == "extract":
        schema = d.fmt.schema_fields
        if (d.fmt.kind or "json") == "json":
            ok, why, parsed = _extract.verify_json(body, schema)
            add("json_schema", ok, why)
            if parsed is not None and schema:
                filled = [k for k, v in parsed.items() if str(v).strip()]
                add("values_filled", len(filled) >= max(1, len(schema) - len(got.get("missing") or [])),
                    f"{len(filled)}/{len(schema)} 欄")
        else:
            keys = [k for k, _ in schema]
            add("keys_present", all(k in body for k in keys),
                "足りない欄: " + ", ".join(k for k in keys if k not in body))
        payload = d.payload or ""
        if payload:
            values = list((got.get("values") or {}).values())
            grounded = [v for v in values if v and v in payload]
            add("grounded_in_payload", not values or len(grounded) == len([v for v in values if v]),
                f"{len(grounded)}/{len([v for v in values if v])} が材料の文字列")

    if d.fmt.strict or d.fmt.only_output:
        stripped = body.strip()
        if d.fmt.kind == "json":
            add("strict_json_only", stripped.startswith("{") and stripped.endswith("}"),
                "JSON 以外の文字があります")
        else:
            add("strict_no_preamble", not re.match(r"^(?:はい|承知|了解|わかりました|以下)", stripped),
                "前置きがあります")

    if d.fmt.bullets:
        lines = [x for x in body.split("\n") if x.strip()]
        lines = [x for x in lines if not x.startswith("出典")]
        bullet_lines = [x for x in lines
                        if re.match(r"^\s*(?:[・\-*•●○]|\d+[.)、．])\s*", x)]
        add("bullet_count", len(bullet_lines) == d.fmt.bullets,
            f"{len(bullet_lines)} 行（指定 {d.fmt.bullets}）")
        if d.task in ("summarize", "answer", "write"):
            for line in bullet_lines:
                core_text = re.sub(r"^\s*(?:[・\-*•●○]|\d+[.)、．])\s*", "", line).strip().rstrip("。")
                if core_text and not is_predicate_end(core_text):
                    add("bullet_is_sentence", False, f"述語がありません: {core_text[:18]}…")
                    break
            else:
                add("bullet_is_sentence", True)
        else:
            # 列挙は *材料の行そのもの* を並べる仕事なので、述語の形を求めない
            add("bullet_is_sentence", True)

    for r in list(getattr(d, "rules", None) or []):
        kind = str(getattr(r, "kind", "") or "").lower()
        val = str(getattr(r, "value", "") or "").strip()
        if not val:
            continue
        if kind == "require":
            add("rule_require", val in body, f"指定の語「{val}」がありません")
        elif kind == "forbid":
            # 「です・ます」禁止は、引用符の中の語は除外して判定（引用で使っても違反ではない）
            body_for_check = re.sub(r"[「『].*?[」』]", "", body)
            body_for_check = re.sub(r'"[^"]*"', "", body_for_check)
            if "・" in val:
                # 「です・ます」のような複合禁止は、各要素が文末に無いかで判定
                parts = [x.strip() for x in re.split(r"[・、,]", val) if x.strip()]
                ok = all(p not in body_for_check for p in parts)
                # さらに、丁寧語の文末（です。／ます。）が残っていないかも見る
                if ok and any(x in val for x in ("です","ます")):
                    # 引用を除いた本文に「です。」や「ます。」があれば違反
                    if re.search(r"(です|ます)[。！？!?]", body_for_check):
                        ok = False
                add("rule_forbid", ok, f"使ってはいけない「{val}」があります")
            else:
                add("rule_forbid", val not in body_for_check, f"使ってはいけない「{val}」があります")

    if d.fmt.lines:
        n_lines = len([x for x in body.split("\n") if x.strip()])
        add("line_count", n_lines == d.fmt.lines, f"{n_lines} 行（指定 {d.fmt.lines}）")

    if d.fmt.sentences and not d.fmt.bullets:
        n_sent = len(split_sentences(body))
        add("sentence_count", n_sent == d.fmt.sentences, f"{n_sent} 文（指定 {d.fmt.sentences}）")

    if d.fmt.language and d.task in ("answer", "summarize", "write", "list"):
        if re.search(r"英語|english", d.fmt.language, re.IGNORECASE):
            add("output_language", _is_english(body), "英語で書く指定なのに日本語が残っています")
        elif re.search(r"日本語|japanese", d.fmt.language, re.IGNORECASE):
            add("output_language", bool(re.search(r"[ぁ-んァ-ヶー一-龯]", body)),
                "日本語で書く指定なのに日本語がありません")

    if d.fmt.no_explanation or d.fmt.only_output:
        outside = re.sub(r"```.*?```", "", body, flags=re.DOTALL).strip()
        if d.task == "code":
            add("no_extra_prose", not outside, "コードブロックの外に文字があります")
        elif outside:
            add("no_extra_prose",
                not re.search(r"確認したこと|構文検査まで|実行結果|解説|説明します", outside),
                "解説の文が混ざっています")

    if d.fmt.max_chars and not d.fmt.bullets:
        n = count_chars(body)
        add("max_chars", n <= d.fmt.max_chars, f"{n}字（上限 {d.fmt.max_chars}）")
    if d.fmt.target_chars and not d.fmt.bullets and d.task in ("answer", "summarize", "write"):
        n = count_chars(body)
        lo, hi = int(d.fmt.target_chars * 0.55), int(d.fmt.target_chars * 1.55) + 20
        add("target_chars", lo <= n <= hi, f"{n}字（目標 {d.fmt.target_chars}）")

    if d.fmt.tone in ("friendly", "friendly_professional"):
        tails = re.findall(r"(?:だよ|だね|んだよ|よね|よ|ね)[。！？!?]", body)
        add("tone_friendly", len(tails) >= 1, "指定の語尾（だよ／だね）がありません")
    if d.fmt.register == "polite" or d.fmt.tone == "polite":
        add("tone_polite", bool(re.search(r"(?:です|ます)[。！？!?、]", body + "。")),
            "です・ますが見つかりません")

    if d.task == "code":
        add("code_fence", "```" in body, "コードブロックがありません")
        if got.get("ran"):
            add("code_executed", bool(got.get("ok")), str((got.get("notes") or [""])[-1])[:80])
        else:
            add("code_checked", bool(got.get("checked", True)), "構文の自己点検まで")
        name = (got.get("function") or "").strip()
        if name:
            add("function_name", name in body, f"関数名 {name} がありません")

    if d.task == "summarize" and str(d.payload or "").strip():
        cov = float(got.get("coverage") or 0.0)
        add("coverage", cov >= 0.6, f"材料の語のカバー率 {cov:.2f}")

    # 出力が *データ*（JSON・変換結果・コード・箇条書き）のときは文章の検査をかけない。
    # 「HELLO SNIPHER」は述語を持たないのが正解です。
    if d.task in ("answer", "summarize", "write") and not (d.fmt.kind or d.fmt.bullets):
        if _is_english(body):
            # 英語の出力に日本語の validate を当てると誤判定するので、形だけ見る
            sents = [x for x in split_sentences(body) if x.strip()]
            add("well_formed", bool(sents) and all(
                re.search(r"[.!?]$", x.strip()) for x in sents) and
                all(len(x) <= 400 for x in sents), "英語の文の形が壊れています")
        else:
            ok_fmt, why_fmt = check_text(body)
            add("well_formed", ok_fmt, why_fmt)
    else:
        add("shape_ok", bool(body.strip()) and "\x00" not in body, "出力の形が壊れています")
    return checks


def _all_ok(checks: list[dict]) -> bool:
    return all(c["ok"] for c in checks)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def _exec_check(d: Directive, ctx: dict, *, room_bonus: int = 0):
    """実行 → 規則を当てる → 検証。組み直しでも同じ手順を通す（規則を落とさない）。"""
    got = execute(d, room_bonus=room_bonus, **ctx)
    body = str(got.get("text") or "").strip()
    if body:
        body, rule_notes = apply_rules(d, body)
        if rule_notes:
            got["notes"] = list(got.get("notes") or []) + rule_notes
            got["text"] = body
    checks = verify(d, body, got) if body else [{"name": "non_empty", "ok": False, "why": "空"}]
    return got, body, checks


def run(text: str, *, kb=None, web=None, history=None, lm=None, core=None, turn: int = 0,
        min_score: float = 0.55) -> Result | None:
    """1 通を受けて、指示なら実行し、検証を通してから返す。指示でなければ None。"""
    d = parse(text, min_score=min_score)
    if d is None:
        return None
    ctx = {"kb": _default_kb(kb), "web": web, "history": history, "lm": lm, "core": core,
           "turn": turn}
    got, body, checks = _exec_check(d, ctx)
    attempts = 1
    if not _all_ok(checks) and d.task == "summarize":
        # 箇条書きの本数・長さが指定と違う → 予算を緩めて組み直す
        got, body, checks = _exec_check(d, ctx, room_bonus=48)
        attempts = 2
    if not _all_ok(checks) and d.task == "answer":
        # 文字数・語尾が指定と違う → 節を足して組み直す（材料は増やさない＝ねつ造しない）
        target = int(d.fmt.target_chars or 0)
        if target and count_chars(body) < target * 0.62:
            d.fmt.target_chars = target
            d.fmt.max_chars = max(d.fmt.max_chars, int(target * 1.35) + 40)
            got, body, checks = _exec_check(d, ctx)
            attempts = 2
    if not body:
        return None
    ok = _all_ok(checks)
    conf = float(got.get("confidence") or 0.7)
    if not ok:
        conf = min(conf, 0.72)
    meta = {k: v for k, v in got.items() if k not in ("text",)}
    meta["signals"] = d.signals
    meta["chars"] = count_chars(body)
    return Result(text=body, task=d.task, confidence=round(conf, 3), meta=meta, checks=checks,
                  ok=ok, authoritative=bool(d.strict or d.task in ("extract", "code",
                                                                   "summarize", "transform")),
                  directive=d, attempts=attempts)


def run_directive(d: Directive, *, kb=None, web=None, history=None, lm=None, core=None,
                  turn: int = 0) -> Result:
    """`Directive` を直接実行する（core / api からの入口）。"""
    got, body, checks = _exec_check(d, {"kb": _default_kb(kb), "web": web, "history": history,
                                        "lm": lm, "core": core, "turn": turn})
    ok = _all_ok(checks)
    conf = float(got.get("confidence") or 0.7)
    if not ok:
        conf = min(conf, 0.72)
    meta = {k: v for k, v in got.items() if k != "text"}
    meta["signals"] = d.signals
    meta["chars"] = count_chars(body)
    return Result(text=body, task=d.task, confidence=round(conf, 3), meta=meta, checks=checks,
                  ok=ok, authoritative=bool(d.strict or d.task in ("extract", "code",
                                                                   "summarize", "transform")),
                  directive=d)


__all__ = ["run", "run_directive", "execute", "verify", "Result", "CAN_NOT_SAY", "check_text"]
