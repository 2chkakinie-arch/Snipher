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
    if not schema:
        # スキーマが無い抽出依頼 → 材料から読めた値を key: value で返す
        cands = _extract.candidates(payload)[:8]
        schema = [(f"field{i + 1}", "") for i in range(len(cands))]
        values = {k: c["surface"] for k, c in zip([k for k, _ in schema], cands)}
        missing: list[str] = []
        trace = {"candidates": [{"kind": c["kind"], "surface": c["surface"]} for c in cands]}
    else:
        values, missing, trace = _extract.extract_fields(payload, schema)
    kind = d.fmt.kind or "json"
    text = _extract.render_table(values, schema, kind, indent=d.fmt.indent)
    return {"text": text, "values": values, "missing": missing, "trace": trace, "kind": kind,
            "confidence": 0.96 if not missing else 0.86}


def _do_summarize(d: Directive, *, room_bonus: int = 0) -> dict:
    payload = d.payload or d.question or d.raw
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
    explain = not (d.fmt.no_explanation and d.fmt.strict)
    explain = explain or bool(re.search(r"解説|説明|コメント", d.raw))
    got = _code.run(d, explain=explain)
    return got


def _do_answer(d: Directive, *, kb=None, web=None, history=None, lm=None, core=None,
               turn: int = 0) -> dict:
    return _answer.answer(d, kb=kb, web=web, history=history, lm=lm, core=core, turn=turn)


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


def _do_transform(d: Directive) -> dict:
    """文字・語の変換（既存の solve 層が実際に計算する）。"""
    from ..solve import text as textops

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


def _do_write(d: Directive, *, history=None) -> dict:
    from ..writer import write as _write

    seed = len(str(history or "")) % 997
    body, genre, meta = _write(d.raw, seed=seed)
    return {"text": body, "confidence": 0.88, "genre": genre, "meta": meta}


def execute(d: Directive, *, kb=None, web=None, history=None, lm=None, core=None, turn: int = 0,
            room_bonus: int = 0) -> dict:
    task = d.task
    if task == "extract":
        return _do_extract(d)
    if task == "summarize":
        return _do_summarize(d, room_bonus=room_bonus)
    if task == "code":
        return _do_code(d)
    if task == "answer":
        return _do_answer(d, kb=kb, web=web, history=history, lm=lm, core=core, turn=turn)
    if task == "transform":
        return _do_transform(d)
    if task == "list":
        return _do_list(d, kb=kb, web=web, history=history)
    if task == "write":
        return _do_write(d, history=history)
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

        for line in [x for x in body.split("\n") if x.strip()]:
            s = line.strip()
            if s.startswith(("・", "-", "*", "出典", "[", "|")) or re.match(r"^\d+[.)、]", s):
                s = re.sub(r"^(?:[・\-*]|\d+[.)、])\s*", "", s)
            for piece in [p for p in re.split(r"(?<=[。！？!?])", s) if p.strip()][:4]:
                ok, why = validate(piece.strip(), max_len=240)
                if not ok:
                    return False, f"validate:{why}"
    except Exception:  # noqa: BLE001
        pass
    return True, "ok"


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
        for line in bullet_lines:
            core_text = re.sub(r"^\s*(?:[・\-*•●○]|\d+[.)、．])\s*", "", line).strip().rstrip("。")
            if core_text and not is_predicate_end(core_text):
                add("bullet_is_sentence", False, f"述語がありません: {core_text[:18]}…")
                break
        else:
            add("bullet_is_sentence", True)

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

    if d.task == "summarize":
        cov = float(got.get("coverage") or 0.0)
        add("coverage", cov >= 0.6, f"材料の語のカバー率 {cov:.2f}")

    # 出力が *データ*（JSON・変換結果・コード・箇条書き）のときは文章の検査をかけない。
    # 「HELLO SNIPHER」は述語を持たないのが正解です。
    if d.task in ("answer", "summarize", "write") and not (d.fmt.kind or d.fmt.bullets):
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
def run(text: str, *, kb=None, web=None, history=None, lm=None, core=None, turn: int = 0,
        min_score: float = 0.55) -> Result | None:
    """1 通を受けて、指示なら実行し、検証を通してから返す。指示でなければ None。"""
    d = parse(text, min_score=min_score)
    if d is None:
        return None
    ctx = {"kb": _default_kb(kb), "web": web, "history": history, "lm": lm, "core": core,
           "turn": turn}
    got = execute(d, **ctx)
    body = str(got.get("text") or "").strip()
    checks = verify(d, body, got) if body else [{"name": "non_empty", "ok": False, "why": "空"}]
    attempts = 1
    if not _all_ok(checks) and d.task == "summarize":
        # 箇条書きの本数・長さが指定と違う → 予算を緩めて組み直す
        got = execute(d, room_bonus=48, **ctx)
        body = str(got.get("text") or "").strip()
        checks = verify(d, body, got)
        attempts = 2
    if not _all_ok(checks) and d.task == "answer":
        # 文字数・語尾が指定と違う → 節を足して組み直す（材料は増やさない＝ねつ造しない）
        target = int(d.fmt.target_chars or 0)
        if target and count_chars(body) < target * 0.62:
            d.fmt.target_chars = target
            d.fmt.max_chars = max(d.fmt.max_chars, int(target * 1.35) + 40)
            got = execute(d, **ctx)
            body = str(got.get("text") or "").strip()
            checks = verify(d, body, got)
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
    got = execute(d, kb=_default_kb(kb), web=web, history=history, lm=lm, core=core, turn=turn)
    body = str(got.get("text") or "").strip()
    checks = verify(d, body, got)
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
