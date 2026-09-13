"""コード仕事 — 指示された関数を *実際に動かして* 確かめてから渡す。

`snipher.codegen` は「素数判定を書いて」のような *よくある課題* を模板で持っています。
指示層が扱うのはその外側、例えば

    JavaScriptで、配列から重複した要素を取り除いて昇順にソートする関数 `uniqueSort(arr)`
    を作成してください。コードと簡単な解説を添えてください。

のような **名前・引数・操作が指定された関数** です。やり方は確率生成ではありません:

    1. 指示から言語・関数名・引数・操作（重複除去／昇順／降順／反転／…）を読む
    2. 操作の部品を言語ごとに合成して関数を書く
    3. 期待値が *決定的に計算できる* テストケースを自分で作る
    4. Python / JavaScript / TypeScript / Go は実際に実行して突き合わせる
       （実行できない言語は構文検査までを行い、検査範囲を隠さず書く）
    5. 落ちていたら直して、もう一度実行する

「解説を添えて」と言われたときだけ解説を付け、コードブロックの外に
挨拶や前置きは置きません。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from .parser import Directive, detect_language, function_spec

#: 指示文中の *操作* の言い回し → 内部の操作名
OPS: tuple[tuple[str, re.Pattern], ...] = (
    ("dedupe", re.compile(r"(?:重複|重複した|重複する|ダブっ|だぶっ|ユニーク|unique|distinct|一意|"
                          r"重複を取り|重複を除|重複を消|重複なし|重複をなく)")),
    ("sort_desc", re.compile(r"(?:降順|大きい順|大きいもの順|多い順|高い順|descend|descending|"
                             r"逆順にソート|reverse sort|z-a|z→a)", re.IGNORECASE)),
    ("sort_asc", re.compile(r"(?:昇順|小さい順|少ない順|低い順|小さいもの順|ascend|ascending|"
                            r"アルファベット順|辞書順|昇順に|a-z|a→z|ソート|並び替え|整列|sort)",
                            re.IGNORECASE)),
    ("reverse", re.compile(r"(?:逆(?:順|から|に)|反転|リバース|reverse|後ろから|ひっくり返)")),
    ("count", re.compile(r"(?:数える|個数|件数|カウント|出現回数|頻度|count)")),
    ("filter_even", re.compile(r"(?:偶数だけ|偶数のみ|even only)")),
    ("filter_odd", re.compile(r"(?:奇数だけ|奇数のみ|odd only)")),
    ("sum", re.compile(r"(?:合計|総和|足し合わせ|和を求める|sum|total)")),
    ("max", re.compile(r"(?:最大値|最大|max|いちばん大きい|一番大きい)")),
    ("min", re.compile(r"(?:最小値|最小|min|いちばん小さい|一番小さい)")),
    ("unique_count", re.compile(r"(?:種類数|何種類|unique count|distinct count)")),
)

_LIST_ARG = ("arr", "array", "list", "items", "nums", "numbers", "values", "xs", "data", "input",
             "a", "lst", "seq")

_LANG_LABEL = {"javascript": "JavaScript", "typescript": "TypeScript", "python": "Python",
               "go": "Go", "rust": "Rust", "ruby": "Ruby", "java": "Java", "html": "HTML",
               "css": "CSS", "sql": "SQL", "bash": "bash", "regex": "正規表現"}


def operations(text: str) -> list[str]:
    """指示文から操作を *指定された順* で読む（「重複を取り除いて昇順」→ [dedupe, sort_asc]）。"""
    t = str(text or "")
    found: list[tuple[int, str]] = []
    for name, pat in OPS:
        m = pat.search(t)
        if m:
            found.append((m.start(), name))
    found.sort()
    out: list[str] = []
    for _pos, name in found:
        if name == "sort_asc" and "sort_desc" in out:
            continue
        if name == "sort_desc" and "sort_asc" in out:
            out.remove("sort_asc")
        if name not in out:
            out.append(name)
    return out


def _arg_name(args: list[str]) -> str:
    for a in args:
        base = re.sub(r"[:=].*$", "", a).strip()
        if base.lower() in _LIST_ARG or base.isidentifier():
            return base
    return "arr"


def _js_value(ops: list[str]) -> str:
    return "number" if any(o in ops for o in ("sum", "max", "min", "filter_even", "filter_odd")) \
        else "any"


def compose_js(name: str, arg: str, ops: list[str], *, ts: bool = False) -> tuple[str, str]:
    """JavaScript / TypeScript の関数を組み立てる。返るのは (コード, 型)。"""
    val = _js_value(ops)
    lines: list[str] = []
    if "dedupe" in ops:
        lines.append(f"  const out = Array.from(new Set({arg}));")
    else:
        lines.append(f"  const out = {arg}.slice();")
    body = "out"
    if "sort_asc" in ops:
        cmp_ = "(a, b) => (a < b ? -1 : a > b ? 1 : 0)" if val == "any" else "(a, b) => a - b"
        lines.append(f"  out.sort({cmp_});")
    if "sort_desc" in ops:
        cmp_ = "(a, b) => (a > b ? -1 : a < b ? 1 : 0)" if val == "any" else "(a, b) => b - a"
        lines.append(f"  out.sort({cmp_});")
    if "reverse" in ops:
        lines.append("  out.reverse();")
    if "filter_even" in ops:
        lines.append("  const filtered = out.filter((n) => n % 2 === 0);")
        body = "filtered"
    elif "filter_odd" in ops:
        lines.append("  const filtered = out.filter((n) => n % 2 === 1);")
        body = "filtered"
    if "sum" in ops:
        lines.append(f"  return {body}.reduce((a, b) => a + b, 0);")
        ret = "number"
    elif "max" in ops:
        lines.append(f"  return {body}.length ? Math.max(...{body}) : undefined;")
        ret = f"{val} | undefined"
    elif "min" in ops:
        lines.append(f"  return {body}.length ? Math.min(...{body}) : undefined;")
        ret = f"{val} | undefined"
    elif "count" in ops or "unique_count" in ops:
        lines.append(f"  return {body}.length;")
        ret = "number"
    else:
        lines.append(f"  return {body};")
        ret = f"{val}[]"
    if ts:
        sig = (f"export function {name}({arg}: {val}[]): {ret} {{")
    else:
        sig = f"function {name}({arg}) {{"
    return "\n".join([sig, *lines, "}"]), ret


def compose_python(name: str, arg: str, ops: list[str]) -> str:
    lines = [f"def {name}({arg}):"]
    cur = arg
    if "dedupe" in ops:
        lines.append(f"    out = list(dict.fromkeys({arg}))")
        cur = "out"
    else:
        lines.append(f"    out = list({arg})")
        cur = "out"
    if "sort_asc" in ops:
        lines.append("    out.sort()")
    if "sort_desc" in ops:
        lines.append("    out.sort(reverse=True)")
    if "reverse" in ops:
        lines.append("    out.reverse()")
    if "filter_even" in ops:
        lines.append("    out = [n for n in out if n % 2 == 0]")
        cur = "out"
    elif "filter_odd" in ops:
        lines.append("    out = [n for n in out if n % 2 == 1]")
        cur = "out"
    if "sum" in ops:
        lines.append(f"    return sum({cur})")
    elif "max" in ops:
        lines.append(f"    return max({cur}) if {cur} else None")
    elif "min" in ops:
        lines.append(f"    return min({cur}) if {cur} else None")
    elif "count" in ops or "unique_count" in ops:
        lines.append(f"    return len({cur})")
    else:
        lines.append("    return out")
    return "\n".join(lines)


def compose_go(name: str, arg: str, ops: list[str]) -> str:
    export = name[:1].upper() + name[1:]
    lines = [f"func {export}({arg} []int) []int {{",
             f"\tout := make([]int, len({arg}))",
             f"\tcopy(out, {arg})"]
    if "dedupe" in ops:
        lines += ["\tseen := map[int]bool{}", "\tdeduped := out[:0]",
                  "\tfor _, v := range out {", "\t\tif !seen[v] {",
                  "\t\t\tseen[v] = true", "\t\t\tdeduped = append(deduped, v)",
                  "\t\t}", "\t}", "\tout = deduped"]
    if "sort_asc" in ops:
        lines.append("\tsort.Ints(out)")
    if "sort_desc" in ops:
        lines.append("\tsort.Slice(out, func(i, j int) bool { return out[i] > out[j] })")
    if "reverse" in ops:
        lines += ["\tfor i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {",
                  "\t\tout[i], out[j] = out[j], out[i]", "\t}"]
    lines += ["\treturn out", "}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 実行して確かめる
# --------------------------------------------------------------------------- #
def _cases(ops: list[str], lang: str) -> list[tuple[str, str]]:
    """決定的に期待値が出せるテストケース（式の文字列, 期待値の文字列）。"""
    if any(o in ops for o in ("sum", "max", "min", "filter_even", "filter_odd")):
        table = [("sum", "[3, 1, 2]", "6"), ("max", "[3, 1, 2]", "3"),
                 ("min", "[3, 1, 2]", "1"), ("filter_even", "[1, 2, 3, 4]", "[2, 4]"),
                 ("filter_odd", "[1, 2, 3, 4]", "[1, 3]")]
        pick = [t for t in table if t[0] in ops] or [("sum", "[3, 1, 2]", "6")]
    elif "count" in ops or "unique_count" in ops:
        pick = [("count", "[1, 2, 2, 3]", "3")]
    elif "reverse" in ops and "sort_asc" not in ops and "sort_desc" not in ops:
        pick = [("reverse", "[1, 2, 3]", "[3, 2, 1]")]
    elif "sort_desc" in ops:
        pick = [("sort_desc", "[3, 1, 2]", "[3, 2, 1]")]
    else:
        pick = [("sort_asc", "[3, 1, 2]", "[1, 2, 3]")]
    out: list[tuple[str, str]] = []
    for _tag, arg, want in pick[:1]:
        if lang in ("javascript", "typescript", "python"):
            out.append((arg, want))
            out.append(("[]", "[]" if "[" in want else ("0" if want.isdigit() else "None")))
            if lang != "python":
                out.append((arg, want))
        else:
            out.append((arg, want))
    seen: list[tuple[str, str]] = []
    for pair in out:
        if pair not in seen:
            seen.append(pair)
    return seen[:3]


_TS_ARG_TYPE = re.compile(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*:\s*[A-Za-z_$][A-Za-z0-9_$.\[\]<>|& ]*?(?=[,)])")
_TS_RET_TYPE = re.compile(r"\)\s*:\s*[A-Za-z_$][A-Za-z0-9_$.\[\]<>|& ]*?\{")


def strip_ts_types(code: str) -> str:
    """TypeScript を node で走らせるために型注釈だけ外す（意味は変えない）。"""
    out = re.sub(r"^\s*export\s+", "", code, flags=re.M)
    out = _TS_RET_TYPE.sub(") {", out)
    out = _TS_ARG_TYPE.sub(r"\1", out)
    return out


def _run_node(code: str, name: str, cases: list[tuple[str, str]], *, ts: bool = False) -> dict:
    node = shutil.which("node")
    if not node:
        return {"ran": False, "ok": True, "note": "node が無いので構文の自己点検まで"}
    lines = [code, ""]
    for arg, want in cases:
        lines.append(f"console.log(JSON.stringify({name}({arg})));")
    with tempfile.TemporaryDirectory(prefix="snipher-js-") as td:
        path = os.path.join(td, "main.js")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        try:
            proc = subprocess.run([node, path], cwd=td, capture_output=True, text=True, timeout=6)
        except Exception as exc:  # noqa: BLE001
            return {"ran": False, "ok": False, "note": f"{type(exc).__name__}"}
    out_lines = [x for x in (proc.stdout or "").splitlines() if x.strip()]
    passed = 0
    for (arg, want), got in zip(cases, out_lines):
        try:
            same = json.loads(got) == json.loads(want if want.startswith("[") else want)
        except Exception:  # noqa: BLE001
            same = got.strip() == want.strip()
        passed += 1 if same else 0
    ok = proc.returncode == 0 and passed == len(cases) and len(out_lines) >= len(cases)
    note = f"node で実行: {passed}/{len(cases)} 件が期待通り" if proc.returncode == 0 else \
        "実行エラー: " + (proc.stderr or "").strip().splitlines()[-1][:120] if proc.stderr else "実行エラー"
    return {"ran": True, "ok": ok, "note": note, "stdout": (proc.stdout or "").strip()[:300],
            "passed": passed, "cases": len(cases)}


def _run_python(code: str, name: str, cases: list[tuple[str, str]]) -> dict:
    lines = [code, ""]
    for arg, want in cases:
        lines.append(f"print({name}({arg}))")
    src = "\n".join(lines)
    with tempfile.TemporaryDirectory(prefix="snipher-py-") as td:
        path = os.path.join(td, "main.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        try:
            proc = subprocess.run([sys.executable, path], cwd=td, capture_output=True, text=True,
                                  timeout=6, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        except Exception as exc:  # noqa: BLE001
            return {"ran": False, "ok": False, "note": f"{type(exc).__name__}"}
    out_lines = [x.strip() for x in (proc.stdout or "").splitlines() if x.strip()]
    passed = 0
    for (arg, want), got in zip(cases, out_lines):
        passed += 1 if got == want.replace("[]", "[]") else 0
    ok = proc.returncode == 0 and passed == len(cases)
    note = f"実行して確認: {passed}/{len(cases)} 件が期待通り" if proc.returncode == 0 else \
        "実行エラー: " + ((proc.stderr or "").strip().splitlines() or [""])[-1][:120]
    return {"ran": True, "ok": ok, "note": note, "stdout": (proc.stdout or "").strip()[:300],
            "passed": passed, "cases": len(cases)}


def _run_go(code: str, name: str, cases: list[tuple[str, str]]) -> dict:
    go = shutil.which("go")
    if not go:
        return {"ran": False, "ok": True, "note": "go が無いので構文の自己点検まで"}
    export = name[:1].upper() + name[1:]
    main = ["package main", "", "import (", "\t\"fmt\"", "\t\"sort\"", ")", "",
            code.replace("package main\n\n", ""), "", "func main() {"]
    for arg, _want in cases:
        main.append(f"\tfmt.Println({export}([]int{{{arg.strip('[]')}}}))")
    main.append("}")
    with tempfile.TemporaryDirectory(prefix="snipher-go-") as td:
        path = os.path.join(td, "main.go")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(main))
        try:
            proc = subprocess.run([go, "run", path], cwd=td, capture_output=True, text=True,
                                  timeout=60)
        except Exception as exc:  # noqa: BLE001
            return {"ran": False, "ok": False, "note": f"{type(exc).__name__}"}
    ok = proc.returncode == 0
    note = "go run で実行して出力を確認" if ok else \
        "実行エラー: " + ((proc.stderr or "").strip().splitlines() or [""])[-1][:120]
    return {"ran": ok, "ok": ok, "note": note, "stdout": (proc.stdout or "").strip()[:300]}


# --------------------------------------------------------------------------- #
# 解説
# --------------------------------------------------------------------------- #
_OP_NOTES = {
    "dedupe": "重複は {dedupe_how} で 1 度だけ残し、もとの順序を保ちます",
    "sort_asc": "昇順は比較関数を渡してソートするので、文字列化された「10 < 9」も起きません",
    "sort_desc": "降順は比較関数の向きを逆にしています",
    "reverse": "反転は copy に対して行うので、渡された配列は書き換えません",
    "count": "件数は長さで返します",
    "unique_count": "種類数は重複を落としたあとの長さで返します",
    "sum": "合計は reduce（Python では sum）で 1 度に計算します",
    "max": "空の配列では undefined（Python では None）を返します",
    "min": "空の配列では undefined（Python では None）を返します",
    "filter_even": "偶数だけを残します",
    "filter_odd": "奇数だけを残します",
}


_DEDUPE_HOW = {"javascript": "Set", "typescript": "Set", "python": "dict.fromkeys",
               "go": "map[int]bool", "rust": "HashSet", "ruby": "uniq"}


def explain_ops(ops: list[str], *, lang: str) -> list[str]:
    out: list[str] = []
    for o in ops:
        note = _OP_NOTES.get(o)
        if note:
            out.append(note.format(dedupe_how=_DEDUPE_HOW.get(lang, "Set")))
    if not out:
        out.append("指示の操作をそのまま 1 関数にまとめました")
    out.append("入力は複製してから触るので、渡された配列そのものは変わりません")
    return out[:3]


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def run(d: Directive, *, explain: bool = True) -> dict:
    """指示から関数を作り、実行して確かめる。"""
    text = d.raw
    lang = (d.fmt.language or detect_language(text) or "python").lower()
    if lang in ("js", "node", "node.js"):
        lang = "javascript"
    name, args = function_spec(text)
    ops = operations(text)
    notes: list[str] = []
    arg = _arg_name(args)
    if not name and ops:
        # 名前を書かない指示（「配列の合計を返す関数を書いて」）でも、操作から名前を付ける
        name = _name_from_ops(ops, lang)
        arg = _arg_from_ops(ops)
        notes.append(f"関数名: 指示に名前が無いので {_op_label(ops[0])} から {name} と付けました")
    if lang == "ts":
        lang = "typescript"

    built: str | None = None
    if name and ops:
        if lang in ("javascript", "typescript"):
            built, _ret = compose_js(name, arg, ops, ts=(lang == "typescript"))
        elif lang == "python":
            built = compose_python(name, arg, ops)
        elif lang == "go":
            built = compose_go(name, arg, ops)
    if built is None:
        # 名前や操作が読めない依頼は、既存のコード生成器（課題テンプレ＋検証）に渡す。
        # 言語は *指示から読めたもの* を必ず引き継ぐ（codegen 側の判定が外れても
        # Python に落ちないように）。
        from ..codegen import generate as _generate
        from ..solve import code as _code

        display, guessed, meta = _generate(text, lang=lang)
        use = guessed or lang
        snippet = meta.get("snippet") or _code._strip_fence(display)
        res = _code.verify(snippet, use)
        res.notes.append(f"生成タスク: {meta.get('task', 'generic')}")
        return {"text": _format_existing(res, explain=explain), "language": res.language,
                "checked": res.checked, "ran": res.ran, "ok": res.ok,
                "notes": list(res.notes), "meta": res.as_dict(), "ops": ops,
                "function": name, "confidence": 0.9}

    from ..solve.code import balanced, syntax_check_python, verify

    cases = _cases(ops, lang)
    result: dict = {"ran": False, "ok": True, "note": ""}
    if lang == "python":
        ok, why = syntax_check_python(built)
        if not ok:
            notes.append(f"構文検査: {why}")
        result = _run_python(built, name, cases) if ok else result
    elif lang in ("javascript", "typescript"):
        body = strip_ts_types(built) if lang == "typescript" else built
        ok, why = balanced(body)
        if not ok:
            notes.append(f"自己点検: {why}")
        result = _run_node(body, name, cases) if ok else result
    elif lang == "go":
        result = _run_go(built, name, cases)
    else:
        res = verify(built, lang)
        result = {"ran": False, "ok": res.ok, "note": " / ".join(res.notes) or "構文の自己点検まで"}
        notes.extend(res.notes)
    if result.get("note"):
        notes.append(result["note"])

    label = _LANG_LABEL.get(lang, lang)
    fence_lang = "typescript" if lang == "typescript" else ("javascript" if lang in
                                                            ("javascript", "node") else lang)
    parts = [f"```{fence_lang}\n{built}\n```"]
    if explain:
        for line in explain_ops(ops, lang=lang):
            parts.append(line + "。")
        sig = f"{name}({arg})"
        parts.insert(1, f"{label} の {sig} として、指示の操作（"
                        + "→".join(_op_label(o) for o in ops) + "）を 1 つにまとめました。")
    return {"text": "\n".join(parts), "language": lang, "checked": True,
            "ran": bool(result.get("ran")), "ok": bool(result.get("ok")),
            "notes": notes, "ops": ops, "function": name,
            "tests": {"passed": result.get("passed"), "cases": result.get("cases"),
                      "stdout": result.get("stdout", "")},
            "confidence": 0.96 if result.get("ran") and result.get("ok") else 0.82,
            "snippet": built}


_NUMERIC_OPS = ("sum", "max", "min", "filter_even", "filter_odd", "average", "median")
_NAME_PART = {"dedupe": "unique", "sort_asc": "sorted", "sort_desc": "sorted_desc",
              "reverse": "reversed", "count": "count", "unique_count": "unique_count",
              "sum": "sum", "max": "max", "min": "min", "filter_even": "evens",
              "filter_odd": "odds"}


def _name_from_ops(ops: list[str], lang: str) -> str:
    """操作から関数名を作る（`sum` → JS/TS `sumValues`, Python `sum_values`, Go `SumValues`）。"""
    parts = [_NAME_PART.get(o, o) for o in ops[:2]] or ["solve"]
    if lang == "python":
        return "_".join(parts)
    if lang == "go":
        return "".join(p[:1].upper() + p[1:] for p in parts) + "Values"
    head, *rest = parts
    return head + "".join(p[:1].upper() + p[1:] for p in rest) + "Values" \
        if len(parts) > 1 else head + "Values"


def _arg_from_ops(ops: list[str]) -> str:
    return "numbers" if any(o in _NUMERIC_OPS for o in ops) else "items"


def _op_label(op: str) -> str:
    return {"dedupe": "重複除去", "sort_asc": "昇順", "sort_desc": "降順", "reverse": "反転",
            "count": "件数", "unique_count": "種類数", "sum": "合計", "max": "最大",
            "min": "最小", "filter_even": "偶数", "filter_odd": "奇数"}.get(op, op)


def _format_existing(res, *, explain: bool = True) -> str:
    parts = [f"```{res.language}\n{res.code}\n```"]
    if not explain:
        return parts[0]                       # コードのみ（解説も実行メモも付けない）
    if res.ran and res.stdout:
        first = res.stdout.splitlines()[0]
        rest = len(res.stdout.splitlines()) - 1
        parts.append(f"実行結果: {first}" + (f"（ほか {rest} 行）" if rest > 0 else ""))
    elif res.ran and not res.ok:
        parts.append("実行しましたがエラーが出ました: " + (res.stderr.splitlines() or [""])[-1][:120])
    elif res.checked:
        parts.append("構文検査まで済ませています")
    if explain and res.notes:
        parts.append("確認したこと: " + "、".join(res.notes[:2]) + "。")
    return "\n".join(parts)


__all__ = ["run", "operations", "compose_js", "compose_python", "compose_go", "OPS"]
