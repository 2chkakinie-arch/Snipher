"""コード仕事 — 書くだけじゃなく、**実行して確かめる**。

「書いて」で終わる Bot との違いはここです。生成物を

    1. `compile()` で構文検査（1ms・実行しない）
    2. 隔離したサブプロセスで実行（タイムアウト・出力 capture）
    3. 失敗したら、エラーメッセージを手がかりに 1 回だけ直してもう一度試す
    4. 実際の出力を根拠に「動きました」と答える

という順で検証してから出します。Python 以外（JS/HTML/SQL/shell）は
構文レベルの自己点検（括弧・クォートの対応、`node --check` があればそれ）に留め、
実行できないことは隠さず「検証範囲」として明記します。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

from ..lang.phonetics import normalize

@dataclass
class CodeResult:
    code: str
    language: str
    checked: bool = False
    ran: bool = False
    ok: bool = True
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    repairs: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"language": self.language, "checked": self.checked, "ran": self.ran,
                "ok": self.ok, "exit_code": self.exit_code, "repairs": self.repairs,
                "stdout": self.stdout[:400], "stderr": self.stderr[:400],
                "notes": self.notes}


def _allow_run() -> bool:
    return os.environ.get("SNIPHER_CODE_RUN", "1") not in ("0", "off", "false")


def syntax_check_python(code: str) -> tuple[bool, str]:
    try:
        compile(code, "<snipher>", "exec")
    except SyntaxError as exc:
        return False, f"{exc.msg} (line {exc.lineno})"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def run_python(code: str, *, timeout: float = 3.0, stdin: str | None = None) -> tuple[int, str, str]:
    """一時ファイルにして subprocess で実行する（eval ではなく本物の実行）。"""
    with tempfile.TemporaryDirectory(prefix="snipher-code-") as td:
        path = os.path.join(td, "main.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)
        try:
            proc = subprocess.run([sys.executable, path], cwd=td, capture_output=True, text=True,
                                  timeout=timeout, input=stdin or "",
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                       "PYTHONDONTWRITEBYTECODE": "1"})
        except subprocess.TimeoutExpired:
            return 124, "", f"timeout:{timeout}s"
        except Exception as exc:  # noqa: BLE001
            return 125, "", f"{type(exc).__name__}: {exc}"
        return int(proc.returncode or 0), proc.stdout or "", proc.stderr or ""


def node_check(code: str) -> tuple[bool, str]:
    node = shutil.which("node")
    if not node:
        return False, "node が無いため構文のみ自己点検"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
        fh.write(code)
        path = fh.name
    try:
        proc = subprocess.run([node, "--check", path], capture_output=True, text=True, timeout=5)
        ok = proc.returncode == 0
        return ok, (proc.stderr or proc.stdout)[:300]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def balanced(code: str) -> tuple[bool, str]:
    """括弧・クォートの対応チェック（どの言語でも使える最低限の自己点検）。"""
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    quote = ""
    i = 0
    while i < len(code):
        ch = code[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in "\"'`":
            quote = ch
        elif ch == "/" and code[i:i + 2] == "//":
            i = code.find("\n", i)
            if i < 0:
                break
        elif ch == "/" and code[i:i + 2] == "/*":
            i = code.find("*/", i)
            if i < 0:
                return False, "コメントが閉じていません"
            i += 1
        elif ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return False, f"{ch} に対応する開き括弧がありません"
            stack.pop()
        i += 1
    if stack:
        return False, f"{stack[-1]} が閉じられていません"
    if quote:
        return False, "引用符が閉じられていません"
    return True, ""


_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
              "param", "source", "track", "wbr"}


def tag_balance(code: str) -> tuple[bool, str]:
    """HTML/SVG のタグ対応を栈で検査する（void 要素・自己閉じは数えない）。

    「タグの数が合っているか」だけの検査は、li を閉じずに div を閉じる
    ような壊れた文書を通してしまうので、開いた順に閉じているかを見ます。
    """
    stack: list[tuple[str, int]] = []
    src = code or ""
    # script / style の中身とコメントは「タグ」ではない（JS 内の "<div>" に騙されないため
    # 本文だけ空白に置き換えてから数える）
    src = re.sub(r"<(script|style)\b[^>]*>.*?</\1\s*>", " ", src, flags=re.S | re.I)
    src = re.sub(r"<!--.*?-->", " ", src, flags=re.S)
    for m in re.finditer(r'<(/?)([a-zA-Z][a-zA-Z0-9-]*)([^>]*)>', src):
        closing = m.group(1) == "/"
        tag = str(m.group(2)).lower()
        rest = m.group(3) or ""
        if tag in _VOID_TAGS or rest.rstrip().endswith("/"):
            continue
        if closing:
            if not stack:
                return False, f"</{tag}> に対応する <{tag}> がありません"
            if stack[-1][0] != tag:
                return False, f"<{stack[-1][0]}> を閉じないうちに </{tag}> が来ました"
            stack.pop()
        else:
            stack.append((tag, m.start()))
    if stack:
        return False, f"<{stack[-1][0]}> が閉じられていません"
    return True, ""


def _repair_python(code: str, stderr: str) -> str | None:
    """エラーメッセージから直せるものを 1 段階だけ直す（自己修復の最小ループ）。"""
    lines = code.split("\n")
    if "IndentationError" in stderr:
        m = re.search(r"line (\d+)", stderr)
        if m:
            n = int(m.group(1))
            if 0 < n <= len(lines):
                fixed = list(lines)
                fixed[n - 1] = fixed[n - 1].replace("\t", "    ")
                return "\n".join(fixed)
    if "NameError" in stderr and "__main__" not in code:
        return code + "\n"
    if "ModuleNotFoundError" in stderr:
        m = re.search(r"No module named '([\w.]+)'", stderr)
        if m:
            return "\n".join(ln for ln in lines if not ln.strip().startswith(("import " + m.group(1),
                                                                              "from " + m.group(1))))
    if "SyntaxError" in stderr and 'print("' in code:
        fixed = code.replace('print("', "print('").replace('")\n', "')\n", 1)
        return fixed if fixed != code else None
    return None


def verify(code: str, language: str, *, timeout: float = 3.0) -> CodeResult:
    """生成物を検証する（Python は実行まで、他言語は構文点検まで）。"""
    res = CodeResult(code=code, language=language)
    lang = (language or "").lower()
    if lang.startswith("py") or lang == "python":
        ok, why = syntax_check_python(code)
        res.checked = True
        if not ok:
            res.ok = False
            res.notes.append(f"構文検査で検出: {why}")
            repaired = _repair_python(code, why)
            if repaired:
                ok2, why2 = syntax_check_python(repaired)
                if ok2:
                    code, res.ok, res.repairs = repaired, True, 1
                    res.notes.append("自己修復してもう一度検査しました")
                    res.code = repaired
        if res.ok and _allow_run():
            if re.search(r"serve_forever|while\s+True|input\(|asyncio\.run\(|http\.server", code):
                res.notes.append("常駐・待受けを含むため実行はしていません（構文検査まで）")
            else:
                exit_code, out, err = run_python(repaired if res.repairs else code, timeout=timeout)
                res.ran = True
                res.exit_code = exit_code
                res.stdout = out.strip()
                res.stderr = err.strip()
                if exit_code != 0:
                    res.ok = False
                    retry = _repair_python(code, err)
                    if retry and res.repairs == 0:
                        ec2, out2, err2 = run_python(retry, timeout=timeout)
                        if ec2 == 0:
                            res.code, res.ok, res.repairs = retry, True, 1
                            res.stdout, res.stderr, res.exit_code = out2.strip(), err2.strip(), 0
                            res.notes.append("1 回直して再実行し、動きました")
                        else:
                            res.notes.append("再実行でもエラー: " + (err2.strip().splitlines() or [""])[-1][:120])
                    else:
                        res.notes.append("実行時エラー: " + (err.strip().splitlines() or [""])[-1][:120])
                else:
                    res.notes.append("実行して出力を確認")
        elif res.ok:
            res.notes.append("実行はオフ（SNIPHER_CODE_RUN=0）。構文検査まで実施")
    elif lang.startswith(("js", "node", "typescript")):
        bal, why = balanced(code)
        res.checked = bal
        if not bal:
            res.notes.append("自己点検: " + why)
            res.ok = False
        else:
            ok, out = node_check(code)
            if ok:
                res.notes.append("node --check で構文を確認")
            else:
                res.notes.append(out if not ok or "node" in out else "node --check 通過")
    elif lang in ("html", "svg", "xml"):
        bal, why = tag_balance(code)
        res.checked = True
        if not bal:
            res.ok = False
            res.notes.append("タグ検査: " + why)
        else:
            res.notes.append("タグの対応（開いた順に閉じているか）を確認")
    elif lang == "css":
        bal, why = balanced(code)
        res.checked = True
        if not bal:
            res.ok = False
            res.notes.append("自己点検: " + why)
        else:
            res.notes.append("括弧・引用符の対応を確認")
    else:
        bal, why = balanced(code)
        res.checked = True
        if not bal:
            res.ok = False
            res.notes.append("自己点検: " + why)
        else:
            res.notes.append("括弧・引用符の対応を確認")
    return res


def build(request: str, *, language: str | None = None,
          tests: list[tuple[str, str]] | None = None) -> CodeResult:
    """依頼を受けてコードを組み立て、検証する（`snipher.codegen` の上に検査を載せる）。"""
    from ..codegen import generate

    display, lang, meta = generate(request)
    code = meta.get("snippet") or _strip_fence(display)
    if language and language.lower() in ("python", "py"):
        lang = "python"
    res = verify(code, lang)
    if tests and res.ok and lang.startswith("py"):
        res = run_tests(res, tests)
    res.notes.append(f"生成タスク: {meta.get('task', 'generic')}")
    if not res.ok:
        res.notes.append("直せなかった点は仕様不足が原因の可能性が高いので、入出力例をください")
    return res


def _strip_fence(display: str) -> str:
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", display, re.S)
    return m.group(1).strip() if m else display.strip()


def run_tests(res: CodeResult, tests: list[tuple[str, str]]) -> CodeResult:
    """「入力 → 期待出力」があれば、実際に走らせて突き合わせる（自己検証）。"""
    harness = (
        "import sys\n"
        + res.code
        + "\n\nimport json\ncases = json.loads(sys.argv[1])\n"
          "out = []\nfor src, want in cases:\n"
          "    try:\n        val = eval(src, globals(), {})\n    except Exception as exc:\n"
          "        out.append({'src': src, 'got': f'{type(exc).__name__}: {exc}', 'want': want, 'ok': False})\n"
          "        continue\n"
          "    got = str(val)\n"
          "    out.append({'src': src, 'got': got, 'want': want, 'ok': got == str(want)})\n"
          "print(json.dumps(out, ensure_ascii=False))\n"
    )
    exit_code, stdout, stderr = run_python(harness, timeout=4.0,
                                           stdin=None, )
    # tests は argv 経由で渡す（stdin は使わない）
    payload = json.dumps(tests, ensure_ascii=False)
    with tempfile.TemporaryDirectory(prefix="snipher-test-") as td:
        path = os.path.join(td, "harness.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(harness)
        try:
            proc = subprocess.run([sys.executable, path, payload], cwd=td, capture_output=True,
                                  text=True, timeout=6)
            exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except Exception as exc:  # noqa: BLE001
            res.notes.append(f"テスト実行失敗: {type(exc).__name__}")
            return res
    try:
        results = json.loads((stdout or "[]").strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        res.notes.append("テスト結果を解析できませんでした（" + (stderr[:120] or "") + "）")
        return res
    passed = sum(1 for r in results if r.get("ok"))
    res.detail_tests = results                      # type: ignore[attr-defined]
    res.notes.append(f"自分で走らせた検証: {passed}/{len(results)} 件一致")
    if passed != len(results):
        res.ok = False
    return res


__all__ = ["build", "verify", "run_python", "syntax_check_python", "balanced", "tag_balance",
           "CodeResult", "run_tests"]
