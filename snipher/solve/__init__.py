"""道具ではなく「考え方の部品」。計算・文字・暦・コードの検証可能な作業を担う。

これらの出力は *根拠のある主張（Claim）* として composer に渡ります。だから
生成の確率で上書きされることはなく、それでも文章の組み立ては 1 つの通路（mind）を通る。
"""

from __future__ import annotations

from . import code, facts, math as mathlab, text as textops


class SolveResult:
    __slots__ = ("kind", "solution", "authoritative")

    def __init__(self, kind: str, solution, *, authoritative: bool = True):
        self.kind = kind
        self.solution = solution
        self.authoritative = authoritative

    def as_dict(self) -> dict:
        d = dict(self.solution.detail or {})
        d.update({"kind": self.kind, "answer": self.solution.answer,
                  "steps": self.solution.steps, "verified": self.solution.verified})
        return d


def solve_math(text: str) -> SolveResult | None:
    sol = mathlab.solve(text)
    return SolveResult("math", sol) if sol else None


def solve_facts(text: str) -> SolveResult | None:
    sol = facts.handle(text)
    return SolveResult("facts", sol) if sol else None


def solve_text(text: str) -> SolveResult | None:
    sol = textops.handle(text)
    return SolveResult("text", sol) if sol else None


def solve_code(text: str) -> SolveResult | None:
    from ..codegen import is_code_request

    if not is_code_request(text):
        return None
    res = code.build(text)
    sol = mathlab.Solution(answer=_code_answer(res), steps=list(res.notes), kind="code",
                           verified=bool(res.ok),
                           detail={"language": res.language, "ran": res.ran,
                                   "stdout": res.stdout[:300], "exit_code": res.exit_code})
    return SolveResult("code", sol, authoritative=True), res


def _code_answer(res) -> str:
    bits = [f"```{res.language}\n{res.code}\n```"]
    if res.ran and res.stdout:
        first = res.stdout.splitlines()[0]
        rest = len(res.stdout.splitlines()) - 1
        bits.append(f"実行結果: {first}" + (f"（ほか {rest} 行）" if rest > 0 else ""))
    elif res.ran and not res.ok:
        bits.append("実行しましたがエラーが出ました: " + (res.stderr.splitlines() or [""])[-1][:120])
    elif res.checked:
        bits.append("構文検査まで済ませています")
    return "\n".join(bits)


def handle(text: str, *, hint: str = "") -> SolveResult | None:
    """発話 1 つを受けて、検証できる仕事なら解いて返す。"""
    for fn in (solve_facts, solve_math, solve_text):
        try:
            got = fn(text)
        except Exception:  # noqa: BLE001
            got = None
        if got is not None:
            return got
    if hint == "code":
        try:
            got, _res = solve_code(text)
            return got
        except Exception:  # noqa: BLE001
            return None
    return None


__all__ = ["handle", "solve_math", "solve_text", "solve_facts", "solve_code", "SolveResult",
           "mathlab", "textops", "facts", "code"]
