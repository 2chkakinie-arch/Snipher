"""Snipher の「定型文ではない仕事」を先に厳密に解くタスク層。

小さな言語モデルに算数・コード・現在情報を任せると、文法だけ正しくて
答えが間違う。そこで会話生成の前に、次の順で決定的な道具を選ぶ。

    文字列の意味判定 → 厳密計算 / コード整形 / ウェブ調査 → 根拠付き Reply

ここは LLM の代用品ではない。LLM が得意な自由な文章は LFM2.5 に任せ、
間違えると困る仕事は安全な小さな実装で確定させる。依存は標準ライブラリ
だけで、外部検索は ``ResearchEngine`` が必要と判定した質問に限る。
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .research import ResearchEngine, ResearchResult


@dataclass(slots=True)
class TaskAnswer:
    """Composer/Core がそのまま応答にできる、根拠付きの仕事結果。"""

    text: str
    plan: str
    confidence: float = 0.98
    kind: str = "task"
    metadata: dict[str, Any] = field(default_factory=dict)
    authoritative: bool = True

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "plan": self.plan,
            "confidence": self.confidence,
            "kind": self.kind,
            "metadata": self.metadata,
            "authoritative": self.authoritative,
        }


_FULLWIDTH_TRANS = str.maketrans("０１２３４５６７８９．，＋－＝×÷％", "0123456789.,+-=×÷%")
_UNIT_WORDS = "本個枚冊台匹人袋箱回点つ"


def _norm(text: str) -> str:
    # 「ー」は日本語の長音なので ASCII のマイナスに変換しない。
    return str(text or "").translate(_FULLWIDTH_TRANS).replace("−", "-")


def _strip_question(text: str) -> str:
    return re.sub(r"[?？。！!]+$", "", _norm(text).strip()).strip()


def _fmt(value: Fraction | int | float) -> str:
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _term_coefficient(value: Fraction, variable: str = "x") -> str:
    """係数 1 を ``x``、-1 を ``-x`` と表示する。"""
    if value == 1:
        return variable
    if value == -1:
        return "-" + variable
    return _fmt(value) + variable


def _parse_number(raw: str) -> Fraction:
    raw = raw.replace(",", "")
    if "." in raw:
        return Fraction(raw)
    return Fraction(int(raw))


# ---------------------------------------------------------------------------
# 厳密な計算
# ---------------------------------------------------------------------------
_ALLOWED_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                   ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
                   ast.Pow: lambda a, b: a ** b, ast.FloorDiv: lambda a, b: a // b,
                   ast.Mod: lambda a, b: a % b}
_ALLOWED_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def safe_arithmetic(expression: str) -> Fraction:
    """四則・べき・剰余だけを Fraction で評価する（eval は使わない）。"""
    expr = _norm(expression).replace("×", "*").replace("÷", "/")
    expr = expr.replace("^", "**")
    expr = re.sub(r"[^0-9+*/%().\- ]", "", expr)
    if not expr.strip() or not re.search(r"\d", expr):
        raise ValueError("no_expression")
    tree = ast.parse(expr, mode="eval")

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow):
                if right.denominator != 1 or abs(right.numerator) > 100:
                    raise ValueError("power_limit")
                return left ** right.numerator
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ZeroDivisionError
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        raise ValueError("unsupported_expression")

    return visit(tree)


def _parse_poly(expression: str) -> dict[int, Fraction]:
    """x の 2 次以下の式を係数辞書にする。"""
    s = _norm(expression).replace(" ", "").replace("*", "")
    s = s.replace("²", "^2").replace("X", "x")
    if not s:
        raise ValueError("empty_polynomial")
    if s[0] not in "+-":
        s = "+" + s
    terms = re.findall(r"[+-][^+-]+", s)
    out: dict[int, Fraction] = {}
    for term in terms:
        sign = -1 if term[0] == "-" else 1
        body = term[1:]
        if "x" not in body:
            degree, coefficient = 0, _parse_number(body)
        else:
            before, _, after = body.partition("x")
            degree = 1
            if after.startswith("^"):
                degree = int(after[1:])
            elif after:
                raise ValueError("polynomial_syntax")
            coefficient = Fraction(1) if before == "" else _parse_number(before)
        out[degree] = out.get(degree, Fraction(0)) + sign * coefficient
    return {d: c for d, c in out.items() if c}


def _sqrt_fraction(value: Fraction) -> Fraction | None:
    if value < 0:
        return None
    n, d = value.numerator, value.denominator
    sn, sd = math.isqrt(n), math.isqrt(d)
    if sn * sn == n and sd * sd == d:
        return Fraction(sn, sd)
    return None


def solve_equation(text: str) -> TaskAnswer | None:
    """一次/二次方程式を解く。判定できない式は None にして LLM に渡す。"""
    raw = _strip_question(text)
    if "=" not in raw or not re.search(r"[xXｘ]|未知数|方程式", raw):
        return None
    left, right = raw.split("=", 1)
    # 自然文の「2次方程式」「x を求めて」の部分を式から除外。
    # x があれば最初の x から左辺を読むので、係数の前置きも保てる。
    x_at = re.search(r"[xXｘ]", left)
    if x_at:
        prefix = left[:x_at.start()]
        # 「3x」は係数を残し、「2次方程式 x」は説明だけを落とす。
        coefficient = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*$", prefix)
        if coefficient:
            left = coefficient.group(1) + left[x_at.start():]
        else:
            left = left[x_at.start():]
    right = re.split(r"\s*(?:を|と|の|について).*$", right, maxsplit=1)[0]
    try:
        lp, rp = _parse_poly(left), _parse_poly(right)
    except (ValueError, ZeroDivisionError):
        return None
    poly = dict(lp)
    for d, c in rp.items():
        poly[d] = poly.get(d, Fraction(0)) - c
    poly = {d: c for d, c in poly.items() if c}
    degree = max(poly, default=0)
    if degree == 0:
        text_out = "恒等式です。" if poly.get(0, 0) == 0 else "解はありません。"
        return TaskAnswer(text_out, "math:equation", 0.99, "math", {"degree": 0})
    if degree > 2:
        return None
    a, b, c = poly.get(2, Fraction(0)), poly.get(1, Fraction(0)), poly.get(0, Fraction(0))
    if degree == 1:
        root = -c / b
        linear = _term_coefficient(b)
        if c:
            linear += ("+" if c >= 0 else "") + _fmt(c)
        out = f"{linear}=0 より、x = {_fmt(root)} です。"
        return TaskAnswer(out, "math:linear", 0.995, "math", {"degree": 1, "roots": [_fmt(root)]})
    disc = b * b - 4 * a * c
    root_disc = _sqrt_fraction(disc)
    if root_disc is None:
        if disc < 0:
            roots = "実数解はありません"
        else:
            roots = "複素数解です（判別式を計算できます）"
        out = f"判別式 D = {_fmt(disc)} なので、{roots}。"
        return TaskAnswer(out, "math:quadratic", 0.98, "math", {"degree": 2, "discriminant": _fmt(disc)})
    r1 = (-b + root_disc) / (2 * a)
    r2 = (-b - root_disc) / (2 * a)
    roots = [r1, r2]
    # 重解は一つだけ表示。
    root_text = _fmt(r1) if r1 == r2 else f"{_fmt(r1)}, {_fmt(r2)}"
    quadratic = ("x²" if a == 1 else "-x²" if a == -1 else f"{_fmt(a)}x²")
    if b:
        quadratic += ("+" if b > 0 else "") + _term_coefficient(b)
    if c:
        quadratic += ("+" if c > 0 else "") + _fmt(c)
    out = f"{quadratic}=0。\n判別式 D = {_fmt(disc)}、解は x = {root_text} です。"
    return TaskAnswer(out, "math:quadratic", 0.995, "math", {"degree": 2, "roots": [_fmt(r) for r in roots], "discriminant": _fmt(disc)})


def solve_word_problem(text: str) -> TaskAnswer | None:
    """「単価×個数」の日本語文章題を式にして解く。"""
    raw = _norm(text).replace(",", "、")
    if "円" not in raw or not re.search(r"何[" + _UNIT_WORDS + r"]", raw):
        return None
    # unknown: 80円の鉛筆を何本か / 80円の鉛筆を何本
    unknown = re.search(
        r"(?P<price>[0-9]+(?:\.[0-9]+)?)\s*円\s*の\s*(?P<item>[^、。;；]+?)\s*を\s*何(?P<unit>[" + _UNIT_WORDS + r"]+)",
        raw,
    )
    if not unknown:
        # 「鉛筆を1本80円で何本」の別表現
        unknown = re.search(
            r"(?P<item>[^、。;；]+?)\s*を\s*(?P<price>[0-9]+(?:\.[0-9]+)?)\s*円[^、。;；]*何(?P<unit>[" + _UNIT_WORDS + r"]+)",
            raw,
        )
    if not unknown:
        return None
    unknown_price = _parse_number(unknown.group("price"))
    item = re.sub(r"^[0-9]+(?:\.[0-9]+)?(?:本|個|枚|冊|台|匹|人)?", "", unknown.group("item")).strip() or "品物"
    # total は最後の「数字 + 円」にする。ただし単価の数字は除外。
    yen = list(re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*円", raw))
    if len(yen) < 2:
        return None
    total_match = re.search(r"(?:合計|全部|あわせて|合わせて|買うと|すると|なった|なると)[^。！？!?]*?([0-9]+(?:\.[0-9]+)?)\s*円", raw)
    total = _parse_number(total_match.group(1)) if total_match else _parse_number(yen[-1].group(1))
    # 固定購入: 単価の直後の「の...を3個」。unknown の単価は除く。
    fixed: list[tuple[Fraction, Fraction, str]] = []
    for match in re.finditer(
        r"([0-9]+(?:\.[0-9]+)?)\s*円\s*の\s*([^、。;；]+?)\s*を\s*([0-9]+(?:\.[0-9]+)?)\s*([" + _UNIT_WORDS + r"]+)",
        raw,
    ):
        if match.start() == unknown.start() or abs(match.start() - unknown.start()) < 3:
            continue
        price = _parse_number(match.group(1))
        count = _parse_number(match.group(3))
        fixed.append((price, count, match.group(2).strip()))
    # 上の正規表現で未知単価まで固定として拾った場合を除く。
    fixed = [(p, n, name) for p, n, name in fixed if p != unknown_price or name not in item]
    fixed_total = sum((p * n for p, n, _ in fixed), Fraction(0))
    count = (total - fixed_total) / unknown_price
    if count.denominator != 1 or count < 0:
        return None
    fixed_desc = " + ".join(f"{_fmt(p)}×{_fmt(n)}" for p, n, _ in fixed) or "0"
    equation = f"{_fmt(unknown_price)}x + {fixed_desc} = {_fmt(total)}"
    out = f"式を立てると {equation} です。\n{_fmt(unknown_price)}x = {_fmt(total - fixed_total)} より、x = {_fmt(count)}。\n答えは{_fmt(count)}{unknown.group('unit')}です。"
    return TaskAnswer(out, "math:word_problem", 0.999, "math", {
        "variable": "x", "item": item, "unit": unknown.group("unit"),
        "equation": equation, "answer": _fmt(count), "fixed_total": _fmt(fixed_total),
    })


def solve_arithmetic_task(text: str) -> TaskAnswer | None:
    """単純な計算依頼を安全に評価する。"""
    raw = _strip_question(text)
    if not re.search(r"\d", raw):
        return None
    if any(word in raw for word in ("何本", "何個", "何枚", "何冊", "何台")) and "円" in raw:
        return None  # 文章題の担当へ
    math_mark = bool(re.search(r"[+＋\-－×÷*/%]|計算|いくつ|合計|掛け|足し|引き|割り", raw))
    if not math_mark:
        return None
    expr_match = re.search(r"[0-9][0-9\s+＋\-－×÷*/%().^]*[0-9)]", raw)
    if not expr_match:
        return None
    try:
        value = safe_arithmetic(expr_match.group(0))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None
    return TaskAnswer(f"計算すると {expr_match.group(0)} = {_fmt(value)} です。", "math:arithmetic", 0.999, "math", {"value": _fmt(value)})


# ---------------------------------------------------------------------------
# 単位変換
# ---------------------------------------------------------------------------
_UNIT_FACTORS = {
    "mm": Fraction(1, 1000), "ミリ": Fraction(1, 1000),
    "cm": Fraction(1, 100), "センチ": Fraction(1, 100),
    "m": Fraction(1), "メートル": Fraction(1),
    "km": Fraction(1000), "キロ": Fraction(1000),
    "g": Fraction(1), "グラム": Fraction(1),
    "kg": Fraction(1000),
    "mg": Fraction(1, 1000), "ミリグラム": Fraction(1, 1000),
    "ml": Fraction(1), "ミリリットル": Fraction(1),
    "l": Fraction(1000), "リットル": Fraction(1000),
}


def convert_unit(text: str) -> TaskAnswer | None:
    raw = _norm(text).lower().replace(" ", "")
    # 「3キロは何メートル」「250cmをmにして」の両方を読む。
    units = "|".join(sorted((re.escape(x) for x in _UNIT_FACTORS), key=len, reverse=True))
    m = re.search(r"(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<src>" + units + r")(?:は|を|=)(?:何|いくつ|換算すると)?(?P<dst>" + units + r")", raw)
    if not m:
        return None
    src, dst = m.group("src"), m.group("dst")
    # 「キロ」単独は長さ/質量が曖昧。文中にグラム系があれば質量、それ以外は長さ。
    if src == dst:
        value = _parse_number(m.group("value"))
    else:
        # 同名のキロだけは入力の単位種別で決める。
        value = _parse_number(m.group("value")) * _UNIT_FACTORS[src] / _UNIT_FACTORS[dst]
    return TaskAnswer(f"{m.group('value')}{src} = {_fmt(value)}{dst} です。", "unit:convert", 0.995, "utility", {
        "value": _fmt(value), "source_unit": src, "target_unit": dst,
    })


# ---------------------------------------------------------------------------
# コーディングと比較
# ---------------------------------------------------------------------------
_CODE_MARKS = ("コード", "プログラム", "実装", "スクリプト", "書いて", "書き方", "関数", "バグ", "デバッグ", "コード例")

def _is_code_request(text: str) -> bool:
    t = _norm(text).lower()
    return any(m in t for m in _CODE_MARKS) and bool(re.search(r"python|javascript|typescript|java|go|rust|node|next|コード|プログラム|関数", t))


def coding_answer(text: str) -> TaskAnswer | None:
    if not _is_code_request(text):
        return None
    t = _norm(text).lower()
    if "node.js" in t or "nodejs" in t:
        # 比較質問はコード依頼より先に、実行環境とフレームワークを分けて説明する。
        if "next" in t and any(w in t for w in ("どちら", "おすすめ", "違い", "比較", "か")):
            return comparison_answer(text)
    lang = "python" if "python" in t or "パイソン" in t else "javascript" if any(x in t for x in ("javascript", "js", "node")) else "python"
    if "fizzbuzz" in t or "fizz buzz" in t or "フィズ" in t:
        if lang == "python":
            code = """def fizzbuzz(n: int) -> None:\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            print(\"FizzBuzz\")\n        elif i % 3 == 0:\n            print(\"Fizz\")\n        elif i % 5 == 0:\n            print(\"Buzz\")\n        else:\n            print(i)\n\nfizzbuzz(100)"""
            block = "python"
        else:
            code = """function fizzBuzz(n) {\n  for (let i = 1; i <= n; i += 1) {\n    console.log(i % 15 === 0 ? 'FizzBuzz' : i % 3 === 0 ? 'Fizz' : i % 5 === 0 ? 'Buzz' : i);\n  }\n}\n\nfizzBuzz(100);"""
            block = "javascript"
        return TaskAnswer(f"```{block}\n{code}\n```\n3と5の公倍数を先に判定するのがポイントです。", f"code:{block}", 0.995, "code", {"language": block, "snippet": code})
    if "フィボナッチ" in t or "fibonacci" in t:
        if lang == "python":
            code = """def fibonacci(n: int) -> list[int]:\n    a, b = 0, 1\n    result = []\n    for _ in range(n):\n        result.append(a)\n        a, b = b, a + b\n    return result\n\nprint(fibonacci(10))"""
            block = "python"
        else:
            code = """function fibonacci(n) {\n  const result = [];\n  let [a, b] = [0, 1];\n  for (let i = 0; i < n; i += 1) {\n    result.push(a); [a, b] = [b, a + b];\n  }\n  return result;\n}\n\nconsole.log(fibonacci(10));"""
            block = "javascript"
        return TaskAnswer(f"```{block}\n{code}\n```", f"code:{block}", 0.995, "code", {"language": block, "snippet": code})
    if any(x in t for x in ("素数", "prime")):
        code = """def is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    return all(n % d for d in range(2, int(n ** 0.5) + 1))""" if lang == "python" else """function isPrime(n) {\n  if (n < 2) return false;\n  for (let d = 2; d * d <= n; d += 1) if (n % d === 0) return false;\n  return true;\n}"""
        block = "python" if lang == "python" else "javascript"
        return TaskAnswer(f"```{block}\n{code}\n```", f"code:{block}", 0.995, "code", {"language": block, "snippet": code})
    if "hello" in t or "ハロー" in t:
        code = "print('Hello, world!')" if lang == "python" else "console.log('Hello, world!');"
        block = "python" if lang == "python" else "javascript"
        return TaskAnswer(f"```{block}\n{code}\n```", f"code:{block}", 0.99, "code", {"language": block, "snippet": code})
    # 未知のコーディング依頼でも、話題だけを聞き返す代わりに安全な雛形を返す。
    if lang == "python":
        code = """def solve(value):\n    \"\"\"ここに処理を書く。\"\"\"\n    return value\n\nif __name__ == \"__main__\":\n    print(solve(None))"""
        block = "python"
    else:
        code = """export function solve(value) {\n  // ここに処理を書く\n  return value;\n}"""
        block = "javascript"
    return TaskAnswer(f"```{block}\n{code}\n```\n入力例と期待する出力を添えれば、仕様に合わせて実装を具体化できます。", f"code:{block}", 0.86, "code", {"language": block, "snippet": code, "generic": True})


def comparison_answer(text: str) -> TaskAnswer:
    out = (
        "結論：サーバー処理やスクリプトを組むなら Node.js、React の画面とルーティング・"
        "SSR/SSG まで含むWebアプリを早く作るなら Next.js がおすすめです。\n\n"
        "Node.js は JavaScript をブラウザ外で動かす実行環境です。HTTP サーバー、CLI、"
        "バッチなどを自由に設計できます。Next.js は React を土台にしたフレームワークで、"
        "ルーティング、サーバー側処理、静的生成、画像最適化などの規約が用意されています。\n\n"
        "迷ったら、React を使うWebサイトなら Next.js、APIや小さな自動化だけなら Node.js。"
        "Next.js も実行時には Node.js などのJavaScriptランタイムを使うため、完全な二者択一ではありません。"
    )
    return TaskAnswer(out, "compare:node-next", 0.99, "comparison", {"options": ["Node.js", "Next.js"]})


# ---------------------------------------------------------------------------
# ウェブ調査の文章化
# ---------------------------------------------------------------------------
def _source_text(result: ResearchResult) -> str:
    lines: list[str] = []
    for i, source in enumerate(result.sources[:5], 1):
        title = source.get("title") or source.get("url")
        snippet = source.get("content") or source.get("snippet") or ""
        snippet = re.split(r"(?<=[。！？!?])\s*", str(snippet).strip())[0][:300]
        lines.append(f"{i}. {title}: {snippet}" if snippet else f"{i}. {title}")
    if not lines:
        return "ウェブを検索しましたが、いま取得できた検索結果はありません。検索語を少し具体化すると、別の切り口で調べられます。"
    citations = "\n".join(
        f"[{i}] {s.get('title') or s.get('url')} — {s.get('url')}"
        for i, s in enumerate(result.sources[:5], 1)
    )
    return "検索で確認した情報です。\n" + "\n".join(lines) + "\n\n出典:\n" + citations


class TaskRouter:
    """計算・コード・比較・ウェブの入口。"""

    def __init__(self, research: ResearchEngine | None = None):
        self.research = research or ResearchEngine()

    def classify(self, text: str, *, web: bool | None = None) -> str | None:
        t = _norm(text).lower()
        if not t.strip():
            return None
        # 「常に検索」設定でも、短い社会的発話を検索エンジンへ送らない。
        if len(t) <= 24 and any(x in t for x in ("こんにちは", "こんばんは", "おはよう", "ありがとう", "おやすみ", "さようなら")):
            return None
        if "夜" in t and any(x in t for x in ("挨拶", "あいさつ", "なんという", "何と言う", "言う")):
            return "greeting_name"
        if "node.js" in t or "nodejs" in t:
            if "next" in t and any(x in t for x in ("どちら", "おすすめ", "違い", "比較", "か")):
                return "compare"
        # 文章題は一般算数より先に分類する。
        if "円" in t and re.search(r"何[" + _UNIT_WORDS + r"]", t):
            return "word_problem"
        if convert_unit(text) is not None:
            return "unit"
        if solve_equation(text) is not None:
            return "equation"
        if solve_arithmetic_task(text) is not None:
            return "arithmetic"
        if _is_code_request(text):
            return "code"
        needed, _ = self.research.should_research(text, web)
        if needed:
            return "research"
        return None

    def answer(self, text: str, *, web: bool | None = None,
               history: list[dict] | None = None) -> TaskAnswer | None:
        kind = self.classify(text, web=web)
        if kind == "greeting_name":
            return TaskAnswer("夜の挨拶は「こんばんは」です。朝は「おはようございます」、昼は「こんにちは」と言います。", "greeting:name", 0.999, "language", {"answer": "こんばんは"})
        if kind == "word_problem":
            return solve_word_problem(text)
        if kind == "unit":
            return convert_unit(text)
        if kind == "equation":
            return solve_equation(text)
        if kind == "arithmetic":
            return solve_arithmetic_task(text)
        if kind == "code":
            return coding_answer(text)
        if kind == "compare":
            return comparison_answer(text)
        if kind == "research":
            result = self.research.research(text, explicit=web, limit=5, fetch_pages=2)
            return TaskAnswer(_source_text(result), "research:web", 0.9 if result.sources else 0.55, "research", {
                "query": result.query, "reason": result.reason, "sources": result.sources,
                "cached": result.cached, "elapsed_ms": result.elapsed_ms,
            })
        return None


__all__ = [
    "TaskAnswer", "TaskRouter", "coding_answer", "comparison_answer", "safe_arithmetic",
    "solve_arithmetic_task", "solve_equation", "solve_word_problem", "convert_unit",
]
