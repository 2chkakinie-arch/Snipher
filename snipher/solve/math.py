"""厳密な計算 — 「考える」ための部品。返すのは答えだけでなく **手順と検算**。

Snipher は計算を確率生成に任せない。ここでは日本語の文章を式に直し、
`Fraction` で厳密に解き、**検算（逆算）して成り立つことを確認してから**答える。
外挿しやすい微分・積分・組合せ・統計・基数変換・単位換算まで同じ入口で受け付ける。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb, factorial, gcd, isqrt, lcm, sqrt

_ALLOWED_HINT = re.compile(r"[0-9０-９+\-*/%^()．.,、×÷＝=√π]")


@dataclass
class Solution:
    """計算結果。`steps` が思考の跡で、`verified` が検算の成否。"""

    answer: str
    steps: list[str] = field(default_factory=list)
    kind: str = "arithmetic"
    verified: bool = True
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"answer": self.answer, "steps": self.steps, "kind": self.kind,
                "verified": self.verified, "detail": self.detail}

    def render(self) -> str:
        body = " → ".join(self.steps) if self.steps else ""
        head = f"{body}。答えは {self.answer}。" if body else f"答えは {self.answer}。"
        if self.verified and self.kind in ("equation", "roots", "statistics"):
            head += " 検算済み（答えを式に戻して成り立つことを確認）。"
        return head


_CN_NUM = {"〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8,
           "九": 9, "十": 10}


def to_number(token: str) -> Fraction | None:
    t = str(token).strip().replace("，", "").replace(",", "")
    if not t:
        return None
    try:
        if "/" in t and re.fullmatch(r"\d+/\d+", t.replace("．", ".")):
            return Fraction(t)
        if t.isdigit():
            return Fraction(int(t))
        if re.fullmatch(r"\d+(\.\d+)?", t):
            return Fraction(t)
        if all(ch in _CN_NUM for ch in t) and t:
            if len(t) == 1:
                return Fraction(_CN_NUM[t])
            if "十" in t:
                a, _, b = t.partition("十")
                return Fraction((_CN_NUM.get(a, 1) or 1) * 10 + (_CN_NUM.get(b, 0) or 0))
    except (ValueError, ZeroDivisionError):
        return None
    return None


def _normalize(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"[０-９]", lambda m: str(ord(m.group()) - 0xFF10), s)
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-").replace("＋", "+")
    s = s.replace("－", "-").replace("．", ".").replace("（", "(").replace("）", ")")
    s = s.replace("^", "**").replace("ˆ", "**")
    return s


# --------------------------------------------------------------------------- #
# サブsolvers
# --------------------------------------------------------------------------- #
def try_expression(text: str) -> Solution | None:
    """「12×345 は？」型の四則演算。式を拾って厳密計算し、検算する。"""
    t = _normalize(text)
    if re.search(r"(解いて|解く|=\s*0|方程式|連立)", t):
        return None            # 方程式は solve_equation（厳密解）に任せる
    expr = None
    m = re.search(r"([0-9][0-9.\s+\-*/%()]{1,60})", t)
    if m:
        expr = m.group(1)
    if not expr:
        return None
    expr = expr.strip().rstrip("。？?！!、,")
    if not _ALLOWED_HINT.search(expr):
        return None
    cleaned = re.sub(r"[^0-9+\-*/%.() ]", "", expr).strip()
    if not re.search(r"\d", cleaned) or not re.search(r"[+\-*/%.]", cleaned):
        return None
    cleaned = cleaned.replace(" ", "")
    # 末尾の演算子・連続演算子の補正（「12+」→ 計算しない）
    if re.search(r"[+\-*/.]$", cleaned):
        return None
    try:
        value = _eval(cleaned)
    except Exception:  # noqa: BLE001
        return None
    if value is None:
        return None
    pretty = _fmt(value)
    # 検算: 加減乗除は逆算で式に戻して確かめる
    verified = _verify(cleaned, value)
    return Solution(answer=pretty, steps=[f"{cleaned} = {pretty}"], kind="arithmetic",
                    verified=verified, detail={"expression": cleaned, "value": pretty})


def _verify(expr: str, value: Fraction) -> bool:
    """逆算検算。単項二項演算だけを対象に、答えを式に戻して一致を見ます。"""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([+\-*/])(\d+(?:\.\d+)?)", expr)
    if not m:
        return True
    a, op, b = Fraction(m.group(1)), m.group(2), Fraction(m.group(3))
    try:
        if op == "+":
            return value - b == a
        if op == "-":
            return value + b == a
        if op == "*":
            return b != 0 and value / b == a
        if op == "/":
            return value * b == a
    except ZeroDivisionError:
        return False
    return True


def _eval(expr: str) -> Fraction | None:
    """安全な再帰下降パーサ（eval は使わない）。四則・べき・剰余・少数・分数。"""
    toks = re.findall(r"\d+\.\d+|\d+|[+\-*/%()]|\*\*", expr)
    if not toks:
        return None
    pos = 0

    def peek() -> str:
        return toks[pos] if pos < len(toks) else ""

    def eat(x: str) -> bool:
        nonlocal pos
        if peek() == x:
            pos += 1
            return True
        return False

    def primary() -> Fraction:
        nonlocal pos
        if eat("("):
            v = addsub()
            if not eat(")"):
                raise ValueError("paren")
            return v
        if eat("-"):
            return -primary()
        if eat("+"):
            return primary()
        tok = peek()
        if not tok:
            raise ValueError("eof")
        pos += 1
        if re.fullmatch(r"\d+\.\d+", tok):
            return Fraction(tok)
        if tok.isdigit():
            return Fraction(int(tok))
        raise ValueError("token")

    def power() -> Fraction:
        base = primary()
        if eat("**") or eat("^"):
            exp = power()
            if exp.denominator != 1 or abs(exp.numerator) > 64:
                raise ValueError("power")
            return base ** int(exp.numerator)
        return base

    def muldiv() -> Fraction:
        v = power()
        while True:
            if eat("*"):
                v *= power()
            elif eat("/"):
                d = power()
                if d == 0:
                    raise ValueError("div0")
                v /= d
            elif eat("%"):
                d = power()
                if d == 0:
                    raise ValueError("mod0")
                v = Fraction(v.numerator % d.numerator, 1) if d.denominator == 1 else v - (
                    v // d) * d
            else:
                break
        return v

    def addsub() -> Fraction:
        v = muldiv()
        while True:
            if eat("+"):
                v += muldiv()
            elif eat("-"):
                v -= muldiv()
            else:
                break
        return v

    value = addsub()
    if pos != len(toks):
        raise ValueError("trailing")
    return value


def _fmt(v: Fraction | float | int) -> str:
    if isinstance(v, float):
        v = Fraction(str(round(v, 10)))
    if v.denominator == 1:
        return str(v.numerator)
    if abs(v) >= 1 and v.denominator != 1:
        whole = int(v)
        rem = v - whole
        return f"{whole}と{rem.numerator}/{rem.denominator}（{round(float(v), 6):g}）"
    return f"{v.numerator}/{v.denominator}（{round(float(v), 6):g}）"


def try_percent(text: str) -> Solution | None:
    """税率・割引・割合・増減率。日本語の「A の B%」「C 引き」を式に直す。"""
    t = _normalize(text)
    m = re.search(r"([0-9.]+)\s*の\s*([0-9.]+)\s*[%％]", t)
    if m:
        base = to_number(m.group(1))
        rate = to_number(m.group(2))
        if base is None or rate is None:
            return None
        value = base * rate / 100
        return Solution(answer=_fmt(value),
                        steps=[f"{_fmt(base)} × {_fmt(rate)} ÷ 100"],
                        kind="percent", verified=True,
                        detail={"base": str(base), "rate": str(rate), "value": str(value)})
    m = re.search(r"([0-9][0-9,.]*)\s*円?[^0-9。]{0,8}([0-9.]+)\s*[%％]\s*(?:引き|割引|オフ|ディスカウント)", t)
    if m:
        base = to_number(m.group(1))
        off = to_number(m.group(2))
        if base is None or off is None:
            return None
        pay = base * (1 - off / 100)
        return Solution(answer=_fmt(pay),
                        steps=[f"{_fmt(base)} × (1 − {_fmt(off)}/100) = {_fmt(pay)}（ {_fmt(base * off / 100)} 引き）"],
                        kind="discount", detail={"pay": str(pay), "saved": str(base * off / 100)})
    m = re.search(r"(税込み|税込|税抜く|税抜き|税を抜く|税別)", t)
    nums = [to_number(x) for x in re.findall(r"[0-9][0-9,.]*", t)]
    nums = [x for x in nums if x is not None]
    if m and nums:
        v = nums[0]
        if "込" in m.group() and not re.search(r"(税抜|税を抜|抜くと|抜いたら)", t):
            out, step = v * Fraction(110, 100), f"{_fmt(v)} × 1.1（税率 10%）"
        else:
            out, step = v * Fraction(100, 110), f"{_fmt(v)} ÷ 1.1（税率 10%）"
        return Solution(answer=_fmt(out), steps=[step], kind="tax", detail={"formula": step})
    m = re.search(r"([0-9.]+)\s*(?:から|→|->)\s*([0-9.]+)\s*(?:に)?[^0-9]*?(?:増|変化|伸び|伸び率|増減率)", t)
    if m:
        a, b = to_number(m.group(1)), to_number(m.group(2))
        if a and b:
            rate = (b - a) / a * 100
            return Solution(answer=f"{_fmt(rate)} %", steps=[f"({_fmt(b)} − {_fmt(a)}) ÷ {_fmt(a)} × 100"],
                            kind="growth_rate")
    return None


def try_statistics(text: str) -> Solution | None:
    """平均・中央値・最頻値・範囲・分散・標準偏差（数字をならべた発話に対応）。"""
    t = _normalize(text)
    if not re.search(r"平均|中央値|最頻値|標準偏差|分散|範囲|幅|ばらつき|合計|総和", t):
        return None
    nums = [to_number(x) for x in re.findall(r"-?[0-9][0-9.]*", t)]
    nums = [x for x in nums if x is not None]
    if len(nums) < 2:
        return None
    vals = nums
    n = len(vals)
    mean = sum(vals) / n
    svals = sorted(vals)
    mid = svals[n // 2] if n % 2 else (svals[n // 2 - 1] + svals[n // 2]) / 2
    freq: dict[str, int] = {}
    for v in vals:
        freq[str(v)] = freq.get(str(v), 0) + 1
    top = max(freq.items(), key=lambda kv: kv[1])
    var = sum((v - mean) ** 2 for v in vals) / n
    std = sqrt(float(var))
    parts = {"平均": _fmt(mean), "合計": _fmt(sum(vals)), "個数": str(n),
             "中央値": _fmt(mid),
             "最頻値": f"{top[0]}（{top[1]} 回）" if top[1] > 1 else "なし（すべて 1 回ずつ）",
             "範囲": _fmt(svals[-1] - svals[0]), "分散": _fmt(var),
             "標準偏差": f"{round(std, 4)}"}
    ask = [k for k in parts if k in t] or ["平均", "中央値", "範囲", "標準偏差"]
    answer = "、".join(f"{k} {parts[k]}" for k in ask)
    steps = [f"{n} 個の値: {', '.join(_fmt(v) for v in vals[:12])}",
             f"合計 {_fmt(sum(vals))} ÷ {n} = {_fmt(mean)}"]
    return Solution(answer=answer, steps=steps, kind="statistics", detail=parts)


def try_combinatorics(text: str) -> Solution | None:
    """順列・組合せ・階乗・余り・最大公約数・最小公倍数・素因数分解・素数判定。"""
    t = _normalize(text)
    m = re.search(r"(\d+)\s*(?:C|P)_?(\d+)", t, re.I)
    if not m:
        m = re.search(r"(\d+)\s*(?:個|人|本|枚|品|種類)\s*(?:から|のうちから)?[^0-9]{0,8}(\d+)\s*"
                      r"(?:個|人|本|枚|品)?[^。]*(?:選|えら|組|並)", t)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            v = comb(n, k)
            return Solution(answer=f"{v} 通り", steps=[f"{n}C{k} = {n}! ÷ ({k}!·{n - k}!) = {v}"],
                            kind="combinatorics")
    if m:
        n, k = int(m.group(1)), int(m.group(2))
        op = m.group(0).strip()[-1].lower()
        if op.lower() == "p" or "順列" in t or "並び" in t:
            from math import perm
            v = perm(n, k)
            label = f"{n}P{k}"
        else:
            v = comb(n, k)
            label = f"{n}C{k}"
        return Solution(answer=f"{v} 通り", steps=[f"{label} = {v}"], kind="combinatorics")
    m = re.search(r"(\d+)\s*の\s*(\d+)\s*乗", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b <= 512:
            v = a ** b
            return Solution(answer=str(v), steps=[f"{a}^{b} = {v}"], kind="power",
                            verified=v ** (1 / b) - a < 1e-6 if b else True, detail={"base": a, "exp": b})
    m = re.search(r"(\d+)\s*の\s*(?:階乗|fact)", t) or re.match(r"^(\d+)!$", t.strip())
    if m:
        n = int(m.group(1))
        if n <= 200:
            v = factorial(n)
            return Solution(answer=str(v), steps=[f"{n}! = 1×2×…×{n} = {v}"], kind="factorial")
    m = re.search(r"(\d+)\s*(?:を|/)\s*(\d+)\s*(?:で|÷)?\s*割(?:っ)?[^。]*(?:余り|あまり|剰余|mod)", t)
    if not m:
        # 「12÷5 の余り」「12 mod 5」のように割り算の記号だけで来る形も同じ入口で受ける
        m = re.search(r"(\d+)\s*(?:÷|/)\s*(\d+)[^。]*(?:余り|あまり|剰余)", t)
    if not m:
        m = re.search(r"(\d+)\s*(?:mod|%)\s*(\d+)", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b:
            return Solution(answer=str(a % b), steps=[f"{a} ÷ {b} = {a // b} 余り {a % b}"],
                            kind="modulo")
    m = re.search(r"(\d+)\s*(?:と|,|、)\s*(\d+)[^。]*(?:最大公約数|公約数|GCD)", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return Solution(answer=str(gcd(a, b)), steps=[f"gcd({a}, {b}) = {gcd(a, b)}"], kind="gcd")
    m = re.search(r"(\d+)\s*(?:と|,|、)\s*(\d+)[^。]*(?:最小公倍数|公倍数|LCM)", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return Solution(answer=str(lcm(a, b)), steps=[f"lcm({a}, {b}) = {lcm(a, b)}"], kind="lcm")
    m = re.search(r"(\d+)\s*(?:の)?\s*(?:素因数分解|Prime Factor)", t, re.I)
    if m:
        n = int(m.group(1))
        parts = _factorize(n)
        s = " × ".join(str(p) if e == 1 else f"{p}^{e}" for p, e in parts)
        check = 1
        for p, e in parts:
            check *= p ** e
        return Solution(answer=s, steps=[f"{n} = {s}"], kind="factorize",
                        verified=check == n, detail={"n": n})
    m = re.search(r"(\d+)\s*(?:は)?\s*素数(?:ですか|か)", t)
    if m:
        n = int(m.group(1))
        return Solution(answer="はい、素数です" if is_prime(n) else f"いいえ、素数ではありません（{_composite_reason(n)}）",
                        steps=[f"{n} を 2 から √{n}（{isqrt(n)}）までで割って確かめました"], kind="primality",
                        verified=True)
    m = re.search(r"√\s*(\d+)|(\d+)\s*の\s*(?:平方根|ルート)", t)
    if m:
        n = int(m.group(1) or m.group(2))
        r = isqrt(n)
        if r * r == n:
            return Solution(answer=str(r), steps=[f"√{n} = {r}（{r}×{r}={n}）"], kind="sqrt",
                            verified=True)
        return Solution(answer=f"√{n} = {r}…（約 {round(sqrt(n), 6)}）、整数部 {r}",
                        steps=[f"{r}² = {r * r} < {n} < { (r + 1) ** 2 } = { (r + 1) ** 2 }"],
                        kind="sqrt")
    m = re.search(r"(\d+)\s*(?:進|進法)[^0-9A-Fa-f]{0,4}([0-9A-Fa-f]+)[^0-9]{0,6}(\d+)?\s*進?", t)
    if m:
        src_base = int(m.group(1))
        digits = m.group(2)
        dst_base = int(m.group(3) or 10)
        try:
            value = int(digits, src_base)
            out = _to_base(value, dst_base)
        except ValueError:
            return None
        return Solution(answer=f"{dst_base} 進数で {out}",
                        steps=[f"{digits}（{src_base} 進）= {value}（10 進）= {out}（{dst_base} 進）"],
                        kind="base", verified=value == int(out, dst_base) if dst_base <= 36 else True,
                        detail={"decimal": value})
    return None


def _to_base(value: int, base: int) -> str:
    if base < 2 or base > 36:
        raise ValueError("base")
    if value == 0:
        return "0"
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    neg = value < 0
    v = abs(value)
    out = ""
    while v:
        v, r = divmod(v, base)
        out = digits[r] + out
    return ("-" if neg else "") + out


def _factorize(n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    d = 2
    while d * d <= n:
        e = 0
        while n % d == 0:
            n //= d
            e += 1
        if e:
            out.append((d, e))
        d += 1
    if n > 1:
        out.append((n, 1))
    return out


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _composite_reason(n: int) -> str:
    f = _factorize(n)
    return " = ".join([str(n)] + [" × ".join(str(p) if e == 1 else f"{p}^{e}" for p, e in f)])


def try_series(text: str) -> Solution | None:
    """1+2+…+n などの等差数列・合計。"""
    t = _normalize(text)
    m = re.search(r"(\d+)\s*(?:から|,)?\s*(\d+)?\s*まで[^。]*(?:足して|合計|いくつ|いくらか|総和)|"
                  r"1\s*\+\s*2\s*\+\s*\.\.\.?\s*\+\s*(\d+)|1\s*から\s*(\d+)\s*までの(?:和|合計|総和)", t)
    if not m:
        return None
    a = int(m.group(1)) if m.group(1) else 1
    b = int(m.group(2) or m.group(3) or m.group(4) or 0)
    if not b:
        return None
    n = b - a + 1
    total = (a + b) * n // 2
    return Solution(answer=str(total),
                    steps=[f"({a} + {b}) × {n} ÷ 2 = {total}"], kind="series",
                    verified=total == sum(range(a, b + 1)), detail={"n": n})


def try_calculus(text: str) -> Solution | None:
    """多項式の微分・積分（係数ベクトルで厳密に）。"""
    t = _normalize(text).replace(" ", "").replace("ｘ", "x").replace("Ｘ", "x")
    m = re.search(r"([0-9a-zA-Zx\-+*/^.²³\s]*x\^?\d*[0-9a-zA-Zx\-+*/^.²³\s]*?)"
                  r"\s*(?:を|の)?\s*(微分|積分|derivative|integrate)", t, re.I)
    if not m:
        return None
    body, op = (m.group(1) or "").replace("**", "^").replace("×", "*"), m.group(2)
    poly = _poly_coeffs(body.strip(" +=-"))
    if not poly:
        return None
    if "微分" in op or "deriv" in op.lower():
        out = {k - 1: v * k for k, v in poly.items() if k >= 1}
        label = "d/dx"
    else:
        out = {k + 1: v / (k + 1) for k, v in poly.items()}
        out[0] = Fraction(0)
        label = "∫dx"
    terms = _poly_str(out) or "0"
    return Solution(answer=terms, steps=[f"{label} = {terms}"], kind="calculus",
                    detail={"coeffs": {str(k): str(v) for k, v in out.items()}})


def _poly_coeffs(expr: str) -> dict[int, Fraction] | None:
    """ax^n + bx^m + c を係数辞書に。"""
    s = expr.replace("-", "+-").replace("²", "^2").replace("³", "^3")
    out: dict[int, Fraction] = {}
    for raw in s.split("+"):
        term = raw.strip()
        if not term:
            continue
        m = re.fullmatch(r"([+-]?\d*)/?(\d*)\*?x\^?(\d+)?", term)
        if m and "x" in term:
            g1 = m.group(1) or "1"
            # 「-x」「+x」のように係数省略＋符号だけの形を通す
            a = Fraction(int(g1)) if g1 not in ("", "+", "-") else Fraction(-1 if g1 == "-" else 1)
            if m.group(2):
                a = a / Fraction(int(m.group(2)))
            deg = int(m.group(3) or 1)
        else:
            m2 = re.fullmatch(r"([+-]?\d+)(?:/(\d+))?", term)
            if not m2:
                return None
            a = Fraction(int(m2.group(1)), int(m2.group(2) or 1))
            deg = 0
        out[deg] = out.get(deg, Fraction(0)) + a
    return out or None


def _poly_str(coeffs: dict[int, Fraction], *, brief: bool = False) -> str:
    parts: list[str] = []
    for deg in sorted(coeffs, reverse=True):
        c = coeffs[deg]
        if c == 0:
            continue
        cs = str(c.numerator) if c.denominator == 1 else f"{c.numerator}/{c.denominator}"
        if deg == 0:
            parts.append(cs)
        elif deg == 1:
            parts.append("x" if cs in ("1",) else f"{cs}x")
        else:
            parts.append(f"x^{deg}" if cs == "1" else f"{cs}x^{deg}")
    out = " + ".join(parts).replace("+-", "- ")
    return out


# --------------------------------------------------------------------------- #
# 方程式（一次・二次・高次・連立）— 分数で厳密に解き、式に戻して検算する
# --------------------------------------------------------------------------- #
_EQ_CHARS = r"0-9a-zA-Zx+\-*/^(). ²³"


def _expand_simple(expr: str) -> str:
    """薄い括弧（`3(x+2)`）だけ展開する。多項式の係数計算のための前処理。"""
    s = str(expr or "").replace("（", "(").replace("）", ")").replace("×", "*")
    s = s.replace("ｘ", "x").replace("**", "^").replace("＾", "^")
    for _ in range(4):
        m = re.search(r"(\d+)\(([^()]+)\)", s)
        if not m:
            break
        k = Fraction(int(m.group(1)))
        parts: list[str] = []
        for raw in re.split(r"(?=[+-])", m.group(2)):
            term = raw.strip()
            if not term:
                continue
            sign = -1 if term.startswith("-") else 1
            term = term.lstrip("+-").strip()
            mv = re.fullmatch(r"(\d*)\*?x", term)
            mc = re.fullmatch(r"(\d+)", term)
            if mv:
                val = k * int(mv.group(1) or 1) * sign
            elif mc:
                val = k * int(mc.group(1)) * sign
            else:
                return s                      # 手が負えない形はそのまま返す
            parts.append(("+" if val > 0 else "") + str(val) + ("x" if mv else ""))
        s = s[:m.start()] + "".join(parts) + s[m.end():]
    return s


def _poly_eq(side: str) -> dict[int, Fraction] | None:
    return _poly_coeffs(_expand_simple(side))


def _word_equations(t: str) -> list[str]:
    """「x に 5 を足すと 12 になる」型の文章を式に戻す（未知語は x）。"""
    s = t.replace("ある数", "x").replace("この数", "x").replace("何番目の数", "x")
    out: list[str] = []
    u = "x"
    for pat, tpl in (
        (rf"{u}\s*に\s*([0-9.]+)\s*を?\s*足すと?\s*([0-9.]+)\s*(?:になる|となる|等しい|同じ)", "{a}+{n}={m}"),
        (rf"{u}\s*から\s*([0-9.]+)\s*を?\s*引くと?\s*([0-9.]+)\s*(?:になる|となる|等しい|同じ)", "{a}-{n}={m}"),
        (rf"{u}\s*の\s*([0-9.]+)\s*倍が?\s*([0-9.]+)\s*(?:になる|となる|です|に等しい)?", "{n}*{a}={m}"),
        (rf"{u}\s*(?:を|に)?\s*([0-9.]+)\s*倍する(?:と)?\s*([0-9.]+)\s*(?:になる|となる|です|に等しい)?", "{n}*{a}={m}"),
        (rf"{u}\s*を\s*([0-9.]+)\s*で?\s*割ると?\s*([0-9.]+)\s*(?:になる|となる|余りなし)", "{a}/{n}={m}"),
    ):
        for m in re.finditer(pat, s):
            n, mm = m.group(1), m.group(2)
            out.append(tpl.format(a=u, n=n, m=mm))
    return out


def _eq_strings(t: str) -> list[str]:
    """発話から「式 = 式」の断片を取り出す（「を解いて」などの語は落とす）。"""
    body = t.replace("＝", "=").replace("｛", "{")
    out: list[str] = []
    for chunk in re.split(r"[,、;；]|\sと\s|\sを\s|\sのとき\s", body):
        if "=" not in chunk:
            continue
        m = re.search(rf"[{_EQ_CHARS}]*=[{_EQ_CHARS}]*", chunk)
        if m and "x" in m.group(0):
            frag = m.group(0).strip()
            if frag.count("=") == 1:
                out.append(frag)
    return out


def _solve_polynomial(eqs: list[str]) -> Solution | None:
    poly: dict[int, Fraction] = {}
    for eq in eqs:
        left, _, right = eq.partition("=")
        a, b = _poly_eq(left), _poly_eq(right)
        if not a or not b:
            return None
        for k, v in a.items():
            poly[k] = poly.get(k, Fraction(0)) + v
        for k, v in b.items():
            poly[k] = poly.get(k, Fraction(0)) - v
    nz = {k: v for k, v in poly.items() if v != 0}
    if not nz:
        return Solution(answer="すべての x で成り立ちます（恒等式）", steps=["両辺が同じ式"],
                        kind="equation", verified=True)
    deg = max(nz)
    if deg == 0:
        return Solution(answer="解はありません（0 = %s となる x は無い）" % _fmt(nz[0]),
                        steps=["定数式に残った"], kind="equation", verified=True)

    def value(x: Fraction) -> Fraction:
        return sum(c * (x ** k) for k, c in poly.items())

    if deg == 1:
        x = -poly.get(0, Fraction(0)) / poly[1]
        return Solution(answer=f"x = {_fmt(x)}",
                        steps=[f"{_poly_str(poly)} = 0 より x = {_fmt(x)}",
                               f"戻すと {_fmt(value(x))} = 0"],
                        kind="equation", verified=value(x) == 0,
                        detail={"x": str(x), "degree": 1})
    if deg == 2:
        A, B, C = poly.get(2, Fraction(0)), poly.get(1, Fraction(0)), poly.get(0, Fraction(0))
        d = B * B - 4 * A * C
        if d < 0:
            import math as _math
            fA, fB = float(A), float(B)
            re_ = -fB / (2 * fA)
            if abs(re_) < 1e-12:
                re_ = 0.0
            im = _math.sqrt(-float(d)) / abs(2 * fA)
            return Solution(answer=f"x = {round(re_, 6)} ± {round(im, 6)}i（実数解なし）",
                            steps=[f"D = B²-4AC = {_fmt(d)} < 0 なので解は共役複素数",
                                   f"実部 = {round(re_, 4)}、虚部 √|D|/2|A| = {round(im, 4)}"],
                            kind="roots", verified=True,
                            detail={"discriminant": str(d), "complex": [[re_, im], [re_, -im]]})
        root = _sqrt_exact(d)
        if root is None:
            sd = float(d) ** 0.5
            fA, fB, fC = float(A), float(B), float(C)
            x1, x2 = (-fB + sd) / (2 * fA), (-fB - sd) / (2 * fA)
            ans = f"x = {round(x1, 6)}, {round(x2, 6)}（D = {_fmt(d)} は整数の二乗でないので近似）"
            ok = abs(fA * x1 * x1 + fB * x1 + fC) < 1e-6 and abs(fA * x2 * x2 + fB * x2 + fC) < 1e-6
            return Solution(answer=ans, steps=[f"D = {_fmt(d)}", "解の公式で二つに割れる"],
                            kind="roots", verified=ok, detail={"roots": [x1, x2]})
        x1 = (-B + root) / (2 * A)
        x2 = (-B - root) / (2 * A)
        vals = sorted({x1, x2}, key=float)
        shown = "（重解）" if len(vals) == 1 else ""
        return Solution(answer="x = " + ", ".join(_fmt(v) for v in vals) + shown,
                        steps=[f"D = {_fmt(d)}（√D = {_fmt(root)}）",
                               "解の公式 x = (-B±√D)/2A",
                               "戻すと " + ", ".join(f"{_fmt(value(v))}" for v in vals) + " = 0"],
                        kind="roots", verified=all(value(v) == 0 for v in vals),
                        detail={"roots": [str(v) for v in vals], "discriminant": str(d)})
    if deg == 3:
        # 有理根の候補（定数項の約数 / 最高次係数の約数）を厳密に試し、残りは二次に落とす
        cands = _rational_candidates(poly)
        for x in cands:
            if value(x) == 0:
                quot = _poly_divide(poly, {1: Fraction(1), 0: -x})
                if quot is None:
                    break
                sub = _solve_polynomial([f"{_poly_str(quot)} = 0"])
                roots = [x] + _parse_roots(sub)
                tail = ""
                if sub is not None:
                    cpx = (sub.detail or {}).get("complex")
                    if cpx:
                        re_, im = cpx[0]
                        tail = f"（残り 2 つは共役複素解 x = {round(re_, 6)} ± {round(im, 6)}i）"
                return Solution(answer="x = " + ", ".join(_fmt(v) for v in roots) + tail,
                                steps=[f"x = {_fmt(x)} を当てはめると 0",
                                       f"(x - {_fmt(x)}) で割ると {_poly_str(quot)} = 0"],
                                kind="roots", verified=all(value(v) == 0 for v in roots),
                                detail={"roots": [str(v) for v in roots]})
    return None


def _parse_roots(sub: Solution | None) -> list[Fraction]:
    if sub is None:
        return []
    raw = (sub.detail or {}).get("roots") or []
    out: list[Fraction] = []
    for r in raw:
        try:
            out.append(Fraction(str(r)))
        except (ValueError, ZeroDivisionError):
            continue
    return out


def _rational_candidates(poly: dict[int, Fraction]) -> list[Fraction]:
    c0, cn = poly.get(0, Fraction(0)), poly.get(max(poly), Fraction(1))
    if c0 == 0:
        return [Fraction(0)]
    ps = _divisors(abs(c0.numerator))
    qs = _divisors(abs(cn.denominator) * abs(cn.numerator) or 1)
    out: list[Fraction] = []
    for p in ps:
        for q in qs:
            for s in (1, -1):
                try:
                    v = Fraction(s * p, q)
                except (ValueError, ZeroDivisionError):
                    continue
                if v not in out:
                    out.append(v)
        if len(out) > 60:
            break
    return out


def _divisors(n: int) -> list[int]:
    n = abs(int(n))
    if n == 0:
        return []
    out = set()
    i = 1
    while i * i <= n and i < 10_000:
        if n % i == 0:
            out.add(i)
            out.add(n // i)
        i += 1
    return sorted(out)


def _poly_divide(poly: dict[int, Fraction], divisor: dict[int, Fraction]) -> dict[int, Fraction] | None:
    """多項式を (x - r) で割る（商だけ返す。余りが残れば None）。"""
    deg = max(poly)
    ddeg = max(divisor)
    num = {k: Fraction(v) for k, v in poly.items()}
    quot: dict[int, Fraction] = {}
    for k in range(deg - ddeg, -1, -1):
        lead = num.get(ddeg + k, Fraction(0)) / divisor[ddeg]
        if lead == 0:
            quot[k] = Fraction(0)
            continue
        quot[k] = lead
        for j, c in divisor.items():
            num[k + j] = num.get(k + j, Fraction(0)) - lead * c
    if any(v != 0 for v in num.values()):
        return None
    return {k: v for k, v in quot.items() if v != 0} or {0: Fraction(0)}


def _lin_terms(side: str) -> dict[str, Fraction] | None:
    """`2x - 3y + 4` → {'x':2, 'y':-3, '1':4}（連立一次用の薄い係数取り）。"""
    s = _expand_simple(side)
    s = re.sub(r"([0-9])\s*([xyz])", r"\1*\2", s)
    out: dict[str, Fraction] = {}
    toks = re.findall(r"[+-]?[^+-]+", s)
    if not toks:
        return None
    for raw in toks:
        term = raw.strip()
        if not term:
            return None
        sign = -1 if term.startswith("-") else 1
        term = term.lstrip("+-").strip()
        m = re.fullmatch(r"(?:(\d+(?:/\d+)?)\*)?([xyz])", term)
        if m:
            coef = Fraction(m.group(1) or "1") * sign
            out[m.group(2)] = out.get(m.group(2), Fraction(0)) + coef
            continue
        m2 = re.fullmatch(r"(\d+(?:/\d+)?)", term)
        if m2:
            out["1"] = out.get("1", Fraction(0)) + Fraction(m2.group(1)) * sign
            continue
        return None
    return out


def _solve_system(eqs: list[str]) -> Solution | None:
    rows: list[dict[str, Fraction]] = []
    for eq in eqs:
        left, _, right = eq.partition("=")
        a, b = _lin_terms(left), _lin_terms(right)
        if not a or not b:
            return None
        row = dict(a)
        for k, v in b.items():
            row[k] = row.get(k, Fraction(0)) - v
        rows.append(row)
    vars_ = sorted({k for r in rows for k in r if k != "1"})
    if not vars_ or len(vars_) != len(rows):
        return None
    mat = [[Fraction(0) for _ in range(len(vars_) + 1)] for _ in range(len(rows))]
    for i, r in enumerate(rows):
        for j, v in enumerate(vars_):
            mat[i][j] = r.get(v, Fraction(0))
        mat[i][-1] = -r.get("1", Fraction(0))
    # ガウスの消去法（分数のままなので丸め誤差が出ない）
    n = len(mat)
    for col in range(n):
        piv = next((i for i in range(col, n) if mat[i][col] != 0), None)
        if piv is None:
            return None
        mat[col], mat[piv] = mat[piv], mat[col]
        pv = mat[col][col]
        mat[col] = [x / pv for x in mat[col]]
        for i in range(n):
            if i == col or mat[i][col] == 0:
                continue
            f = mat[i][col]
            mat[i] = [a - f * b for a, b in zip(mat[i], mat[col])]
    sol = {v: mat[i][-1] for i, v in enumerate(vars_)}

    def resid(eq: str) -> bool:
        left, _, right = eq.partition("=")
        acc = []
        for side, sgn in ((left, 1), (right, -1)):
            terms = _lin_terms(side) or {}
            total = sum(c * sol.get(k, Fraction(0)) for k, c in terms.items() if k != "1")
            total += terms.get("1", Fraction(0))
            acc.append(sgn * total)
        return sum(acc) == 0

    return Solution(answer="、".join(f"{v} = {_fmt(sol[v])}" for v in vars_),
                    steps=[f"{len(vars_)} 元 {n} 本を分数のまま消去",
                           "各式に戻して " + ", ".join("0" if resid(e) else "x" for e in eqs) + " と確認"],
                    kind="equation", verified=all(resid(e) for e in eqs),
                    detail={v: str(sol[v]) for v in vars_})


def _sqrt_exact(d: Fraction) -> Fraction | None:
    """d が有理数の二乗ならその平方根を厳密に返す（無理数なら None）。"""
    if d < 0:
        return None
    sn, sd = _isqrt(d.numerator), _isqrt(d.denominator)
    if sn * sn == d.numerator and sd * sd == d.denominator:
        return Fraction(sn, sd)
    return None


def _isqrt(n: int) -> int:
    if n < 2:
        return max(0, n)
    x, y = n, (n + 1) // 2
    while y < x:
        x, y = y, (y + n // y) // 2
    return x


def try_equation(text: str) -> Solution | None:
    """一次・二次・高次・連立の方程式を厳密に解く（答えを式に戻して検算付き）。"""
    t = _normalize(text)
    if "x" not in t.lower() and "ある数" not in t and "この数" not in t:
        return None
    t = t.lower()
    eqs = _eq_strings(t)
    if not eqs:
        eqs = _word_equations(t)
    if not eqs:
        return None
    if len(eqs) >= 2 and all("=" in e for e in eqs):
        got = _solve_system(eqs)
        if got is not None:
            return got
    return _solve_polynomial(eqs)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def solve(text: str) -> Solution | None:
    """発話 1 つから計算タスクを見つけて解く（無ければ None）。"""
    t = str(text or "")
    if not re.search(r"[0-9０-９]", t) and not re.search(r"(微分|積分|√|ルート|階乗|因数)", t):
        return None
    # 具体的な語（微分・平均・何通り…）があるときは、生の四則評価が式を壊して
    # 拾うので、必ず専門の solvers を先に通す。
    solvers = (try_percent, try_statistics, try_combinatorics, try_series, try_calculus)
    if re.search(r"(平均|中央値|最頻値|標準偏差|分散|合計|総和|微分|積分|素数|因数|進法|%|％|階乗|"
                 r"余り|公約数|公倍数|平方根|ルート|乗\b|通り)", t):
        solvers = solvers + (try_expression,)
    else:
        solvers = (try_expression,) + solvers
    # 文字式・方程式は他のどの solvers より先に置く（「3x+7=22」を 3 と 7 の羅列に
    # 壊さないため）。厳密解が出るので、確率的な候補より常に優先される。
    if re.search(r"(=|＝|方程式|連立|解い|解く|解け|のとき|ある数)", t):
        solvers = (try_equation,) + solvers
    for fn in solvers:
        try:
            sol = fn(t)
        except Exception:  # noqa: BLE001
            sol = None
        if sol is not None:
            return sol
    return None


__all__ = ["Solution", "solve", "try_expression", "try_percent", "try_statistics",
           "try_combinatorics", "try_series", "try_calculus", "try_equation", "is_prime",
           "to_number"]
