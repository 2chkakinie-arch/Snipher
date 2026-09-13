"""解く層（snipher.solve）— 検算できる仕事は生成に聞かない。

算数・数学・文章・暦・コードの各ソルバは「答え 1 個」ではなく
*答え + 思考の跡（steps）+ 検算（verified）* を返す。verified=False のものを
応答に出さないことが、この project の「断言できないことは言わない」の実装。
"""

from __future__ import annotations

import pytest

from snipher import solve
from snipher.solve import code as C
from snipher.solve import facts as F
from snipher.solve import math as M
from snipher.solve import text as T


# --------------------------------------------------------------------------- #
# 数学
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q,answer", [
    ("3×7", "21"),
    ("12÷5 の余りはいくつ？", "2"),
    ("2の10乗", "1024"),
    ("5 の階乗", "120"),
])
def test_arithmetic_is_exact(q: str, answer: str) -> None:
    sol = M.solve(q)
    assert sol is not None and sol.verified
    assert answer in sol.answer


@pytest.mark.parametrize("eq,x", [
    ("x+3=7", "4"),
    ("3x+7=22 を解いて", "5"),
    ("2x+5=15", "5"),
    ("3(x+2)=21", "5"),
])
def test_linear_equations_are_solved_with_check(eq: str, x: str) -> None:
    sol = M.solve(eq)
    assert sol is not None and sol.verified, sol
    assert f"x = {x}" == sol.answer.split("（")[0]


def test_quadratic_roots_are_exact_when_rational() -> None:
    sol = M.solve("x^2-5x+6=0")
    assert sol is not None and sol.kind == "roots"
    assert sol.verified and "2" in sol.answer and "3" in sol.answer


def test_irreducible_quadratic_reports_complex_or_approx() -> None:
    sol = M.solve("x^2+1=0")
    assert sol is not None and sol.verified
    assert "i" in sol.answer                       # 虚数解まで言い切る


def test_cubic_factors_over_a_rational_root() -> None:
    sol = M.solve("x^3-8=0 を解いて")
    assert sol is not None and sol.verified
    assert "2" in sol.answer and "複素" in sol.answer


def test_simultaneous_linear_system_is_solved_exactly() -> None:
    sol = M.solve("x+y=10, x-y=2")
    assert sol is not None and sol.verified
    assert "x = 6" in sol.answer and "y = 4" in sol.answer


def test_word_problem_becomes_an_equation() -> None:
    sol = M.solve("ある数を 4 倍すると 52 になる。この数は？")
    assert sol is not None and sol.answer.endswith("13")


def test_percent_and_statistics() -> None:
    assert M.solve("240円の15%引き").answer.strip().endswith("204")
    st = M.solve("3, 5, 7, 9 の平均")
    assert st is not None and "6" in st.answer
    tot = M.solve("1,2,3,4,5 の合計")
    assert tot is not None and "15" in tot.answer


def test_combinatorics_and_base_conversion() -> None:
    assert "10" in M.solve("5C3").answer
    assert "11111111" in M.solve("10進数255を2進数に").answer


def test_calculus_on_polynomials() -> None:
    assert M.solve("2x^2+3x を微分して").answer.replace(" ", "") == "4x+3"


def test_verified_flag_reflects_a_real_check() -> None:
    # 検算は「答えを式に戻す」こと。割って壊れた式は verified にならない。
    sol = M._solve_polynomial(["x^2+2x+1 = 0"])
    assert sol is not None and sol.verified
    assert M._verify("2+2", M.Fraction(5)) is False


# --------------------------------------------------------------------------- #
# 暦・時計
# --------------------------------------------------------------------------- #
def test_now_answers_from_the_clock_not_the_knowledge_base() -> None:
    sol = F.handle("今は西暦何年？")
    assert sol is not None
    assert "20" in sol.answer                      # 年を含む
    assert sol.detail.get("year")                   # 機械が読める値も出す
    assert "曜日" in sol.detail.get("weekday", "")


def test_elapsed_and_age_are_computed() -> None:
    sol = F.handle("2020年1月1日から何日経った？")
    assert sol is not None and "日" in sol.answer
    age = F.try_age("1990年3月12日生まれの年齢は？")
    assert age is not None and age.answer.endswith("歳")


def test_clock_math() -> None:
    sol = F.try_clock_math("1時間45分は何分？")
    assert sol is not None and "105" in sol.answer


# --------------------------------------------------------------------------- #
# 文字・語彙
# --------------------------------------------------------------------------- #
def test_counts_and_case() -> None:
    sol = T.count_words("apple banana cherry")
    assert "3" in sol.answer
    up = T.case_convert("hello world を大文字に")
    assert up is not None and "HELLO WORLD" in up.answer


def test_transliteration_round_trip() -> None:
    sol = T.transliterate("りんご をローマ字に")
    assert sol is not None and "ringo" in sol.answer.replace("'", "")


def test_base64_and_hex_and_rot() -> None:
    sol = T.encode_decode("hello を base64 でエンコードして")
    assert sol is not None and "aGVsbG8" in sol.answer


def test_word_length_query_uses_the_wordbank() -> None:
    sol = T.word_length_query("5文字の英単語を教えて")
    assert sol is not None and sol.answer


def test_dictionary_query_answers_with_lexicon_facts() -> None:
    sol = T.dictionary_query("「猫」の読みと拍を教えて")
    assert sol is not None
    assert "ねこ" in sol.answer


# --------------------------------------------------------------------------- #
# コード（実際に走らせて検証する）
# --------------------------------------------------------------------------- #
def test_python_syntax_check_catches_errors() -> None:
    ok, _msg = C.syntax_check_python("def f(:\n  pass\n")
    assert ok is False
    assert C.syntax_check_python("def f():\n    return 1\n")[0] is True


def test_html_tag_balance_is_verified() -> None:
    assert C.tag_balance("<ul><li>a</li></ul>")[0] is True
    ok, why = C.tag_balance("<ul><li>a</ul>")
    assert ok is False and "li" in why                      # 開いた順に閉じていない
    assert C.tag_balance("<br><img src=x><hr>")[0] is True  # void 要素は数えない
    res = C.verify("<div><p>x</p></div>", "html")
    assert res.ok and any("タグ" in n for n in res.notes)


@pytest.mark.skipif(not C._allow_run(), reason="SNIPHER_CODE_RUN=0")
def test_generated_code_is_actually_run() -> None:
    res = C.verify("def is_prime(n):\n    return n > 1\nprint([x for x in range(2, 10) if is_prime(x)])\n",
                   "python")
    assert res.ran and res.ok
    assert "2" in res.stdout


def test_solve_handle_returns_authoritative_results() -> None:
    got = solve.handle("3×7は？")
    assert got is not None and got.authoritative
    assert got.solution.answer.strip() == "21"
    d = got.as_dict()
    assert d["kind"] == "math" and d["verified"] is True


def test_code_request_is_solved_with_execution_result() -> None:
    pair = solve.solve_code("Pythonで1から10までの素数を列挙するコードを書いて")
    if pair is None:                      # is_code_request が拾わない書き方ならスキップ
        pytest.skip("この入力はコード依頼と判定されませんでした")
    res, raw = pair
    assert "```" in res.solution.answer
    if raw.ran:
        assert "実行結果" in res.solution.answer
