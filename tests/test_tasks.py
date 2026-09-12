"""計算・コード・比較を知識ベースより先に解くテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher.composer import Composer  # noqa: E402
from snipher.tasks import (  # noqa: E402
    TaskRouter,
    safe_arithmetic,
    solve_equation,
    solve_word_problem,
)


def test_safe_arithmetic_does_not_execute_python():
    assert safe_arithmetic("20 + 3 * 4") == 32
    assert safe_arithmetic("(10 - 2) / 4") == 2
    try:
        safe_arithmetic("__import__('os').system('echo bad')")
    except ValueError:
        pass
    else:  # pragma: no cover - a regression would make this test fail
        raise AssertionError("arbitrary Python must not be evaluated")


def test_japanese_word_problem_is_solved_with_derivation():
    query = "1本80円の鉛筆を何本か買い、1個120円の消しゴムを3個買うと600円になった。鉛筆は何本買ったでしょうか。"
    answer = solve_word_problem(query)
    assert answer is not None
    assert answer.metadata["answer"] == "3"
    assert "80x" in answer.text
    assert "答えは3本" in answer.text


def test_unit_conversion_is_exact():
    answer = TaskRouter().answer("3キロは何メートル", web=False)
    assert answer is not None
    assert "3000メートル" in answer.text


def test_linear_and_quadratic_equations():
    linear = solve_equation("3x + 2 = 11を解いて")
    quadratic = solve_equation("2次方程式 x^2 - 5x + 6 = 0 を解いて")
    assert linear is not None and "x = 3" in linear.text
    assert quadratic is not None and quadratic.metadata["roots"] == ["3", "2"]


def test_common_coding_request_is_not_a_book_topic():
    import ast

    answer = TaskRouter().answer("PythonでFizzBuzzを書いて", web=False)
    assert answer is not None
    assert answer.kind == "code"
    assert "def fizzbuzz" in answer.text
    assert "```python" in answer.text
    snippet = answer.metadata["snippet"]
    ast.parse(snippet)


def test_node_and_next_are_compared_by_role_not_keyword_lookup():
    answer = TaskRouter().answer("Node.jsとNext.js、どちらがおすすめ？", web=False)
    assert answer is not None
    assert answer.kind == "comparison"
    assert "実行環境" in answer.text
    assert "フレームワーク" in answer.text


def test_night_greeting_is_answered_directly():
    answer = Composer().compose("夜の挨拶をなんという？", web=False)
    assert answer.plan == "greeting:name"
    assert "こんばんは" in answer.text
    assert answer.notes["authoritative"] is True
