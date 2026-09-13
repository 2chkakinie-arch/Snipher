"""Snipher の指示層 — 「指示文」と「材料」を分けて、指示された仕事を確実にこなす。

    from snipher.instruction import run
    got = run("次のテキストから情報を抽出し、JSON 形式のみで出力してください。…")
    got.text        # → {"origin": "東京", ...}（JSON だけ）
    got.checks      # → [{"name": "json_schema", "ok": true}, …]（指示通りの検査記録）

中身は 4 つのモジュールです。

    parser.py    … 指示の読み取り（指示部／材料部／出力仕様／役割）
    extract.py   … 情報抽出（材料の文字列をそのまま値にする）
    summarize.py … 要約（件数と短さを守り、材料の語だけで書く）
    code.py      … コード生成（書いた関数を実際に実行して確かめる）
    answer.py    … 問いへの応答（役割・口調・文字数を守る）
    style.py     … 文体と長さ（述語を辞書で引き直して口調を組み替える）
    run.py       … 実行 → 検証 → 組み直しの 1 本道

指示でなければ `run()` は None を返し、いつもの会話経路（mind）が引き継ぎます。
"""

from __future__ import annotations

from .parser import Directive, FormatSpec, explain, parse
from .run import CAN_NOT_SAY, Result, execute, run, run_directive, verify

__all__ = ["parse", "run", "run_directive", "execute", "verify", "explain", "Directive",
           "FormatSpec", "Result", "CAN_NOT_SAY"]
