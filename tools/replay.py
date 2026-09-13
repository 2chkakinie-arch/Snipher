#!/usr/bin/env python3
"""ユーザーの失敗例プロンプトを SnipherCore.stream_reply に流して応答を印字する。

    .venv/bin/python tools/replay.py            # 同梱ケース全件
    .venv/bin/python tools/replay.py -t "黄金比について教えて"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CASES = [
    ("answer", '以下の質問に一言で答えてください。余計な解説は不要です。\n\n質問：日本の首都はどこですか？'),
    ("classify", "以下の単語を「果物」か「野菜」かで分類してください。\n\nりんご:\n\nトマト:\n\nバナナ:"),
    ("json_fill", '次のキーに対する数値を埋めてJSON形式で出力してください。\n\n{\n\n"one": 1,\n\n"two":\n\n}'),
    ("continue", '次の文章の続きを1文で書いてください。\n\n「今日は朝から雨が降っていたので、」'),
    ("roleplay", 'あなたは語尾に「〜ロボ」をつけるロボットです。\n\n挨拶をしてください。'),
    ("explain", "日本について詳しく教えてください"),
    ("price", "電球　平均値段"),
    ("unknown", "GLM5.3とは"),
    ("small_lm", "「web検索しろよ」   世界で一番小さい言語モデル"),
    ("qa_short", "日本の国旗は何色ですか？"),
    ("translate", "「猫」を英語に訳して"),
]


def collect(engine, text: str, history: list[dict] | None = None, web: bool = False) -> tuple[str, dict]:
    msgs = (history or []) + [{"role": "user", "content": text}]
    out: list[str] = []
    meta: dict = {}
    for event in engine.stream_reply(msgs, mode="auto", web=web):
        t = event.get("type")
        if t == "delta":
            out.append(event.get("text", ""))
        elif t == "done":
            meta = event.get("stats") or {}
    return "".join(out), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--text", action="append", default=None)
    ap.add_argument("--web", action="store_true", help="web裏取りをONにする")
    args = ap.parse_args()

    from snipher.core import SnipherCore

    core = SnipherCore()
    cases = [(None, t) for t in args.text] if args.text else CASES
    for tag, text in cases:
        t0 = time.perf_counter()
        try:
            reply, meta = collect(core, text, web=args.web)
        except Exception as e:  # noqa: BLE001
            reply, meta = f"<EXC {type(e).__name__}: {e}>", {}
        dt = (time.perf_counter() - t0) * 1000
        print(f"\n{'=' * 72}\n[case {tag}] {dt:.1f}ms  route={meta.get('route')}  ok={meta.get('ok')}\n--- prompt ---\n{text}\n--- reply ---\n{reply}")
    core.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
