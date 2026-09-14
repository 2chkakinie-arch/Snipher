#!/usr/bin/env python3
"""Snipher v8 の実測 — 実運用で落ちた 10 問 + ループ禁止 + 神経系の検査。

    python tools/bench_v8.py          # 表で要約
    python tools/bench_v8.py --json   # 機械が読める形
    python tools/bench_v8.py --runs 3 # 1 問あたり何回繰り返すか

測るのは 3 つ。

  A. 実運用の 10 問 … ユーザーが投げて落ちた 10 問（プロンプトはログのまま）。
     見るのは「頼まれた形で返せたか」だけ（答えの文字列ではなく性質で検査）。
  B. ループ禁止     … 同じ文・同じ定型句（「〜の知識で答えます」等）の繰り返しが
     10 問の出力のどこにも出ないこと。`snipher/guard.py` が物理的に禁じる。
  C. 神経系（lm8）  … パラメータだけの生成（小説 2000 字・テンプレート不使用）と
     言語モデルとしての健全性（perplexity）。重みが無ければ skip。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench_following import TEMPLATE_JUNK, check_case  # noqa: E402

# --------------------------------------------------------------------------- #
# A. 実運用で落ちた 10 問（プロンプトはログのまま）
# --------------------------------------------------------------------------- #
CASES: list[dict] = [
    {
        "name": "ひらがなだけ（全文変換・記号なし）",
        "prompt": "「空が青い」という文章を、ひらがなだけで出力してください。他の文字や記号は含めないでください。",
        "expect": {"exact": "そらがあおい"},
    },
    {
        "name": "カンマ区切りのリスト（改行・説明なし）",
        "prompt": "「りんご、ゴリラ、ラッパ」をカンマ区切りのリストにしてください。改行や説明は不要です。",
        "expect": {"exact": "りんご,ゴリラ,ラッパ"},
    },
    {
        "name": "選択肢 + 字数制限（東京/大阪・2字以内）",
        "prompt": "日本の首都はどこですか？「東京」か「大阪」のどちらか1文字以上2文字以内で答えてください。",
        "expect": {"exact": "東京"},
    },
    {
        "name": "条件分岐（たら→単語で）",
        "prompt": "「雨が降ったら傘をさす。晴れたら帽子をかぶる。」いま外は晴れています。何を持っていきますか？単語で答えてください。",
        "expect": {"exact": "帽子"},
    },
    {
        "name": "計算（数字だけ）",
        "prompt": "昨日の晩ごはんはカレーでした。ところで、1 + 1 はいくつですか？数字だけで答えてください。",
        "expect": {"exact": "2"},
    },
    {
        "name": "好みの解決（好き/嫌い→どちら）",
        "prompt": "私は犬が好きで、猫は嫌いです。私が好きな動物はどちらですか？",
        "expect": {"exact": "犬"},
    },
    {
        "name": "JSON 抽出（品名・価格）",
        "prompt": "次の文から情報を抽出してJSON形式で出力してください。「品名: リンゴ, 価格: 100円」",
        "expect": {"json_eq": {"品名": "リンゴ", "価格": "100円"}},
    },
    {
        "name": "JSON 抽出（キーを user_name に）",
        "prompt": "「名前：みのお」という情報を、キーを user_name にしたJSONで出力してください。",
        "expect": {"json_eq": {"user_name": "みのお"}},
    },
    {
        "name": "数の比較（大きい方・数字だけ）",
        "prompt": "10 と 5 はどちらが大きいですか？大きい方の数字だけを答えてください。",
        "expect": {"exact": "10"},
    },
    {
        "name": "反転（オフ+1回→オン）",
        "prompt": "電気のスイッチがオフになっています。スイッチを1回押すとどうなりますか？「オン」か「オフ」で答えてください。",
        "expect": {"exact": "オン"},
    },
]


def check_v8(case: dict, text: str) -> list[str]:
    """v8 の 1 問を検査する（check_case + JSON 等価 + ひらがな限定）。"""
    failed = check_case(case, text)
    exp = case.get("expect") or {}
    body = str(text or "").strip()
    if "json_eq" in exp:
        try:
            obj = json.loads(body)
        except Exception:
            failed.append("JSON として読めない")
            obj = None
        if obj is not None and obj != exp["json_eq"]:
            failed.append(f"値が違う: {obj!r}（期待 {exp['json_eq']!r}）")
    if case["name"].startswith("ひらがなだけ"):
        import re

        if re.search(r"[^ぁ-ん]", body):
            failed.append("ひらがな以外の字がある")
    return failed


# --------------------------------------------------------------------------- #
# B. ループ禁止（同じ文・定型句の繰り返しは物理的に禁止）
# --------------------------------------------------------------------------- #
def check_loops(texts: list[str]) -> list[str]:
    failed: list[str] = []
    try:
        from snipher.guard import BANNED_PHRASES, find_repeats
    except Exception:  # noqa: BLE001
        return ["guard.py が無い"]
    for i, text in enumerate(texts):
        body = str(text or "")
        for phrase in BANNED_PHRASES:
            if phrase in body:
                failed.append(f"Q{i + 1}: 禁止の定型句「{phrase}」が出た")
        for rep in find_repeats(body):
            failed.append(f"Q{i + 1}: 繰り返し「{rep[:24]}…」")
    for junk in TEMPLATE_JUNK:
        for i, text in enumerate(texts):
            if junk in str(text or ""):
                failed.append(f"Q{i + 1}: テンプレートの漏れ {junk}")
    return failed


# --------------------------------------------------------------------------- #
# C. 神経系（lm8）— 重みがあれば測る
# --------------------------------------------------------------------------- #
def measure_neural() -> dict:
    out: dict = {"status": "skip", "reason": "重みが無い（tools/lm8_pretrain.py で学習）"}
    try:
        from snipher.lm8.evaluate import quick_report
    except Exception as e:  # noqa: BLE001
        out["reason"] = f"lm8 が無い: {e}"
        return out
    try:
        out = {"status": "ok", **quick_report()}
    except Exception as e:  # noqa: BLE001
        out = {"status": "error", "reason": str(e)[:200]}
    return out


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #
def _ask(core, prompt: str, *, web: bool, mode: str) -> tuple[str, dict]:
    text = ""
    stats: dict = {}
    for ev in core.stream_reply([{"role": "user", "content": prompt}], mode=mode, web=web):
        kind = ev.get("type")
        if kind == "delta" and not text:
            text += str(ev.get("text") or "")
        elif kind == "done":
            text = str(ev.get("text") or text)
            stats = ev.get("stats") or {}
    return text, stats


def measure(runs: int = 1, *, mode: str = "auto", web: bool = False,
            neural: bool = True) -> dict:
    from snipher.core import SnipherCore

    os.environ.setdefault("SNIPHER_WEB", "off")
    core = SnipherCore()
    rows: list[dict] = []
    timings: list[float] = []
    for _ in range(max(1, runs)):
        for case in CASES:
            t0 = time.perf_counter()
            text, stats = _ask(core, case["prompt"], web=web, mode=mode)
            dt = (time.perf_counter() - t0) * 1000
            timings.append(dt)
            failed = check_v8(case, text)
            rows.append({"case": case["name"], "ok": not failed, "failed": failed,
                         "ms": round(dt, 1), "route": stats.get("route"),
                         "plan": stats.get("plan"),
                         "text": text[:120].replace("\n", " / ")})
    loop_failed = check_loops([r["text"] for r in rows])
    neural_rep = measure_neural() if neural else {"status": "skip"}
    ok_n = sum(1 for r in rows if r["ok"])
    return {"cases": rows, "follow_rate": ok_n / max(1, len(rows)),
            "ok": ok_n, "n": len(rows),
            "median_ms": round(statistics.median(timings), 1) if timings else 0,
            "p95_ms": round(sorted(timings)[max(0, int(len(timings) * 0.95) - 1)], 1)
            if timings else 0,
            "max_ms": round(max(timings), 1) if timings else 0,
            "loop_failed": loop_failed, "loops_ok": not loop_failed,
            "neural": neural_rep}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Snipher v8 の実測")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--no-neural", action="store_true")
    args = ap.parse_args(argv)
    data = measure(args.runs, mode=args.mode, web=args.web, neural=not args.no_neural)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0 if data["ok"] == data["n"] and data["loops_ok"] else 1
    print(f"{'case':36s} {'ok':4s} {'ms':>8s} {'route':8s} text")
    print("-" * 100)
    for r in data["cases"]:
        print(f"{r['case']:36s} {'o' if r['ok'] else 'x':4s} {r['ms']:8.1f} "
              f"{str(r['route']):8s} {r['text'][:60]}")
        for f in r["failed"]:
            print(f"    ! {f}")
    print("-" * 100)
    print(f"A. 指示追従率 {data['ok']}/{data['n']} = {data['follow_rate'] * 100:.1f}%"
          f"（中央値 {data['median_ms']} ms, p95 {data['p95_ms']} ms, 最大 {data['max_ms']} ms）")
    print(f"B. ループ禁止 {'OK（0 件）' if data['loops_ok'] else 'NG'}")
    for f in data["loop_failed"][:10]:
        print(f"    ! {f}")
    neu = data["neural"]
    if neu.get("status") == "ok":
        print(f"C. 神経系 ppl={neu.get('ppl')} novel={neu.get('novel_chars')}字 "
              f"repeat={neu.get('novel_repeat')} template_free={neu.get('template_free')}")
    else:
        print(f"C. 神経系 skip（{neu.get('reason', '')}）")
    good = data["ok"] == data["n"] and data["loops_ok"]
    print(f"判定: {'PASS' if good else 'NG'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
