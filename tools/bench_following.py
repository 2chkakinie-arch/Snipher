#!/usr/bin/env python3
"""指示追従の実測（実運用で落ちた 14 問）— 「頼まれた形で返せたか」を数字で残す。

    python tools/bench_following.py          # 表で要約
    python tools/bench_following.py --json   # 機械が読める形
    python tools/bench_following.py --runs 3 # 1 問あたり何回繰り返すか

`tools/bench_instruction.py` が「指示の型 × 出力仕様」を広く測るのに対して、こちらは
*実際に投げられて落ちた 14 問*（ログに残っているもの）を、経路（core）を通して 1 問ずつ
検査します。見るのは 4 つです。

  1. 指示追従率 … その問いの検査（形・禁止語・字数）を全部通した割合
  2. 経路        … instruction / tool / knowledge のどれで答えたか（理由の記録）
  3. 速度        … 1 問あたりの中央値 / p95 / 最大
  4. 捏造        … 材料が無いのに断定していないか（空欄【 】・予報の断言・別話題への逃げ）

検査は `check_case()` に集約してあります。期待（expect）は「答えの文字列」ではなく
*満たすべき性質*（1 文字である・禁止語を含まない・材料に無い値を言わない）で書きます。
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

# --------------------------------------------------------------------------- #
# 実際に落ちた 14 問（プロンプトはログのまま）
# --------------------------------------------------------------------------- #
CASES: list[dict] = [
    {
        "name": "助詞の穴埋め（1 文字で）",
        "prompt": "次の文の空欄に入る最も適切な助詞を1文字で答えてください。「私（ ）公園へ行く。",
        "expect": {"exact": "は", "max_chars": 2,
                   "absent": ["日本語", "ひらがな", "公園へ行く"]},
    },
    {
        "name": "カタカナ変換（出力だけ）",
        "prompt": "「りんご」をカタカナに変換して出力してください。",
        "expect": {"exact": "リンゴ", "max_chars": 8, "absent": ["果物", "植物"]},
    },
    {
        "name": "定義を一言で",
        "prompt": "「太陽」とは何ですか？一言で説明してください。",
        "expect": {"contains_any": ["恒星", "太陽系の中心"], "max_chars": 60,
                   "absent": ["満ち欠け", "月齢"]},
    },
    {
        "name": "同じ意味の語（類義語）",
        "prompt": "「嬉しい」と同じような意味を持つ言葉を1つ挙げてください。",
        "expect": {"contains_any": ["幸せ", "しあわせ", "嬉しい", "楽しい", "喜ばしい", "ハッピー"],
                   "max_chars": 40, "absent": ["について", "【"]},
    },
    {
        "name": "カテゴリで選ぶ（果物だけ）",
        "prompt": "次の単語の中から「果物」だけを選んでください：[ 犬, りんご, 車, バナナ ]",
        "expect": {"contains_all": ["りんご", "バナナ"], "absent": ["犬", "車"],
                   "max_chars": 60},
    },
    {
        "name": "短文 10 文字以内",
        "prompt": "「猫」についての短文を、ちょうど10文字以内で書いてください。",
        "expect": {"contains": "猫", "max_chars": 10, "absent": ["【", "について"]},
    },
    {
        "name": "計算結果だけを数字で",
        "prompt": "3 + 5 の計算結果だけを数字で答えてください。",
        "expect": {"exact": "8"},
    },
    {
        "name": "全称命題（はい / いいえ）",
        "prompt": "人間は必ず息をします。太郎は人間です。太郎は息をしますか？"
                  "「はい」か「いいえ」で答えてください",
        "expect": {"exact": "はい"},
    },
    {
        "name": "規則の適用（赤信号）",
        "prompt": "「赤信号では止まれ、青信号では進め。」いま信号は赤です。どうすればいいですか？",
        "expect": {"contains": "止ま", "absent": ["進め", "進みます", "青"]},
    },
    {
        "name": "禁止語つきの礼（ありがとう以外）",
        "prompt": "「ありがとう」を使わずに、感謝の気持ちを表す短い返答をしてください。",
        "expect": {"absent": ["ありがとう"], "max_chars": 40,
                   "contains_any": ["助かり", "感謝", "お礼", "恐縮", "おかげ", "ありたく",
                                    "受け止め", "うれしい", "心より"]},
    },
    {
        "name": "ひらがな書き（実辞書の読み）",
        "prompt": "山羊をひらがなにしてください。",
        "expect": {"exact": "やぎ", "absent": ["山羊"]},
    },
    {
        "name": "反対語（対義語）",
        "prompt": "「嬉しい」の反対語を1つ挙げてください。",
        "expect": {"contains_any": ["悲しい", "悲しみ", "憂い", "つらい", "辛い"],
                   "max_chars": 40, "absent": ["について", "【"]},
    },
    {
        "name": "語の写し（英語）",
        "prompt": "犬と猫を英語に訳してください。",
        "expect": {"contains_all": ["dog", "cat"], "max_chars": 40},
    },
    {
        "name": "語の写し（対応表に無い語は音写まで）",
        "prompt": "「うれしい」を英語にして",
        "expect": {"contains_any": ["ureshii", "happy", "glad", "うれしい"],
                   "absent": ["できません", "分かりません", "【"]},
    },
    {
        "name": "現在の天気（材料が無いときの正直さ）",
        "prompt": "明日の天気は？",
        "expect": {"contains_any": ["天気", "空模様", "予報"],
                   "absent": ["晴れます", "雨が降ります", "曇りです", "【"]},
    },
]

#: どの答えにも残ってはいけない *テンプレートの漏れ*（v6 まで出ていたもの）
TEMPLATE_JUNK = (
    "について、要点をまとめます。",
    "について、お知らせします。",
    "ご確認のうえ、必要であれば",
    "内容は【内容】で",
    "ついての短文について",
    "同じような意味について、",
)


def check_case(case: dict, text: str) -> list[str]:
    """1 問の出力を検査し、外れた理由の一覧を返す（空なら合格）。"""
    exp = case.get("expect") or {}
    body = str(text or "")
    strip = body.strip()
    failed: list[str] = []

    for junk in TEMPLATE_JUNK:
        if junk in body:
            failed.append(f"テンプレートの漏れ: {junk}")
    if "【" in body and "】" in body:
        failed.append("材料に無い空欄【 】が残っている")
    if not strip:
        failed.append("出力が空")
        return failed
    if "exact" in exp and strip != exp["exact"]:
        failed.append(f"形が違う: {strip[:24]!r}（期待 {exp['exact']!r}）")
    if "contains" in exp and exp["contains"] not in body:
        failed.append(f"「{exp['contains']}」が無い")
    for w in exp.get("contains_all") or []:
        if w not in body:
            failed.append(f"「{w}」が無い")
    if exp.get("contains_any") and not any(w in body for w in exp["contains_any"]):
        failed.append("どれも無い: " + "/".join(exp["contains_any"]))
    for w in exp.get("absent") or []:
        if w in body:
            failed.append(f"禁止「{w}」が出た")
    limit = int(exp.get("max_chars") or 0)
    if limit and len(strip.replace("\n", "")) > limit:
        failed.append(f"{len(strip)} 文字（上限 {limit}）")
    return failed


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


def measure(runs: int = 1, *, mode: str = "auto", web: bool = False) -> dict:
    from snipher.core import SnipherCore

    os.environ.setdefault("SNIPHER_WEB", "off")
    core = SnipherCore()
    rows: list[dict] = []
    timings: list[float] = []
    per_case: dict[str, list[float]] = {}
    for _ in range(max(1, runs)):
        for case in CASES:
            t0 = time.perf_counter()
            text, stats = _ask(core, case["prompt"], web=web, mode=mode)
            dt = (time.perf_counter() - t0) * 1000
            timings.append(dt)
            per_case.setdefault(case["name"], []).append(dt)
            failed = check_case(case, text)
            rows.append({
                "case": case["name"], "ok": not failed, "failed": failed,
                "ms": round(dt, 1), "route": stats.get("route"), "plan": stats.get("plan"),
                "engine": stats.get("engine"), "secs": stats.get("seconds"),
                "chars": len((text or "").replace("\n", "")), "text": (text or "")[:160],
            })
    ok = sum(1 for r in rows if r["ok"])
    return {
        "cases": len(rows),
        "ok": ok,
        "follow_rate": round(ok / len(rows), 4) if rows else 0.0,
        "runs": max(1, runs),
        "ms": {"median": round(statistics.median(timings), 1),
               "p95": round(sorted(timings)[max(0, int(len(timings) * 0.95) - 1)], 1),
               "max": round(max(timings), 1)},
        "rows": rows,
    }


def print_table(res: dict) -> None:
    print(f"{'case':38s} {'ok':>3s} {'ms':>8s} {'route':>10s}  text")
    print("-" * 108)
    for r in res["rows"]:
        mark = " o " if r["ok"] else " x "
        text = r["text"].replace("\n", " ⏎ ")
        print(f"{r['case'][:38]:38s}{mark}{r['ms']:8.1f} {str(r['route'])[:10]:>10s}  {text[:44]}")
        for why in r["failed"]:
            print(f"{'':44s}↳ {why}")
    print("-" * 108)
    print(f"指示追従率 {res['ok']}/{res['cases']} = {res['follow_rate'] * 100:.1f}%"
          f"（{res['runs']} 回ずつ / 中央値 {res['ms']['median']} ms, "
          f"p95 {res['ms']['p95']} ms, 最大 {res['ms']['max']} ms）")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="実運用で落ちた指示の追従率を測る")
    ap.add_argument("--json", action="store_true", help="JSON で出す")
    ap.add_argument("--runs", type=int, default=1, help="1 問あたりの反復回数")
    ap.add_argument("--mode", default="auto", help="core の mode（auto/fast/light…）")
    ap.add_argument("--web", action="store_true", help="ウェブ裏取りを有効にして測る")
    args = ap.parse_args(argv)
    res = measure(args.runs, mode=args.mode, web=args.web)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print_table(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
