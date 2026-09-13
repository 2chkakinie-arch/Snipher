#!/usr/bin/env python3
"""指示追従の実測 — 「頼まれた仕事を、頼まれた形で返せたか」を数字で残す。

    python tools/bench_instruction.py          # 表で要約
    python tools/bench_instruction.py --json   # 機械が読める形
    python tools/bench_instruction.py --runs 3 # 1 指示あたり何回繰り返すか

測るのは次の 5 つ。

  1. 指示追従率   … 出力仕様の検査（JSON スキーマ／箇条書きの本数／文字数／語尾／
                    コードの実行）を *全部* 通した割合
  2. 仕事の実行   … タスクごとに「実際にやったか」（抽出は値が材料と一致、要約は
                    材料の語だけ、コードは実行、応答は根拠つき）
  3. 速度         … 1 指示あたりの中央値 / p95 / 最大（ニューラルは起動しない）
  4. 誤検出       … 会話の 1 通を指示と誤読しなかったか（0 が正解）
  5. 禁止表現     … 「できません」系の逃げが 1 つも出ていないか
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# 指示のバッテリー（タスク × 出力仕様）
# --------------------------------------------------------------------------- #
SUITE: list[dict] = [
    {"task": "extract", "name": "JSON 抽出（形式のみ・4 欄）", "expect": {"checks": True},
     "prompt": """次のテキストから情報を抽出し、必ず指定のJSON形式のみで出力してください。余計な挨拶や解説は不要です。

テキスト: 「東京から京都まで新幹線で約2時間15分。料金は指定席で約14,000円です。」

JSONフォーマット:
{
  "origin": "出発地",
  "destination": "目的地",
  "duration": "所要時間",
  "fare": "料金"
}"""},
    {"task": "extract", "name": "JSON 抽出（別の材料・3 欄）", "expect": {"checks": True},
     "prompt": """次のテキストから情報を抽出し、JSONのみで出力してください。

テキスト: 「大阪から博多まで飛行機で約1時間30分。料金は片道約18,000円です。」

JSONフォーマット:
{"from": "出発地", "to": "目的地", "time": "所要時間"}"""},
    {"task": "summarize", "name": "要約（箇条書き 3・短く）", "expect": {"checks": True},
     "prompt": """以下の文章を読み、重要なポイントを3つの箇条書きで短く要約してください。

文章：
オセロや将棋などの完全情報ゲームにおいて、AIは探索アルゴリズムを用いて最適な手を選択します。ミニマックス法は相手が最善を尽くすことを前提に自分の利益を最大化する手法であり、α-β枝刈りを組み合わせることで無駄な探索を削減できます。さらに評価関数を工夫することで、深い読みを行わなくても強い着手を実現できます。"""},
    {"task": "summarize", "name": "要約（箇条書き 2・番号付き）", "expect": {"checks": True},
     "prompt": """以下の文章を2つの箇条書きで要約してください。番号を付けてください。

文章：
ラーメンは中華麺とスープからなる麺料理です。スープは豚骨や鶏ガラ、魚介、野菜を組み合わせて作ります。麺とスープを別々に仕上げ、合わせる直前に温度を揃えると味がぼけません。"""},
    {"task": "code", "name": "JS 関数（重複除去 + 昇順）", "expect": {"checks": True},
     "prompt": ("JavaScriptで、配列から重複した要素を取り除いて昇順にソートする関数 "
                "`uniqueSort(arr)` を作成してください。コードと簡単な解説を添えてください。")},
    {"task": "code", "name": "Python 関数（重複除去 + 降順）", "expect": {"checks": True},
     "prompt": "Pythonで、リストの重複を除いて降順に並べ替える関数 dedupeDesc(items) を作成してください。"},
    {"task": "code", "name": "TypeScript 関数（合計）", "expect": {"checks": True},
     "prompt": "TypeScriptで、数値配列の合計を返す関数 total(nums: number[]): number を作成してください。"},
    {"task": "answer", "name": "応答（役割 + 口調 + 200 字）", "expect": {"checks": True},
     "prompt": """あなたは「頼れるベテランエンジニアのアシスタント」です。

以下の質問に対して、専門的でありながら親しみやすい口調（〜だよ、〜だね）で200文字程度で簡潔に答えてください。

質問：WebAssembly（Wasm）をブラウザで動かす一番のメリットは何ですか？"""},
    {"task": "answer", "name": "応答（知識ベースの話題・100 字）", "expect": {"checks": True},
     "prompt": "ラーメンとは何ですか？100文字程度で答えてください。"},
    {"task": "answer", "name": "応答（指示文が材料を差し出している）", "expect": {"checks": True},
     "prompt": """以下の質問に答えてください。

資料：しりとりは、前の語の最後の音で始まる語を順に言っていく遊びです。

質問：しりとりのルールは何ですか？"""},
    {"task": "list", "name": "列挙（材料から 3 件）", "expect": {"checks": True},
     "prompt": """以下の項目を3つの箇条書きで列挙してください。

項目：
りんごは赤い果物です。みかんは冬に出回る柑橘です。ぶどうは房になって実ります。"""},
    {"task": "transform", "name": "文字列の変換（大文字化）", "expect": {"checks": True},
     "prompt": '次のテキストを大文字に変換してください。\n\nテキスト: "hello snipher"'},
]

#: 指示と誤読してはいけない会話の 1 通（ここで Directive が出たら誤検出）
CONVERSATIONAL = [
    "こんにちは", "いま何時？", "ラーメンとは何ですか？", "しりとりしよう", "ありがとう",
    "12+7 はいくつ？", "今日はいい天気だね", "名前は何ていうの？", "もう少し詳しく",
    "pythonって何", "疲れた", "明日の予定は？", "自己紹介して", "何ができるの？",
    "猫", "67", "asdfgh", "GLM5.3とは", "三毛猫とは", "「うれしい」を英語にして",
]


def _check_map(res) -> dict[str, bool]:
    return {c["name"]: bool(c["ok"]) for c in (res.checks if res else [])}


def measure(runs: int = 1) -> dict:
    t_import = time.perf_counter()
    os.environ.setdefault("SNIPHER_WEB", "off")          # 検索を挟まない生の速度を見る
    from snipher.instruction import parse, run           # noqa: E402
    from snipher.instruction.run import CAN_NOT_SAY      # noqa: E402

    rows: list[dict] = []
    timings: list[float] = []
    refusals: list[str] = []
    per_task: dict[str, list[float]] = {}

    for _ in range(max(1, runs)):
        for case in SUITE:
            t0 = time.perf_counter()
            res = run(case["prompt"])
            dt = (time.perf_counter() - t0) * 1000
            timings.append(dt)
            per_task.setdefault(case["task"], []).append(dt)
            checks = _check_map(res)
            text = (res.text if res else "") or ""
            banned = [b for b in CAN_NOT_SAY if b in text]
            if banned:
                refusals.append(f"{case['name']} → {banned[0]}")
            spec_ok = bool(res is not None and res.ok and text.strip())
            if not spec_ok:
                failed = [k for k, v in checks.items() if not v]
                rows.append({"case": case["name"], "task": case["task"], "ok": False,
                             "ms": round(dt, 1), "chars": len(text.replace("\n", "")),
                             "failed": failed})
            else:
                rows.append({"case": case["name"], "task": case["task"], "ok": True,
                             "ms": round(dt, 1), "chars": len(text.replace("\n", "")),
                             "failed": []})

    # 誤検出（会話を指示と読まないか）
    false_positives = []
    t_fp: list[float] = []
    for text in CONVERSATIONAL:
        t0 = time.perf_counter()
        d = parse(text)
        t_fp.append((time.perf_counter() - t0) * 1000)
        if d is not None:
            false_positives.append(f"{text!r} → {d.task} ({d.confidence:.2f})")

    n = len(rows)
    ok_n = sum(1 for r in rows if r["ok"])
    return {
        "import_seconds": round(time.perf_counter() - t_import, 3),
        "suite_size": len(SUITE),
        "runs": max(1, runs),
        "n_cases": n,
        "instruction_follow_rate": round(ok_n / max(1, n), 4),
        "cases_failed": [r for r in rows if not r["ok"]][:10],
        "per_task_ok": {t: round(sum(1 for r in rows if r["task"] == t and r["ok"])
                                 / max(1, sum(1 for r in rows if r["task"] == t)), 3)
                        for t in sorted({r["task"] for r in rows})},
        "ms_median": round(statistics.median(timings), 2) if timings else 0.0,
        "ms_p95": round(sorted(timings)[max(0, int(len(timings) * 0.95) - 1)], 2) if timings else 0.0,
        "ms_max": round(max(timings), 2) if timings else 0.0,
        "ms_per_task_median": {t: round(statistics.median(v), 2) for t, v in sorted(per_task.items())},
        "cases_per_second": round(len(timings) / max(1e-9, sum(timings)) * 1000, 1),
        "parse_ms_median": round(statistics.median(t_fp), 3) if t_fp else 0.0,
        "refusals": refusals[:6],
        "refusal_count": len(refusals),
        "false_positives": false_positives[:6],
        "false_positive_count": len(false_positives),
        "conversational_probes": len(CONVERSATIONAL),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    gc.disable()
    data = measure(runs=a.runs)
    gc.enable()
    if a.json:
        print(json.dumps(data, ensure_ascii=False, indent=1))
        return 0
    print(f"=== 指示追従 実測（{data['n_cases']} 指示 / {data['suite_size']} 種 × {data['runs']} 回）")
    print(f"  指示追従率 {data['instruction_follow_rate'] * 100:.1f}%"
          f"（タスク別 {data['per_task_ok']}）")
    print(f"  1 指示: 中央値 {data['ms_median']}ms / p95 {data['ms_p95']}ms"
          f" / 最大 {data['ms_max']}ms  （{data['cases_per_second']} 指示/秒）")
    print(f"  タスク別中央値: {data['ms_per_task_median']}")
    print(f"  誤検出 {data['false_positive_count']}/{data['conversational_probes']}"
          f"（読み取り 中央値 {data['parse_ms_median']}ms）"
          f" / 禁止表現 {data['refusal_count']}")
    for line in data["false_positives"]:
        print("    ! 誤検出", line)
    for line in data["refusals"]:
        print("    ! 逃げ", line)
    for row in data["cases_failed"]:
        print(f"    ! {row['case']}: {row['failed']}")
    # 合格線は「全指示が仕様どおり・誤検出ゼロ・逃げゼロ」。1 つでも外したら NG。
    ok = (data["instruction_follow_rate"] >= 1.0 and data["false_positive_count"] == 0
          and data["refusal_count"] == 0)
    print("  判定:", "PASS" if ok else "NG")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
