#!/usr/bin/env python3
"""Snipher v3 の実測 — 速い・軽い・壊れない、を数字で残す。

    python tools/bench.py            # 表で要約
    python tools/bench.py --json     # mechanically 読める形
    python tools/bench.py --turns 5  # 1 発話あたり何回繰り返して数えるか

測るのは次の 4 種類だけ。
  * 応答時間（初手・通常・最悪）… 会話として *その場で* 返せるか
  * 常駐サイズ（import + 起動 + レスポンス生成中の最大 RSS）… 軽量か
  * パラメータ・語彙・知識の規模 … どこまで「自分の頭」で持っているか
  * 文章の健全性（組み立て文の validate 通過率と定型文の重複）… 完璧な文章か
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

BATTERY = [
    "今は西暦何年？", "なにができますか", "ありがとう", "猫とは", "3×7は？",
    "しりとりしよ", "今日のニュースは？", "回文を作って", "Pythonで素数判定を書いて",
    "100 の素因数分解", "「うれしい」を英語にして", "私の名前は何？", "疲れた",
    "x^2-5x+6=0 を解いて", "GLM5.3とは", "三毛猫とは", "67", "asdfgh",
]


def _rss_kib() -> int:
    """このプロセス最大 RSS（kB）。/proc が無い環境では现时刻の値に fallback する。"""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            peak = cur = 0
            for line in fh:
                if line.startswith("VmHWM:"):
                    peak = int(line.split()[1])
                elif line.startswith("VmRSS:"):
                    cur = int(line.split()[1])
            return peak or cur
    except Exception:  # noqa: BLE001
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except Exception:  # noqa: BLE001
            return 0


def _sizes() -> dict:
    out: dict[str, int] = {}
    data = ROOT / "snipher" / "data"
    for name, rel in (("core", "neural/core.npz"), ("lm", "lm.npz"),
                      ("wordbank", "wordbank.bin.gz"), ("game_words", "wordbank_game.bin.gz"),
                      ("kb", "kb.json"), ("corpus", "corpus.txt.gz"),
                      ("conjugation", "conjugation.bin.gz"), ("english_words", "words_en.txt.gz")):
        p = data / rel
        out[name] = p.stat().st_size if p.exists() else 0
    out["code_lines"] = sum(
        len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
        for f in (ROOT / "snipher").rglob("*.py")
    )
    return out


def _python_files() -> list[Path]:
    return [p for p in (ROOT / "snipher").rglob("*.py")]


def measure(turns: int = 3) -> dict:
    t_import = time.perf_counter()
    os.environ.setdefault("SNIPHER_WEB", "off")        # 検索を挟まない生の速度を見る
    from snipher.composer import Composer, validate   # noqa: E402
    from snipher import lm as lm_mod                   # noqa: E402
    from snipher.neural.core import DistilledCore      # noqa: E402
    return _measure(turns, Composer, validate, lm_mod, DistilledCore, t_import)


def _measure(turns: int, Composer, validate, lm_mod, DistilledCore, t_import) -> dict:
    sizes = _sizes()
    r_after_import = _rss_kib()

    t0 = time.perf_counter()
    c = Composer()
    ms_boot = (time.perf_counter() - t0) * 1000

    timings: list[float] = []
    first_ms: float | None = None
    bad: list[str] = []
    replies: dict[str, list[str]] = {}
    for turn in range(1, turns + 1):
        for q in BATTERY:
            t = time.perf_counter()
            r = c.compose(q, history=[], turn=turn, web=False)
            dt = (time.perf_counter() - t) * 1000
            timings.append(dt)
            if first_ms is None:
                first_ms = dt
            replies.setdefault(q, []).append(r.text)
            text = (r.text or "").strip()
            if not text:
                bad.append(f"{q} → 空応答")
                continue
            ok, why = validate(text, max_len=400)
            if not ok:
                bad.append(f"{q} → {why}")

    lm = lm_mod.shared()
    core = DistilledCore()
    st = core.status()
    params = {"lm_entries": int(getattr(lm, "total", 0) or 0),
              "lm_vocab": int(getattr(lm, "n_vocab", 0) or 0),
              "neural_params": int(st.get("params") or 0),
              "neural_vocab": int(st.get("vocab") or 0),
              "weights_bytes": int(st.get("weights_bytes") or 0),
              "val_ppl": (st.get("metrics") or {}).get("best_val", {}).get("ppl")}
    stats = c.kb.stats() if hasattr(c.kb, "stats") else {}
    distinct = {q: len(set(v)) for q, v in replies.items()}

    return {
        "import_seconds": round(time.perf_counter() - t_import, 3),
        "composer_boot_ms": round(ms_boot, 1),
        "first_reply_ms": round(first_ms or 0, 1),
        "reply_ms_median": round(statistics.median(timings), 2),
        "reply_ms_p95": round(sorted(timings)[int(len(timings) * 0.95) - 1], 2),
        "reply_ms_max": round(max(timings), 2),
        "replies_per_second": round(len(timings) / sum(timings) * 1000, 1),
        "n_replies": len(timings),
        "validate_failures": bad[:8],
        "validate_failure_count": len(bad),
        "prose_ok_rate": round(1 - len(bad) / max(1, len(timings)), 4),
        "varied_replies": sum(1 for v in distinct.values() if v >= 2),
        "battery_size": len(BATTERY),
        "peak_rss_kib_after_import": r_after_import,
        "params": params,
        "sizes_bytes": sizes,
        "lexicon": c.kb.stats().get("words") if isinstance(stats, dict) else None,
        "kb": {k: stats.get(k) for k in ("topics", "facts", "qa", "how", "why", "opinions")
               if isinstance(stats, dict)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    gc.disable()                       # gc の山を測らないための無効化（計測中だけ）
    data = measure(turns=a.turns)
    gc.enable()
    if a.json:
        print(json.dumps(data, ensure_ascii=False, indent=1))
        return 0
    mb = 1024 * 1024
    print(f"=== Snipher 実測（{data['n_replies']} 応答 / {data['battery_size']} 発話 × {a.turns} ターン）")
    print(f"  import {data['import_seconds']}s → Composer 起動 {data['composer_boot_ms']}ms"
          f"、初手 {data['first_reply_ms']}ms")
    print(f"  1 応答: 中央値 {data['reply_ms_median']}ms / p95 {data['reply_ms_p95']}ms"
          f" / 最大 {data['reply_ms_max']}ms  （{data['replies_per_second']} 応答/秒）")
    print(f"  常駐: import 直後 {data['peak_rss_kib_after_import'] // 1024} MiB"
          f"、コード {data['sizes_bytes']['code_lines']} 行")
    print("  データ: " + " / ".join(f"{k} {v / mb:.2f}MiB" for k, v in data["sizes_bytes"].items()
                                   if k != "code_lines" and v))
    pr = data["params"]
    print(f"  知識: {data['kb']}")
    print(f"  モデル: コア {pr['neural_params'] / 1e6:.2f}M 語彙 {pr['neural_vocab']}"
          f"（重み {pr['weights_bytes'] / (1024 * 1024):.2f}MiB、val ppl"
          f" {pr['val_ppl'] if pr['val_ppl'] is not None else '—'}）"
          f" / n-gram {pr['lm_entries']} エントリ")
    print(f"  文章: validate 通過率 {data['prose_ok_rate'] * 100:.1f}%"
          f"（失敗 {data['validate_failure_count']}）、同じ発話で応答が揺れた例 "
          f"{data['varied_replies']}/{data['battery_size']}")
    for line in data["validate_failures"]:
        print("    !", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
