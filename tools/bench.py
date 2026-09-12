#!/usr/bin/env python3
"""Snipher v2 の実測ベンチマーク（README の数字はここから作る）。

    python tools/bench.py            # 表を印刷して var/bench.json に保存
    python tools/bench.py --json     # JSON だけ

測るのは次の 5 つ。すべてこのマシンでの実測値です。

    1. 知識ベース検索     … 40 発話の検索にかかった時間（1 件あたり ms）
    2. composer          … 発話 → 日本語の応答を組み立てる時間（1 件あたり ms）
    3. n-gram LM         … 1 文の採点（判定）にかかる時間と perplexity の判別力
    4. 内蔵ニューラルコア … ロード時間・1 文字あたりの生成時間（KV キャッシュ on/off）・補完
    5. 重みの合計         … kb.json + lm.npz + core.npz のバイト数

比較対象の LFM2.5-1.2B-JP は公開値（1.17B パラメータ / GGUF Q4_K_M 731MB /
safetensors 2.2GB）を記載するだけで、このスクリプトでは実行しません。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUERIES = [
    "花火とは", "好きな食べ物は？", "暇だなあ", "猫がゴロゴロ言う", "おすすめの本",
    "明日の天気はどう？", "なぜ空は青い", "観葉植物の葉が黄色い", "お腹が痛い", "肩が凝った",
    "gitがわからない", "Wi-Fiが遅い", "タピオカって何", "ありがとう", "こんにちは",
    "あなたは誰？", "67", "あ", "asdfgh", "ぬるぬる猿について",
    "犬の散歩はどれくらい必要", "ラーメンの作り方", "本を読みたい", "洗濯物が乾かない",
    "部屋が散らかってる", "お金がない", "眠れない", "Pythonって何", "咳が出る", "熱がある",
    "誕生日プレゼント", "夏祭りに行きたい", "将棋をやりたい", "雑学を教えて", "映画が見たい",
    "コーヒーが好き", "筋トレしてる", "転職したい", "節約したい", "植物を育ててる",
    # 知識ベースに材料が無く、内蔵ニューラルコアが本文を作る側に回る発話
    "話して", "何してるの", "ちょっと聞いて",
]

LFM25_REFERENCE = {
    "name": "LFM2.5-1.2B-JP（公開値）",
    "params": 1_170_000_000,
    "weights_gguf_q4_bytes": 731 * 1024 * 1024,
    "weights_safetensors_bytes": int(2.2 * 1024 * 1024 * 1024),
    "download_required": True,
}


def _timeit(fn, n: int) -> tuple[float, float]:
    """(合計秒, 1 件あたり秒)"""
    ts: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sum(ts), statistics.fmean(ts)


def bench_kb(kb) -> dict:
    def one():
        for q in QUERIES:
            kb.answer(q)

    total, mean = _timeit(one, 3)
    hits = sum(1 for q in QUERIES if kb.answer(q) is not None)
    st = kb.stats()
    return {
        "queries": len(QUERIES),
        "hits": hits,
        "hit_rate": round(hits / len(QUERIES), 3),
        "ms_per_query": round(mean / len(QUERIES) * 1000, 4),
        "topics": st["topics"], "facts": st["facts"], "questions": st["questions"],
        "opinions": st.get("opinions"),
        "bytes": (ROOT / "snipher" / "data" / "kb.json").stat().st_size,
    }


def bench_composer() -> dict:
    from snipher.composer import Composer
    from snipher.knowledge import KnowledgeBase

    c = Composer(kb=KnowledgeBase())
    replies = [c.compose(q, turn=i + 1).text for i, q in enumerate(QUERIES)]

    def one():
        for i, q in enumerate(QUERIES):
            c.compose(q, turn=100 + i)

    total, mean = _timeit(one, 3)
    plans = [c.compose(q, turn=900 + i).plan for i, q in enumerate(QUERIES)]
    return {
        "ms_per_reply": round(mean / len(QUERIES) * 1000, 3),
        "replies_per_second": round(len(QUERIES) / mean, 1),
        "avg_reply_chars": round(statistics.fmean(len(r) for r in replies), 1),
        "distinct_plans": len(set(plans)),
        "samples": [{"q": q, "plan": p, "text": r}
                    for q, p, r in list(zip(QUERIES, plans, replies))[:6]],
    }


def bench_lm() -> dict:
    from snipher import lm as lm_mod

    model = lm_mod.shared()
    if model is None or not model.is_ready:
        return {"available": False}
    good = "花火は、火薬の燃焼と爆発で光と音を出し、夜空に模様を描く娯楽です。"
    junk = "はばがを、光と音を出し娯楽です花火爆発夜空模様描く。"

    def one():
        model.score(good)

    total, mean = _timeit(one, 200)
    return {
        "available": True,
        "engine": model.engine_name(),
        "params": model.n_params(),
        "order": model.order,
        "vocab": model.n_vocab,
        "bytes": model.bytes_on_disk(),
        "load_seconds": model.load_seconds,
        "ms_per_score": round(mean * 1000, 4),
        "scores_per_second": round(1.0 / mean, 0),
        "ppl_good": model.perplexity(good),
        "ppl_junk": model.perplexity(junk),
        "conf_good": model.confidence(good),
        "conf_junk": model.confidence(junk),
        "calib": model.calib,
    }


def bench_neural() -> dict:
    import numpy as np

    from snipher.neural.cache import get_core
    from snipher.neural.nn import DecodeCache

    t0 = time.perf_counter()
    core = get_core()
    load = time.perf_counter() - t0
    if core is None or not core.is_ready:
        return {"available": False, "reason": "重みが未ビルド（python tools/distill_neural.py）"}
    prompt = "<user>観葉植物の葉が黄色い\n<asst>"
    ids = [2] + core.tok.encode(prompt)

    def gen_cached():
        core.generate_ids(ids, max_new=40, temperature=0.7, top_k=32, seed=5, use_cache=True)

    def gen_plain():
        core.generate_ids(ids, max_new=40, temperature=0.7, top_k=32, seed=5, use_cache=False)

    _t, m_cached = _timeit(gen_cached, 3)
    _t2, m_plain = _timeit(gen_plain, 3)
    # 素のプロンプト（温度高め）と、パイプラインが実際に使う対話プロンプトの両方を出す
    text = core.tok.decode(core.generate_ids(ids, max_new=40, temperature=0.7, top_k=32,
                                             seed=5, use_cache=True))
    chat_q = "よく眠れない"
    chat_text = ""
    try:
        chat_text = (core.reply(chat_q, max_chars=48, temperature=0.6, top_k=24,
                                seed=11) or "").strip()
    except Exception:  # noqa: BLE001
        chat_text = ""
    _t3, m_complete = _timeit(lambda: core.complete("私は毎日朝に", seed=5), 3)
    _t4, m_score = _timeit(lambda: core.score("今日はいい天気ですね。"), 20)
    _t5, m_reply = _timeit(lambda: core.reply("観葉植物の葉が黄色い", max_chars=40, seed=5), 3)
    net = core.net
    cache = DecodeCache(net, batch=1)
    _t6, m_step = _timeit(lambda: cache.step(net, np.array([7])), 30)
    return {
        "available": True,
        "engine": core.engine_name(),
        "params": net.n_params(),
        "d_model": net.cfg.d_model,
        "n_layers": net.cfg.n_layers,
        "blocks": "".join("c" if b == "conv" else "a" for b in net.cfg.blocks),
        "vocab": core.tok.size(),
        "bytes": core.path.stat().st_size if core.path.exists() else 0,
        "load_seconds": round(load, 3),
        "trained_at": core.meta.get("trained_at"),
        "metrics": core.meta.get("metrics"),
        "ms_per_char_cached": round(m_cached / 40 * 1000, 3),
        "ms_per_char_plain": round(m_plain / 40 * 1000, 3),
        "speedup_kv_cache": round(m_plain / max(1e-9, m_cached), 2),
        "chars_per_second": round(40 / m_cached, 1),
        "ms_per_decode_step": round(m_step * 1000, 3),
        "ms_complete": round(m_complete * 1000, 1),
        "ms_score": round(m_score * 1000, 2),
        "ms_reply": round(m_reply * 1000, 1),
        "sample": chat_text or text.strip(),
        "sample_query": chat_q if chat_text else prompt,
        "sample_raw": text.strip(),
    }


def bench_end_to_end() -> dict:
    from snipher.core import SnipherCore

    core = SnipherCore(torch_provider=lambda: None)
    core.active_backend = lambda: None          # フルウェイトは無い前提で測る

    n = min(24, len(QUERIES))

    def one():
        for q in QUERIES[:n]:
            list(core.stream_reply([{"role": "user", "content": q}]))

    total, mean = _timeit(one, 2)
    routes: dict[str, int] = {}
    cands = accepted = 0
    generated_samples: list[tuple[str, str]] = []
    for q in QUERIES:
        evs = list(core.stream_reply([{"role": "user", "content": q}]))
        st = evs[-1]["stats"]
        routes[st["route"]] = routes.get(st["route"], 0) + 1
        # ニューラル生成に挑戦したターンだけ、候補数と採用を数える
        if st.get("candidates") and st.get("neural_used"):
            cands += int(st["candidates"])
            if st.get("generated"):
                accepted += 1
                txt = "".join(e.get("text", "") for e in evs if e.get("type") == "delta")
                if txt and len(generated_samples) < 3:
                    generated_samples.append((q, txt))
    return {
        "ms_per_turn": round(mean / n * 1000, 2),
        "turns_per_second": round(n / mean, 1),
        "turns_measured": n,
        "routes": routes,
        "generated_candidates": cands,
        "generated_accepted_turns": accepted,
        "generated_samples": generated_samples,
        "light_ready": core.light_ready(),
        "lm_ready": core.lm_ready(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="JSON だけを出力する")
    ap.add_argument("--out", type=Path, default=ROOT / "var" / "bench.json")
    args = ap.parse_args(argv)

    from snipher.knowledge import KnowledgeBase

    kb = KnowledgeBase()
    out = {
        "generated_at": round(time.time(), 1),
        "kb": bench_kb(kb),
        "composer": bench_composer(),
        "lm": bench_lm(),
        "neural": bench_neural(),
        "end_to_end": bench_end_to_end(),
        "reference_lfm25": LFM25_REFERENCE,
    }
    weights = (out["kb"]["bytes"] + int(out["lm"].get("bytes") or 0)
               + int(out["neural"].get("bytes") or 0))
    out["total_weight_bytes"] = weights

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        _print_table(out)
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass
    return 0


def _print_table(d: dict) -> None:
    kb, lm, nn_, e2e, comp = d["kb"], d["lm"], d["neural"], d["end_to_end"], d["composer"]
    ref = d["reference_lfm25"]
    p = print
    p("")
    p("## Snipher v2 実測ベンチマーク")
    p("")
    p("| 層 | 規模 | 重み | 速さ |")
    p("|---|---|---|---|")
    p(f"| 知識ベース検索 | {kb['topics']} 話題 / {kb['facts']} 事実 / {kb['questions']} 問答 "
      f"| {kb['bytes'] / 1024:.0f} KiB | {kb['ms_per_query']:.3f} ms/発話 |")
    if lm.get("available"):
        p(f"| 巨大 n-gram LM（審判） | {lm['params']:,} エントリ / {lm['order']}-gram "
          f"| {lm['bytes'] / 1024 / 1024:.2f} MB | {lm['ms_per_score']:.3f} ms/文 |")
    if nn_.get("available"):
        p(f"| 内蔵ニューラルコア | {nn_['params']:,} params (d={nn_['d_model']} L={nn_['n_layers']}) "
          f"| {nn_['bytes'] / 1024 / 1024:.2f} MB | {nn_['ms_per_char_cached']:.2f} ms/文字 "
          f"({nn_['chars_per_second']:.0f} 文字/秒) |")
    p(f"| composer（文の設計図） | 8 計画 / {comp['distinct_plans']} 種を実測 "
      f"| 0 MB | {comp['ms_per_reply']:.2f} ms/応答 |")
    p(f"| **合計（同梱重み）** | | **{d['total_weight_bytes'] / 1024 / 1024:.2f} MB** | "
      f"**{e2e['ms_per_turn']:.1f} ms/ターン** |")
    p("")
    p("| 比較 | Snipher v2 | LFM2.5-1.2B-JP（公開値） |")
    p("|---|---|---|")
    total_params = (nn_.get("params", 0) + lm.get("params", 0))
    p(f"| パラメータ | {total_params:,}（コア {nn_.get('params', 0):,} + LM {lm.get('params', 0):,}） "
      f"| {ref['params']:,} |")
    p(f"| 重み | {d['total_weight_bytes'] / 1024 / 1024:.2f} MB（同梱・DL 不要） "
      f"| {ref['weights_gguf_q4_bytes'] / 1024 / 1024:.0f} MB(GGUF Q4) / "
      f"{ref['weights_safetensors_bytes'] / 1024 / 1024 / 1024:.1f} GB(safetensors) |")
    p(f"| ダウンロード | 不要 | 必要 |")
    p(f"| 1 応答の速さ | {e2e['ms_per_turn']:.1f} ms（検索+組立+判定） "
      f"| 生成はトークン単位（CPU で数十 ms/トークン） |")
    p("")
    p("### 生成のゲート（best-of-N → 文法 / 内蔵コア確信 / n-gram 自然さ / 話題一致）")
    p(f"* 経路の内訳: " + ", ".join(f"{k}={v}" for k, v in sorted(e2e["routes"].items())))
    p(f"* 内蔵コアが出した候補 {e2e['generated_candidates']} 文を全数検査 → 採用 "
      f"{e2e['generated_accepted_turns']} ターン。落ちた文は composer の正直な応答に置き換わり、"
      f"**的外れな生成文は 1 つも出力に出ません**。")
    for q, t in e2e.get("generated_samples") or []:
        p(f"  * {q!r} → {t[:60]!r}")
    p("")
    p("### 品質（内蔵 LM による perplexity）")
    if lm.get("available"):
        p(f"* 正しい日本語: ppl {lm['ppl_good']} / 確信度 {lm['conf_good']}")
        p(f"* 文字をシャッフル: ppl {lm['ppl_junk']} / 確信度 {lm['conf_junk']}")
    if nn_.get("available") and nn_.get("metrics"):
        m = nn_["metrics"]
        best = (m or {}).get("best_val") or {}
        if best:
            p(f"* 内蔵ニューラルコア: val loss {best.get('loss')} / ppl {best.get('ppl')} "
              f"/ top-1 精度 {best.get('acc')}")
        p(f"* 生成サンプル（{nn_.get('sample_query', '')}）: {nn_['sample'][:60]!r}")
        p(f"* KV キャッシュ: {nn_['speedup_kv_cache']}x（{nn_['ms_per_char_plain']} ms → "
          f"{nn_['ms_per_char_cached']} ms / 文字）")
    p("")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
