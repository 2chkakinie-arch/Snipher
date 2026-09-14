"""Snipher v8（再帰的思考モデル）のベンチマーク。

測るもの（すべて実測・テンプレートや後処理の助けなし）:

  * アーキテクチャ: パラメータ数 / GQA の KV 削減比 / MoE の expert 構成
  * 思考コンプライアンス: <think>…</think> を正しく閉じる率
  * ドラフト検証: 校正モデルC の受理率・却下理由
  * 反復封印: 生成中に同一 n-gram が周回しない率（100% が目標）
  * 長文生成: テンプレート無しで到達した文字数・定型表現ゼロ
  * 速度: 1 文字あたりの生成時間

使い方:

    python tools/bench_v8.py                 # 出荷済み v8*.npz を測る
    python tools/bench_v8.py --train-tiny    # 重みが無ければ tiny を即席学習して測る
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEMPLATE_MARKERS = ("の知識で答えます", "としてお答えします", "以下の通りです",
                    "ご質問ありがとうございます", "お役に立てれば幸いです")


def _load_mind():
    from snipher.mind.recurrent import RecurrentMind

    try:
        mind = RecurrentMind()
        if mind.available:
            return mind
    except Exception:  # noqa: BLE001
        pass
    return None


def _train_tiny_mind():
    """重みが無い環境でも測れるように、tiny をその場で学習する。"""
    from snipher.neural.moe import MoEConfig, MoENet
    from snipher.neural.moe_core import MoECore
    from snipher.neural.moe_train import MoETrainer
    from snipher.neural.tokenizer import BOS, CharTokenizer, EOS
    from snipher.neural.train import TextDataset, TrainConfig
    from snipher.mind.recurrent import RecurrentMind
    import numpy as np

    corpus = [
        "<user>赤信号ではどうする？\n<asst><think>信号の意味を思い出す。赤は停止。だから止まる。</think>\n止まります。",
        "<user>3×7は？\n<asst><think>3を7回足すと21。</think>\n21",
        "<user>私は猫が\n<asst>好きです。毎日なでています。",
        "<user>雨の日は何をする？\n<asst><think>濡れない工夫を考える。傘を持つ。部屋で本を読む。</think>\n家で本を読みます。",
        "<user>好きな食べ物は？\n<asst><think>好きな物を思い浮かべる。果物とごはん。</think>\n果物が好きです。",
    ] * 60
    tok = CharTokenizer.from_text("\n".join(corpus), max_vocab=160)
    ids: list[int] = []
    for t in corpus:
        ids += [BOS] + tok.encode(t) + [EOS]
    cfg = MoEConfig(n_vocab=tok.size(), d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                    n_experts=4, top_k=2, expert_dim=40, max_pos=128)
    net = MoENet.random(cfg, seed=7)
    ds = TextDataset(np.array(ids, dtype=np.int64), 48, np.random.default_rng(0))
    tc = TrainConfig(seq_len=48, batch=32, epochs=3, lr=8e-3, min_lr=1e-4, warmup_ratio=0.0)
    MoETrainer(net, ds, ds, tc).train()
    return RecurrentMind(cores={"A": MoECore(net=net, tok=tok), "B": MoECore(net=net, tok=tok),
                                "C": MoECore(net=net, tok=tok)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tiny", action="store_true", help="重みが無い場合 tiny を学習して測る")
    ap.add_argument("--novel-chars", type=int, default=2000)
    ap.add_argument("--probes", type=int, default=20)
    ap.add_argument("--json", action="store_true", help="JSON で出力")
    args = ap.parse_args(argv)

    mind = _load_mind()
    trained_here = False
    if mind is None and args.train_tiny:
        mind = _train_tiny_mind()
        trained_here = True
    if mind is None:
        print("v8 重みがありません。先に `python tools/train_v8.py --phase all --profile v8b` "
              "（または --train-tiny）", file=sys.stderr)
        return 2

    st = mind.status()
    res: dict = {"model": "Snipher v8 (recurrent)", "trained_here": trained_here,
                 "status": st, "sections": {}}

    # ---- 1) アーキテクチャ -------------------------------------------------
    core_b = mind.cores["B"]
    cfg = core_b.net.cfg
    arch = {
        "params": core_b.net.n_params(),
        "d_model": cfg.d_model, "layers": cfg.n_layers,
        "blocks": list(cfg.blocks),
        "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
        "gqa_kv_reduction": round(cfg.n_heads / cfg.n_kv_heads, 1),
        "n_experts": cfg.n_experts, "top_k": cfg.top_k, "expert_dim": cfg.expert_dim,
        "vocab": cfg.n_vocab, "rope": True, "rmsnorm": True,
    }
    res["sections"]["architecture"] = arch

    # ---- 2) 思考コンプライアンス -------------------------------------------
    probes = ["赤信号ではどうする？", "3×7は？", "雨の日は何をする？",
              "好きな食べ物は？", "私は猫が"] * (args.probes // 5 + 1)
    probes = probes[:args.probes]
    closed = 0
    nonempty = 0
    for i, p in enumerate(probes):
        th = mind.think(p, seed=i)
        if th["text"]:
            nonempty += 1
        if th["closed"]:
            closed += 1
    res["sections"]["think_compliance"] = {
        "probes": len(probes),
        "nonempty": nonempty,
        "closed": closed,
        "close_rate": round(closed / len(probes), 3),
    }

    # ---- 3) ドラフト検証（受理率・却下理由） -------------------------------
    accepted = 0
    rounds_total = 0
    reason_counts: dict[str, int] = {}
    for i, p in enumerate(probes):
        r = mind.respond(p, max_chars=48, seed=i)
        accepted += 1 if r["accepted"] else 0
        rounds_total += r["rounds"]
        for reason in r["reasons"]:
            key = reason.split(":")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
    res["sections"]["draft_verification"] = {
        "probes": len(probes),
        "accepted": accepted,
        "accept_rate": round(accepted / len(probes), 3),
        "mean_rounds": round(rounds_total / len(probes), 2),
        "reject_reasons": reason_counts,
    }

    # ---- 4) 反復封印 --------------------------------------------------------
    def max_ngram_repeat(text: str, n: int = 4) -> int:
        seen: dict[str, int] = {}
        worst = 0
        for i in range(len(text) - n + 1):
            g = text[i:i + n]
            seen[g] = seen.get(g, 0) + 1
            worst = max(worst, seen[g])
        return worst

    t0 = time.time()
    novel = mind.generate_novel("小さな町のはずれに古い時計台がありました。",
                                max_chars=args.novel_chars, temperature=0.9, seed=3)
    gen_secs = time.time() - t0
    worst4 = max_ngram_repeat(novel, 4)
    worst5 = max_ngram_repeat(novel, 5)
    templates = [m for m in TEMPLATE_MARKERS if m in novel]
    markers = [m for m in ("<think>", "</think>", "<user>", "<asst>", "<sys>") if m in novel]
    res["sections"]["novel_generation"] = {
        "target_chars": args.novel_chars,
        "reached_chars": len(novel),
        "reach_rate": round(len(novel) / args.novel_chars, 3),
        "template_markers": templates,
        "control_markers": markers,
        "max_4gram_repeat": worst4,
        "max_5gram_repeat": worst5,
        # 反復封印は 5-gram で掛かる（no_repeat_ngram=5）。5-gram が 1 回以下なら周回なし
        "loop_free": worst5 <= 1,
        "seconds": round(gen_secs, 2),
        "chars_per_sec": round(len(novel) / max(gen_secs, 1e-3), 1),
    }

    # ---- 5) サンプル --------------------------------------------------------
    res["samples"] = {
        "think": mind.think("赤信号ではどうする？", seed=0)["text"],
        "respond": mind.respond("3×7は？", max_chars=48, seed=0)["text"],
        "novel_head": novel[:120],
    }

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0

    print("=== Snipher v8 ベンチマーク ===")
    a = arch
    print(f"[1] アーキテクチャ   params={a['params']:,}  d={a['d_model']} L={a['layers']} "
          f"GQA {a['n_heads']}/{a['n_kv_heads']} (KV×1/{a['gqa_kv_reduction']}) "
          f"MoE {a['n_experts']}x top{a['top_k']} (dim {a['expert_dim']})  vocab={a['vocab']} "
          f"RoPE+RMSNorm")
    tc = res["sections"]["think_compliance"]
    print(f"[2] 思考コンプライアンス  閉じ率 {tc['close_rate']*100:.0f}% "
          f"({tc['closed']}/{tc['probes']})  非空 {tc['nonempty']}/{tc['probes']}")
    dv = res["sections"]["draft_verification"]
    print(f"[3] ドラフト検証  受理率 {dv['accept_rate']*100:.0f}% ({dv['accepted']}/{dv['probes']}) "
          f"平均ラウンド {dv['mean_rounds']}  却下理由 {dv['reject_reasons'] or 'なし'}")
    ng = res["sections"]["novel_generation"]
    print(f"[4] 長文生成  {ng['reached_chars']}/{ng['target_chars']} chars "
          f"({ng['reach_rate']*100:.0f}%)  5-gram 最大反復 {ng['max_5gram_repeat']} "
          f"(周回なし={ng['loop_free']}) 定型表現 {ng['template_markers'] or 'なし'} "
          f"制御残骸 {ng['control_markers'] or 'なし'} "
          f"{ng['seconds']}s ({ng['chars_per_sec']} chars/s)")
    print(f"[5] サンプル  think={res['samples']['think']!r}")
    print(f"                respond={res['samples']['respond']!r}")
    print(f"                novel={res['samples']['novel_head']!r}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
