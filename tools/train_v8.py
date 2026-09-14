"""Snipher v8（再帰的思考モデル）の 3 フェーズ学習 CLI。numpy のみ。

ユーザー設計図の学習パイプラインを、そのまま 3 段で回します:

    Phase 1 継続事前学習 (Continual Pre-training)
        … 次の 1 トークンを正確に予測する素地（perplexity 低下）
    Phase 2 SFT + CoT（思考プロセス付き指示チューニング）
        … <think>…</think> を応答文頭に仕込んだ教師で「考えてから答える癖」を焼き付け
    Phase 3 DPO / GRPO-lite（思考力とフォーマット遵守の強化学習）
        … 「<think> で整理して答えた回答」を好み、ルール報酬で褒める

使い方:

    python tools/train_v8.py --phase pretrain --profile tiny --out var/v8.npz
    python tools/train_v8.py --phase all --profile v8b --out snipher/data/neural/v8.npz
    python tools/train_v8.py --phase all --profile v8a --out snipher/data/neural/v8_a.npz

モデルA（思考）・B（文章化）・C（校正）を役割別に 3 つ焼きたい場合は、--profile を
切り替えて 3 回動かします。CPU 2 コアでも tiny/base は数十秒〜数分で回ります。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROFILES = {
    # 本番（モデルB: 文章化）… GQA 6/2 + MoE 4x top2、約 0.5M パラメータ
    "v8b": {"d_model": 96, "n_layers": 3, "n_heads": 6, "n_kv_heads": 2,
            "n_experts": 4, "top_k": 2, "expert_dim": 192, "epochs": 2,
            "batch": 64, "seq_len": 64, "docs": 6000, "max_vocab": 1400,
            "cot_repeat": 6, "pretrain_repeat": 3},
    # モデルA（思考）・C（校正）用の軽量プロファイル
    "v8ac": {"d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
             "n_experts": 4, "top_k": 2, "expert_dim": 128, "epochs": 2,
             "batch": 64, "seq_len": 64, "docs": 4000, "max_vocab": 1000,
             "cot_repeat": 6, "pretrain_repeat": 3},
    # CI / スモーク用（数秒〜数十秒）
    "tiny": {"d_model": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
             "n_experts": 4, "top_k": 2, "expert_dim": 40, "epochs": 1,
             "batch": 32, "seq_len": 32, "docs": 400, "max_vocab": 160,
             "cot_repeat": 2, "pretrain_repeat": 2},
}


def build_docs(prof: dict, seed: int) -> tuple[list[str], list[str]]:
    """Phase 1（素地）と Phase 2（CoT 込み）の文書列を返す。"""
    from snipher.lexicon import Lexicon
    from snipher.neural.corpus import CorpusBuilder
    from snipher.neural.cot import build_cot_docs

    lex = Lexicon()
    builder = CorpusBuilder(lex, seed=seed)
    authored = builder.authored_docs(repeat=1)
    kb = builder.kb_docs()
    grammar = builder.build(prof["docs"]) if prof.get("docs") else []
    pretrain = authored * prof["pretrain_repeat"] + kb * 2 + grammar
    cot = build_cot_docs(seed=seed)
    sft = pretrain + cot * prof["cot_repeat"]
    return pretrain, sft


def make_datasets(docs: list[str], tok, prof: dict, seed: int):
    import numpy as np

    from snipher.neural.tokenizer import BOS, EOS
    from snipher.neural.train import TextDataset

    rng = np.random.default_rng(seed)
    order = list(range(len(docs)))
    rng.shuffle(order)
    split = max(1, int(len(order) * 0.95))
    train_ids: list[int] = []
    val_ids: list[int] = []

    def stream(idx_list, sink):
        for i in idx_list:
            sink.append(BOS)
            sink.extend(tok.encode(docs[i]))
            sink.append(EOS)

    stream(order[:split], train_ids)
    stream(order[split:], val_ids)
    if len(val_ids) < 1024:
        take = min(1024, max(256, len(train_ids) // 20))
        val_ids = train_ids[-take:]
        train_ids = train_ids[:-take]
    tr = TextDataset(np.array(train_ids, dtype=np.int64), prof["seq_len"], rng)
    va = TextDataset(np.array(val_ids, dtype=np.int64), prof["seq_len"], np.random.default_rng(1))
    return tr, va, len(train_ids)


def run_pretrain(net, tok, docs, prof: dict, seed: int, on_log) -> dict:
    from snipher.neural.moe_train import MoETrainer
    from snipher.neural.train import TrainConfig

    tr, va, ntok = make_datasets(docs, tok, prof, seed)
    tc = TrainConfig(seq_len=prof["seq_len"], batch=prof["batch"], epochs=prof["epochs"])
    trainer = MoETrainer(net, tr, va, tc)
    return trainer.train(on_log=on_log)


def run_dpo(net, tok, prof: dict, seed: int, steps: int = 60) -> dict:
    import copy

    import numpy as np

    from snipher.neural.align import dpo_step
    from snipher.neural.cot import build_dpo_pairs
    from snipher.neural.tokenizer import BOS

    pairs = build_dpo_pairs(seed=seed)
    rng = np.random.default_rng(seed)
    ref = copy.deepcopy(net)
    history: list[dict] = []
    for s in range(steps):
        pair = pairs[rng.integers(0, len(pairs))]
        ids_c = [BOS] + tok.encode(pair["prompt"] + "<asst>" + pair["chosen"])
        ids_r = [BOS] + tok.encode(pair["prompt"] + "<asst>" + pair["rejected"])
        res = dpo_step(net, ref, ids_c, ids_r, beta=0.3)
        history.append({"step": s, "loss": res["loss"], "margin": res["margin"]})
        if s and s % max(1, steps // 6) == 0:
            ref = copy.deepcopy(net)   # 参照モデルを定期更新
    return {"steps": steps, "final_loss": history[-1]["loss"], "history": history}


def run_grpo(net, tok, prof: dict, seed: int, steps: int = 8) -> dict:
    from snipher.neural.align import grpo_step
    from snipher.neural.moe_core import MoECore
    from snipher.neural.sample import SamplingControls

    core = MoECore(net=net, tok=tok)
    prompts = [f"<user>{a}×{b}は？\n<asst>" for a, b in
               [(3, 7), (6, 4), (5, 9), (8, 3), (7, 6), (4, 8)]]
    expected = [str(a * b) for a, b in
                [(3, 7), (6, 4), (5, 9), (8, 3), (7, 6), (4, 8)]]

    def gen(_net, prompt):
        body = prompt.rsplit("<asst>", 1)[0]
        ctrl = SamplingControls(temperature=0.8, top_k=40, no_repeat_ngram=4)
        return "<think>" + core.generate(body + "<asst>", max_chars=48,
                                          controls=ctrl) + "</think>" + \
               core.generate(body + "<asst>", max_chars=24, controls=ctrl)

    hist: list[dict] = []
    for s in range(steps):
        r = grpo_step(net, prompts, tok=tok, k=4, lr=2e-4, expected=expected, gen=gen)
        hist.append({"step": s, "mean_reward": r["mean_reward"]})
    return {"steps": steps, "final_mean_reward": hist[-1]["mean_reward"], "history": hist}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Snipher v8 再帰的思考モデルの学習")
    ap.add_argument("--phase", default="all",
                    choices=["all", "pretrain", "sft", "dpo", "grpo"])
    ap.add_argument("--profile", default="v8b", choices=sorted(PROFILES))
    ap.add_argument("--out", default=str(ROOT / "snipher" / "data" / "neural" / "v8.npz"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--docs", type=int, default=None)
    ap.add_argument("--dpo-steps", type=int, default=60)
    ap.add_argument("--grpo-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--report", default=None, help="学習レポート(JSON)の書き出し先")
    args = ap.parse_args(argv)

    prof = dict(PROFILES[args.profile])
    if args.epochs is not None:
        prof["epochs"] = args.epochs
    if args.docs is not None:
        prof["docs"] = args.docs
    if args.lr is not None:
        prof["lr"] = args.lr

    t0 = time.time()
    from snipher.neural.moe import MoEConfig, MoENet
    from snipher.neural.store import save
    from snipher.neural.tokenizer import CharTokenizer

    pretrain_docs, sft_docs = build_docs(prof, args.seed)
    print(f"docs: pretrain {len(pretrain_docs)} / sft {len(sft_docs)} ({time.time() - t0:.0f}s)")

    tok = CharTokenizer.from_text("\n".join(sft_docs), max_vocab=prof["max_vocab"])
    print(f"vocab: {tok.size()} chars")

    cfg = MoEConfig(n_vocab=tok.size(), d_model=prof["d_model"], n_layers=prof["n_layers"],
                    n_heads=prof["n_heads"], n_kv_heads=prof["n_kv_heads"],
                    n_experts=prof["n_experts"], top_k=prof["top_k"],
                    expert_dim=prof["expert_dim"], max_pos=288)
    net = MoENet.random(cfg, seed=args.seed)
    print(f"model: {net.n_params():,} params (d={cfg.d_model} L={cfg.n_layers} "
          f"GQA {cfg.n_heads}/{cfg.n_kv_heads} MoE {cfg.n_experts}x top{cfg.top_k})")

    last_log = {}

    def log(kind, m):
        last_log.update(m)
        if kind == "eval":
            print(f"  epoch {m['epoch']:>2} loss={m['loss']:.4f} ppl={m['ppl']:.2f} "
                  f"acc={m['acc']:.3f} ({m['sec']:.0f}s)", flush=True)

    report: dict = {"profile": args.profile, "params": net.n_params(),
                    "vocab": tok.size(), "phases": {}}

    if args.phase in ("all", "pretrain", "sft"):
        print(f"[Phase 1] 継続事前学習 ({len(pretrain_docs)} docs)…", flush=True)
        report["phases"]["pretrain"] = run_pretrain(net, tok, pretrain_docs, prof, args.seed, log)
        print(f"[Phase 2] SFT + CoT ({len(sft_docs)} docs)…", flush=True)
        report["phases"]["sft"] = run_pretrain(net, tok, sft_docs, prof, args.seed, log)

    if args.phase in ("all", "dpo"):
        print(f"[Phase 3a] DPO ({args.dpo_steps} steps)…", flush=True)
        report["phases"]["dpo"] = run_dpo(net, tok, prof, args.seed, steps=args.dpo_steps)

    if args.phase in ("all", "grpo"):
        print(f"[Phase 3b] GRPO-lite ({args.grpo_steps} steps)…", flush=True)
        report["phases"]["grpo"] = run_grpo(net, tok, prof, args.seed, steps=args.grpo_steps)

    # ---- 生成サンプル（学習の成果） ------------------------------------- #
    from snipher.neural.moe_core import MoECore

    core = MoECore(net=net, tok=tok)
    probes = ["赤信号ではどうする？", "3×7は？", "私は猫が"]
    samples = {p: core.generate(f"<user>{p}\n<asst>", max_chars=48, temperature=0.7)
               for p in probes}
    for k, v in samples.items():
        print(f"  sample {k!r} → {v!r}")

    extra = {
        "trained_at": round(time.time(), 1),
        "profile": args.profile,
        "params": net.n_params(),
        "docs": {"pretrain": len(pretrain_docs), "sft": len(sft_docs)},
        "report": report,
        "samples": samples,
    }
    info = save(args.out, net, tok.vocab, extra=extra)
    print(f"saved: {info['path']} ({info['bytes'] / 1024:.0f} KiB, {info['params']:,} params)")
    if args.report:
        Path(args.report).write_text(json.dumps(extra, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
