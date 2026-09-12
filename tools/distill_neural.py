"""Snipher 内蔵ニューラルコア（LFM2.5 アーキテクチャ蒸留）のビルド CLI。

    python tools/distill_neural.py                    # 本番スナップショット
    python tools/distill_neural.py --profile tiny     # 数秒（テスト/CI 用）
    python tools/distill_neural.py --epochs 12 --docs 20000

教師データは `snipher/data/*.json`（語彙テーブル）と `kb.json`（知識ベース）から
自動生成するので、外部データのダウンロードもファイルのアップロードも不要です。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DEFAULT = ROOT / "snipher" / "data" / "neural" / "core.npz"

PROFILES = {
    "tiny": {"d_model": 48, "n_layers": 2, "n_heads": 4, "conv_kernel": 3, "epochs": 2,
             "batch": 32, "seq_len": 32, "docs": 1200, "max_vocab": 260},
    "base": {"d_model": 192, "n_layers": 6, "n_heads": 6, "conv_kernel": 4, "epochs": 8,
             "batch": 96, "seq_len": 64, "docs": 14000, "max_vocab": 720},
    "big": {"d_model": 256, "n_layers": 8, "n_heads": 8, "conv_kernel": 4, "epochs": 12,
            "batch": 128, "seq_len": 80, "docs": 26000, "max_vocab": 900},
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Snipher 内蔵ニューラルコアの蒸留ビルド")
    ap.add_argument("--profile", default="base", choices=sorted(PROFILES))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--docs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--report", default=None, help="学習レポート(JSON)の書き出し先")
    args = ap.parse_args(argv)

    import numpy as np

    from snipher.lexicon import Lexicon
    from snipher.neural.corpus import CorpusBuilder
    from snipher.neural.nn import MicroNet, NNConfig
    from snipher.neural.store import save
    from snipher.neural.tokenizer import BOS, EOS, CharTokenizer
    from snipher.neural.train import TextDataset, TrainConfig, Trainer

    prof = dict(PROFILES[args.profile])
    for key in ("epochs", "batch", "seq_len", "docs"):
        v = getattr(args, key if key != "seq_len" else "seq_len")
        if v is not None:
            prof[key] = v
    if args.lr is not None:
        prof["lr"] = args.lr

    t0 = time.time()
    lex = Lexicon()
    builder = CorpusBuilder(lex, seed=args.seed)
    docs = builder.build(prof["docs"]) + builder.kb_docs()
    docs = [d for d in docs if d.strip()]
    print(f"corpus: {len(docs)} docs / {sum(len(d) for d in docs)} chars "
          f"({time.time() - t0:.1f}s)")

    tok = CharTokenizer.from_text("\n".join(docs), max_vocab=prof["max_vocab"])
    print(f"vocab: {tok.size()} chars")

    cfg = NNConfig(n_vocab=tok.size(), d_model=prof["d_model"], n_layers=prof["n_layers"],
                   n_heads=prof["n_heads"], conv_kernel=prof["conv_kernel"],
                   max_pos=max(64, prof["seq_len"] * 4))
    net = MicroNet.random(cfg, seed=args.seed)
    print(f"model: {net.n_params():,} params "
          f"(d={cfg.d_model} L={cfg.n_layers} blocks={''.join('c' if b=='conv' else 'a' for b in cfg.blocks)})")

    # ---- token stream -------------------------------------------------- #
    rng = np.random.default_rng(args.seed)
    order = list(range(len(docs)))
    rng.shuffle(order)
    split = max(1, int(len(order) * 0.98))
    train_ids, val_ids = [], []

    def stream(idx_list, sink):
        for i in idx_list:
            sink.append(BOS)
            sink.extend(tok.encode(docs[i]))
            sink.append(EOS)

    stream(order[:split], train_ids)
    stream(order[split:], val_ids)
    if len(val_ids) < 512:              # 検証データが薄すぎたら訓練末尾を分捕る
        val_ids = train_ids[-4096:]
        train_ids = train_ids[:-4096]
    tr = TextDataset(np.array(train_ids, dtype=np.int64), prof["seq_len"], rng)
    va = TextDataset(np.array(val_ids, dtype=np.int64), prof["seq_len"], np.random.default_rng(1))
    print(f"tokens: train={len(train_ids):,} val={len(val_ids):,}")

    tc = TrainConfig(seq_len=prof["seq_len"], batch=prof["batch"], epochs=prof["epochs"])
    if args.lr is not None:
        tc.lr = args.lr
    trainer = Trainer(net, tr, va, tc)
    last = {}

    def log(kind, m):
        nonlocal last
        last = m
        if kind == "eval":
            print(f"  epoch {m['epoch']:>2}  loss={m['loss']:.4f} ppl={m['ppl']:.2f} "
                  f"acc={m['acc']:.3f}  ({m['sec']:.0f}s)", flush=True)
        elif m["step"] % max(1, prof.get("log_every", 200)) == 0:
            print(f"    step {m['step']:>5} loss={m['loss']:.3f} lr={m['lr']:.2e} ({m['sec']:.0f}s)", flush=True)

    res = trainer.train(on_log=log)

    # ---- 生成サンプル --------------------------------------------------- #
    from snipher.neural.core import DistilledCore

    core = DistilledCore(net=net, tok=tok)
    probes = ["明日は", "私は猫が", "量子コンピュータ", "<user>よく眠れない\n<asst>"]
    samples = {p: core.generate(p, max_chars=42, temperature=0.7) for p in probes}
    comp = core.complete("私は毎日朝に")
    print("samples:")
    for k, v in samples.items():
        print(f"  {k!r:34} → {v!r}")
    print(f"  complete('私は毎日朝に') → {comp!r}")

    extra = {
        "trained_at": round(time.time(), 1),
        "profile": args.profile,
        "params": net.n_params(),
        "docs": len(docs),
        "train_tokens": len(train_ids),
        "metrics": res,
        "samples": samples,
        "corpus_seed": args.seed,
    }
    info = save(args.out, net, tok.vocab, extra=extra)
    print(f"saved: {info['path']} ({info['bytes'] / 1024:.0f} KiB, {info['params']:,} params)")
    if args.report:
        Path(args.report).write_text(json.dumps(extra, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
