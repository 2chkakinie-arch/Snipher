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
    # v3big … v3 より語彙と幅を広げた最終候補（d=384 / L=8 / 語彙 3,000）
    "v3big": {"d_model": 384, "n_layers": 8, "n_heads": 8, "conv_kernel": 4, "epochs": 3,
              "batch": 40, "seq_len": 112, "docs": 14000, "max_vocab": 3000,
              "authored_repeat": 6, "kb_repeat": 3, "corpus_docs": 26000},
    # v3    … 実辞書の語彙で組んだ大規模コーパスで育てる本番プロファイル
    #         （d=288 / L=6 / 語彙 2,400 字。int8 量子化で 7 MB 前後・1 文字 2ms 台）
    "v3": {"d_model": 288, "n_layers": 6, "n_heads": 8, "conv_kernel": 4, "epochs": 2,
           "batch": 48, "seq_len": 96, "docs": 12000, "max_vocab": 2400,
           "authored_repeat": 5, "kb_repeat": 3, "corpus_docs": 24000},
    # tiny  … CI/スモーク用（数十秒）
    "tiny": {"d_model": 48, "n_layers": 2, "n_heads": 4, "conv_kernel": 3, "epochs": 2,
             "batch": 32, "seq_len": 32, "docs": 1200, "max_vocab": 260,
             "authored_repeat": 2, "kb_repeat": 1},
    # base  … 軽いスナップショット（約 1M パラメータ）
    "base": {"d_model": 192, "n_layers": 6, "n_heads": 6, "conv_kernel": 4, "epochs": 8,
             "batch": 96, "seq_len": 64, "docs": 14000, "max_vocab": 720,
             "authored_repeat": 4, "kb_repeat": 2},
    # big   … 中間（約 2.9M パラメータ）
    "big": {"d_model": 256, "n_layers": 8, "n_heads": 8, "conv_kernel": 4, "epochs": 8,
            "batch": 64, "seq_len": 80, "docs": 22000, "max_vocab": 900,
            "authored_repeat": 6, "kb_repeat": 3},
    # huge  … 本番スナップショット（約 5.6M パラメータ / int8 で約 5MB）
    #         LFM2.5-1.2B-JP の 1/214 の重さで、1 文字 2ms（KV キャッシュ使用）
    "huge": {"d_model": 384, "n_layers": 10, "n_heads": 8, "conv_kernel": 4, "epochs": 6,
             "batch": 48, "seq_len": 88, "docs": 14000, "max_vocab": 1150,
             "authored_repeat": 8, "kb_repeat": 3},
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Snipher 内蔵ニューラルコアの蒸留ビルド")
    ap.add_argument("--profile", default="huge", choices=sorted(PROFILES))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--docs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--corpus", default=str(ROOT / "snipher" / "data" / "corpus.txt.gz"))
    ap.add_argument("--corpus-docs", type=int, default=0, help="0=すべて")
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

    # ---- 教師データの内訳（質の高い順に重みを付ける） ---------------------- #
    authored = builder.authored_docs(repeat=1)          # 人が書いた対話（最良）
    kb_docs = builder.kb_docs()                         # 知識ベース（事実・Q&A・手順）
    grammar = builder.build(prof["docs"]) if prof.get("docs") else []   # 文法生成（文型の網羅）
    if not args.corpus_docs:
        args.corpus_docs = int(prof.get("corpus_docs") or 0)
    if args.corpus and Path(args.corpus).exists():
        import gzip as _gzip

        fp = Path(args.corpus)
        opener = _gzip.open if fp.suffix == ".gz" else open
        with opener(fp, "rt", encoding="utf-8") as fh:   # type: ignore[operator]
            lines = [ln.strip() for ln in fh if ln.strip()]
        if args.corpus_docs:
            lines = lines[: args.corpus_docs]
        grammar = [f"<asst>{' '.join(lines[i:i + 3])}" for i in range(0, len(lines) - 2, 3)] + grammar
        print(f"corpus: {args.corpus} → {len(grammar):,} docs")
    ar, kr = int(prof.get("authored_repeat", 4)), int(prof.get("kb_repeat", 2))
    docs = authored * ar + kb_docs * kr + grammar
    docs = [d for d in docs if d and d.strip()]
    print(f"corpus: authored {len(authored)}x{ar} + kb {len(kb_docs)}x{kr} + "
          f"grammar {len(grammar)} = {len(docs)} docs / {sum(len(d) for d in docs)} chars "
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
    split = max(1, int(len(order) * 0.96))
    train_ids, val_ids = [], []

    def stream(idx_list, sink):
        for i in idx_list:
            sink.append(BOS)
            sink.extend(tok.encode(docs[i]))
            sink.append(EOS)

    stream(order[:split], train_ids)
    stream(order[split:], val_ids)
    if len(val_ids) < 4096:             # 検証データが薄すぎたら訓練末尾を分捕る
        take = min(4096, max(512, len(train_ids) // 20))
        val_ids = train_ids[-take:]
        train_ids = train_ids[:-take]
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

    # 各 epoch の終わりにスナップショットを書く（途中で止めても重みは使える）
    def snapshot(_net, m):
        info = save(args.out, _net, tok.vocab, extra={
            "trained_at": round(time.time(), 1), "profile": args.profile,
            "params": _net.n_params(), "docs": len(docs), "partial": True,
            "epoch": m.get("epoch"), "metrics": {"best_val": m}, "kv_cache": True,
            "corpus_seed": args.seed,
        })
        print(f"  snapshot → {info['path']} ({info['bytes'] / 1024:.0f} KiB) "
              f"epoch {m.get('epoch')} val_ppl={m.get('ppl')}", flush=True)

    res = trainer.train(on_log=log, on_epoch=snapshot)

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
        "corpus_mix": {"authored": len(authored) * ar, "kb": len(kb_docs) * kr,
                       "grammar": len(grammar)},
        "kv_cache": True,
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
