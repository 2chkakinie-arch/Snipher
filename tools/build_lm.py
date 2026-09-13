#!/usr/bin/env python3
"""巨大 n-gram 言語モデル（snipher/data/lm.npz）をビルドする。

    python tools/build_lm.py [--grammar 30000] [--order 5] [--out snipher/data/lm.npz]

学習データは 3 種類、すべてリポジトリ内の素材だけ:

1. **人が書いた日本語** … `kb.json`（196 話題・約 3000 文）と `dialogues.json`（対話 220 組）
2. **文法で生成した文** … `CorpusBuilder` が語彙テーブルから作る文（数万〜数十万）
3. **対話文書**          … `<user>` 発話 → `<asst>` 応答のペア

これで「事実を言う文」「相手に返す文」「文末を畳む文」の 3 つの分布を同時に学びます。
ビルド後は保持データ／ランダム文字列／文法を壊した文で perplexity を比較し、
日本語の良し悪しを判別できるかをその場で検証します。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snipher import knowledge                      # noqa: E402
from snipher.lm import CharNgramLM, normalize       # noqa: E402
from snipher.neural.corpus import CorpusBuilder     # noqa: E402


def strip_markers(doc: str) -> list[str]:
    """<user>/<asst> 文書を、そのまま学習できる平文に割る。"""
    out: list[str] = []
    for line in doc.split("\n"):
        line = line.strip()
        for tag in ("<user>", "<asst>"):
            if line.startswith(tag):
                line = line[len(tag):].strip()
        if line.startswith("続き:"):
            line = line[len("続き:"):].strip()
        if line:
            out.append(line)
    return out


def read_corpus(path: Path | None, limit: int) -> list[str]:
    """tools/build_corpus.py が作った実辞書ベースの文（corpus.txt.gz）を読む。"""
    if not path or not Path(path).exists():
        return []
    import gzip as _gzip

    fp = Path(path)
    opener = _gzip.open if fp.suffix == ".gz" else open
    out: list[str] = []
    with opener(fp, "rt", encoding="utf-8") as fh:      # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if 4 <= len(line) <= 140:
                out.append(line)
            if len(out) >= limit:
                break
    return out


def collect(grammar: int, seed: int = 20250912,
            corpus: Path | None = None) -> tuple[list[str], list[str], list[str]]:
    """(学習文, 評価用の保持文, 語彙構築用の全文) を返す。

    語彙（文字集合）は保持文も含めて作る。文字種を知らないだけで perplexity が
    跳ね上がるので、そこは評価を歪めないために先に揃えておく。
    """
    kb = knowledge.KnowledgeBase()
    authored: list[str] = []
    dpath = Path(kb.path).parent / "dialogues.json" if hasattr(kb, "path") else None
    if dpath is None or not dpath.exists():
        dpath = ROOT / "snipher" / "data" / "dialogues.json"
    if dpath.exists():
        data = json.loads(dpath.read_text(encoding="utf-8"))
        for pair in data.get("dialogues", []):
            for k in ("user", "asst"):
                s = str(pair.get(k, "")).strip()
                if s:
                    authored.append(s)

    kb_sents = [s for s in kb.all_sentences() if 4 <= len(s) <= 140]

    rng = random.Random(seed)
    held = rng.sample(kb_sents, min(400, len(kb_sents)))
    held_set = set(held)
    kb_train = [s for s in kb_sents if s not in held_set]

    cb = CorpusBuilder(seed=seed)
    docs = cb.build(max(1000, grammar)) if grammar else []
    # 指示追従（SFT）の形も審判に読ませる。JSON・箇条書き・短い聞き返しを
    # 「ありえない日本語」として減点すると、応答の組み立てが型から逃げます。
    sft_lines: list[str] = []
    try:
        for doc in cb.sft_docs():
            for line in strip_markers(doc):
                if 4 <= len(line) <= 200:
                    sft_lines.append(line)
    except Exception:  # noqa: BLE001
        sft_lines = []
    gen: list[str] = []
    for doc in docs:
        for line in strip_markers(doc):
            if 4 <= len(line) <= 140:
                gen.append(line)
    # 実辞書の語で組み立てた文（語彙が一桁多い。流暢さの審判の主食）
    gen.extend(read_corpus(corpus, 400_000))
    rng.shuffle(gen)

    # 人が書いた日本語は 3 倍に重み付け（分布の中心に置く）
    texts = authored * 3 + kb_train * 2 + sft_lines * 2 + gen
    rng.shuffle(texts)
    # 語彙は学習文すべて（保持文も含む）から作る
    return texts, held, authored + kb_sents + gen


def evaluate(lm: CharNgramLM, held: list[str], authored: list[str]) -> dict:
    """保持データ／自作対話／文字シャッフル／文法破壊の 4 分布を比べて検証する。

    perplexity は中央値で見る（稀な文字を含む 1 文が平均を壊すため）。
    """
    import math
    import statistics

    def ppls(texts: list[str]) -> list[float]:
        out = []
        for t in texts:
            if not t:
                continue
            lp, n = lm.logprob(t)
            if n:
                out.append(min(1e6, math.exp(-lp)))
        return out or [1e6]

    rng = random.Random(7)
    shuffled = []
    for t in held[:120]:
        chars = list(t)
        rng.shuffle(chars)
        shuffled.append("".join(chars))
    broken = []
    for t in held[:120]:
        s = t.replace("です", "ですです").replace("ます", "ますます")
        if s == t:
            s = (t[:-2] + "をに") if len(t) > 4 else (t + "をに")
        broken.append(s)
    trunc = [t[: max(3, len(t) // 2)] for t in held[:120]]

    h, a, sh, br, tr = (ppls(held), ppls(authored[:300]), ppls(shuffled),
                        ppls(broken), ppls(trunc))
    med = statistics.median
    lo = sorted(h)[max(0, int(len(h) * 0.25) - 1)]
    hi = med(sh)
    # 確信度はこの校正値を当ててから測る（当てないと汎用カーブになってしまう）
    lm.calib = {"lo": round(float(lo), 4), "hi": round(float(hi), 4), "source": "build_lm"}
    confs = [lm.confidence(t) for t in held if t]
    return {
        "ppl_lo": round(float(lo), 3),
        "ppl_hi": round(float(hi), 3),
        "ppl_kb_heldout": round(float(med(h)), 3),
        "ppl_authored": round(float(med(a)), 3),
        "ppl_shuffled_chars": round(float(med(sh)), 3),
        "ppl_broken_grammar": round(float(med(br)), 3),
        "ppl_truncated": round(float(med(tr)), 3),
        "mean_conf_kb": round(statistics.fmean(confs), 4) if confs else 0.0,
        "median_conf_kb": round(float(med(confs)), 4) if confs else 0.0,
        "frac_conf_ge_06": round(sum(1 for c in confs if c >= 0.6) / max(1, len(confs)), 4),
        "mean_conf_shuffled": round(statistics.fmean([lm.confidence(t) for t in shuffled]), 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grammar", type=int, default=30000, help="文法生成文の上限")
    ap.add_argument("--order", type=int, default=5, help="n-gram の次数（既定 5）")
    ap.add_argument("--min-count", type=str, default="1,1,1,2,2,3",
                    help="次数ごとの最小出現回数（既定 1,1,1,2,2,3）")
    ap.add_argument("--max-vocab", type=int, default=3500, help="文字語彙の上限")
    ap.add_argument("--out", type=Path, default=ROOT / "snipher" / "data" / "lm.npz")
    ap.add_argument("--corpus", type=Path, default=ROOT / "snipher" / "data" / "corpus.txt.gz",
                    help="tools/build_corpus.py の生成文（無ければ無視）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    texts, held, all_text = collect(args.grammar, corpus=args.corpus)
    if not args.quiet:
        chars = sum(len(normalize(t)) for t in texts)
        print(f"[lm] 学習文 {len(texts):,} / {chars:,} 文字 / 保持 {len(held)} 文")

    mc = tuple(int(x) for x in str(args.min_count).split(","))
    lm = CharNgramLM(order=args.order)
    lm.build_vocab(all_text, max_vocab=args.max_vocab)
    stats = lm.train(texts, min_counts=mc, build_vocab=False)
    if not args.quiet:
        print(f"[lm] 学習 {stats['seconds']}s / vocab {stats['vocab']:,} / entries {stats['entries']}")

    ev = evaluate(lm, held, texts[:400])
    lm.calib = {"lo": ev["ppl_lo"], "hi": ev["ppl_hi"], "source": "build_lm"}
    lm.metrics["eval"] = ev
    info = lm.save(args.out)
    info.update(ev)
    info["seconds"] = round(time.time() - t0, 2)
    info["params"] = lm.n_params()
    if not args.quiet:
        print(f"[lm] 保存 {info['path']} ({info['bytes'] / 1e6:.2f} MB) / "
              f"{info['params']:,} entries / {info['seconds']}s")
        print(f"[lm] 校正 lo={ev['ppl_lo']} hi={ev['ppl_hi']}")
        print(f"[lm] ppl(中央値) 保持KB={ev['ppl_kb_heldout']} 自作対話={ev['ppl_authored']} "
              f"文法破壊={ev['ppl_broken_grammar']} 途中切断={ev['ppl_truncated']} "
              f"文字シャッフル={ev['ppl_shuffled_chars']}")
        print(f"[lm] conf 保持KB 中央値={ev['median_conf_kb']} (0.6以上 {ev['frac_conf_ge_06']:.0%}) "
              f"シャッフル={ev['mean_conf_shuffled']}")
        if ev["ppl_shuffled_chars"] <= ev["ppl_kb_heldout"] * 3:
            print("[lm] 警告: 判別力が低い（学習データ不足の疑い）")
        if ev["median_conf_kb"] < 0.5:
            print("[lm] 警告: 保持KBの確信度が低い（語彙の取りこぼしか）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
