#!/usr/bin/env python3
"""欧文の語彙リスト（`snipher/data/words_en.txt.gz`）を作る。

`english-words`（MIT / 語彙データは SCOWL）が入っている環境でのみ動く *ビルド時* の道具。
実行時依存にはしない — 生成物をリポジトリに同梱するので、Snipher は numpy すら要りません。

    python tools/build_words_en.py            # → snipher/data/words_en.txt.gz

これで「5 文字の英単語を 3 つ」「A で始まる語」「Z で終わる語」「anagram にできる語」
といった欧文の語遊び・スペル確認も、辞書引きとして成立します。
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "snipher" / "data" / "words_en.txt.gz"
WORD = re.compile(r"^[a-z][a-z'-]{1,15}$")


def load_words() -> tuple[list[str], str]:
    try:
        import english_words                                   # type: ignore

        raw = english_words.get_english_words_set(["web2"], alpha=True, lower=True)
        out = sorted({str(w).strip() for w in raw if WORD.match(str(w).strip())})
        return out, "english-words (MIT) / SCOWL word list"
    except Exception as exc:  # noqa: BLE001
        print(f"english-words が使えないため語彙リストを作りません: {exc}", file=sys.stderr)
        return [], ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    words, src = load_words()
    words = [w for w in words if args.min_len <= len(w) <= args.max_len]
    if not words:
        print(json.dumps({"words": 0, "skipped": True}))
        return 1
    with gzip.GzipFile(str(args.out), "wb", compresslevel=9, mtime=0) as fh:
        fh.write("\n".join(words).encode("utf-8"))
    by_len: dict[str, int] = {}
    for w in words:
        by_len[str(len(w))] = by_len.get(str(len(w)), 0) + 1
    meta = {"words": len(words), "bytes": args.out.stat().st_size, "source": src,
            "by_length": by_len}
    (args.out.parent / "words_en.meta.json").write_text(json.dumps(meta, ensure_ascii=False,
                                                                   indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
