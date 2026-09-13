#!/usr/bin/env python3
"""`public/` ビルド — Cloudflare Pages に置く静的 UI と *Pages Function 用の知識ファイル* を作る。

Cloudflare Pages は静的アセット + JS/Wasm の Functions を動かす場所で、Python プロセスを
載せられません。一方 Snipher の会話の中核（索引 → 証拠 → 文の組み立て）は
*重みが要らない検索＋合成* なので、JS に移植して Pages 側で完結させます。

生成物:
  public/index.html      … 会話 UI（snipher/web/chat.html と同じ画面。API が無い環境では内蔵 engine で答える）
  public/engine.mjs      … 検索と合成（Pages Function とブラウザの両方から import される 1 本）
  public/kb.json         … kb.json から *JS が使う欄だけ* を抜いた軽量版
  public/kb.meta.json    … 規模の表示用
  public/_headers        … キャッシュ方針（ハッシュ付きアセットは不変、HTML は no-store）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_KB = ROOT / "snipher" / "data" / "kb.json"
SRC_ENGINE = ROOT / "snipher" / "web" / "engine.mjs"
SRC_HTML = ROOT / "snipher" / "web" / "chat.html"
SRC_PAGES = ROOT / "snipher" / "web" / "pages"
OUT = ROOT / "public"
FUNCS = ROOT / "functions"

# Pages Function が *同期で読む* ファイルなので、大きくしすぎない（Workers の memory 上限）
MAX_KB_BYTES = 4 * 1024 * 1024
KEEP = ("id", "topic", "cat", "aliases", "def", "facts", "why", "how", "tips",
        "opinion", "qa", "followups", "related", "tags")


def slim_kb(path: Path) -> tuple[list[dict], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items: list[dict] = []
    for it in data.get("items") or []:
        row = {k: it.get(k) for k in KEEP if it.get(k)}
        if not row.get("def") and not row.get("facts"):
            continue
        row["aliases"] = [str(a).lower() for a in (row.get("aliases") or [])][:24]
        row["qa"] = [[str(q), str(a)] for q, a in (row.get("qa") or [])[:6]]
        row["followups"] = [str(x) for x in (row.get("followups") or [])[:4]]
        items.append(row)
    meta = {"version": data.get("version") or 1, "topics": len(items),
            "facts": sum(len(i.get("facts") or []) for i in items),
            "qa": sum(len(i.get("qa") or []) for i in items)}
    return items, meta


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def build(*, check_only: bool = False) -> int:
    if not SRC_KB.exists():
        print(f"kb がありません: {SRC_KB}（先に `python tools/build_kb.py`）", file=sys.stderr)
        return 2
    items, meta = slim_kb(SRC_KB)
    kb_text = json.dumps({"meta": meta, "items": items}, ensure_ascii=False,
                         separators=(",", ":"))
    if len(kb_text.encode("utf-8")) > MAX_KB_BYTES:
        print(f"public/kb.json が大きすぎます: {len(kb_text.encode('utf-8'))} bytes", file=sys.stderr)
        return 2
    engine = SRC_ENGINE.read_text(encoding="utf-8")
    html = SRC_HTML.read_text(encoding="utf-8")
    # 静的デモ（Pages）では /api が無いので、内蔵 engine で答える。フォールバックを 1 行だけ仕込む。
    # UI は /api が無い環境向けに ./engine.mjs を読む（Pages の静的デモで同じ答えが返る）
    headers = (
        "/*\n  Cache-Control: public, max-age=0, must-revalidate\n"
        "  X-Content-Type-Options: nosniff\n"
        "  Referrer-Policy: same-origin\n"
        "/kb.json\n  Cache-Control: public, max-age=300\n"
        "/engine.mjs\n  Cache-Control: public, max-age=300\n"
        "/assets/*\n  Cache-Control: public, max-age=31536000, immutable\n"
    )
    built = {
        "kb.json": kb_text,
        "engine.mjs": engine,
        "index.html": html,
        "kb.meta.json": json.dumps({**meta, "sha": digest(kb_text)}, ensure_ascii=False, indent=2),
        "_headers": headers,
    }
    funcs = {
        "_engine/engine.mjs": engine,
        "api/[[path]].js": (SRC_PAGES / "[[path]].js").read_text(encoding="utf-8"),
    }
    if check_only:
        bad = [f"public/{name}" for name, text in built.items()
               if not (OUT / name).exists() or (OUT / name).read_text(encoding="utf-8") != text]
        bad += [f"functions/{name}" for name, text in funcs.items()
                if not (FUNCS / name).exists() or (FUNCS / name).read_text(encoding="utf-8") != text]
        if bad:
            print("public/ が最新ではありません: " + ", ".join(sorted(bad))
                  + "  →  python tools/build_public.py", file=sys.stderr)
            return 1
        print(f"public/ は最新です（topics={meta['topics']} facts={meta['facts']}）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in built.items():
        (OUT / name).write_text(text, encoding="utf-8")
    for name, text in funcs.items():
        dest = FUNCS / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    if (OUT / "assets").exists():
        shutil.rmtree(OUT / "assets")
    (OUT / "assets").mkdir(exist_ok=True)
    total = sum(len(t.encode("utf-8")) for t in list(built.values()) + list(funcs.values()))
    print(f"wrote public/+functions/ : topics={meta['topics']} facts={meta['facts']} qa={meta['qa']} "
          f"kb={len(kb_text.encode('utf-8')) / 1024:.0f} KiB total={total / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="差分があるとき exit 1")
    args = ap.parse_args()
    raise SystemExit(build(check_only=args.check))
