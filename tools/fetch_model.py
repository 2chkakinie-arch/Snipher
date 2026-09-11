"""LFM2.5-1.2B-JP のモデル取得ツール（オフライン環境向けフォールバック付き）。

このサンドボックスのように HuggingFace への外向き接続が遮断されている環境では
transformers の自動ダウンロードが失敗する。そのため:

1. このツールを**インターネットのあるマシン**で実行してモデルを取得し、
   できあがったディレクトリを ``SNIPHER_LFM_MODEL`` に指定する、または
2. ブラウザ UI の「モデル管理」からモデルファイルをアップロードする
   (``GET /api/model/import`` で必要ファイルの判定ができる)。

ソースは上から順に試行する(最初に成功したところから取得):

    - https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606      (公式)
    - https://hf-mirror.com/LiquidAI/LFM2.5-1.2B-JP-202606       (公式のミラー)

ダウンロードは Range リクエストによる**レジューム対応**(中断しても再実行すれば
続きから再開)。依存は Python 標準ライブラリのみ。

使い方:
    python tools/fetch_model.py --dest var/models/LFM2.5-1.2B-JP
    python tools/fetch_model.py --dest var/models/x --only small   # 重み以外のみ(検証用)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"

SOURCES = [
    "https://huggingface.co/{model_id}/resolve/main/{filename}",
    "https://hf-mirror.com/{model_id}/resolve/main/{filename}",
]

# 必須ファイル(軽量順)。model.safetensors が本体の重み(~2.4GB)。
FILES = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "model.safetensors",
]

CHUNK = 1024 * 1024  # 1MB


def _open_with_resume(url: str, dest: Path) -> int:
    """Range リクエストでレジュームダウンロード。書き込んだ総バイト数を返す。"""
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "snipher-fetch/1.0"}
    if done:
        headers["Range"] = f"bytes={done}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        total = resp.headers.get("Content-Length")
        total = int(total) + done if total else None
        mode = "ab" if done and resp.status in (206, 200) and resp.status == 206 else "wb"
        with open(part, mode) as out:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    sys.stdout.write(f"\r    {dest.name}: {done/1e6:.1f}/{total/1e6:.1f} MB ({pct:4.1f}%)")
                else:
                    sys.stdout.write(f"\r    {dest.name}: {done/1e6:.1f} MB")
                sys.stdout.flush()
    sys.stdout.write("\n")
    part.rename(dest)
    return done


def fetch_file(filename: str, dest_dir: Path) -> tuple[bool, str]:
    """1 ファイルを取得。→ (成功?, メッセージ)"""
    dest = dest_dir / filename
    if dest.exists():
        return True, f"skip (存在: {filename})"
    last_err = "no source"
    for tmpl in SOURCES:
        url = tmpl.format(model_id=MODEL_ID, filename=filename)
        print(f"  {filename} <- {url.split('/resolve/')[0]}")
        for attempt in range(3):
            try:
                _open_with_resume(url, dest)
                return True, f"ok: {filename}"
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}"
                if e.code in (403, 404):
                    break  # 次のソースへ
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                last_err = str(e)[:200]
            time.sleep(2 * (attempt + 1))
    return False, f"fail: {filename} ({last_err})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="var/models/LFM2.5-1.2B-JP")
    ap.add_argument("--only", choices=["small", "all"], default="all",
                    help="small=重み以外の小さいファイルのみ(検証用)")
    ap.add_argument("--model-id", default=MODEL_ID)
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = list(FILES)
    if args.only == "small":
        files = [f for f in files if not f.endswith(".safetensors")]

    print(f"モデル: {args.model_id}")
    print(f"保存先: {dest.resolve()}\n")
    failed = []
    for f in files:
        ok, msg = fetch_file(f, dest)
        print(f"  -> {msg}")
        if not ok:
            failed.append(f)
    print()
    if failed:
        print("未取得:", ", ".join(failed))
        print("→ すべてのソースに接続できませんでした。ネットワークのあるマシンで実行するか、")
        print("  ブラウザから「モデル管理」パネルでアップロードしてください。")
        return 1
    print("完了! 起動時に次を指定してください:")
    print(f"  SNIPHER_LFM_MODEL={dest} uvicorn snipher.api:app --host 0.0.0.0 --port 8000")
    print(f"または UI の「モデル管理」→「アップロード済みモデルをロード」({dest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
