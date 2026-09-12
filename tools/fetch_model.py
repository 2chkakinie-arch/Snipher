"""LFM2.5-1.2B-JP のモデル取得 CLI（サーバー起動時の自動取得と同じロジック）。

通常このツールは不要 — Snipher は起動時にモデルを**全自動**で取得する
（snipher/lfm/acquire.py）。この CLI は事前取得やデバッグ用。

ソースは自動試行（レジューム対応）:
    キャッシュ → SNIPHER_LFM_URLS → 共有ミラー(ギガワタス) → HuggingFace 公式 → hf-mirror

使い方:
    python tools/fetch_model.py                          # GGUF(推奨・最速)を自動取得
    python tools/fetch_model.py --backend torch          # transformers 用一式を取得
    python tools/fetch_model.py --backend gguf --quant Q8_0
    python tools/fetch_model.py --dest var/models/x --only small   # 検証用(重み以外)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher.lfm.acquire import build_acquirer  # noqa: E402
from snipher.lfm.config import DEFAULT_GGUF_QUANT, DEFAULT_MODEL_ID, LfmConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["gguf", "torch"], default="gguf",
                    help="gguf=llama.cpp 用(既定・最速) / torch=transformers 用")
    ap.add_argument("--quant", default=DEFAULT_GGUF_QUANT,
                    help=f"GGUF の量子化(既定 {DEFAULT_GGUF_QUANT}: Q4_0/Q4_K_M/Q5_K_M/Q6_K/Q8_0/F16)")
    ap.add_argument("--dest", default="", help="保存先ディレクトリ(既定: var/models)")
    ap.add_argument("--only", choices=["small", "all"], default="all",
                    help="small=重み以外の小さいファイルのみ(検証用・torch のみ)")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = ap.parse_args()

    import os

    if args.dest:
        os.environ["SNIPHER_LFM_CACHE_DIR"] = args.dest
    os.environ["SNIPHER_LFM_GGUF_QUANT"] = args.quant
    cfg = LfmConfig()
    if args.model_id != DEFAULT_MODEL_ID:
        cfg.model_source = args.model_id
    acq = build_acquirer(cfg)
    if hasattr(acq, "model_id"):
        acq.model_id = args.model_id
    if args.backend == "gguf":
        acq.gguf_file = f"LFM2.5-1.2B-JP-202606-{args.quant}.gguf"

    plans = acq.plan_gguf() if args.backend == "gguf" else acq.plan_torch()
    if args.only == "small":
        plans = [p for p in plans if not p.filename.endswith((".safetensors", ".gguf"))]

    print(f"モデル: {args.model_id} ({args.backend})")
    print(f"保存先: {Path(cfg.cache_dir).resolve()}\n")

    # 進捗表示スレッド
    import threading

    stop = threading.Event()

    def _progress():
        while not stop.is_set():
            st = acq.status()
            if st["phase"] == "downloading" and st["bytes_total"]:
                pct = st["bytes_done"] / st["bytes_total"] * 100
                mb = st["bytes_done"] / 1e6
                tot = st["bytes_total"] / 1e6
                spd = st["speed_bps"] / 1e6
                sys.stdout.write(f"\r    {st['current_file']}: {mb:.1f}/{tot:.1f} MB ({pct:4.1f}%) {spd:.1f} MB/s [{st['current_source']}]")
                sys.stdout.flush()
            time.sleep(0.4)

    th = threading.Thread(target=_progress, daemon=True)
    th.start()
    ok, paths, err = acq.ensure_plan(plans, args.backend)
    stop.set()
    sys.stdout.write("\n")

    for line in acq.status()["log"]:
        print(" ", line)
    if not ok:
        print("\n未取得:", err)
        print("→ すべてのソースに接続できませんでした。ネットワークのある環境では")
        print("  サーバー起動時に自動で再試行されます（UI からも再試行できます）。")
        return 1
    print("\n完了! 取得先:", ", ".join(str(p) for p in paths))
    print("サーバー起動時に自動的にこのキャッシュが使われます:")
    print("  uvicorn snipher.api:app --host 0.0.0.0 --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
