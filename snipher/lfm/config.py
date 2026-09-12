"""LFM ニューラルコアの設定（環境変数で上書き可能）。

Snipher の内部構造として LFM2.5-1.2B-JP を動かすための設定を集約する。

バックエンド（自動選択が既定）:
    - ``gguf`` : llama.cpp（llama-cpp-python または llama-server）+ 公式 GGUF。
      CPU で最速。LFM2.5-1.2B-JP-202606-Q4_K_M.gguf (~731MB) を自動取得する。
    - ``torch``: transformers + 動的 INT8 量子化。model.safetensors (~2.2GB) を
      自動取得する。未知文字の埋め込み学習（予約トークン）はこのバックエンド
      でのみ可能。
    - ``off``  : ニューラルコアを無効化（超小型エンジンのみ）。

モデルの取得は完全に自動（tools なし・アップロード不要）:
    キャッシュ → SNIPHER_LFM_URLS → ギガワタス共有(公式の代替ミラー) →
    HuggingFace 公式 → hf-mirror の順に試行し、レジューム付きで取得する。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# HuggingFace への接続が遮断されている環境(サンドボックス等)では
# 自動ダウンロードが長時間固まるので、既定で短いタイムアウトを設定する。
# 本物のネットワークが遅い環境では環境変数で上書きできる。
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "15")

# Liquid AI の日本語チャットモデル（LFM2.5 アーキテクチャ / 1.17B / 32K context）
DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"
# 公式 GGUF リポジトリ（llama.cpp 用・CPU 推論に最適化済み）
DEFAULT_GGUF_REPO = "LiquidAI/LFM2.5-1.2B-JP-202606-GGUF"
# 既定の量子化。Q4_K_M = 731MB で品質と速度のバランスが最も良い。
DEFAULT_GGUF_QUANT = os.environ.get("SNIPHER_LFM_GGUF_QUANT", "Q4_K_M").strip() or "Q4_K_M"

# ユーザー提供のモデル共有ミラー（model.safetensors 2.18GB）。
# HuggingFace に直接繋がらない環境向けの重量ファイルの代替ソースとして、
# 自動取得の試行リストに組み込まれている（期限切れの場合は自動でスキップ）。
GIGA_WATASU_SHARE_URL = os.environ.get(
    "SNIPHER_LFM_SHARE_PAGE", "https://giga-watasu.jp/d/aabdc5853ed6a0696ee4a965"
).strip()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_DIR = REPO_ROOT / "var" / "learned_vocab"
DEFAULT_CACHE_DIR = REPO_ROOT / "var" / "models"

DEFAULT_SYSTEM_PROMPT = (
    "あなたは Snipher という汎用の日本語 AI 会話パートナーです。"
    "雑談、日常の相談、学習、数学、プログラミング、文章作成を依頼の目的に合わせて扱います。"
    "質問には結論と理由を、計算には途中式を、コードには実行可能なコードブロックを示します。"
    "現在の出来事・価格・天気・最新仕様は、与えられた検索材料だけを根拠にし、推測で断定しません。"
    "自然で明快な日本語を使い、不要な前置きや『私は分かりません』だけの返答で終わらせません。"
)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def is_serverless() -> bool:
    """サーバーレス（Vercel 等）環境では 731MB の自動取得を行わない。"""
    for k in ("VERCEL", "AWS_LAMBDA_FUNCTION_NAME", "NETLIFY", "LAMBDA_TASK_ROOT"):
        if os.environ.get(k):
            return True
    return False


@dataclass
class LfmConfig:
    """環境変数 SNIPHER_LFM_* で上書きできる設定。"""

    # モデルソース: ローカルディレクトリ / .gguf ファイル / HuggingFace モデルID。
    # 未指定なら自動取得（acquire.py がキャッシュへダウンロードする）。
    model_source: str = field(
        default_factory=lambda: os.environ.get("SNIPHER_LFM_MODEL", "").strip()
        or DEFAULT_MODEL_ID
    )
    # 明示的な GGUF パス（指定すると gguf バックエンドがそれをそのまま使う）
    gguf_path: str = field(default_factory=lambda: os.environ.get("SNIPHER_LFM_GGUF", "").strip())
    # バックエンド: auto | gguf | torch | off
    backend: str = field(
        default_factory=lambda: os.environ.get("SNIPHER_LFM_BACKEND", "auto").strip().lower() or "auto"
    )
    # llama-server バイナリの場所（llama-cpp-python が無い場合の代替）
    llama_server_bin: str = field(
        default_factory=lambda: os.environ.get("SNIPHER_LLAMA_SERVER", "").strip()
    )
    # 初回起動時に自動ロード（+ 必要なら自動ダウンロード）するか
    autostart: bool = field(default_factory=lambda: _env_bool("SNIPHER_LFM_AUTOSTART", True))
    # 自動取得を有効にするか（off ならローカル/環境変数のモデルだけを使う）。
    # サーバーレス環境では既定 off: 関数サイズ/時間制限に対して 731MB は大きすぎるため、
    # 内蔵ニューラルコア（蒸留スナップショット）とリモート委譲で知能を確保する。
    auto_fetch: bool = field(default_factory=lambda: _env_bool("SNIPHER_LFM_AUTO_FETCH",
                                                               not is_serverless()))
    # 内蔵ニューラルコア（LFM2.5 を蒸留した NumPy スナップショット）: auto|on|off
    light_core: str = field(default_factory=lambda: (
        os.environ.get("SNIPHER_LIGHT_CORE", "auto").strip().lower() or "auto"))
    # 軽量ニューラルコアの生成上限（文字数）
    light_max_chars: int = field(default_factory=lambda: _env_int("SNIPHER_LIGHT_MAX_CHARS", 64))
    # 蒸留コアの確信度がこの値を切ったら、重い LFM2.5 の起動を裏で進める（段階昇格）
    light_gate: float = field(default_factory=lambda: _env_float("SNIPHER_LIGHT_GATE", 0.34))
    # LFM2.5-1.2B-JP のフルウェイトを常駐させたホストへの委譲（任意・ゼロ設定）
    remote_url: str = field(default_factory=lambda: os.environ.get("SNIPHER_LFM_REMOTE_URL", "").strip())
    remote_token: str = field(default_factory=lambda: os.environ.get("SNIPHER_LFM_REMOTE_TOKEN", "").strip())
    # 推論時の動的 INT8 量子化（torch バックエンドの CPU 高速化）
    quantize_int8: bool = field(default_factory=lambda: _env_bool("SNIPHER_LFM_QUANTIZE", True))
    # 未知文字学習用の予約トークン枠
    reserved_tokens: int = field(default_factory=lambda: _env_int("SNIPHER_LFM_RESERVED", 256))
    # プロンプト予算（トークン）。超過すると古い履歴から落とす
    prompt_budget: int = field(default_factory=lambda: _env_int("SNIPHER_LFM_PROMPT_BUDGET", 1024))
    # 生成の既定値（LFM2.5 公式推奨: temperature 0.1 / top_k 50 / rep 1.05）
    max_new_tokens: int = field(default_factory=lambda: _env_int("SNIPHER_LFM_MAX_NEW_TOKENS", 128))
    temperature: float = field(default_factory=lambda: _env_float("SNIPHER_LFM_TEMPERATURE", 0.1))
    top_k: int = field(default_factory=lambda: _env_int("SNIPHER_LFM_TOP_K", 50))
    repetition_penalty: float = field(
        default_factory=lambda: _env_float("SNIPHER_LFM_REPETITION_PENALTY", 1.05)
    )
    # 学習（未知文字の深学習 = 埋め込みのみ少数ステップの勾配更新）
    learn_steps: int = field(default_factory=lambda: _env_int("SNIPHER_LFM_LEARN_STEPS", 12))
    learn_lr: float = field(default_factory=lambda: _env_float("SNIPHER_LFM_LEARN_LR", 3e-3))
    # 学習済み語彙の保存先
    store_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SNIPHER_LFM_STORE_DIR", "").strip() or DEFAULT_STORE_DIR
        )
    )
    # 自動取得したモデルのキャッシュ先
    cache_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SNIPHER_LFM_CACHE_DIR", "").strip() or DEFAULT_CACHE_DIR
        )
    )
    # 取得失敗時の再試行間隔（秒）
    fetch_retry_seconds: int = field(
        default_factory=lambda: _env_int("SNIPHER_LFM_FETCH_RETRY", 300)
    )
    # llama.cpp のスレッド数（0 = 自動: 物理コア数の半分、最低1）
    n_threads: int = field(default_factory=lambda: _env_int("SNIPHER_LLM_THREADS", 0))

    @property
    def is_local(self) -> bool:
        p = Path(self.model_source)
        return p.exists() and (p.is_dir() or p.is_file())

    def extra_urls(self) -> list[str]:
        """SNIPHER_LFM_URLS に指定された追加の直接ダウンロード URL。"""
        raw = os.environ.get("SNIPHER_LFM_URLS", "")
        return [u.strip() for u in raw.replace(",", "\n").splitlines() if u.strip()]
