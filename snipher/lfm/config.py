"""LFM ニューラルエンジンの設定（環境変数で上書き可能）。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# HuggingFace への接続が遮断されている環境(サンドボックス等)では
# 自動ダウンロードが長時間固まるので、既定で短いタイムアウトを設定する。
# 本物のネットワークが遅い環境では環境変数で上書きできる。
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "15")

# Liquid AI の日本語チャットモデル（LFM2 アーキテクチャ / 1.17B / 32K context）
DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_DIR = REPO_ROOT / "var" / "learned_vocab"

DEFAULT_SYSTEM_PROMPT = (
    "あなたは親しみやすい日本語の会話パートナーです。"
    "日常会話を 自然で短めの返答（1〜3文）で返します。"
    "難しい説明より、 相手の話に共感して会話を続けることを優先します。"
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


@dataclass
class LfmConfig:
    """環境変数 SNIPHER_LFM_* で上書きできる設定。"""

    # モデルソース: ローカルディレクトリ or HuggingFace モデルID
    model_source: str = field(
        default_factory=lambda: os.environ.get("SNIPHER_LFM_MODEL", "").strip()
        or DEFAULT_MODEL_ID
    )
    # 初回起動時に自動ロードするか
    autostart: bool = field(default_factory=lambda: _env_bool("SNIPHER_LFM_AUTOSTART", True))
    # 推論時の動的 INT8 量子化（CPU 高速化）
    quantize_int8: bool = field(default_factory=lambda: _env_bool("SNIPHER_LFM_QUANTIZE", True))
    # 未知文字学習用の予約トークン枠
    reserved_tokens: int = field(default_factory=lambda: _env_int("SNIPHER_LFM_RESERVED", 256))
    # プロンプト予算（トークン）。超過すると古い履歴から落とす
    prompt_budget: int = field(default_factory=lambda: _env_int("SNIPHER_LFM_PROMPT_BUDGET", 1024))
    # 生成の既定値
    max_new_tokens: int = field(default_factory=lambda: _env_int("SNIPHER_LFM_MAX_NEW_TOKENS", 128))
    temperature: float = field(default_factory=lambda: _env_float("SNIPHER_LFM_TEMPERATURE", 0.3))
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

    @property
    def is_local(self) -> bool:
        p = Path(self.model_source)
        return p.exists() and p.is_dir()
