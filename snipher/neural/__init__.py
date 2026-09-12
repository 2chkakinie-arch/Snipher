"""Snipher の**内蔵ニューラルコア**（LFM2.5 アーキテクチャの蒸留モデル）。

このパッケージの目的は「LFM2.5-1.2B-JP を巨大な別プロセスとして読む」ことでは
ありません。サーバーレス（Vercel）では 731MB の GGUF も 2.18GB の safetensors も
**バンドルに同梱できず、実行時にダウンロードもできない**（500MB 制限・
永続ディスクなし・コールドスタート）。

そこで Snipher は LFM2.5 と**同じ構造family**を持つ小型ニューラルコアを
自前のパラメータとして内蔵します。

    LFM2.5-1.2B-JP                     Snipher 内蔵ニューラルコア
    ──────────────────────            ─────────────────────────────
    短距離畳み込み + 注意の Hybrid  →  ShortConv / 注意の交互スタック（同じ系譜）
    RMSNorm                          →  RMSNorm
    SwiGLU 風ゲート                  →  SiLU ゲート付き畳み込み
    語彙 ~65k BPE                    →  文字レベル語彙（未知文字が原理的に出ない）
    1.17B 浮遊パラメータ              →  ~1M の int8 量子化パラメータ（約 1MB）

役割は LFM2.5 と同じ「確率的に不安な部分の担当」に限定します。

    * 下書きの不確実スロットの書き換え（文章生成）
    * 助動詞の補い（断片文の補完・文末の確定）
    * 確信度（perplexity）によるゲート → 本物の LFM2.5 に渡すかの判断
    * LFM2.5 が居ない環境（Vercel）での最終生成

依存は numpy のみ。 torch / transformers / llama.cpp も、モデルの
ダウンロードもアップロードも不要で、import した瞬間から動きます。
"""

from __future__ import annotations

from .tokenizer import CharTokenizer, SPECIAL_TOKENS
from .nn import NNConfig, MicroNet

__all__ = [
    "CharTokenizer",
    "SPECIAL_TOKENS",
    "NNConfig",
    "MicroNet",
    "core",
    "neural_available",
]


def numpy_available() -> bool:
    try:  # pragma: no cover - 環境依存
        import numpy  # noqa: F401
    except Exception:
        return False
    return True


def core():
    """プロセス共通の内蔵ニューラルコア（重みが無い／numpy が無い場合は None）。"""
    from .cache import get_core

    return get_core()


def neural_available() -> bool:
    return core() is not None
