"""LFM2.5-1.2B-JP ニューラルエンジン（オプション）。

torch / transformers が入っていない環境（Vercel など軽量デプロイ先）では
import に失敗せず、Snipher 従来の超小型エンジンへフォールバックできるよう、
このパッケージは常時 import 可能である必要がある。重い依存は各モジュール内で
遅延 import する。
"""

from __future__ import annotations

__all__ = ["LLM_DEPS_AVAILABLE", "LfmEngine", "get_engine"]


def _check_deps() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


LLM_DEPS_AVAILABLE = _check_deps()


def get_engine():
    """プロセス共通の LfmEngine シングルトンを返す（依存が無ければ None）。"""
    if not LLM_DEPS_AVAILABLE:
        return None
    from .engine import get_engine as _get

    return _get()
