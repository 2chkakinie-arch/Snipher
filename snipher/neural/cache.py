"""内蔵ニューラルコアのプロセス共通インスタンス（遅延ロード・スレッドセーフ）。"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

_lock = threading.Lock()
_core = None


def get_core(force: bool = False):
    """重みが無ければ None を返す（高速コアだけで動き続けるための安全網）。"""
    global _core
    if _core is not None and not force:
        return _core
    with _lock:
        if _core is not None and not force:
            return _core
        try:
            from .core import DistilledCore, available

            if not available():
                return None
            t0 = time.time()
            _core = DistilledCore()
            if not _core.is_ready:
                log.warning("内蔵ニューラルコアのロードに失敗: %s", _core.error)
                _core = None
                return None
            log.info("内蔵ニューラルコア準備完了 (%.0fms / %s params)",
                     (time.time() - t0) * 1000, f"{_core.net.n_params():,}" if _core.net else "?")
            return _core
        except Exception as exc:  # noqa: BLE001
            log.warning("内蔵ニューラルコアを使えません: %s", exc)
            return None


def reset() -> None:
    """重みを差し替えたとき（再蒸留後）に呼ぶ。"""
    global _core
    with _lock:
        _core = None
