"""リアルタイム Web 検索 — 出力中に走らせ、確率の波として生成に干渉させる。

要件:
    Web検索を挨拶とか簡単な受け答え以外の「あらゆる」プロンプトで出力中に走らせ、
    検索結果を確率の波にして生成中に干渉させることで、現在の高速推論を維持しながら
    実質無限の知識のAIを再現して

設計:
    - 入力受信と同時にバックグラウンドで検索を開始 (ノンブロッキング)
    - 取得した検索結果の文を `SteeringBus` に確率バイアスとして注入
    - 生成ループは高速推論を止めず、検索結果が届き次第、滑らかに知識が反映される
    - 挨拶・礼・謝罪・短い相槌は検索対象外 (高速経路を維持)
    - それ以外はすべて検索 (無限知識の再現)

このモジュールは `WebGrounding` (同期) のリアルタイム版。
同期版が「検索→待つ→生成」であるのに対し、こちらは「検索∥生成」。
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass

from ..lang.phonetics import normalize

# 検索対象外の簡易パターン (挨拶・礼・謝罪・超短文)
_SIMPLE_RE = re.compile(
    r"^(こんにちは|こんばんは|おはよう|はじめまして|やあ|もしもし|"
    r"ありがとう|ありがと|感謝|すみません|ごめん|おつかれ|お疲れ|"
    r"さようなら|さよなら|またね|バイバイ|おやすみ|"
    r"はい|いいえ|うん|ええ|そう|なるほど|わかった|了解|おけ|ok|"
    r"草|w+|笑)+[。！？!?.]*$",
    re.IGNORECASE,
)

# 検索をスキップする短い相槌 (2文字以下など)
_SHORT_RE = re.compile(r"^[ぁ-んァ-ヶー]{1,2}[？?。]*$")

_EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="snipher-realtime-web")


@dataclass
class RealtimeResult:
    query: str
    sentences: list[str]
    sources: list[dict]
    elapsed_ms: float
    ok: bool


def should_search_realtime(text: str) -> bool:
    t = normalize(str(text or "")).strip()
    if not t or len(t) < 2:
        return False
    # 挨拶・礼・謝罪・超短文はスキップ
    if _SIMPLE_RE.match(t):
        return False
    if _SHORT_RE.match(t):
        return False
    # 1-2文字の感嘆のみもスキップ
    if len(t) <= 3 and re.fullmatch(r"[ぁ-んァ-ヶ一-龯！？?。、]+", t):
        # ただし「とは」「なぜ」など問いの形なら検索する
        if not re.search(r"(とは|なぜ|どう|何|いつ|どこ|誰|いくら|方法|仕組み)", t):
            return False
    return True


class RealtimeWebGrounding:
    """出力中に並走する Web 検索 (確率波として生成に干渉)。

    使用例:
        rwg = RealtimeWebGrounding(web_grounding)
        # 入力と同時に検索開始 (ノンブロッキング)
        fut = rwg.search_async("ロシアとは", on_result=lambda sents: bus.inject(sents))
        # 生成ループはそのまま走る。結果が届いたら bus 経由で確率に乗る。
    """

    def __init__(self, web_grounding=None):
        self.web = web_grounding
        self._futures: dict[int, Future] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def search_async(self, query: str, *, on_result=None,
                     extra_queries: tuple[str, ...] = ()) -> tuple[int, Future | None]:
        query = str(query or "").strip()
        if not query or not should_search_realtime(query):
            # 検索不要 → 即座に空を返す (高速経路を維持)
            fut = _EXECUTOR.submit(lambda: RealtimeResult(query=query, sentences=[],
                                                          sources=[], elapsed_ms=0, ok=False))
            if on_result:
                fut.add_done_callback(lambda f: None)
            return -1, fut

        if self.web is None or not getattr(self.web, "enabled", True):
            fut = _EXECUTOR.submit(lambda: RealtimeResult(query=query, sentences=[],
                                                          sources=[], elapsed_ms=0, ok=False))
            return -1, fut

        with self._lock:
            self._counter += 1
            tid = self._counter

        def _do():
            t0 = time.time()
            try:
                grounding = self.web.gather(query, explicit=False,
                                            extra_queries=extra_queries)
                sents = []
                for ev in getattr(grounding, "evidence", [])[:4]:
                    txt = getattr(ev, "text", str(ev))
                    if txt and len(txt) >= 12:
                        sents.append(str(txt).strip())
                sources = getattr(grounding, "sources", [])[:3]
                elapsed = (time.time() - t0) * 1000
                result = RealtimeResult(query=query, sentences=sents,
                                        sources=sources, elapsed_ms=elapsed,
                                        ok=bool(sents))
                # 確率波として注入
                if sents:
                    try:
                        from ..lfm.steering import get_steering_bus
                        bus = get_steering_bus()
                        bus.inject_evidence(sents, strength=1.8)
                    except Exception:
                        pass
                if on_result:
                    try:
                        on_result(sents, sources)
                    except Exception:
                        pass
                return result
            except Exception as e:
                elapsed = (time.time() - t0) * 1000
                return RealtimeResult(query=query, sentences=[], sources=[],
                                      elapsed_ms=elapsed, ok=False)

        fut = _EXECUTOR.submit(_do)
        with self._lock:
            self._futures[tid] = fut

        def _cleanup(f: Future):
            with self._lock:
                self._futures.pop(tid, None)

        fut.add_done_callback(_cleanup)
        return tid, fut

    def search_sync(self, query: str, *, timeout: float = 1.2) -> RealtimeResult | None:
        _, fut = self.search_async(query)
        if fut is None:
            return None
        try:
            return fut.result(timeout=timeout)
        except Exception:
            return None

    def pending(self) -> int:
        with self._lock:
            return len(self._futures)

    def status(self) -> dict:
        with self._lock:
            return {"pending": len(self._futures), "counter": self._counter}
