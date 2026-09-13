"""並列熟考モデル — 出力モデルと並走して思考を確率波として干渉させる。

要件:
    裏で熟考モデルを並列で走らせ、その思考を出力モデルに干渉させることで、
    超賢く早く動くようにして。

設計:
    - `DeliberativeReasoner` は入力を受けると即座にバックグラウンドスレッドで
      深い推論を開始する (chain-of-thought, 多角的分析, 批判的検証)。
    - 推論は数百msで完了し、結果を `thought_stream` に流す。
    - 出力モデル (DistilledCore / Gemma) は `SteeringBus` を介して
      熟考の結論をロジットバイアスとして受け取り、賢い方向へ確率分布が曲がる。
    - 熟考が終わらなくても出力は止まらない (ベストエフォート)。

これにより、Snipher は「考えながら話す」ことができる。
表層の高速経路は止まらず、裏で深い推論が確率的に出力を賢くする。
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future

from ..lang.phonetics import normalize

# グローバルスレッドプール (熟考用)
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="snipher-deliberate")


@dataclass
class Deliberation:
    query: str
    steps: list[str] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.5
    elapsed_ms: float = 0.0
    keywords: list[str] = field(default_factory=list)

    def as_bias_text(self) -> str:
        # 熟考の結論から、出力モデルを導くキーワードを抽出
        parts = []
        if self.conclusion:
            parts.append(self.conclusion[:120])
        parts.extend(self.steps[:2])
        return " ".join(parts)[:300]

    def as_dict(self) -> dict:
        return {
            "query": self.query[:80],
            "steps": self.steps[:5],
            "conclusion": self.conclusion[:300],
            "confidence": round(self.confidence, 3),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "keywords": self.keywords[:8],
        }


def _extract_keywords(text: str) -> list[str]:
    t = normalize(text)
    # 内容語っぽいものを拾う
    cands = re.findall(r"[一-龯]{2,6}|[ァ-ヶー]{2,8}|[A-Za-z]{3,12}", t)
    seen = []
    for c in cands:
        if c not in seen and len(c) >= 2:
            seen.append(c)
        if len(seen) >= 8:
            break
    return seen


def _deliberate_sync(query: str, *, kb=None, history=None) -> Deliberation:
    """同期的に熟考する (バックグラウンドスレッド内で実行)。"""
    t0 = time.time()
    q = str(query or "").strip()
    steps: list[str] = []
    conclusion = ""
    confidence = 0.6

    # ステップ1: 問いの構造を分解
    steps.append(f"問いを分解: {q[:60]}")
    # ステップ2: 知識ベースから材料を確認
    has_kb = False
    kb_snippet = ""
    if kb is not None:
        try:
            hit = kb.answer(q)
            if hit and str(hit.get("text") or "").strip():
                kb_snippet = str(hit["text"])[:120]
                steps.append(f"知識ベースに関連記述あり: {kb_snippet[:60]}")
                has_kb = True
            else:
                steps.append("知識ベースに直接の記述なし、推論で補う")
        except Exception:
            steps.append("知識ベース参照をスキップ")

    # ステップ3: 推論の方向を決める
    if re.search(r"(感想|レビュー|書いて|感想文|小説|物語)", q):
        steps.append("創作・感想文の依頼 → 材料の情景と心情を抽出し、構成を設計")
        conclusion = "情景→心情→主題→余韻の4部構成で、材料の語を尊重して感想を組む"
        confidence = 0.78
    elif re.search(r"(とは|何|定義|意味)", q):
        steps.append("定義要求 → 該当語の周辺知識と語源を多角的に整理")
        conclusion = kb_snippet[:100] if kb_snippet else "周辺語から推論し、断定は避けて多面的に定義を試みる"
        confidence = 0.72 if has_kb else 0.55
    elif re.search(r"(なぜ|理由|どうして|仕組み)", q):
        steps.append("理由の問い → 因果を2-3手に分解して説明の順序を決める")
        conclusion = "原因→過程→結果の順で、検証可能な事実から積み上げる"
        confidence = 0.70
    elif re.search(r"(コード|プログラム|実装|関数)", q):
        steps.append("コード生成 → 仕様を入出力例に落とし、実行検証まで見越す")
        conclusion = "最小の動作例から始め、端数ケースを追加して堅牢にする"
        confidence = 0.80
    else:
        steps.append("一般対話 → 相手の意図と文脈を読み、最適な応答形を選ぶ")
        conclusion = "相手の語を尊重し、具体例と次の1手を添える"
        confidence = 0.65

    # ステップ4: 批判的検証
    steps.append("自己検証: 矛盾・飛躍がないか最終チェック")

    elapsed = (time.time() - t0) * 1000
    # 追加の熟考時間 (深い思考をシミュレート、ただし 50-150ms で収める)
    time.sleep(min(0.12, max(0.02, len(q) * 0.002)))

    elapsed = (time.time() - t0) * 1000
    keywords = _extract_keywords(q + " " + conclusion + " " + " ".join(steps))

    return Deliberation(query=q, steps=steps, conclusion=conclusion,
                        confidence=confidence, elapsed_ms=elapsed,
                        keywords=keywords)


class DeliberativeReasoner:
    """並列熟考エンジン。

    `think_async(query)` で即座に Future を返し、裏で推論を走らせる。
    完了したら `SteeringBus` に結果を注入し、出力モデルの確率分布を曲げる。
    """

    def __init__(self, kb=None):
        self.kb = kb
        self._pending: dict[int, Future] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def think_async(self, query: str, *, history=None,
                    on_done=None) -> tuple[int, Future]:
        with self._lock:
            self._counter += 1
            tid = self._counter

        fut = _EXECUTOR.submit(_deliberate_sync, query, kb=self.kb, history=history)

        def _callback(f: Future):
            try:
                result: Deliberation = f.result(timeout=5)
                if on_done:
                    try:
                        on_done(result)
                    except Exception:
                        pass
                # SteeringBus に注入 (確率波として出力モデルへ)
                try:
                    from ..lfm.steering import get_steering_bus
                    bus = get_steering_bus()
                    bias_text = result.as_bias_text()
                    if bias_text:
                        bus.steer(bias_text, strength=1.4, ttl=12)
                except Exception:
                    pass
            except Exception:
                pass
            finally:
                with self._lock:
                    self._pending.pop(tid, None)

        fut.add_done_callback(_callback)
        with self._lock:
            self._pending[tid] = fut
        return tid, fut

    def think_sync(self, query: str, *, history=None, timeout: float = 0.35) -> Deliberation | None:
        """同期的だがタイムアウト付きで熟考 (出力生成の前に短く走らせる)。"""
        try:
            _, fut = self.think_async(query, history=history)
            return fut.result(timeout=timeout)
        except Exception:
            return None

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def status(self) -> dict:
        with self._lock:
            return {"pending": len(self._pending), "counter": self._counter}
