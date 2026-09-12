"""Snipher の統括エンジン。解析と生成を組み合わせた公開 API を提供する。"""

from __future__ import annotations

from .lexicon import Lexicon
from .parser import Parser
from .probability import ProbabilityModel
from .generator import Generator


class SnipherEngine:
    """Snipher のファサード。"""

    def __init__(self, data_dir: str | None = None, seed: int | None = None):
        self.lexicon = Lexicon(data_dir)
        self.parser = Parser(self.lexicon)
        self.model = ProbabilityModel(self.lexicon, seed=seed)
        self.generator = Generator(self.lexicon, self.model, seed=seed)

    # ------------------------------------------------------------------ #
    # 解析
    # ------------------------------------------------------------------ #
    def analyze(self, text: str) -> dict:
        """日本語文を解析する(文構造・助動詞・要点・トピック)。"""
        return self.parser.parse(text)

    # ------------------------------------------------------------------ #
    # 生成
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str | None = None, register: str | None = None,
                 tense: str = "nonpast", n: int = 1, seed: int | None = None) -> dict:
        """確率的に日本語文を生成する。生成文を解析した結果も同梱する。"""
        if n <= 1:
            result = self.generator.generate(prompt=prompt, register=register, tense=tense, seed=seed)
            result["analysis"] = self.parser.parse(result["text"])
            return result

        sentences = [
            self.generator.generate(
                prompt=prompt, register=register, tense=tense,
                seed=None if seed is None else seed + i,
            )
            for i in range(n)
        ]
        for s in sentences:
            s["analysis"] = self.parser.parse(s["text"])
        return {"sentences": sentences, "count": n}

    # ------------------------------------------------------------------ #
    # モデル情報
    # ------------------------------------------------------------------ #
    def knowledge(self) -> dict | None:
        """知識ベース（BM25 検索）の統計。使えない環境では None。"""
        try:
            from .knowledge import KnowledgeBase

            kb = KnowledgeBase.shared()
            return kb.stats() if kb is not None else None
        except Exception:  # noqa: BLE001
            return None

    def neural(self) -> dict:
        """内蔵ニューラルコア（LFM2.5 蒸留スナップショット）の情報。"""
        out = {"available": False, "params": 0, "vocab": 0, "weights_bytes": 0, "engine": None}
        try:
            from .neural import numpy_available

            if not numpy_available():
                out["reason"] = "numpy が未インストールです"
                return out
            from .neural.cache import get_core

            core = get_core()
            if core is None:
                from .neural.core import DEFAULT_PATH

                out["reason"] = f"重みが未ビルドです（python tools/distill_neural.py）: {DEFAULT_PATH}"
                return out
            out.update({"available": True, "params": core.net.n_params(),
                        "vocab": core.tok.size(), "engine": core.engine_name(),
                        "weights_bytes": core.path.stat().st_size if core.path.exists() else 0,
                        "trained_at": core.meta.get("trained_at"),
                        "metrics": core.meta.get("metrics")})
        except Exception as exc:  # noqa: BLE001
            out["reason"] = str(exc)
        return out

    def language_model(self) -> dict:
        """巨大 n-gram 言語モデル（流暢さの審判）の情報。"""
        out = {"available": False, "params": 0, "order": 0, "vocab": 0, "bytes": 0, "engine": None}
        try:
            from . import lm as lm_mod

            model = lm_mod.shared()
            if model is None or not model.is_ready:
                out["reason"] = f"重みが未ビルドです（python tools/build_lm.py）: {lm_mod.DEFAULT_PATH}"
                return out
            out.update({"available": True, "params": model.n_params(), "order": model.order,
                        "vocab": model.n_vocab, "bytes": model.bytes_on_disk(),
                        "engine": model.engine_name(), "tokens": int(model.total),
                        "load_seconds": model.load_seconds, "calib": model.calib,
                        "trained_at": model.trained_at})
        except Exception as exc:  # noqa: BLE001
            out["reason"] = str(exc)
        return out

    def info(self) -> dict:
        """モデルのパラメータ数と設計情報を返す。"""
        cfg = self.lexicon.config
        stats = self.lexicon.stats()
        table_params = sum(
            len(getattr(self.lexicon, k))
            for k in ("verbs", "adjectives", "nouns", "particles", "auxiliaries",
                      "adverbs", "conjunctions", "interjections", "adnominals",
                      "patterns", "corpus")
        )
        weight_params = len(cfg.get("weights", {}))
        kb_stats = self.knowledge() or {}
        kb_params = int(sum(kb_stats.get(k, 0) for k in ("topics", "facts", "questions", "answers")))
        net = self.neural()
        net_params = int(net.get("params") or 0)
        lm = self.language_model()
        lm_params = int(lm.get("params") or 0)
        return {
            "name": "Snipher",
            "version": "2.0.0",
            "architecture": (
                f"知識ベース検索({int(kb_stats.get('topics') or 0)} 話題) + composer(文の設計図) "
                "+ 巨大 n-gram 言語モデル(流暢さの審判) "
                "+ LFM2.5 アーキテクチャの蒸留ニューラルコア(KV キャッシュ付き) "
                "+ ルールベース構文解析/確率的テーブル生成"
            ),
            "language": "日本語のみ",
            "total_parameters": table_params + weight_params + kb_params + net_params + lm_params,
            "parameter_breakdown": {
                "lexicon_table_entries": table_params,
                "probability_weights": weight_params,
                "knowledge_base_entries": kb_params,
                "neural_core_weights": net_params,
                "language_model_entries": lm_params,
            },
            "table_entries": table_params,
            "probability_weights": weight_params,
            "knowledge_base": kb_stats or None,
            "neural_core": {k: v for k, v in net.items() if k != "metrics"},
            "language_model": lm,
            "weights": cfg.get("weights", {}),
            "temperature": cfg.get("temperature"),
            "lexicon_stats": stats,
            "formula": (
                "S(w) = alpha*logP_freq + beta*logP_trans + gamma*Q_role "
                "+ delta*Q_inflect + epsilon*Q_register + zeta*Q_topic"
            ),
        }
