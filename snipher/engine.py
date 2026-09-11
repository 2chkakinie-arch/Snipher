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
        return {
            "name": "Snipher",
            "version": "0.1.0",
            "architecture": "ルールベース構文解析 + 確率的テーブル生成(if構文ベース)",
            "language": "日本語のみ",
            "total_parameters": table_params + weight_params,
            "table_entries": table_params,
            "probability_weights": weight_params,
            "weights": cfg.get("weights", {}),
            "temperature": cfg.get("temperature"),
            "lexicon_stats": stats,
            "formula": (
                "S(w) = alpha*logP_freq + beta*logP_trans + gamma*Q_role "
                "+ delta*Q_inflect + epsilon*Q_register + zeta*Q_topic"
            ),
        }
