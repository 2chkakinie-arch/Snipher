"""辞書データのロードと検索を担当するモジュール。

Snipher の「知識」はすべて JSON テーブルとして持つ。
パラメータはテーブルのエントリ数 + 確率式の重み(6個)のみであり、
これが「量子化より圧倒的に少ないパラメータ」の正体である。
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


class Lexicon:
    """単語テーブルをメモリに展開して検索する。"""

    def __init__(self, data_dir: Path | str | None = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.verbs: list[dict] = []
        self.adjectives: list[dict] = []
        self.nouns: list[dict] = []
        self.particles: list[dict] = []
        self.auxiliaries: list[dict] = []
        self.adverbs: list[dict] = []
        self.conjunctions: list[dict] = []
        self.interjections: list[dict] = []
        self.adnominals: list[dict] = []
        self.patterns: list[dict] = []
        self.corpus: list[dict] = []
        self.config: dict = {}
        self._load()

    # ------------------------------------------------------------------ #
    # ロード
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        self.verbs = self._read("verbs.json")
        self.adjectives = self._read("adjectives.json")
        self.nouns = self._read("nouns.json")
        self.particles = self._read("particles.json")
        self.auxiliaries = self._read("auxiliaries.json")
        other = self._read("other.json")
        self.adverbs = other.get("adverbs", [])
        self.conjunctions = other.get("conjunctions", [])
        self.interjections = other.get("interjections", [])
        self.adnominals = other.get("adnominals", [])
        self.patterns = self._read("patterns.json")
        self.corpus = self._read("corpus.json")
        self.config = self._read("config.json")

    def _read(self, name: str) -> list | dict:
        path = self.data_dir / name
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------ #
    # 検索ヘルパ
    # ------------------------------------------------------------------ #
    @staticmethod
    def _by_surface(entries: list[dict], surface: str) -> dict | None:
        for e in entries:
            if e.get("s") == surface:
                return e
        return None

    def find_verb(self, surface: str) -> dict | None:
        return self._by_surface(self.verbs, surface)

    def find_adjective(self, surface: str) -> dict | None:
        return self._by_surface(self.adjectives, surface)

    def find_noun(self, surface: str) -> dict | None:
        return self._by_surface(self.nouns, surface)

    def find_particle(self, surface: str) -> dict | None:
        return self._by_surface(self.particles, surface)

    def find_auxiliary(self, surface: str) -> dict | None:
        return self._by_surface(self.auxiliaries, surface)

    def nouns_by_tag(self, tag: str) -> list[dict]:
        return [n for n in self.nouns if tag in n.get("t", [])]

    def verbs_by_tag(self, tag: str) -> list[dict]:
        return [v for v in self.verbs if tag in v.get("t", [])]

    def adjectives_by_tag(self, tag: str) -> list[dict]:
        return [a for a in self.adjectives if tag in a.get("t", [])]

    # ------------------------------------------------------------------ #
    # 統計
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        return {
            "verbs": len(self.verbs),
            "adjectives": len(self.adjectives),
            "nouns": len(self.nouns),
            "particles": len(self.particles),
            "auxiliaries": len(self.auxiliaries),
            "adverbs": len(self.adverbs),
            "conjunctions": len(self.conjunctions),
            "patterns": len(self.patterns),
            "corpus_sentences": len(self.corpus),
        }
