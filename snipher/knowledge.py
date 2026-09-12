"""知識ベース検索（BM25 + 文字バイグラム）。

`snipher/data/kb.json`（`tools/build_kb.py` で生成）を、単語分割に頼らない
**文字バイグラム + 半角英字語** の逆引きインデックスで引く。

    クエリ → 正規化 → 索引文書（topic/aliases/facts/questions）と BM25 スコア
           → 上位トピック + そのトピックの事実・回答 + 会話を広げる質問

高速コア（数ミリ秒）で動かすため、語彙数に対する前処理はロード時に 1 回だけ。
推論は numpy 不要・dict 走査のみで、数万件でも軽量に動く。
"""

from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data" / "kb.json"

_ASCII_WORD = re.compile(r"[A-Za-z0-9_+#.\-]{2,}")
_JP = re.compile(r"[ぁ-んァ-ヶ一-龯ー々〆〇]")
_STRIP = re.compile(r"[\s。、！？!?・…「」『』()（）:：;；〜~\-_]")


def tokenize(text: str) -> list[str]:
    """検索用のトークン列（日本語は 2-gram、英数字は小文字化して 1 語）。"""
    t = _STRIP.sub("", str(text or "").lower())
    out: list[str] = []
    # 英数字の塊を先に抜いて、残りの日本語文字列を 2-gram 化
    spans = [(m.start(), m.group()) for m in _ASCII_WORD.finditer(t)]
    for pos, w in spans:
        out.append("w:" + w)
    t2 = _ASCII_WORD.sub(" ", t)
    buf: list[str] = []
    for ch in t2:
        if _JP.match(ch):
            buf.append(ch)
        else:
            out.extend(_grams(buf))
            buf = []
    out.extend(_grams(buf))
    return out


def _grams(chars: list[str]) -> list[str]:
    if not chars:
        return []
    if len(chars) == 1:
        return ["g:" + chars[0]]
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


class KnowledgeBase:
    """kb.json を引く小さな検索エンジン。"""

    _cache: dict[str, "KnowledgeBase"] = {}
    _lock = threading.Lock()

    def __init__(self, path: str | Path | None = None):
        p = Path(path) if path else _DATA
        self.path = p
        self.items: list[dict] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len: list[int] = []
        self.avg_len = 0.0
        self.version = 0
        self._load(p)

    # ------------------------------------------------------------------ #
    def _load(self, p: Path) -> None:
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        items = raw.get("items", []) if isinstance(raw, dict) else []
        self.version = int(raw.get("version", 0)) if isinstance(raw, dict) else 0
        df: dict[str, set[int]] = {}
        for i, it in enumerate(items):
            blob = " ".join(
                [it.get("topic", "")]
                + [a * 2 for a in it.get("aliases", [])]
                + it.get("verbs", [])
                + it.get("facts", [])
                + it.get("questions", [])
                + it.get("answers", [])
                + it.get("followups", [])
            )
            toks = tokenize(blob)
            self.doc_len.append(max(1, len(toks)))
            tf: dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, n in tf.items():
                df.setdefault(tok, set()).add(i)
                self.postings.setdefault(tok, []).append((i, n))
            it["_qtexts"] = list(it.get("questions", []))
        self.items = items
        # BM25 の idf を先に計算
        N = max(1, len(items))
        self.avg_len = (sum(self.doc_len) / N) if N else 1.0
        self.idf = {t: math.log(1.0 + (N - len(d) + 0.5) / (len(d) + 0.5)) for t, d in df.items()}
        for t in list(self.postings):
            self.postings[t].sort(key=lambda x: -x[1])
        # 全トピックに近いくらい配られるトークン（機能語バイグラム）は証拠に数えない
        N = max(1, len(items))
        self.common = {t for t, d in df.items() if len(d) / N > 0.34}
        self._alias_index: dict[str, list[int]] = {}
        for i, it in enumerate(items):
            for a in [it.get("topic", "")] + list(it.get("aliases", [])):
                key = _STRIP.sub("", str(a).lower())
                if len(key) >= 1:
                    self._alias_index.setdefault(key, []).append(i)

    # ------------------------------------------------------------------ #
    @classmethod
    def shared(cls, path: str | Path | None = None) -> "KnowledgeBase":
        key = str(path or _DATA)
        with cls._lock:
            kb = cls._cache.get(key)
            if kb is None:
                kb = cls(path)
                cls._cache[key] = kb
            return kb

    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 3, k1: float = 1.4, b: float = 0.72) -> list[dict]:
        """クエリに効く順にトピックを返す。score は 0..1 に正規化。"""
        if not self.items:
            return []
        toks = tokenize(query)
        if not toks:
            return []
        qfreq: dict[str, int] = {}
        for t in toks:
            qfreq[t] = qfreq.get(t, 0) + 1
        scores: dict[int, float] = {}
        matched: dict[int, set[str]] = {}
        for tok, qn in qfreq.items():
            post = self.postings.get(tok)
            if not post:
                continue
            idf = self.idf.get(tok, 0.0)
            for i, tf in post:
                denom = tf + k1 * (1 - b + b * self.doc_len[i] / max(1e-6, self.avg_len))
                scores[i] = scores.get(i, 0.0) + idf * (tf * (k1 + 1)) / denom
                matched.setdefault(i, set()).add(tok)
        if not scores:
            return []
        # 発話中のエイリアス完全一致は強い証拠（短い発話でも取り逃さない）
        norm_q = _STRIP.sub("", str(query or "").lower())
        for key, idxs in self._alias_index.items():
            if key and key in norm_q:
                # 複数文字の一致は強い証拠、1 文字は弱く効かせる
                # 発話に現れた語彙の直接一致は、bigram 統計より強い証拠とみなす
                bonus = (3.0 + 2.0 * len(key)) if len(key) >= 2 else 5.0
                for i in idxs:
                    scores[i] = scores.get(i, 0.0) + bonus
                    matched.setdefault(i, set()).add("a:" + key)
        mx = max(scores.values()) or 1.0
        nq = len(set(toks)) or 1
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        out: list[dict] = []
        for i, sc in ranked:
            hit = matched.get(i, set())
            grams = {t for t in toks if not t.startswith(("w:", "a:")) and t not in self.common}
            covered = len(grams & hit) if grams else 0
            alias = any(t.startswith("a:") for t in hit)
            out.append({
                "item": self.items[i],
                "index": i,
                "score": round(sc / mx, 4),
                "raw": round(sc, 4),
                "coverage": round(covered / nq, 4),
                "covered_n": int(covered),
                "alias_hit": alias,
                "matched": sorted(hit)[:12],
            })
        return out

    # ------------------------------------------------------------------ #
    def _best_answer(self, item: dict, query: str) -> tuple[str | None, str | None]:
        """questions/answers が併記されていれば、近い質問に対応する答えを返す。"""
        qs = item.get("questions", [])
        as_ = item.get("answers", [])
        if not as_:
            return None, None
        if not qs or len(qs) != len(as_):
            return (as_[0], qs[0] if qs else None)
        qn = set(tokenize(query))
        best_i, best_ov = 0, -1
        for i, q in enumerate(qs):
            ov = len(qn & set(tokenize(q)))
            if ov > best_ov:
                best_i, best_ov = i, ov
        return as_[best_i], qs[best_i]

    def answer(self, query: str, min_score: float = 0.34, min_cov: float = 0.24) -> dict | None:
        """会話に使える 1 返信の材料を返す。無ければ None。

        戻り値:
            text      … 提案される返答文（まだ polisher を通す前の下書き）
            confidence… テーブルヒットの強さ（0..1）。高速コア即答の可否に使う
            topic/score/facts/usage
        """
        hits = self.search(query, top_k=3)
        if not hits:
            return None
        top = hits[0]
        # 相対スコアだけでなく、発話をどれだけ説明できたか（被覆率）で採否を決める
        # alias 一致が無ければ、少なくとも 2 つの実語が実際に一致していないと採用しない
        strong = bool(top.get("alias_hit")) or (
            float(top.get("coverage", 0.0)) >= min_cov and int(top.get("covered_n", 0)) >= 2)
        if not strong:
            alt = next((h for h in hits[1:] if h.get("alias_hit")), None)
            if alt is not None:
                top, strong = alt, True
        if not strong or float(top["score"]) < min_score:
            return None
        item = top["item"]
        ans, asked = self._best_answer(item, query)
        facts = item.get("facts", [])
        follow = (item.get("followups") or [""])[0]
        if ans:
            text = ans
        elif facts:
            text = f"{item.get('topic', '')}のことですね。{facts[0]}"
        else:
            return None
        if follow and not text.endswith("？") and not text.endswith("?"):
            text = f"{text}{follow}"
        # 関連する事実が 2 つ以上あれば 1 つだけ添えて情報量を出す
        if len(facts) > 1 and not ans:
            text = f"{text}{facts[1]}" if len(text) < 90 else text
        conf = min(0.92, 0.5 + 0.3 * float(top["score"]) + 0.25 * min(1.0, float(top.get("coverage", 0.0))) * 2)
        return {
            "text": text,
            "confidence": round(float(conf), 4),
            "topic": item.get("topic"),
            "score": top["score"],
            "coverage": top.get("coverage", 0.0),
            "fact": facts[0] if facts else None,
            "asked": asked,
            "usage": "answer",
            "related": [h["item"].get("topic") for h in hits[1:]],
        }

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        return {
            "topics": len(self.items),
            "facts": sum(len(i.get("facts", [])) for i in self.items),
            "questions": sum(len(i.get("questions", [])) for i in self.items),
            "answers": sum(len(i.get("answers", [])) for i in self.items),
            "terms": len(self.postings),
            "version": self.version,
        }
