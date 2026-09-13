"""ウェブ裏取り — 検索と html-fetch を「証拠の文」に変える層。

v2 の失敗は、検索結果ページの *案内文*（リワード・サインイン等）を
スニペットとしてそのまま回答に貼っていたことです。ここでの扱いは違います。

1. 検索（Edge UA の HTML。Bing → DuckDuckGo → Wikipedia API）で「どのページを見るか」を決める
2. 上位ページの本文を読み、記事本文の密度でノイズ（案内・広告）を落とす
3. 発話と *語が重なっている文だけ* を証拠として採用する（重なりのない文は捨てる）
4. 出典 URL とともに保持し、応答には「調べた内容の要約」として出す（貼らない）

ネットワークが使えない環境では例外を投げず、空の証拠を返す（上の層がそれを踏まえて動く）。
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field

from ..lang.phonetics import kana_ratio, normalize, to_hiragana
from ..mind.frame import Evidence
from ..research import ResearchEngine

log = logging.getLogger(__name__)

# ページ chrome・案内文・広告の断片。1 つでも入った文は証拠にしない。
BOILERPLATE: tuple[str, ...] = (
    "リワード", "reward", "サインイン", "アカウントを選択", "インテリジェント検索",
    "すばやく見つけ", "見つけられるよう", "Cookie", "クッキー", "利用規約", "プライバシー",
    "ログイン", "会員登録", "ログアウト", "広告", "スポンサー", "pr", "All rights reserved",
    "copyright", "検索結果", "JavaScript", "ジャバスクリプト", "ブラウザ", "シェア", "フォロー",
    "注目記事", "関連記事", "関連リンク", "メニュー", "トップへ", "お問い合わせ",
    "個人情報保護方針", "terms of use", "privacy policy", "ニュースレター", "アプリをダウンロード",
    "表示できません", "ページが見つかりません", "アクセスが集中", "認証してください",
    "別のアカウント", "公平でバランス", "表現の自由", "基本的権利", "情報への自由",
)

_SENT = re.compile(r"(?<=[。！？!?])\s*")
_WS = re.compile(r"\s+")


@dataclass
class Grounding:
    """1 回の裏取りの結果（証拠文 + 出典 + 診断）。"""

    query: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    elapsed_ms: float = 0.0
    cached: bool = False
    error: str | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.evidence)

    def as_dict(self) -> dict:
        return {"query": self.query, "evidence": [e.as_dict() for e in self.evidence[:6]],
                "sources": self.sources[:6], "elapsed_ms": round(self.elapsed_ms, 1),
                "cached": self.cached, "error": self.error, "reason": self.reason,
                "count": len(self.evidence)}


def query_terms(query: str) -> list[str]:
    """発話から「一致を確かめる語」を作る（問いの定型句は落として実語だけ）。"""
    t = normalize(query).lower()
    t = re.sub(r"(とは|って何|とは何|何ですか|なんですか|について|に関して|を教えて|教えて|"
               r"調べて|検索して|教えてもらえますか|？|\?|！|!|。)", " ", t)
    words: list[str] = []
    for w in re.split(r"[\s、,。・()（）「」『』\[\]]+", t):
        w = w.strip("「」『』。、,.!?！？:：;・")
        if len(w) >= 1 and w not in words:
            words.append(w)
    extra: list[str] = []
    for w in words:
        for piece in re.findall(r"[一-龯々]{2,8}|[ァ-ヶー]{2,10}|[A-Za-z][A-Za-z0-9_.+-]{1,}", w):
            if piece not in words and piece not in extra:
                extra.append(piece)
    return (words + extra)[:14]


def is_boilerplate(sent: str) -> bool:
    low = sent.lower()
    return any(b.lower() in low for b in BOILERPLATE)


def split_sentences(text: str, *, lo: int = 12, hi: int = 230) -> list[str]:
    out: list[str] = []
    t = _WS.sub(" ", str(text or "")).strip()
    if not t:
        return out
    for raw in _SENT.split(t):
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"^[、・\-\*•\d.]+", "", s).strip()
        if not (lo <= len(s) <= hi):
            continue
        if is_boilerplate(s):
            continue
        if s.count("http") or s.count("www."):
            continue
        if not re.search(r"[ぁ-んァ-ヶ一-龯A-Za-z]", s):
            continue
        out.append(s)
    return out


def score_sentence(sent: str, terms: list[str], *, pos: int = 0, rank: int = 0,
                   from_body: bool = False) -> float:
    """「発話と本当に同じ話をしている文か」を、語の重なりだけで素直に測る。"""
    body = normalize(sent).lower()
    kana_body = to_hiragana(body)
    score = 0.0
    matched = 0
    for term in terms:
        t = normalize(term).lower()
        if len(t) < 2:
            continue
        if t in body or to_hiragana(t) in kana_body:
            matched += 1
            score += 2.2 + min(2.4, len(t) * 0.35)
    if matched == 0:
        return 0.0
    if re.search(r"(とは|である|です|ます|指す|意味|定義|こと|もの)", body):
        score += 1.1
    if 24 <= len(sent) <= 130:
        score += 1.0
    if from_body:
        score += 1.4                      # 検索スニペットより本文のほうが詳しい
    score += max(0.0, 1.6 - pos * 0.12) + max(0.0, 1.2 - rank * 0.22)
    return score


def _grams(text: str, n: int = 5) -> set[str]:
    t = normalize(text)
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


class WebGrounding:
    """`ResearchEngine` を使って証拠文の集合を作る（Snipher がインターネットに出る唯一の口）。"""

    def __init__(self, engine: ResearchEngine | None = None, *,
                 enabled: bool | None = None, fetch_pages: int = 3, limit: int = 6):
        self.engine = engine or ResearchEngine()
        self.enabled = (os.environ.get("SNIPHER_WEB", "auto") != "off") if enabled is None else enabled
        self.fetch_pages = fetch_pages
        self.limit = limit
        self.last: Grounding | None = None

    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        return bool(self.enabled)

    def gather(self, query: str, *, explicit: bool | None = None, want: int | None = None,
               extra_queries: tuple[str, ...] = (), read_pages: int | None = None) -> Grounding:
        """検索 → 本文 → 証拠文。失敗時は空の Grounding（例外を外に出さない）。"""
        started = time.perf_counter()
        out = Grounding(query=normalize(query))
        if not self.enabled:
            out.error = "web_disabled"
            return out
        want = want or self.limit
        pages = read_pages if read_pages is not None else self.fetch_pages
        queries = [query] + [q for q in extra_queries if q and q != query]
        seen_urls: set[str] = set()
        collected: list[Evidence] = []
        terms_all: list[str] = []
        errors: list[str] = []
        for qi, q in enumerate(queries[:3]):
            try:
                res = self.engine.research(q, explicit=True if explicit else explicit,
                                           limit=max(3, want), fetch_pages=pages)
            except Exception as exc:  # noqa: BLE001
                log.debug("research 失敗: %s", exc)
                errors.append(f"research_error:{type(exc).__name__}")
                continue
            out.cached = out.cached or bool(res.cached)
            out.reason = out.reason or res.reason
            if res.error:
                errors.append(str(res.error))
            terms = query_terms(q)
            for term in terms:
                if term not in terms_all:
                    terms_all.append(term)
            for rank, src in enumerate(res.sources or []):
                url = str(src.get("url") or "")
                title = str(src.get("title") or "")
                if url in seen_urls:
                    continue
                body = str(src.get("content") or "")
                snippet = str(src.get("snippet") or "")
                # タイトルが「検索語とは」そのもの・または chrome だけのソースは捨てる
                if not body and not snippet:
                    continue
                pool: list[tuple[str, bool, int]] = []
                if body and body != snippet:
                    pool += [(s, True, i) for i, s in enumerate(split_sentences(body)[:60])]
                pool += [(s, False, i) for i, s in enumerate(split_sentences(snippet)[:6])]
                if title:
                    pool.append((title, False, 0))
                scored = sorted(
                    ((score_sentence(sent, terms_all, pos=pos, rank=rank, from_body=from_body),
                      sent, from_body) for sent, from_body, pos in pool),
                    key=lambda x: -x[0])
                picked_here: list[Evidence] = []
                for sc, sent, from_body in scored:
                    if sc <= 0:
                        continue
                    if not self._topic_overlap(sent, terms_all):
                        continue
                    picked_here.append(Evidence(text=sent, url=url, title=title,
                                                origin="web", score=sc))
                    if len(picked_here) >= (4 if body else 2):
                        break
                if picked_here:
                    seen_urls.add(url)
                    out.sources.append({"url": url, "title": title, "rank": rank + 1,
                                        "provider": src.get("provider") or "web",
                                        "snippets": [e.text[:120] for e in picked_here[:2]]})
                    collected.extend(picked_here)
            if len(collected) >= want * 2 and qi == 0:
                break
        # 重複除去（4-gram の重なり 0.5 超）+ スコア順
        collected.sort(key=lambda e: -e.score)
        kept: list[Evidence] = []
        grams: set[str] = set()
        for e in collected:
            g = _grams(e.text, 5)
            if g and grams:
                if len(g & grams) / len(g) > 0.5:
                    continue
            grams |= g
            kept.append(e)
            if len(kept) >= want:
                break
        out.evidence = kept
        out.elapsed_ms = (time.perf_counter() - started) * 1000
        out.error = None if kept else (";".join(errors[:2]) or "no_evidence")
        self.last = out
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _topic_overlap(sent: str, terms: list[str]) -> bool:
        """発話の実語との重なりが無い文は「別の記事の chrome」とみなして落とす。"""
        body = normalize(sent).lower()
        kana = to_hiragana(body)
        for t in terms:
            tt = normalize(t).lower()
            if len(tt) < 2:
                continue
            if tt in body or to_hiragana(tt) in kana:
                return True
        # 数字を含む問い合わせ（「何年」等）は数字の一致も手がかりにする
        nums = re.findall(r"\d+", " ".join(terms))
        if nums and any(n in body for n in nums):
            return True
        return False

    def read_url(self, url: str, *, terms: list[str] | None = None) -> list[Evidence]:
        """URL を 1 枚読んで証拠文を出す（/api/fetch と同じ口）。"""
        try:
            page = self.engine.fetch(url)
        except Exception:  # noqa: BLE001
            return []
        if page.error:
            return []
        terms = terms or []
        sents = split_sentences(page.text)
        out: list[Evidence] = []
        for i, s in enumerate(sents):
            sc = score_sentence(s, terms, pos=i) if terms else (2.0 if i < 6 else 0.0)
            if sc <= 0:
                continue
            out.append(Evidence(text=s, url=page.url, title=page.title, origin="web", score=sc))
            if len(out) >= 6:
                break
        return out

    def status(self) -> dict:
        st = {"enabled": self.enabled, "providers": [], "engine": None}
        try:
            st["engine"] = self.engine.status()
            st["providers"] = st["engine"].get("providers", [])
        except Exception:  # noqa: BLE001
            pass
        return st


__all__ = ["WebGrounding", "Grounding", "BOILERPLATE", "query_terms", "split_sentences",
           "score_sentence", "is_boilerplate"]
