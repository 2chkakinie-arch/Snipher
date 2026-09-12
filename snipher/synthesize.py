"""Snipher Web統合シンセサイザー — 検索結果を「答え」に編む。

ResearchEngine が集めた sources (検索スニペット + html-fetch 本文) から、
要求の型(定義/理由/手順/一般)に合う文を抽出・要約して 0 から組み立てる。
コピペではなく「複数ソースの情報を1つの答えに編み直す」層。

方針:
  - ソースの文をそのまま貼らない。重複を除き、短く整形して結合する
  - 定義要求なら「Xとは、〜。」の形で断定する (根拠がある場合のみ)
  - ソースが無い/薄い場合は None を返し、呼び出し側の生成に譲る
"""

from __future__ import annotations

import re


_SENT_SPLIT = re.compile(r"(?<=[。！？!?])\s*")
_NOISE = (
    "cookie", "クッキー", "プライバシー", "利用規約", "ログイン", "会員登録",
    "広告", "メニュー", "ナビ", "検索結果", "関連記事", "おすすめ記事",
    "copyright", "all rights reserved", "フォロー", "シェア",
)


def _sentences(text: str) -> list[str]:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(t) if p and p.strip()]
    out: list[str] = []
    for p in parts:
        # ページパーサーが題名と本文を空白で繋いだ残骸を落とす
        p = re.sub(r"^(.{1,24}について)\s+", "", p).strip()
        if len(p) < 10 or len(p) > 220:
            continue
        low = p.lower()
        if any(n in p or n in low for n in _NOISE):
            continue
        if p.count("、") > 8:
            continue
        out.append(p)
    return out


def _query_words(query: str) -> list[str]:
    q = re.sub(r"(とは|って何|とは何|何ですか|なんですか|について|に関して|を教えて|教えて|とは？|？|\?|！|!|。)", " ", str(query or ""))
    words = [w for w in re.split(r"[\s、。,.]+", q) if w and len(w) >= 1]
    # 漢字・カタカナの塊を優先
    scored: list[str] = []
    for w in words:
        w = w.strip("「」『』()（）[]")
        if len(w) >= 1 and w not in scored:
            scored.append(w)
    # 2文字以上の部分語も追加 (「67ミーム」→「ミーム」)
    extra: list[str] = []
    for w in list(scored):
        m = re.findall(r"[一-龯々]{1,8}|[ァ-ヶー]{2,8}|[A-Za-z]{2,}", w)
        extra.extend(m)
    for w in extra:
        if w not in scored:
            scored.append(w)
    return [w for w in scored if len(w) >= 1][:12]


def _score_sentence(sent: str, words: list[str], pos: int) -> float:
    s = 0.0
    for w in words:
        if len(w) >= 2 and w in sent:
            s += 2.0 + min(2.0, len(w) * 0.3)
        elif len(w) == 1 and w in sent:
            s += 0.3
    # 定義らしさ
    if re.search(r"(とは|である|です|指す|いう|意味|こと|もの)", sent):
        s += 1.0
    # 長すぎ/短すぎを避ける
    if 20 <= len(sent) <= 120:
        s += 1.0
    # ソースの前半ほど重要
    s += max(0.0, 2.0 - pos * 0.15)
    return s


def rank_sentences(sources: list[dict], query: str, limit: int = 8) -> list[dict]:
    words = _query_words(query)
    cands: list[dict] = []
    for si, src in enumerate(sources or []):
        text = str(src.get("content") or src.get("snippet") or "")
        title = str(src.get("title") or "")
        url = str(src.get("url") or "")
        for pos, sent in enumerate(_sentences(text)):
            sc = _score_sentence(sent, words, pos)
            cands.append({"sentence": sent, "score": sc, "title": title, "url": url, "rank": si})
        # タイトル自体も候補に (説明的な題名だけ。問いを繰り返すだけの題名は捨てる)
        if (title and 6 <= len(title) <= 80
                and not any(n in title for n in ("検索", "Google", "Bing"))
                and re.search(r"(とは|である|です|ます|こと|もの|について|。|！|？)", title)):
            core = re.sub(r"(について|とは|って何|とは何か|の意味|の定義|。|！|？|\s)+$", "", title)
            if len(core) > 10 or "、" in title:  # 「Xについて」だけの題名は情報ゼロ
                sent = title if title.endswith(("。", "！", "？")) else title.rstrip("。") + "。"
                cands.append({"sentence": sent, "score": 1.2, "title": title, "url": url, "rank": si})
    # 重複除去 (文字 2-gram の重なりが高いものを落とす)
    cands.sort(key=lambda d: -d["score"])
    picked: list[dict] = []
    seen_grams: set[str] = set()
    for c in cands:
        grams = {c["sentence"][i:i + 4] for i in range(max(0, len(c["sentence"]) - 3))}
        if not grams:
            continue
        overlap = len(grams & seen_grams) / max(1, len(grams))
        if overlap > 0.55:
            continue
        seen_grams |= grams
        picked.append(c)
        if len(picked) >= limit:
            break
    return picked


def _topic_of(query: str) -> str:
    q = str(query or "").strip()
    q = re.sub(r"(とは|って何|とは何|何ですか|なんですか|について|に関して|を教えて|教えて|？|\?|！|!|。)+$", "", q)
    q = re.sub(r"^(.+?)の(意味|定義|概要)$", r"\1", q)
    return q.strip("「」『』、。 ").strip()[:24]


def synthesize_definition(query: str, sources: list[dict]) -> dict | None:
    """「Xとは」型の答えを編む。材料が薄ければ None。"""
    ranked = rank_sentences(sources, query, limit=6)
    if len(ranked) < 1:
        return None
    topic = _topic_of(query) or query.strip()[:20]
    # 本文: 上位文を 3 文まで結合
    body_sents = [c["sentence"] for c in ranked[:3]]
    body = "".join(s if s.endswith(("。", "！", "？")) else s + "。" for s in body_sents)
    if len(body) < 20:
        return None
    # 「Xとは、」で始まっていなければ付ける
    if topic and not body.startswith(topic):
        first = ranked[0]["sentence"]
        # 既に定義形ならそのまま
        if "とは" not in first[:24]:
            body = f"{topic}とは、{body}"
    # 補足: 残りの文から 1-2 文
    extra = ""
    if len(ranked) > 3:
        extra = "".join(s if s.endswith(("。", "！", "？")) else s + "。" for s in [c["sentence"] for c in ranked[3:5]])
    text = body + extra
    # 長すぎたら詰める
    if len(text) > 600:
        text = text[:600].rstrip("、。") + "。"
    used_urls = []
    for c in ranked[:5]:
        if c["url"] and c["url"] not in used_urls:
            used_urls.append(c["url"])
    return {"text": text, "topic": topic, "used": ranked[:5], "urls": used_urls}


def synthesize_general(query: str, sources: list[dict], qtype: str = "general") -> dict | None:
    """定義以外の問い・話題の答えを編む。"""
    if qtype == "def":
        return synthesize_definition(query, sources)
    ranked = rank_sentences(sources, query, limit=6)
    if len(ranked) < 2:
        return synthesize_definition(query, sources) if ranked else None
    topic = _topic_of(query)
    sents = [c["sentence"] for c in ranked[:4]]
    body = "".join(s if s.endswith(("。", "！", "？")) else s + "。" for s in sents)
    if len(body) < 24:
        return None
    if topic and len(topic) <= 20 and topic not in body[:40]:
        body = f"{topic}について調べた内容をまとめます。{body}"
    if len(body) > 700:
        body = body[:700].rstrip("、。") + "。"
    used_urls = []
    for c in ranked[:5]:
        if c["url"] and c["url"] not in used_urls:
            used_urls.append(c["url"])
    return {"text": body, "topic": topic, "used": ranked[:5], "urls": used_urls}


def synthesize(query: str, sources: list[dict], qtype: str = "general") -> dict | None:
    if not sources:
        return None
    try:
        return synthesize_general(query, sources, qtype)
    except Exception:
        return None


__all__ = ["synthesize", "synthesize_definition", "synthesize_general", "rank_sentences"]
