"""出力の番人 — 同じ文・同じ定型句のループを *物理的に* 禁じる（v8）。

「〜の知識で答えます」「〜について、要点をまとめます。」のような定型句や、
同じ文の繰り返しは、どの経路（指示・知識・神経）から出てもここで落とす。
生成側の気分に任せず、出力の最終段で機械的に検査・除去するのが役目。

    BANNED_PHRASES … 1 字でも含んでいれば落とす定型句
    find_repeats    … 繰り返し（同じ文・同じ長い句・同じ語の連呼）の検出
    scrub           … 定型句の除去 + 繰り返しの圧縮（JSON/CSV/コードは壊さない）

神経系（lm8）のサンプラーは、同じ表を *ロジットの段* で禁じる
（`snipher/lm8/infer.py` の banned-phrase ban）。ここはその番人の出力側。
"""

from __future__ import annotations

import re

#: 出力のどこにも残ってはいけない定型句（v6 まで実際に出ていたもの + 類形）
BANNED_PHRASES = (
    "の知識で答えます",
    "知識で答えま",
    "単独の項目を持って",
    "構成語",
    "について、要点をまとめます。",
    "について、お知らせします。",
    "ご確認のうえ、必要であれば",
    "内容は【内容】で",
    "ついての短文について",
    "同じような意味について、",
    "要点をまとめます。内容は",
    "お知らせします。内容は",
    "推測では埋めません。どんな場面で使う語かを一言もらえれば",
)

#: 繰り返しとみなす句の長さ（これ以上の長さの一致を 2 回以上許さない）
REPEAT_GRAM = 12
#: 同じ語の連呼とみなす回数
REPEAT_WORD = 3


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?])|\n", str(text or ""))
    return [p.strip() for p in parts if p.strip()]


def find_repeats(text: str) -> list[str]:
    """繰り返しを見つけて、その句の一覧を返す（無ければ空）。"""
    out: list[str] = []
    body = str(text or "")
    if not body.strip():
        return out
    # 1) 同じ文が 2 回以上（JSON/コードの行は除く）
    seen: set[str] = set()
    for sent in _sentences(body):
        s = sent.strip()
        if len(s) < 8 or s.lstrip().startswith(("{", "[", "|", "-", "*", "・", "#", ">", "```")):
            continue
        if re.match(r"^\d+[.)、]", s):
            continue
        if s in seen and s not in out:
            out.append(s)
        seen.add(s)
    # 2) 同じ長い句（12 字以上）が 2 回以上
    flat = re.sub(r"\s+", "", body)
    if len(flat) >= REPEAT_GRAM * 2:
        grams: dict[str, int] = {}
        for i in range(len(flat) - REPEAT_GRAM + 1):
            g = flat[i:i + REPEAT_GRAM]
            if re.fullmatch(r"[「」『』、。！？!?…―ー\s]+", g):
                continue
            grams[g] = grams.get(g, 0) + 1
        for g, n in grams.items():
            if n >= 2 and all(g not in x for x in out):
                # ほかで既報の文に含まれる句は数えない
                out.append(g)
                if len(out) >= 6:
                    break
    # 3) 同じ語の連呼（「とてもとてもとても」）
    for m in re.finditer(r"([\u3041-\u3096\u30a1-\u30fa\u4e00-\u9fffA-Za-z]{2,8})"
                         r"(?:\s*\1\s*){2,}", body):
        if m.group(0) not in out:
            out.append(m.group(0))
    return out


def scrub(text: str) -> tuple[str, dict]:
    """定型句の除去 + 繰り返しの圧縮。返るのは (本文, 診断)。

    JSON・CSV・表・コードは *形* なので壊さない（その中は検査だけする）。
    """
    body = str(text or "")
    report: dict = {"removed": [], "repeats": 0}
    if not body.strip():
        return body, report
    stripped = body.strip()
    structured = stripped.startswith(("{", "[")) or "```" in stripped \
        or bool(re.search(r"^\s*\|.+\|\s*$", body, re.M))
    if not structured:
        # 区切り（。/改行）を保ったまま文に割る（裸の答えは 1 文・区切り無し）
        tokens = re.split(r"([。！？!?]+|\n+)", body)
        pairs: list[tuple[str, str]] = []
        for i in range(0, len(tokens), 2):
            txt = tokens[i].strip()
            sep = tokens[i + 1] if i + 1 < len(tokens) else ""
            if txt:
                pairs.append((txt, sep))
        # 定型句を含む *文* を落とす（文が無くなりそうなら句だけ落とす）
        kept = [(txt, sep) for txt, sep in pairs
                if not next((p for p in BANNED_PHRASES if p in txt), "")]
        for txt, _sep in pairs:
            hit = next((p for p in BANNED_PHRASES if p in txt), "")
            if hit and (txt, _sep) not in kept:
                report["removed"].append(hit)
        if kept:
            pairs = kept
        else:
            for phrase in BANNED_PHRASES:
                if phrase in body:
                    report["removed"].append(phrase)
                    body = body.replace(phrase, "")
            return body.strip(), report
        # 同じ文の 2 回目以降を落とす
        seen: set[str] = set()
        deduped: list[tuple[str, str]] = []
        for txt, sep in pairs:
            if len(txt) >= 8 and txt in seen:
                report["repeats"] += 1
                continue
            seen.add(txt)
            deduped.append((txt, sep))
        body = "".join(txt + sep for txt, sep in deduped)
        body = re.sub(r"。{2,}", "。", body).strip()
    else:
        # 構造化出力は壊さない（検査だけして診断に残す）
        for phrase in BANNED_PHRASES:
            if phrase in body:
                report["removed"].append(f"structured:{phrase}")
    return body, report


__all__ = ["BANNED_PHRASES", "REPEAT_GRAM", "REPEAT_WORD", "find_repeats", "scrub"]
