"""Snipher 知識ベース v2 の「中身」（人が書いた日本語そのもの）。

`tools/build_kb.py` はここにあるトピック定義を読み込んで `snipher/data/kb.json` を作る。
1 トピック = 1 辞書エントリで、**問いの型**（定義/理由/方法/時期/場所/値段/感想…）ごとに
答えを持つの が v1 との最大の違いです。v1 は「トピックが合えば facts[0] を返す」だけなので、
「量子コンピュータとは」に自己紹介が返るようなズレが起きていた。

書式（コンパクトに書けるよう、ali/verbs/tags はカンマ区切り文字列も許す）:

    T("hanabi", "花火", cat="行事",
      ali="はなび,打ち上げ花火,花火大会,線香花球",
      d="火薬の燃焼と爆発で、夜空に光と音の花を咲かせる夏の娯楽です。",
      f=["色は炎色反応で決まり、ストロンチウムは赤、銅は青緑に見えます。"],
      why=["…"], how=["…"], when="…", where="…", cost="…", tips=["…"],
      opinion="…", qa=[("どうして色が変わるの", "…")],
      fu=["今年の花火はもう見ましたか。"], rel="夏祭り,夏", verbs="上がる,打ち上げる,見る")

フィールドの意味:
    d        … 定義（「Xとは」に答える一文）
    f        … 事実（説明の本文。1 文 1 事実）
    why      … 理由・しくみ（「なぜ」に答える）
    how      … 手順・やり方（「作り方/やり方」に答える。手順順に並べる）
    when     … 時期・時間
    where    … 場所
    who      … 人・主体
    cost     … 値段・費用・量
    tips     … コツ・注意点
    opinion  … 感想・好み（「好き？/おすすめは？」に答える一人称の文）
    qa       … よくある問いと答えの組（型が分からない具体質問はこちらで拾う）
    fu       … 会話を広げる質問（応答の末尾に 1 つだけ付ける）
    rel      … 関連トピック名
"""

from __future__ import annotations

from typing import Any, Iterable

# ---------------------------------------------------------------------- #
# 小さな DSL
# ---------------------------------------------------------------------- #
_LIST_FIELDS = ("f", "why", "how", "tips", "qa", "fu")
_STR_FIELDS = ("d", "when", "where", "who", "cost", "opinion")


def _lst(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()] if "," in v and len(v) < 60 else [v.strip()] if v.strip() else []
    out: list[str] = []
    for x in v:
        if isinstance(x, (tuple, list)):
            out.append(x)  # qa の組はそのまま
        else:
            s = str(x).strip()
            if s:
                out.append(s)
    return out


def _words(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.replace("、", ",").split(",") if x.strip()]
    return [str(x).strip() for x in v if str(x).strip()]


def T(id: str, topic: str, *, cat: str = "", ali: Any = None, tags: Any = None,
      d: str | None = None, f: Any = None, why: Any = None, how: Any = None,
      when: str | None = None, where: str | None = None, who: str | None = None,
      cost: str | None = None, tips: Any = None, opinion: str | None = None,
      qa: Any = None, fu: Any = None, rel: Any = None, verbs: Any = None) -> dict:
    """1 トピックを作る（build_kb.py が検証してから kb.json に書き出す）。"""
    qas: list[tuple[str, str]] = []
    for item in _lst(qa):
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            qas.append((str(item[0]).strip(), str(item[1]).strip()))
        elif isinstance(item, str) and "|" in item:
            q, a = item.split("|", 1)
            qas.append((q.strip(), a.strip()))
    return {
        "id": id,
        "topic": topic,
        "cat": cat,
        "aliases": _words(ali) or [topic],
        "tags": _words(tags),
        "def": (d or "").strip(),
        "facts": _lst(f),
        "why": _lst(why),
        "how": _lst(how),
        "when": (when or "").strip(),
        "where": (where or "").strip(),
        "who": (who or "").strip(),
        "cost": (cost or "").strip(),
        "tips": _lst(tips),
        "opinion": (opinion or "").strip(),
        "qa": qas,
        "followups": _lst(fu),
        "related": _words(rel),
        "verbs": _words(verbs),
    }


# ---------------------------------------------------------------------- #
# 集約
# ---------------------------------------------------------------------- #
MODULES = ("food", "nature", "culture", "tech", "life", "animals", "talk", "netculture")


def all_items() -> list[dict]:
    """全ドメインのトピックを 1 つのリストにまとめる。"""
    import importlib

    out: list[dict] = []
    for name in MODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        items: Iterable[dict] = mod.ITEMS
        for it in items:
            out.append(it)
    return out


def validate(items: list[dict]) -> list[str]:
    """書き出し前の品質チェック。エラー文字列のリストを返す（空なら OK）。"""
    errs: list[str] = []
    seen_id: set[str] = set()
    seen_topic: set[str] = set()
    for it in items:
        tid = it["id"]
        if tid in seen_id:
            errs.append(f"id が重複: {tid}")
        seen_id.add(tid)
        if it["topic"] in seen_topic:
            errs.append(f"topic が重複: {it['topic']}")
        seen_topic.add(it["topic"])
        if not it.get("def"):
            errs.append(f"{tid}: 定義(d)がありません")
        if not it.get("facts") and not it.get("qa"):
            errs.append(f"{tid}: 事実(f)も qa もありません")
        for field in _STR_FIELDS:
            v = it.get(field) or ""
            if v and not v.endswith(("。", "！", "？", "!", "?", "です", "ます")):
                errs.append(f"{tid}.{field}: 文末が閉じていません → {v[-12:]}")
        for field in ("facts", "why", "tips", "followups"):
            for s in it.get(field) or []:
                if len(s) < 6:
                    errs.append(f"{tid}.{field}: 短すぎます → {s}")
        for s in it.get("how") or []:      # 手順は短い句でよい
            if len(s) < 3:
                errs.append(f"{tid}.how: 短すぎます → {s}")
        for q, a in it.get("qa") or []:
            if not q or not a:
                errs.append(f"{tid}.qa: 空の問答があります")
        for a in list(it.get("aliases") or []):
            if len(a) > 24:
                errs.append(f"{tid}: alias が長すぎます → {a}")
    return errs


__all__ = ["T", "all_items", "validate", "MODULES"]
