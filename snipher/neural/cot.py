"""CoT（Chain of Thought）SFT データと DPO 選好ペアの合成。

ユーザー設計図の Phase 2 / Phase 3 に使う「思考プロセス付き」教師データを、
手元の素材（知識ベース・厳密計算・手書きの論理例）から **決定的に** 組み立てる
（seed 固定で再現ビルド可能。外部ダウンロード不要）。

形式はすべて同じワイヤ形式:

    <user>{質問}
    <asst><think>{結論の箇条書き}</think>
    {最終回答}

``<think>`` と ``</think>`` は文字レベル規約（`snipher/mind/recurrent.py` の
`THINK_OPEN/THINK_CLOSE` と同じリテラル文字列）なので、学習時に
「思考区間を先に書いてから答える」という癖がそのまま重みに焼き付きます。
"""

from __future__ import annotations

import random
from pathlib import Path

# 手書きの論理 CoT（ユーザー例そのままの流儀）
_LOGIC_EXAMPLES = [
    ("赤信号ではどうする？", "信号の意味を思い出す。赤は停止。青は進む。現在は赤。だから止まる。", "止まります。"),
    ("雨が降りそうだけど傘がない。どうする？", "濡れるか、買うか、待つかを比べる。買うのが確実。", "コンビニで傘を買います。"),
    ("お腹が空いた。何をする？", "空腹を満たす手段を並べる。食べるのが最短。", "何か食べます。"),
    ("財布を家に忘れた。どうする？", "現金が無いと買えない。取りに戻るか、電子決済を探す。", "家に取りに戻ります。"),
    ("外が暗い。何をする？", "暗いと見えない。灯りをつける。", "電気をつけます。"),
    ("喉が渇いた。どうする？", "水分を補うのが先。水を飲む。", "水を飲みます。"),
    ("宿題が終わっていない。どうする？", "残りを片付ける。優先度は高い。", "今から宿題をします。"),
    ("部屋が散らかっている。どうする？", "片付ける。順番に物を戻す。", "片付けを始めます。"),
]


def _kb() -> list:
    from .. import knowledge

    return knowledge.KnowledgeBase().items


def build_cot_docs(seed: int = 13, max_items: int | None = None) -> list[str]:
    """KB + 算数 + 論理から CoT 文書を作る。重複なし・決定的。"""
    rng = random.Random(seed)
    docs: list[str] = []
    seen: set[str] = set()

    def add(q: str, think: str, answer: str) -> None:
        doc = f"<user>{q}\n<asst><think>{think}</think>\n{answer}"
        if doc in seen:
            return
        seen.add(doc)
        docs.append(doc)

    # 1) 手書きの論理例（錨）
    for q, t, a in _LOGIC_EXAMPLES:
        add(q, t, a)

    # 2) 知識ベース
    items = _kb()
    if max_items:
        items = items[:max_items]
    for it in items:
        topic = str(it.get("topic") or "").strip()
        d = str(it.get("def") or "").strip()
        facts = [str(x).strip() for x in (it.get("facts") or []) if str(x).strip()]
        why = [str(x).strip() for x in (it.get("why") or []) if str(x).strip()]
        how = [str(x).strip() for x in (it.get("how") or []) if str(x).strip()]
        if not topic or not d:
            continue
        if why:
            add(f"{topic}とは？",
                f"まず定義を確認する。{topic}は{d}。次に理由を見る。{why[0]}",
                d)
        elif facts:
            add(f"{topic}について教えて",
                f"要点を整理する。{topic}は{d}。補足すると{facts[0]}",
                f"{d}{facts[0]}")
        else:
            add(f"{topic}とは？", f"定義を思い出す。{topic}は{d}。", d)
        if how:
            steps = "。".join(f"{i + 1}に{x}" for i, x in enumerate(how[:3]))
            add(f"{topic}のやり方は？", f"手順を立てる。{steps}。", how[0])

    # 3) 算数（厳密計算の CoT）
    for _ in range(80):
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
        add(f"{a}×{b}は？", f"{a}を{b}回足すと{a * b}。", str(a * b))
    for _ in range(60):
        a = rng.randint(10, 99)
        b = rng.randint(1, 9)
        add(f"{a}+{b}は？", f"{a}に{b}を足すと{a + b}。", str(a + b))
    for _ in range(40):
        a = rng.randint(10, 99)
        b = rng.randint(1, min(9, a))
        add(f"{a}-{b}は？", f"{a}から{b}を引くと{a - b}。", str(a - b))

    rng.shuffle(docs)
    return docs


def _strip_think(doc: str) -> str:
    """<think>…</think> を外した版（rejected 用）。"""
    out = doc
    if "<think>" in out and "</think>" in out:
        out = out.split("<think>", 1)[0] + out.split("</think>", 1)[1]
    return out


def _loopify(doc: str) -> str:
    """同じ文をループさせた版（rejected 用。n-gram 周回の教師）。"""
    asst = doc.split("<asst>", 1)[1]
    answer = asst.split("</think>", 1)[-1] if "</think>" in asst else asst
    head = doc.split("<asst>", 1)[0] + "<asst>"
    return f"{head}{answer}{answer}"


def _templatize(doc: str) -> str:
    """「〜の知識で答えます」等の定型表現を差した版（rejected 用）。"""
    head, answer = doc.split("<asst>", 1)
    return f"{head}<asst>私の知識で答えます。{answer}"


def build_dpo_pairs(seed: int = 13, max_items: int | None = None) -> list[dict]:
    """(chosen, rejected) の選好ペア。

    chosen   … <think> で整理してから答える（正しいフォーマット）
    rejected … think 無し / 同一文ループ / 定型表現つき のいずれか
    """
    rng = random.Random(seed)
    docs = build_cot_docs(seed=seed, max_items=max_items)
    pairs: list[dict] = []
    for doc in docs:
        if "<think>" not in doc:
            continue
        prompt = doc.split("<asst>", 1)[0].replace("<user>", "").rstrip("\n")
        chosen = doc.split("<asst>", 1)[1]
        variant = rng.randint(0, 2)
        if variant == 0:
            rejected = _strip_think(doc).split("<asst>", 1)[1]
        elif variant == 1:
            rejected = _loopify(doc).split("<asst>", 1)[1]
        else:
            rejected = _templatize(doc).split("<asst>", 1)[1]
        if not chosen or not rejected or chosen == rejected:
            continue
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    rng.shuffle(pairs)
    return pairs
