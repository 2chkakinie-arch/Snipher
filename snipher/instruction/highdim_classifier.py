"""高次元分類器 — 指示文全体を 1024 次元の密ベクトルに写し、タスクを判定する。

従来の `parser.classify_task` は、正規表現で単一キーワード（「抽出」「要約」など）だけを見ていた。
それだと「指示追従」「果物」「文章」といった強い語が 1 つあるだけで、別タスクに引っ張られてしまう。

ここでは、指示文・質問・材料（payload）を合わせた *全文脈* を 1024 次元の
文字 n-gram ハッシュベクトルに埋め込み、各タスクの重心（centroid）との
コサイン類似度で判定する。高次元なので、単一語の影響は全体で薄まる。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List, Tuple

DIM = 1024
TASKS = ["extract", "summarize", "code", "answer", "transform", "list", "write", "classify"]

# タスクごとの代表例（学習データ）。全文脈を使うので、同じ語が別タスクに出ても区別できる。
_EXAMPLES: Dict[str, List[str]] = {
    "extract": [
        "次のテキストから情報を抽出し、JSON形式で出力してください。テキスト: 東京から京都まで...",
        "CSV形式で抽出してください。テキスト: 氏名: 山田太郎, 年齢: 34",
        "次のキーに対する数値を埋めてJSONで出力してください。 {\"one\":1,\"two\":}",
        "文章から人名と日付を抜き出して表にしてください。",
        "以下の文章から origin, destination, duration, fare を抽出してJSONで",
    ],
    "summarize": [
        "以下の文章を3つの箇条書きで要約してください。文章: オセロや将棋などの完全情報ゲームにおいて、AIは探索アルゴリズムを用いて...",
        "文章を2つの箇条書きで要約してください。番号を付けてください。文章: ラーメンは中華麺とスープからなる...",
        "以下の文章を200字程度で要約してください。文章: 科学の進歩について...",
        "重要ポイントを3つにまとめてください。文章: 経済の動向について...",
        "文章の要点だけを短く要約してください。文章: 歴史の出来事について...",
        "以下の文章を読み、重要なポイントを3つの箇条書きで短く要約してください。文章: オセロや将棋...",
    ],
    "code": [
        "JavaScriptで、配列から重複を取り除いて昇順にソートする関数 uniqueSort(arr) を作成してください。",
        "Pythonでリストの重複を除いて降順に並べ替える関数 dedupeDesc(items) を作成してください。",
        "TypeScriptで数値配列の合計を返す関数 total(nums: number[]): number を作成してください。",
        "Goで countItems(items []string) int を実装してください。",
        "関数を作ってください。コードと実行結果を示してください。",
    ],
    "answer": [
        "日本の首都はどこですか？一言で答えてください。",
        "日本について詳しく教えてください。",
        "あなたは語尾に「〜ロボ」をつけるロボットです。挨拶をしてください。",
        "黄金比について教えてください。",
        "WebAssemblyのメリットは何ですか？200字で答えてください。",
        "しりとりのルールは何ですか？資料: しりとりは...",
        "ラーメンとは何ですか？",
        "プロンプトの指示（回答の作成、JSON抽出、要約、コード生成）に対して、モデルが指示通りのタスクを実行せず",
        "果物について教えてください。",
        "文章について教えてください。",
        "指示追従について教えてください。",
        "次のテキストから情報を抽出するタスクについて説明してください。",
        "Snipherの仕組みを3つの箇条書きで説明してください。「です・ます」は使わないこと。",
        "What is photosynthesis? Answer in English in 2 sentences.",
        "Answer in English. What is AI?",
        "Explain in English in 2 sentences.",
        "Snipherについて3つの箇条書きで説明してください。",
        "光合成とは何ですか？2文で答えてください。",
    ],
    "transform": [
        "次のテキストを大文字に変換してください。テキスト: \"hello snipher\"",
        "テキストを逆順にしてください。",
        "次の文章を英語に翻訳してください。テキスト: 今日は良い天気です。",
        "文字列を小文字に変換してください。",
        "次のテキストをローマ字に変換してください。",
        "Translate the following sentence into Japanese. Text: \"The weather is good today, so I went to the park for a walk.\"",
        "Translate into English. Text: 今日は良い天気です。",
        "Please translate the sentence into Japanese.",
        "次のテキストを英語に翻訳してください。",
        "英訳してください。",
        "和訳してください。",
    ],
    "list": [
        "以下の項目を3つの箇条書きで列挙してください。項目: りんごは赤い果物です。",
        "4つの都市を列挙してください。",
        "りんご、みかん、ぶどうを列挙してください。",
        "以下の単語を「果物」か「野菜」かで分類してください。りんご: トマト: バナナ:",
        "リストアップしてください。",
    ],
    "write": [
        "次の文章の続きを1文で書いてください。「今日は朝から雨が降っていたので、」",
        "業務メールを書いてください。お詫びの文面で200字程度。",
        "続きを書いてください。",
        "物語を1文で続けてください。",
        "文章を生成してください。",
    ],
    "classify": [
        "以下の単語を「果物」か「野菜」かで分類してください。りんご: トマト: バナナ:",
        "単語をカテゴリで分類してください。",
        "次の単語を分類してください。",
        "果物か野菜かで仕分けしてください。",
        "分類してください。",
    ],
}

def _embed(text: str) -> List[float]:
    """文字 2-4 gram を 1024 次元にハッシュし、L2 正規化した密ベクトルを返す。"""
    s = re.sub(r"\s+", " ", text or "").strip().lower()
    if not s:
        return [0.0]*DIM
    vec = [0.0]*DIM
    # 1) 文字 n-gram (2,3,4)
    for n in (2,3,4):
        for i in range(len(s)-n+1):
            gram = s[i:i+n]
            h = int(hashlib.sha256(gram.encode("utf-8")).hexdigest(), 16) % DIM
            vec[h] += 1.0
    # 2) 単語 n-gram 的な重み（空白で区切った語のハッシュも足す）
    for w in re.split(r"[、。！？\s]+", s):
        if len(w) >= 2:
            h = int(hashlib.sha256(("w_"+w).encode("utf-8")).hexdigest(), 16) % DIM
            vec[h] += 1.5
    # 3) 長さ正規化
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / norm for x in vec]

# 事前計算した重心（centroid）。起動時に1回だけ計算する（高次元なので単語1つでは動かない）。
_CENTROIDS: Dict[str, List[float]] = {}
_CENTROID_NORMS: Dict[str, float] = {}

def _build_centroids():
    global _CENTROIDS, _CENTROID_NORMS
    if _CENTROIDS:
        return
    for task, examples in _EXAMPLES.items():
        # 全文脈を埋め込む（指示+質問+材料を混ぜた例なので、文脈全体で学習される）
        vecs = [_embed(ex) for ex in examples]
        centroid = [0.0]*DIM
        for v in vecs:
            for i in range(DIM):
                centroid[i] += v[i]
        # 平均
        centroid = [x / len(vecs) for x in centroid]
        # 正規化
        norm = math.sqrt(sum(x*x for x in centroid)) or 1.0
        centroid = [x / norm for x in centroid]
        _CENTROIDS[task] = centroid
        _CENTROID_NORMS[task] = norm

_build_centroids()

def _cosine(a: List[float], b: List[float]) -> float:
    return sum(x*y for x, y in zip(a, b))

def classify_full(text: str, *, instruction: str = "", payload: str = "", question: str = "") -> Tuple[str, float, Dict[str, float]]:
    """全文脈（instruction+payload+question）を高次元で分類する。

    Returns: (task, confidence, scores)
    """
    # 全文脈を1つに繋げて埋め込む（単一キーワードではなく全文で判定）
    full = " ".join([str(instruction or ""), str(payload or ""), str(question or ""), str(text or "")]).strip()
    if not full.strip():
        return "", 0.0, {}
    vec = _embed(full)
    scores: Dict[str, float] = {}
    for task, cent in _CENTROIDS.items():
        # コサイン類似度
        sim = _cosine(vec, cent)
        # 長さや構造の手がかりも少し加える（高次元ベクトルだけでは弱い場合の補助）
        # 例: JSON指示なら extract を少し上げるが、単一語ではなく全文の構造で
        if task == "extract" and re.search(r"json|csv|抽出|スキーマ", full, re.IGNORECASE):
            sim += 0.04
        if task == "code" and re.search(r"関数|function\s+\w+\s*\(|def\s+\w+\s*\(|class\s+\w+|```", full, re.IGNORECASE):
            sim += 0.04
        # English answer should not be confused with code: downweight code if no code artifact
        if task == "code" and re.search(r"Answer in English|What is", full) and not re.search(r"関数|function|def |class |```|implement|create.*function", full, re.IGNORECASE):
            sim -= 0.08
        if task == "answer" and re.search(r"Answer in English|What is", full):
            sim += 0.06
        if task == "summarize" and re.search(r"要約|まとめ|箇条書き", full):
            sim += 0.03
        if task == "classify" and re.search(r"分類|仕分け|カテゴリ", full):
            sim += 0.05
        if task == "transform" and re.search(r"変換|翻訳|大文字|小文字|translate", full, re.IGNORECASE):
            sim += 0.04
        if task == "write" and re.search(r"続き|生成|書いて|作成して", full):
            sim += 0.03
        scores[task] = sim
    # 最も高いものを選ぶ
    best = max(scores, key=lambda k: scores[k])
    # 2番目との差で確信度を測る（単一語に引っ張られないかの指標）
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores)>1 else 0)
    # 信頼度は margin と絶対値で決める。単一語だけでは margin が小さくなる。
    conf = max(0.0, min(1.0, 0.55 + margin*1.8 + (sorted_scores[0]-0.2)*0.3))
    # 低すぎる場合は「分からない」とする（キーワードだけで決めない）
    if sorted_scores[0] < 0.18:
        return "", 0.0, scores
    return best, conf, scores

def classify(text: str) -> Tuple[str, float]:
    """簡易API: テキストだけで判定（後方互換）"""
    task, conf, _ = classify_full(text)
    return task, conf

# デバッグ用: ベクトルの次元数を外部から確認できる
def dim() -> int:
    return DIM

__all__ = ["classify", "classify_full", "dim", "DIM", "TASKS"]
