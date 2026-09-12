"""超小型エンジン用の対話応答器(意図分類 + テーブル応答)。

Snipher-mini の「ランダムに1文を生成するだけ」だった応答を、
ユーザーの発話の意図に合わせた返答に引き上げる。知識はすべて
snipher/data/responses.json のテーブル(= パラメータ)で、
解析は既存の parser / 確率生成は既存の generator を再利用する。

返信の信頼度(confidence):
    - テーブルの定形文      → 0.95(確定。LFM 補正は不要)
    - 確率的生成文を含む場合 → 生成スロットの確率の重み平均(低いと LFM 補正)
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

from .generator import Generator
from .lexicon import Lexicon
from .parser import Parser
from .tasks import TaskRouter

_RULE_CONFIDENCE = 0.95

# 普通体(タメ口)判定の目安。応答は常に丁寧体で返すが、解析の参考にする。
_CASUAL_MARKS = ("だね", "だよ", "じゃん", "だろ", "だな", "わかん", "めっちゃ", "超")


def _load_tables() -> dict:
    for candidate in (
        Path(__file__).parent / "data" / "responses.json",
        resources.files("snipher").joinpath("data/responses.json"),
    ):
        try:
            return json.loads(Path(candidate).read_text(encoding="utf-8"))
        except Exception:
            continue
    return {"intents": [], "question": {}, "fallback": {}}


class Responder:
    """意図に応じた会話返答を作る(テーブル + 確率的補完)。"""

    def __init__(self, lexicon: Lexicon | None = None, generator: Generator | None = None,
                 parser: Parser | None = None):
        self.lex = lexicon or Lexicon()
        self.parser = parser or Parser(self.lex)
        self.gen = generator or Generator(self.lex)
        self.tasks = TaskRouter()
        tables = _load_tables()
        self.intents = tables.get("intents", [])
        self.question_tbl = tables.get("question", {})
        self.fallback_tbl = tables.get("fallback", {})

    # ------------------------------------------------------------------ #
    def reply(self, text: str, seed: int | None = None) -> dict:
        """ユーザー発話 → 返答。{"text","intent","confidence","use_generator","fixes_hint"}"""
        clean = self._normalize(text)
        topic = self._find_topic(text)

        # 計算・コード・比較は、会話テーブルより先に厳密な道具へ渡す。
        # Responder 単体を使う古い統合コードでも、KB の近い話題を誤返答しない。
        try:
            task = self.tasks.answer(text, web=False)
        except Exception:  # noqa: BLE001
            task = None
        if task is not None:
            return {
                "text": task.text,
                "base_text": task.text,
                "intent": "task",
                "topic": task.metadata.get("item") if task.metadata else None,
                "confidence": round(float(task.confidence), 4),
                "use_generator": False,
                "task": task.as_dict(),
            }

        # 1) キーワード一致の意図テーブル
        for intent in self.intents:
            for kw in intent.get("keywords", []):
                if kw and kw in clean:
                    return self._from_frames(
                        intent["id"], intent.get("replies", []), topic,
                        confidence=_RULE_CONFIDENCE, seed=seed,
                    )

        # 2) 疑問文(文構造解析 or 末尾?)
        is_question = clean.endswith("?") or clean.endswith("？") or \
            bool(clean) and (clean.endswith("か。") or clean.endswith("か"))
        if not is_question:
            try:
                is_question = self.parser.parse(text).get("structure", {}).get("name") == "疑問文"
            except Exception:
                is_question = False
        if is_question:
            tbl = self.question_tbl
            key = "with_topic" if topic else "without_topic"
        else:
            tbl = self.fallback_tbl
            key = "with_topic" if topic else "without_topic"
        frames = tbl.get(key, [])

        # 3) テーブルの骨子 + 話題に関する確率的な1文(= 不確実性の源)
        return self._from_frames(
            "question" if is_question else "fallback", frames, topic,
            confidence=None, seed=seed, user_text=text,
        )

    # ------------------------------------------------------------------ #
    def _from_frames(self, intent: str, frames: list, topic: str | None, *,
                     confidence: float | None, seed: int | None = None,
                     user_text: str = "") -> dict:
        import random

        rng = random.Random(seed)
        if not frames:
            frames = [["なるほど。", "もう少し教えてください。"]]

        order = list(range(len(frames)))
        rng.shuffle(order)
        chosen = None
        for idx in order:
            cand = frames[idx]
            if any("{topic}" in part and not topic for part in cand):
                continue
            chosen = cand
            break
        parts = [p.replace("{topic}", topic) if topic else p for p in (chosen or [])]
        base_text = "".join(parts)
        text = base_text
        used_generator = False
        gen_conf = None

        # 定形文だけの意図(挨拶等)はここで確定。fallback/question は
        # 話題に合わせた確率的な1文を追加して会話を広げる。
        if intent in ("fallback", "question") and confidence is None:
            try:
                g = self.gen.generate(
                    prompt=topic or user_text or None,
                    register="polite",
                    seed=seed,
                )
                sent = g.get("text", "").strip()
                if sent:
                    text = f"{text}{sent}"
                    used_generator = True
                    gen_conf = g.get("confidence")
            except Exception:
                pass

        conf = confidence if confidence is not None else (
            gen_conf if gen_conf is not None else _RULE_CONFIDENCE
        )
        return {
            "text": text,
            # base_text = 確率的生成を含まない安全な骨子(LFM が無い環境で
            # confidence が低いときはこちらに退避する)
            "base_text": base_text,
            "intent": intent,
            "topic": topic,
            "confidence": round(float(conf), 4),
            "use_generator": used_generator,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[\s。、！?？!．.]", "", str(text or ""))

    def _find_topic(self, text: str) -> str | None:
        """発話から既知の名詞(話題)を見つける。"""
        try:
            t = self.parser.parse(text).get("topic")
            if t:
                return t
        except Exception:
            pass
        clean = self._normalize(text)
        for n in sorted(self.lex.nouns, key=lambda n: -len(n["s"])):
            if n["s"] in clean:
                return n["s"]
        return None
