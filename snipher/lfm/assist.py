"""ハイブリッド補正: Snipher-mini の速さ + LFM2.5 の賢さ。

考え方:
    1. Snipher-mini(476 パラメータ + 対話テーブル)が**常に**一瞬で下書きを作る
    2. 下書きの各スロット決定には確率(ソフトマックス確率)が付く。
       その重み平均 = confidence が十分高ければ **そのまま返す**(高速経路、
       LFM は 1 トークンも消費しないので速度は現状維持)
    3. confidence が閾値未満(= 確率的に不安な話題・語彙)のときだけ
       LFM2.5-1.2B-JP に「下書きを自然な返答に書き直す」よう依頼する。
       出力は短い(既定 64 トークン上限)ので、応答全体を LFM で作るより
       はるかに速く、品質だけが引き上がる。

さらにどちらの経路でも polisher が助動詞・文体の欠落をルールで補う
(助動詞の補い)。これはマイクロ秒単位なので速度に影響しない。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from ..polisher import Polisher
from ..responder import Responder

DEFAULT_ASSIST_MAX_NEW_TOKENS = 64

# 下書きを自然な返答に書き直させるためのプロンプト(短く・意味保持・丁寧体)
POLISH_SYSTEM = (
    "あなたは日本語の会話アシスタントです。"
    "与えられた下書きを、相手の発話に対する自然で短い返答(1〜2文)に書き直します。"
    "意味はできるだけ保ち、丁寧体(です・ます)で返します。"
    "出力は書き直した返答の本文のみ。記号の説明や前置きは付けません。"
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


@dataclass
class AssistConfig:
    """ハイブリッド補正の設定(環境変数 SNIPHER_ASSIST_* で上書き)。"""

    enabled: bool = True
    threshold: float = 0.35           # confidence がこの値未満なら LFM 補正
    max_new_tokens: int = DEFAULT_ASSIST_MAX_NEW_TOKENS

    @classmethod
    def from_env(cls) -> "AssistConfig":
        return cls(
            enabled=_env_bool("SNIPHER_ASSIST_ENABLED", True),
            threshold=_env_float("SNIPHER_ASSIST_THRESHOLD", 0.35),
            max_new_tokens=int(_env_float("SNIPHER_ASSIST_MAX_NEW_TOKENS", DEFAULT_ASSIST_MAX_NEW_TOKENS)),
        )


class HybridAssist:
    """mini 下書き → (必要なら) LFM 書き直し、の指揮者。"""

    def __init__(self, responder: Responder | None = None, polisher: Polisher | None = None,
                 cfg: AssistConfig | None = None):
        self.responder = responder or Responder()
        self.polisher = polisher or Polisher(self.responder.lex)
        self.cfg = cfg or AssistConfig.from_env()

    # ------------------------------------------------------------------ #
    def draft(self, user_text: str) -> dict:
        """高速経路: 対話テーブル + 確率的補完 + 助動詞の補い。

        数ミリ秒で返る。戻り値の confidence が LFM 補正の判定に使われる。
        ``base_text`` は確率的生成を含まない安全な骨子(LFM が無い環境で
        不確実な生成文をそのまま出さないための退避先)。
        """
        t0 = time.time()
        r = self.responder.reply(user_text)
        base = self.polisher.polish(r.get("base_text", r["text"]), register="polite")
        p = self.polisher.polish(r["text"], register="polite")
        r["text"] = p["text"]
        r["base_text"] = base["text"]
        r["fixes"] = p["fixes"]
        r["draft_seconds"] = round(time.time() - t0, 5)
        return r

    def needs_lfm(self, draft: dict) -> bool:
        if not self.cfg.enabled:
            return False
        conf = float(draft.get("confidence", 1.0))
        return conf < self.cfg.threshold

    # ------------------------------------------------------------------ #
    def polish_messages(self, draft_text: str, user_text: str) -> list[dict]:
        """LFM に渡す「書き直し」プロンプト。"""
        return [
            {"role": "system", "content": POLISH_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"相手の発話: {user_text}\n"
                    f"下書き: {draft_text}\n"
                    "この下書きを自然な返答に書き直してください。"
                ),
            },
        ]

    def polish_with_lfm(self, lfm_engine, draft: dict, user_text: str,
                        max_new_tokens: int | None = None, temperature: float = 0.3):
        """LFM で下書きを書き直す。stream_chat のイベントをそのまま中継する。

        LFM 出力が空のときは下書きに安全フォールバックする。
        """
        msgs = self.polish_messages(draft["text"], user_text)
        out: list[str] = []
        stats: dict = {}
        for ev in lfm_engine.stream_chat(
            msgs,
            max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
            min_new_tokens=4,
            temperature=temperature,
            top_k=50,
            repetition_penalty=1.05,
            use_template=True,
            system_prompt=POLISH_SYSTEM,
        ):
            if ev.get("type") == "delta":
                out.append(ev.get("text", ""))
                yield ev
            elif ev.get("type") == "start":
                yield ev  # UI 用にエンジン名などを中継
            elif ev.get("type") == "done":
                stats = ev.get("stats", {})
            elif ev.get("type") == "error":
                yield ev

        text = "".join(out).strip()
        # モデルが前置きを付けるケースへの保険: 引用ブロックだけ抜く
        if text.startswith(("書き直し", "返答:")):
            text = text.split(":", 1)[-1].strip()
        if not text:
            text = draft["text"]
            stats["assist_fallback"] = "empty_output"
        stats["assist"] = "lfm"
        stats["draft"] = draft["text"]
        stats["draft_confidence"] = draft.get("confidence")
        return text, stats
