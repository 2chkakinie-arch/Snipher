"""チャットテンプレート管理。

優先順位:

1. ``tokenizer`` がネイティブの chat template を持っていればそれを使う
   （LFM2.5-1.2B-JP-202606 は ``chat_template.jinja`` を同梱）
2. 無ければ Snipher 内蔵の ChatML 形式テンプレート（LFM2 系の
   ``<|im_start|>/<|im_end|>`` と同じ形式）で組み立てる
3. それも不可、または明示的に OFF の場合は「テンプレートなし生成」:
   システムプロンプト等を一切付けず、テキストを素で続唱させる
   （ベースモデル的な completion 動作）

どこかで失敗しても必ず生成可能なように、段階的にフォールバックする。
"""

from __future__ import annotations

import logging
from typing import Iterable

log = logging.getLogger(__name__)

MODE_NATIVE = "native"      # tokenizer 同梱のテンプレート
MODE_BUILTIN = "builtin"    # 内蔵 ChatML テンプレート
MODE_RAW = "raw"            # テンプレートなし（素の completion）

Message = dict  # {"role": str, "content": str}

# LFM2 系トークナイザ（<|startoftext|>, <|im_start|>, <|im_end|>）向けの内蔵テンプレート。
# LiquidAI/LFM2.5-1.2B-JP-202606 の chat_template.jinja と同じワイヤ形式。
BUILTIN_TEMPLATE_DESC = "ChatML (LFM2系: <|im_start|>/<|im_end|>)"


def render_builtin_chatml(messages: Iterable[Message], bos: str | None = "<|startoftext|>") -> str:
    """内蔵 ChatML テンプレートを素の Python で描画する（jinja 不要）。"""
    parts: list[str] = []
    if bos:
        parts.append(bos)
    for m in messages:
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


class TemplateManager:
    """tokenizer の chat template の有無を吸収し、常にプロンプトを作れる。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.native_template: str | None = None
        try:
            self.native_template = getattr(tokenizer, "chat_template", None)
        except Exception:
            self.native_template = None

    @property
    def has_native(self) -> bool:
        return bool(self.native_template)

    def default_mode(self) -> str:
        return MODE_NATIVE if self.has_native else MODE_BUILTIN

    # ------------------------------------------------------------------ #
    def apply(
        self,
        messages: list[Message],
        use_template: bool = True,
        add_generation_prompt: bool = True,
    ) -> tuple[str, str]:
        """メッセージ列 → (プロンプト文字列, 使用モード)。

        モードは "native" | "builtin" | "raw"。use_template=False の場合は
        常に "raw"（テンプレートなし生成）。
        """
        if not messages:
            messages = [{"role": "user", "content": ""}]

        if use_template:
            # 1) ネイティブテンプレート
            if self.has_native:
                try:
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                    )
                    if isinstance(text, str) and text:
                        return text, MODE_NATIVE
                except Exception as exc:  # テンプレート実行時エラーにも耐える
                    log.warning("apply_chat_template 失敗、内蔵テンプレートへフォールバック: %s", exc)
            # 2) 内蔵 ChatML
            bos = None
            try:
                bos = self.tokenizer.bos_token or "<|startoftext|>"
            except Exception:
                bos = "<|startoftext|>"
            text = render_builtin_chatml(messages, bos=bos)
            if add_generation_prompt:
                text += "<|im_start|>assistant\n"
            return text, MODE_BUILTIN

        # 3) テンプレートなし生成: 役割ラベルも制御トークンも付けない
        chunks = [str(m.get("content", "")) for m in messages]
        text = "\n".join(c for c in chunks if c)
        return text, MODE_RAW

    def info(self) -> dict:
        return {
            "has_native": self.has_native,
            "native_template": (self.native_template or "")[:400] or None,
            "builtin": BUILTIN_TEMPLATE_DESC,
            "default_mode": self.default_mode(),
        }
