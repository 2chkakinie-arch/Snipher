"""Snipher の思考層（mind）— 見る・集める・決める・書く・検べる の 5 工程。

`think()` だけが外に公開する顔です。ここで決めたことを composer が引き受け、
core / api は経路表示だけを整えます。
"""

from __future__ import annotations

from .frame import Activity, Claim, Entity, Evidence, Frame, Rule, split_sentences
from .parse import build_frame, classify_ask, detect_act
from .play import judge, pick, word_from_turn
from .rules import chain_ok, check, compile_rules, enforce, from_definition
from .state import ConversationState
from .think import Thought, think
from .voice import Rendered, render

__all__ = ["think", "Thought", "build_frame", "classify_ask", "detect_act", "compile_rules",
           "from_definition", "check", "enforce", "chain_ok", "ConversationState", "pick",
           "judge", "word_from_turn", "render", "Rendered", "Frame", "Claim", "Entity",
           "Evidence", "Rule", "Activity", "split_sentences"]
