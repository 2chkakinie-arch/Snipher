"""CoT SFT データと DPO 選好ペアのテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_cot_docs_are_well_formed():
    from snipher.neural.cot import build_cot_docs

    docs = build_cot_docs(seed=1)
    assert len(docs) > 100
    for d in docs:
        assert "<user>" in d and "<asst>" in d
        assert "<think>" in d and "</think>" in d
        # 思考区間が回答より前にある
        assert d.index("<think>") < d.index("</think>")


def test_cot_math_is_correct():
    from snipher.neural.cot import build_cot_docs

    docs = build_cot_docs(seed=2)
    # 3×7=21 が正しい形で含まれる（再現性）
    assert any("<think>3を7回足すと21。</think>\n21" in d for d in docs)


def test_cot_is_deterministic():
    from snipher.neural.cot import build_cot_docs

    a = build_cot_docs(seed=7)
    b = build_cot_docs(seed=7)
    assert a == b


def test_dpo_pairs_have_distinct_rejected():
    from snipher.neural.cot import build_dpo_pairs

    pairs = build_dpo_pairs(seed=3)
    assert len(pairs) > 50
    for p in pairs:
        assert "<think>" in p["chosen"] and "</think>" in p["chosen"]
        assert p["chosen"] != p["rejected"]
        assert p["prompt"] and "think" not in p["prompt"].lower().split("asst")[-1]
