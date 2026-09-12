"""Snipher の単体テスト。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher import Lexicon, Morphology, Parser, ProbabilityModel, Generator, SnipherEngine  # noqa: E402


def test_lexicon_loads():
    lex = Lexicon()
    stats = lex.stats()
    assert stats["verbs"] > 50
    assert stats["nouns"] > 100
    assert stats["adjectives"] > 50
    assert stats["patterns"] >= 10
    assert stats["corpus_sentences"] >= 20


def test_morphology_godan():
    m = Morphology()
    kaku = {"s": "書く", "r": "かく", "c": "五段", "tr": "他動詞", "p": "を", "t": ["行動"]}
    assert m.inflect_verb(kaku, "masu") == "書き"
    assert m.inflect_verb(kaku, "ta") == "書いた"
    assert m.inflect_verb(kaku, "te") == "書いて"
    assert m.inflect_verb(kaku, "nai") == "書かない"

    nomu = {"s": "飲む", "r": "のむ", "c": "五段", "tr": "他動詞", "p": "を", "t": ["食事"]}
    assert m.inflect_verb(nomu, "ta") == "飲んだ"

    kau = {"s": "買う", "r": "かう", "c": "五段", "tr": "他動詞", "p": "を", "t": ["行動"]}
    assert m.inflect_verb(kau, "ta") == "買った"
    assert m.inflect_verb(kau, "nai") == "買わない"

    iku = {"s": "行く", "r": "いく", "c": "五段", "tr": "自動詞", "p": "へ", "t": ["移動"]}
    assert m.inflect_verb(iku, "ta") == "行った"


def test_morphology_ichidan_suru():
    m = Morphology()
    taberu = {"s": "食べる", "r": "taberu", "c": "一段", "tr": "他動詞", "p": "を", "t": ["食事"]}
    assert m.inflect_verb(taberu, "ta") == "食べた"
    assert m.inflect_verb(taberu, "masu") == "食べ"
    assert m.inflect_verb(taberu, "nai") == "食べない"

    benkyou = {"s": "勉強する", "r": "benkyou suru", "c": "サ変", "tr": "他動詞", "p": "を", "t": ["勉強"]}
    assert m.inflect_verb(benkyou, "ta") == "勉強した"
    assert m.inflect_verb(benkyou, "masu") == "勉強し"

    kuru = {"s": "来る", "r": "kuru", "c": "カ変", "tr": "自動詞", "p": "が", "t": ["移動"]}
    assert m.inflect_verb(kuru, "ta") == "来た"
    assert m.inflect_verb(kuru, "nai") == "来ない"


def test_morphology_adjective():
    m = Morphology()
    takai = {"s": "高い", "r": "takai", "c": "い形容詞", "t": ["評価"]}
    assert m.inflect_adjective(takai, "past") == "高かった"
    assert m.inflect_adjective(takai, "negative") == "高くない"

    shizuka = {"s": "静か", "r": "shizuka", "c": "な形容詞", "t": ["自然"]}
    assert m.inflect_adjective(shizuka, "attributive") == "静かな"
    assert m.inflect_adjective(shizuka, "past") == "静かだった"


def test_parser_analyze():
    engine = SnipherEngine(seed=1)
    result = engine.analyze("私は猫が好きです。")
    assert "tokens" in result
    assert result["structure"]["pattern"] in ("state", "nominal", "plain", "topic_subject")
    assert any(t["pos"] == "名詞" for t in result["tokens"])
    assert result["topic"] is not None
    # 助動詞の抽出
    aux = [t["surface"] for t in result["tokens"] if t["pos"] == "助動詞"]
    assert "です" in aux


def test_generate_basic():
    engine = SnipherEngine(seed=42)
    out = engine.generate(n=1)
    text = out["text"]
    assert isinstance(text, str)
    assert text.endswith("。")
    assert out["analysis"]["tokens"]


def test_generate_prompt_topic():
    engine = SnipherEngine(seed=42)
    out = engine.generate(prompt="猫について", n=1)
    # 話題として「猫」が選ばれる
    assert out["topic"] == "猫"


def test_generate_registers():
    engine = SnipherEngine(seed=7)
    polite = engine.generate(register="polite", n=1)["text"]
    casual = engine.generate(register="casual", n=1)["text"]
    # 普通体(だ/た)には丁寧マーカーが現れない
    assert "ます" not in casual
    assert "です" not in casual
    # 丁寧体は「です/ます」または丁寧な推量「でしょう」を含む
    assert any(m in polite for m in ("です", "ます", "でしょう"))


def test_generate_many_and_reproducible():
    engine = SnipherEngine(seed=123)
    a = engine.generate(seed=999, n=1)["text"]
    b = engine.generate(seed=999, n=1)["text"]
    assert a == b
    many = engine.generate(n=5)
    assert many["count"] == 5


def test_probability_sample_deterministic():
    engine = SnipherEngine(seed=3)
    model = ProbabilityModel(engine.lexicon, seed=3)
    cands = [{"score": s} for s in (0.0, 1.0, 2.0)]
    picks = {model.sample(cands)["score"] for _ in range(50)}
    assert len(picks) >= 2  # 温度>0 なので複数の候補が出る


def test_info():
    engine = SnipherEngine()
    info = engine.info()
    assert len(info["weights"]) == 6
    # 高速コア（テーブル + 確率重み）は軽量のまま = 10ms 応答の源泉
    bd = info["parameter_breakdown"]
    assert 1400 <= bd["lexicon_table_entries"] < 40_000
    assert bd["probability_weights"] == 6
    # 知識とニューラルコアも総パラメータに計上される（増やした分だけ増える）
    assert info["knowledge_base"]["facts"] > 150
    assert info["total_parameters"] >= bd["lexicon_table_entries"] + bd["knowledge_base_entries"]
    if info["neural_core"]["available"]:
        assert bd["neural_core_weights"] >= 300_000
        assert info["total_parameters"] >= bd["neural_core_weights"]
    else:
        assert bd["neural_core_weights"] == 0     # 未ビルド環境でも破綻しない
    # 巨大 n-gram LM はリポジトリ同梱（流暢さの審判）。エントリ数＝パラメータ数
    lm = info["language_model"]
    if lm["available"]:
        assert bd["language_model_entries"] > 100_000
        assert lm["order"] >= 4 and lm["vocab"] > 500
        assert info["total_parameters"] >= bd["language_model_entries"]
    else:
        assert bd["language_model_entries"] == 0
    # 高速コア（テーブル + 確率重み）は 4 万未満のまま = 10ms 応答の源泉
    assert bd["lexicon_table_entries"] + bd["probability_weights"] < 40_000
