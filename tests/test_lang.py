"""v3 の言葉の層: 音韻（lang.phonetics）・語彙バンク（lang.lex）・活用（lang.morph）。

これらの層は「話題ごとの分岐」を一切持たない。しりとりも辞書照会も訳語も、
同じ音韻・語彙・活用の Query だけで成り立っていることをここで固定する。
"""

from __future__ import annotations

import pytest

from snipher.lang import morph, phonetics as ph
from snipher.lang.lex import bank, english_by_length, english_words, lookup


# --------------------------------------------------------------------------- #
# 音韻
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kana,romaji", [
    ("きょう", "kyou"),
    ("あいす", "aisu"),
    ("ぱん", "pan"),
    ("しりとり", "shiritori"),
    ("しつれい", "shitsurei"),
    ("まつ", "matsu"),
])
def test_kana_to_romaji(kana: str, romaji: str) -> None:
    assert ph.kana_to_ro(kana) == romaji


@pytest.mark.parametrize("romaji,kana", [
    ("kyou", "きょう"),
    ("aisu", "あいす"),
    ("pan", "ぱん"),
    ("shiritori", "しりとり"),
    ("shitsurei", "しつれい"),
])
def test_romaji_to_kana_roundtrip(romaji: str, kana: str) -> None:
    assert ph.ro_to_kana(romaji) == kana
    assert ph.kana_to_ro(kana) == romaji


def test_mora_split_treats_small_kana_as_part_of_mora() -> None:
    assert ph.mora_split("ありがとう") == ["あ", "り", "が", "と", "う"]
    assert ph.mora_split("きょう") == ["きょ", "う"]
    assert ph.mora_count("日本語") == 4
    assert ph.char_count("日本語") == 3


def test_chain_keys_use_last_mora_and_first_mora() -> None:
    assert ph.chain_key("アイス") == "す"
    assert ph.chain_start("アイス") == "あ"
    assert ph.chain_match("キツネ", "ネギ")
    assert not ph.chain_match("りんご", "キツネ")
    # 「ん」で終わる語はしりとりで負けになる
    assert ph.ends_with_n("パン")
    assert not ph.ends_with_n("さんま")


def test_script_detection() -> None:
    assert ph.is_hiragana("ねこ")
    assert ph.is_katakana("ネコ")
    assert ph.is_kana("ネコ") and not ph.is_all_kana("猫")
    assert ph.is_kanji("猫")
    assert ph.kana_ratio("ねこがすき") > 0.9
    assert ph.ascii_ratio("GLM5.3") > 0.5


# --------------------------------------------------------------------------- #
# 語彙バンク
# --------------------------------------------------------------------------- #
def test_wordbank_is_shipped_and_large() -> None:
    st = bank().stats()
    assert st["words"] >= 100_000, "語彙バンクは実辞書からビルド済みであること"
    assert st["game_words"] >= 40_000
    assert "Apache" in st["source"] or "fallback" in st["source"]


def test_lookup_returns_reading_pos_and_morae() -> None:
    ent = lookup("猫")
    assert ent is not None
    assert ent["surface"] == "猫"
    assert ent["reading"] == "ねこ"
    assert ent["pos"].startswith("名詞")
    assert ent["morae"] == 2
    assert ent["romaji"] == "neko"


def test_prefix_and_suffix_queries_over_readings() -> None:
    pre = bank().by_reading_prefix("ねこ", limit=6)
    assert pre and all(w.reading.startswith("ねこ") for w in pre)
    suf = bank().by_reading_suffix("ねこ", limit=6)
    assert suf and all(w.reading.endswith("ねこ") for w in suf)


def test_mora_length_and_containing_queries() -> None:
    two = bank().by_morae(2, limit=10)
    assert two and all(w.morae == 2 for w in two)
    hits = bank().containing("ねこ", limit=5)
    assert hits and all("ねこ" in (w.reading + w.surface) for w in hits)


def test_chain_candidates_continue_the_previous_word() -> None:
    cands = bank().chain_candidates("キツネ", limit=8)
    assert cands, "前の語の送り音から始まる語が引けること"
    for w in cands:
        assert ph.chain_match("キツネ", w.surface), w.surface


def test_segment_tags_content_words() -> None:
    seg = bank().segment("猫が魚を食べた")
    surfaces = [s for s, _ in seg]
    assert "猫" in surfaces and "魚" in surfaces
    pos = dict(seg)
    assert pos["猫"].startswith("名詞")


def test_english_word_list_is_available() -> None:
    assert len(english_words()) > 100_000
    four = english_by_length(4, startswith="word")
    assert "word" in four


# --------------------------------------------------------------------------- #
# 活用
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("surface,form,expect", [
    ("書く", "te", "書いて"),
    ("食べる", "ta", "食べた"),
    ("する", "masu", "します"),
    ("高い", "ta", "高かった"),
    ("高い", "adverb", "高く"),
    ("書く", "mizen", "書か"),
    ("猫", "masu", "猫です"),
])
def test_inflect(surface: str, form: str, expect: str) -> None:
    assert morph.inflect(surface, form) == expect


@pytest.mark.parametrize("surface,lemma", [
    ("食べた", "食べる"),
    ("走り", "走る"),
    ("勉強した", "勉強する"),
])
def test_lemma_of(surface: str, lemma: str) -> None:
    assert morph.lemma_of(surface) == lemma


def test_register_switch_is_reversible() -> None:
    assert morph.to_polite("猫が魚を食べる") == "猫が魚を食べます。"
    assert morph.to_plain("猫が魚を食べます。") == "猫が魚を食べる。"


def test_conjugation_table_is_real_data() -> None:
    tab = morph.table()
    assert len(tab) > 10_000, "辞書から抽出した活用パラダイムを同梱していること"
    assert "書く" in tab
