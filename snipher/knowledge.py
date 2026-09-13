"""知識ベース検索 v2 — 「話題が本当に合っているか」を証拠で判定する検索。

v1 の失敗（`花火とは` に植物の水やりが返る / `好きな食べ物は何？` に寿司の作法が返る）の
原因は 3 つありました。

1. **文字バイグラムだけの BM25** を根拠にしていた（`花` の 1 文字が植物に当たった）
2. **1 文字の alias 一致に大きな加点**をしていた（被覆率 0.0 でも採用された）
3. **問いの型を見ていなかった**（「Xとは」に自己紹介や雑談が返った）

v2 の判定は「証拠の強さ」の順です。

    A. 話題一致   … 発話に話題名そのものが出た（1 文字の話題名も可）   → 最も強い
    B. alias 一致 … 2 文字以上の索引語が出た（汎用語は索引に入れない）  → 強い
    C. 問答一致   … 発話と KB の「よくある問い」の**文字 2-gram** が重なった → 口語に強い
    D. 内容語被覆 … 助詞を除いた内容語のうち、実際に当たった割合        → 中程度
    E. バイグラム … 上記が全部 0 のときの補助（これ単独では採用しない）

そのうえで **問いの型**（とは / なぜ / 作り方 / いつ / どこ / いくら / 好き?）を判定し、
kb.json の対応する欄（def / why / how / when / where / cost / opinion）だけを使います。
材料が無ければ ``None`` を返し、composer が「知らない」と正直に答えます。
"""

from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data" / "kb.json"

_ASCII_WORD = re.compile(r"[A-Za-z0-9_+#.\-]{2,}")
_JP = re.compile(r"[ぁ-んァ-ヶ一-龯ー々〆〇]")
_STRIP = re.compile(r"[\s。、！？!?・…「」『』()（）:：;；〜~\-_]")

# 索引から外す汎用語（これだけでは話題を特定できない）
GENERIC_ALIASES = {
    "好き", "嫌い", "おすすめ", "良い", "悪い", "楽しい", "嬉しい", "悲しい", "疲れた",
    "相談", "話題", "会話", "雑談", "時間", "生活", "健康", "自然", "文化",
    "技術", "社会", "学問", "娯楽", "趣味", "仕事", "勉強", "移動", "家事", "人間関係",
    "動物", "季節", "行事", "感情", "将来", "性格", "記憶",
    "光", "音", "環境", "科学", "経済", "交通", "地理", "歴史", "情報",
    "言葉", "文章", "数字", "色", "形", "場所", "もの", "こと", "人", "家族", "友人",
    "咲く", "見る", "食べる", "飲む", "行く", "来る", "する", "なる", "ある", "いる",
    "何", "なん", "いつ", "どこ", "だれ", "誰", "いくら", "どんな", "どう", "なぜ",
    "教えて", "知りたい", "わかる", "分かる", "助けて", "つらい", "苦しい", "すごい",
    "ありがとう", "ごめん", "こんにちは", "おはよう", "こんばんは",
}

# 発話から証拠として落とす機能語（助詞・助動詞・指示語）
_FUNC_WORDS = {
    "は", "が", "を", "に", "で", "と", "も", "へ", "や", "の", "から", "まで", "より", "だけ",
    "って", "とは", "です", "ます", "でした", "ですか", "でしょうか", "ください", "教えて",
    "知りたい", "わかる", "分かる", "何", "なん", "なぜ", "どうして", "いつ", "どこ", "だれ",
    "誰", "いくら", "どんな", "どういう", "どう", "おすすめ", "好き", "嫌い", "ある", "いる",
    "する", "した", "して", "これ", "それ", "あれ", "この", "その", "あの", "わたし", "私",
    "あなた", "僕", "俺", "ちょっと", "少し", "いろいろ", "何か", "なんか", "ね", "よ", "さ",
    "かな", "わ", "ぞ", "ぜ", "ば", "たら", "の", "だけ", "だけ", "ので", "けど", "し", "て", "だ", "た", "ない",
    "たい", "でしょう", "みたい", "っぽい", "そう", "とても", "すごく", "どれ", "どの",
    "について", "に関して", "に対して", "につい", "について教えてください",
    "教えてください", "教えて", "説明して", "詳しく", "詳しく教えてください",
    "簡単に", "短く", "長く", "ゆっくり", "教えてください", "説明してください",
}

# 未知語の塊から前後を削る機能語（中身は壊さない）
_TRIM_CHARS = "はがをにでもへやのかねよさなわばだたてし"
_TRIM_WORDS = ("でしょう", "ください", "だから", "から", "まで", "より", "だけ", "って", "とは",
               "です", "ます", "ない", "たい", "ので", "けど", "たら", "みたい", "について",
               "ことで", "という", "って")

# 問いの型 → kb.json の欄
QTYPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("why", ("なぜ", "どうして", "なんで", "理由", "しくみ", "仕組み", "どういうわけ")),
    ("how", ("作り方", "やり方", "作りかた", "やりかた", "どうやる", "どうすれば", "どうしたら",
             "方法", "手順", "コツ", "こつ", "ポイント", "編み方", "飼い方", "育て方", "始め方",
             "過ごし方", "使い方", "選び方", "書き方", "撮り方", "歩き方", "飲み方", "食べ方",
             "対策", "どうやっ", "どのように", "覚え方", "仕込み")),
    ("when", ("いつ", "何時", "時期", "何月", "季節", "旬", "タイミング", "どれくらい", "どのくらい",
              "どのぐらい", "何分", "何時間", "何日", "何年", "何歳")),
    ("where", ("どこ", "何処", "場所", "どのあたり", "どこで")),
    ("who", ("だれ", "誰", "どの人", "何人")),
    ("cost", ("いくら", "値段", "価格", "費用", "料金", "コスト", "何円")),
    ("pros_cons", ("メリット", "デメリット", "利点", "短所", "長所", "欠点", "弱点", "強み",
                   "弱み", "欠所", "よさ", "どこが良い", "何が悪い", "何がダメ")),
    ("opinion", ("好き", "嫌い", "おすすめ", "どう思う", "感想", "どんな感じ", "おいしい",
                 "美味しい", "楽しい", "興味", "推し")),
    ("count", ("いくつ", "何個", "何種類", "何匹", "何本")),
    ("def", ("とは", "って何", "とは何", "何ですか", "なんですか", "どういうもの", "どんなもの",
             "何者", "どういう", "意味", "定義", "どんな", "なんという", "何という", "何と言う",
             "なんて言う", "何て言う", "呼び方")),
)

# 困りごとのサイン（定義より tips / qa を優先する）
_REC_MARKS = ("おすすめ", "何がいい", "なにがいい", "いいですか", "いいもの", "どうすれば",
              "教えて", "ほしい", "欲しい", "選び", "何が", "どんなのが")

# 「困りごと・不調・故障」の印。これが有ると定義ではなく対処（tips）を返す
_TROUBLE_MARKS = (
    "ない", "できない", "分からない", "つらい", "辛い", "疲れた", "困る", "困っ", "失敗",
    "遅い", "重い", "痛い", "眠れない", "足りない", "続く", "崩れ", "合わない", "合わない",
    "減らない", "増える", "壊れ", "落ち", "迷う", "迷っ", "心配", "不安", "嫌", "苦手",
    # 症状・状態の変化
    "黄色", "茶色", "黒ず", "白く", "赤く", "枯れ", "しおれ", "萎れ", "乾い", "乾燥",
    "腫れ", "かゆ", "痒", "咳", "熱", "だるい", "めまい", "寒気", "吐き", "痛く",
    "太っ", "痩せ", "やせ", "起きれ", "起きられ", "寝坊", "遅刻",
    # 故障・汚れ・トラブル
    "動かない", "つかない", "点かない", "消え", "切れ", "詰ま", "漏れ", "滲み", "サビ",
    "錆", "汚れ", "臭い", "匂い", "におい", "カビ", "虫", "破れ", "傷", "ひび", "焦げ",
    "生焼け", "べちゃ", "固い", "硬すぎ", "まずい", "散らか", "溜ま", "たまっ", "止まら",
    "error", "バグ", "クラッシュ", "固ま", "再起動", "繋がら", "つながら", "遅延",
)

_MATCH_TOPIC = 12.0
_MATCH_ALIAS = 8.0
_HEAD_BONUS = 5.0          # 「X は/が/の/を」の X は文の主題
_FIRST_BONUS = 3.0         # 発話の先頭に近い語ほど主題らしい
_FINAL_BONUS = 4.0         # 「犬の散歩」の 散歩 のように、文末の名詞が中心


def normalize(text: str) -> str:
    return str(text or "").lower().replace("　", " ")


def _strip_punct(text: str) -> str:
    return _STRIP.sub("", str(text or "").lower())


def question_type(text: str) -> str:
    """発話の「問いの型」を判定する。"""
    t = normalize(text)
    for qtype, pats in QTYPE_PATTERNS:
        for p in pats:
            if p in t:
                return qtype
    if t.rstrip().endswith(("?", "？")):
        return "general_q"
    return "general"


_QUESTIONISH = re.compile(r"(\?|？|ですか|ですかね|ますか|とは|って何|どう|なぜ|いくら|いつ|どこ|何|方法|"
                          r"手順|使い方|つくり方|作り方|メリット|デメリット|違い|比較|おすすめ)")


def bigrams(text: str) -> set[str]:
    """比較用の文字 2-gram 集合（記号を除く）。"""
    t = _strip_punct(text)
    t = re.sub(r"[^\wぁ-んァ-ヶ一-龯ー]", "", t)
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def dice(a: str, b: str) -> float:
    """文字 2-gram の Dice 係数（0..1）。短い口語の一致判定に使う。"""
    ga, gb = bigrams(a), bigrams(b)
    if not ga or not gb:
        return 0.0
    return 2.0 * len(ga & gb) / (len(ga) + len(gb))


def similarity(a: str, b: str) -> float:
    """v1 互換の名前（実体は Dice 係数）。"""
    return dice(a, b)


def char_set(text: str) -> set[str]:
    t = _STRIP.sub("", normalize(text))
    return {c for c in t if _JP.match(c) or c.isalnum()}


# 1 文字の語を「独立した語」と認める境界（助詞・句読点・文字種の変化）
_BOUNDARY = set("はがをにでもへやのかねよさなわばだたてしとっ、。！？!?…・〜ー ")

# 疑問詞・副詞・時の語。ここで一度区切ると、1 文字の話題名（空/本/猫 …）を拾える
_SPLIT_WORDS = (
    "どうして", "なぜ", "なんで", "どんな", "どの", "だれ", "誰", "何", "なに", "いつ",
    "どこ", "どう", "とても", "すごく", "ちょっ", "もっと", "ぜんぜん", "全然", "少し",
    "一番", "最近", "今日", "明日", "昨日", "今朝", "今夜", "今", "ほか", "もし",
    "やっぱり", "やはり", "初めて", "久しぶ", "例えば", "たとえば", "特に", "とくに",
    "について", "って", "とは", "というのは", "という", "って何", "が好き", "が嫌い",
    "の葉", "の調子", "の具合", "が遅", "が痛い", "が出る",
)
_SPLIT_RE = re.compile("(" + "|".join(
    re.escape(w) for w in sorted(_SPLIT_WORDS, key=len, reverse=True)) + ")")


def _is_boundary(chunk: str, i: int, j: int) -> bool:
    """chunk[i:j] の前後が語の境界になっているか。"""
    left_ok = (i == 0) or (chunk[i - 1] in _BOUNDARY) or (not _JP.match(chunk[i - 1]))
    right_ok = (j >= len(chunk)) or (chunk[j] in _BOUNDARY) or (not _JP.match(chunk[j]))
    return left_ok and right_ok


class WordIndex:
    """話題名・alias・語彙テーブルから作る、最長一致の辞書。

    * 話題名は 1 文字でも証拠になる（猫 / 犬 / 雨 / 花 / 本 …）
    * alias は 2 文字以上、かつ汎用語でないものが証拠になる
    * 1 文字の alias は「前後が助詞で区切られている」ときだけ証拠になる
      （`基本です` の `本` を拾わないため）
    * 語彙テーブルの語は「正しく区切るため」だけに使い、証拠にはしない
    """

    def __init__(self, items: list[dict], extra_words: set[str] | None = None,
                 single_words: set[str] | None = None):
        self.topics: dict[str, set[int]] = {}
        self.aliases: dict[str, set[int]] = {}
        tag_candidates: dict[str, set[int]] = {}
        self.evidence: dict[str, set[int]] = {}
        self.short: dict[str, set[int]] = {}      # 1 文字の alias / 話題名
        self.known: set[str] = set()
        self.known1: set[str] = set()             # 1 文字の語（境界条件付きで使う）
        self.max_len = 1
        for i, it in enumerate(items):
            topic = _strip_punct(it.get("topic", "")).lower()
            if topic:
                self.topics.setdefault(topic, set()).add(i)
            for a in it.get("aliases") or []:
                a = _strip_punct(a).lower()
                if a in GENERIC_ALIASES or a == topic:
                    continue
                if len(a) >= 2:
                    self.aliases.setdefault(a, set()).add(i)
                elif len(a) == 1:
                    self.short.setdefault(a, set()).add(i)
                    self.known1.add(a)      # 暇 / 咳 / 星 … 1 文字の alias も区切れるように
            for t in it.get("tags") or []:
                t = _strip_punct(t).lower()
                if len(t) >= 2 and t not in GENERIC_ALIASES:
                    tag_candidates.setdefault(t, set()).add(i)
        # tag は「その語を名乗る話題が少ない」ときだけ証拠にする（食べ物 → 全料理、を防ぐ）
        for t, ids in tag_candidates.items():
            if len(ids) <= 2:
                self.aliases.setdefault(t, set()).update(ids)
        # 1 文字の語（話題名・alias）も証拠として引けるようにする
        for w, ids in self.short.items():
            self.evidence.setdefault(w, set()).update(ids)
            self.known1.add(w)
        # ASCII 語は表記揺れ（Wi-Fi / wifi / WiFi）を吸収するため、
        # 記号を落とした形も同じ証拠として登録する
        for w, ids in list(self.topics.items()) + list(self.aliases.items()):
            variants = {w}
            if w.isascii():
                variants.add(_strip_punct(w))
                variants.add(w.replace("-", "").replace(".", "").replace("/", ""))
            for v in variants:
                if not v:
                    continue
                self.evidence.setdefault(v, set()).update(ids)
                if v in self.topics:
                    self.topics.setdefault(v, set()).update(ids)
                if len(v) >= 2:
                    self.known.add(v)
                    self.max_len = max(self.max_len, len(v))
                else:
                    self.known1.add(v)
                    self.short.setdefault(v, set()).update(ids)
        for w in single_words or ():
            w = _strip_punct(w).lower()
            if len(w) == 1:
                self.known1.add(w)
        for w in extra_words or ():
            w = _strip_punct(w).lower()
            if len(w) >= 2:
                self.known.add(w)
                self.max_len = max(self.max_len, len(w))
            elif len(w) == 1:
                self.known1.add(w)

    def __len__(self) -> int:
        return len(self.evidence) + len(self.short)

    def kind_of(self, word: str, i: int) -> str:
        if word in self.topics and i in self.topics[word]:
            return "topic"
        if word in self.aliases and i in self.aliases[word]:
            return "alias"
        if word in self.short and i in self.short[word]:
            return "topic" if i in self.topics.get(word, ()) else "alias"
        return "tag"

    def shared(self, word: str) -> int:
        """同じ語を名乗る話題の数（多いほど証拠として弱い）。"""
        return max(1, len(self.evidence.get(word, ())) or len(self.short.get(word, ())))

    def topics_of(self, word: str) -> set[int]:
        return self.evidence.get(word, set()) or self.short.get(word, set())

    # ------------------------------------------------------------------ #
    def segment(self, chunk: str) -> list[tuple[str, bool]]:
        """塊を最長一致で切る。→ [(語, 証拠になるか)]。

        まず疑問詞・副詞（なぜ / どうして / どんな / 最近 …）で区切ります。
        これをしないと「なぜ空は青い」の `空` のように、左が助詞でない
        1 文字の話題名を取りこぼします。
        """
        out: list[tuple[str, bool]] = []
        for piece in _SPLIT_RE.split(chunk):
            if not piece:
                continue
            if _SPLIT_WORDS and piece in _SPLIT_WORDS:
                out.append((piece, False))       # 疑問詞などは証拠にしない
                continue
            out.extend(self._segment_raw(piece))
        return out

    def _segment_raw(self, chunk: str) -> list[tuple[str, bool]]:
        """最長一致の本体。未知の塊は前後を削って断片にする。"""
        out: list[tuple[str, bool]] = []
        i, n = 0, len(chunk)
        unknown: list[str] = []

        def flush() -> None:
            if not unknown:
                return
            s = _trim("".join(unknown))
            unknown.clear()
            if len(s) >= 2:
                out.append((s, False))

        while i < n:
            matched = None
            for length in range(min(self.max_len, n - i), 1, -1):
                cand = chunk[i:i + length]
                if cand in self.known:
                    matched = cand
                    break
            if matched is None:
                one = chunk[i:i + 1]
                # 1 文字の語は、前後が助詞などで区切れているときだけ採用する
                if one in self.known1 and _is_boundary(chunk, i, i + 1):
                    matched = one
            if matched:
                flush()
                out.append((matched, bool(self.topics_of(matched))))
                i += len(matched)
            else:
                ch = chunk[i]
                if _JP.match(ch) or ch.isalnum():
                    unknown.append(ch)
                else:
                    flush()
                i += 1
        flush()
        return out


def _trim(s: str) -> str:
    """未知語の塊から、前後の助詞・助動詞だけを削る（中身は壊さない）。"""
    changed = True
    while changed and s:
        changed = False
        for w in _TRIM_WORDS:
            if s.endswith(w) and len(s) > len(w):
                s, changed = s[: -len(w)], True
            if s.startswith(w) and len(s) > len(w):
                s, changed = s[len(w):], True
        while len(s) >= 2 and s[-1] in _TRIM_CHARS:
            s, changed = s[:-1], True
        while len(s) >= 2 and s[0] in _TRIM_CHARS:
            s, changed = s[1:], True
    return s


def lexicon_words() -> tuple[set[str], set[str]]:
    """語彙テーブルの語（区切り用）。→ (2 文字以上の語, 1 文字の名詞)。無ければ空。"""
    out: set[str] = set()
    single: set[str] = set()
    try:
        from .parser import Parser

        out |= {w for w in Parser()._all_surfaces() if len(w) >= 2}
    except Exception:  # noqa: BLE001 - 辞書が無くても検索は動く
        pass
    try:
        base = Path(__file__).resolve().parent / "data" / "nouns.json"
        for row in json.loads(base.read_text(encoding="utf-8")):
            w = str(row.get("s", "")).strip()
            if len(w) == 1:
                single.add(w)
            elif len(w) >= 2:
                out.add(w)
    except Exception:  # noqa: BLE001
        pass
    return out, single


def content_words(text: str, index: WordIndex) -> list[str]:
    """発話から内容語だけを取り出す（辞書の語 + 未知語の断片、重複なし）。"""
    t = normalize(text)
    out: list[str] = []
    for m in _ASCII_WORD.finditer(t):
        w = m.group()
        out.append(w)
        bare = w.replace("-", "").replace(".", "").replace("/", "").replace("_", "")
        if bare and bare != w:
            out.append(bare)          # wi-fi → wifi も候補に入れる
    rest = _ASCII_WORD.sub(" ", t)
    for chunk in re.split(r"[\s。、！？!?・…「」『』()（）:：;；〜~\-_]+", rest):
        if not chunk:
            continue
        for w, _ev in index.segment(chunk):
            out.append(w)
    cleaned: list[str] = []
    for w in out:
        if not w or w in _FUNC_WORDS:
            continue
        if len(w) == 1 and not _JP.match(w):
            continue
        if w not in cleaned:
            cleaned.append(w)
    return cleaned


def tokenize(text: str) -> list[str]:
    """索引用のトークン化（文字バイグラム + ASCII 語）。"""
    t = _strip_punct(text)
    out: list[str] = []
    for m in _ASCII_WORD.finditer(t):
        out.append("w:" + m.group())
    t2 = _ASCII_WORD.sub(" ", t)
    buf: list[str] = []
    for ch in t2:
        if _JP.match(ch):
            buf.append(ch)
        else:
            out.extend(_grams(buf))
            buf = []
    out.extend(_grams(buf))
    return out


def _grams(chars: list[str]) -> list[str]:
    if not chars:
        return []
    if len(chars) == 1:
        return ["g:" + chars[0]]
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


# 被覆率を数えるとき、同じ語の表記ゆれ（Wi-Fi / wifi）や
# 助詞がくっ付いた断片（が遅 / のに）を 1 つにまとめる。
_PARTICLE_FRAG = re.compile(r"^(が|は|を|に|で|と|も|へ|の|や|から|まで|より|だ|た|で)")


def _content_set(words) -> set[str]:
    out: set[str] = set()
    for w in words or []:
        w = str(w)
        if w in _FUNC_WORDS:
            continue
        w = _PARTICLE_FRAG.sub("", w, count=1)
        key = w.lower().replace("-", "").replace(".", "").replace("_", "").replace(" ", "")
        if key:
            out.add(key)
    return out


class KnowledgeBase:
    """kb.json を引く検索エンジン（v2: 内容語 + 問答の一致 + 問いの型）。"""

    _cache: dict[str, "KnowledgeBase"] = {}
    _lock = threading.Lock()

    def __init__(self, path: str | Path | None = None):
        p = Path(path) if path else _DATA
        self.path = p
        self.items: list[dict] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len: list[int] = []
        self.avg_len = 1.0
        self.version = 0
        self.idf: dict[str, float] = {}
        self.common: set[str] = set()
        self.index = WordIndex([])
        self.qa_index: list[tuple[int, str, str]] = []
        self._load(p)

    # ------------------------------------------------------------------ #
    def _load(self, p: Path) -> None:
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        items = raw.get("items", []) if isinstance(raw, dict) else []
        self.version = int(raw.get("version", 0)) if isinstance(raw, dict) else 0
        df: dict[str, set[int]] = {}
        for i, it in enumerate(items):
            blob = " ".join(
                [it.get("topic", ""), it.get("cat", ""), it.get("def", "")]
                + [a * 3 for a in it.get("aliases", [])]
                + it.get("tags", [])
                + it.get("verbs", [])
                + it.get("facts", [])
                + it.get("why", [])
                + it.get("how", [])
                + it.get("tips", [])
                + [q for q, _ in it.get("qa", [])]
            )
            toks = tokenize(blob)
            self.doc_len.append(max(1, len(toks)))
            tf: dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, n in tf.items():
                df.setdefault(tok, set()).add(i)
                self.postings.setdefault(tok, []).append((i, n))
            for pair in it.get("qa") or []:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2 and str(pair[0]).strip():
                    self.qa_index.append((i, str(pair[0]).strip(), str(pair[1]).strip()))
        self.items = items
        N = max(1, len(items))
        self.avg_len = (sum(self.doc_len) / N) if N else 1.0
        self.idf = {t: math.log(1.0 + (N - len(d) + 0.5) / (len(d) + 0.5)) for t, d in df.items()}
        for t in list(self.postings):
            self.postings[t].sort(key=lambda x: -x[1])
        self.common = {t for t, d in df.items() if len(d) / N > 0.34}
        extra, single = lexicon_words()
        self.index = WordIndex(items, extra_words=extra, single_words=single)

    # ------------------------------------------------------------------ #
    @classmethod
    def shared(cls, path: str | Path | None = None) -> "KnowledgeBase":
        key = str(path or _DATA)
        with cls._lock:
            kb = cls._cache.get(key)
            if kb is None:
                kb = cls(path)
                cls._cache[key] = kb
            return kb

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._cache.clear()

    # ------------------------------------------------------------------ #
    def match_words(self, query: str) -> tuple[list[str], dict[int, float], dict[int, list[str]]]:
        """発話の内容語 → 話題ごとの証拠点。"""
        words = content_words(query, self.index)
        score: dict[int, float] = {}
        hits: dict[int, list[str]] = {}
        t = normalize(query)
        heads = {w for w in words if any(w + p in t for p in ("は", "が", "の", "を", "って", "とは", "について"))}
        first = words[0] if words else ""
        # 文末（ copula・終助詞を除く）に立つ名詞が、その文の中心になる
        tail = words[-1] if words else ""
        stem = t.rstrip("。、！？!?…・〜 　")
        for end in ("です", "だ", "である", "ですか", "ですね", "だね", "だよ", "か", "ね", "よ",
                    "なの", "なんだ", "たい", "ます", "ません", "だった", "らしい", "そう"):
            if stem.endswith(end) and len(stem) > len(end):
                stem = stem[: -len(end)]
                break
        final_head = bool(tail) and stem.endswith(tail) and len(tail) >= 1
        for w in words:
            ids = self.index.topics_of(w)
            if not ids:
                continue
            share = self.index.shared(w)
            for i in ids:
                kind = self.index.kind_of(w, i)
                if kind == "topic":
                    s = _MATCH_TOPIC + 1.5 * len(w)
                elif kind == "alias":
                    s = _MATCH_ALIAS + 1.0 * len(w)
                else:
                    s = 1.0 * len(w)
                s /= math.sqrt(share)             # 同じ語を名乗る話題が多いほど弱く
                if w in heads and not (final_head and w != tail and (w + "の") in t):
                    s += _HEAD_BONUS
                if w == first:
                    s += _FIRST_BONUS
                if final_head and w == tail:
                    s += _FINAL_BONUS
                score[i] = score.get(i, 0.0) + s
                hits.setdefault(i, []).append(w)
        return words, score, hits

    def match_qa(self, query: str, min_dice: float = 0.52, min_shared: int = 3,
                 only: set[int] | None = None):
        """KB の「よくある問い」と発話が重なれば、その問答を返す（口語に強い）。"""
        q = _strip_punct(query)
        # 「暇」「つらい」のような一言発話は、同じ形の問いがあればそれを採用する
        if len(char_set(q)) < 3:
            for i, question, answer in self.qa_index:
                if only is not None and i not in only:
                    continue
                if _strip_punct(question) == q:
                    return (i, question, answer, 1.0)
            return None
        qg = bigrams(q)
        best = None
        for i, question, answer in self.qa_index:
            if only is not None and i not in only:
                continue
            if len(question) < 3:
                continue
            qg2 = bigrams(question)
            shared = len(qg & qg2)
            if shared < min_shared:
                continue
            d = 2.0 * shared / (len(qg) + len(qg2))
            if d >= min_dice and (best is None or d > best[3]):
                best = (i, question, answer, round(d, 4))
        return best

    def exact_topic(self, query: str) -> dict | None:
        """発話全体が 1 つの話題名と一致する場合（「暇」「猫」のような一言発話）。"""
        raw = str(query or "")
        q = _strip_punct(raw)
        q = re.sub(r"(とは|って|は|が|を|の|ですか|でしょうか|について|のこと|何|なん)+$", "", q)
        if not q:
            return None
        # 純粋な数字の一言 (「67」) は年齢・数量の可能性が高く、話題名への
        # 即断はしない。定義マーカー (とは/って何/意味) がある場合のみ通す。
        if re.fullmatch(r"[0-9０-９\s.,．，]+", q):
            if not re.search(r"(とは|って何|意味|定義|何|なん)", raw):
                return None
        for i in self.index.topics_of(q):
            if self.index.kind_of(q, i) in ("topic", "alias", "tag"):
                return self.items[i]
        for i in self.index.short.get(q, ()):      # 「暇」のような 1 文字の一言発話
            return self.items[i]
        return None

    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 3, k1: float = 1.4, b: float = 0.72) -> list[dict]:
        """話題を証拠の強さ順に返す。score は 0..1、coverage は内容語の被覆率。"""
        if not self.items:
            return []
        words, word_score, word_hits = self.match_words(query)
        bm25: dict[int, float] = {}
        matched: dict[int, set[str]] = {}
        toks = tokenize(query)
        qfreq: dict[str, int] = {}
        for tok in toks:
            qfreq[tok] = qfreq.get(tok, 0) + 1
        for tok in qfreq:
            post = self.postings.get(tok)
            if not post:
                continue
            idf = self.idf.get(tok, 0.0)
            if idf <= 0.05:
                continue
            for i, tf in post:
                denom = tf + k1 * (1 - b + b * self.doc_len[i] / max(1e-6, self.avg_len))
                bm25[i] = bm25.get(i, 0.0) + idf * (tf * (k1 + 1)) / denom
                matched.setdefault(i, set()).add(tok)

        total: dict[int, float] = dict(word_score)
        mx_bm = max(bm25.values()) if bm25 else 1.0
        for i, s in bm25.items():
            # バイグラムの相対スコアは「内容語が当たった話題の補強」にしか使わない
            w = 0.30 if word_score else 1.0
            total[i] = total.get(i, 0.0) + w * 6.0 * (s / max(1e-6, mx_bm))
        if not total:
            return []
        mx = max(total.values()) or 1.0
        # 被覆率の分母 = 発話の内容語（辞書の語 + 未知語の断片）。機能語は数えない。
        # 表記ゆれ（Wi-Fi / wifi）と助詞の断片（が遅）は 1 つにまとめる。
        content = _content_set(words)
        ranked = sorted(total.items(), key=lambda kv: -kv[1])[: max(top_k, 1) * 3]
        out: list[dict] = []
        for i, sc in ranked:
            hit_words = set(word_hits.get(i, []))
            covered = len(content & _content_set(hit_words)) if content else 0
            cov = covered / len(content) if content else 0.0
            kinds = {self.index.kind_of(w, i) for w in hit_words}
            out.append({
                "item": self.items[i],
                "index": i,
                "score": round(sc / mx, 4),
                "raw": round(sc, 4),
                "coverage": round(cov, 4),
                "covered_n": int(covered),
                "word_hits": sorted(hit_words)[:8],
                "matched": sorted(matched.get(i, set()))[:12],
                "topic_hit": bool(kinds & {"topic", "alias", "tag"}),
                "kind": ("topic" if "topic" in kinds else "alias" if "alias" in kinds
                         else "word" if kinds else "gram"),
            })
        out.sort(key=lambda h: -h["raw"])
        return out[:top_k]

    # ------------------------------------------------------------------ #
    def _pick_field(self, item: dict, qtype: str, query: str,
                    qa_hit: tuple[str, str, float] | None = None) -> tuple[str, str]:
        """問いの型に合う欄を選ぶ。→ (text, 使った欄)"""
        def first(vals) -> str:
            if isinstance(vals, str):
                return vals.strip()
            for v in vals or []:
                v = str(v).strip()
                if v:
                    return v
            return ""

        trouble = any(m in query for m in _TROUBLE_MARKS)
        # 発話とほぼ同じ「よくある問い」が KB にあれば、それが最も的確な答え
        if qa_hit is not None and qa_hit[2] >= 0.62:
            return qa_hit[1], "qa"
        # 自由応答と困りごとでは、定義より「よくある問い」の答えを優先する
        if qa_hit is not None and (qtype in ("general", "general_q") or trouble):
            return qa_hit[1], "qa"
        if qtype == "def":
            t = first(item.get("def"))
            if t:
                return t, "def"
            if qa_hit is not None:
                return qa_hit[1], "qa"
            return first(item.get("facts")), "fact"
        if qtype == "pros_cons":
            marks = ("メリット", "デメリット", "利点", "短所", "長所", "欠点", "弱点", "強み", "弱み",
                     "速い", "遅い", "向い", "向か", "苦手")
            pairs = item.get("qa") or []
            best = ""
            for q, a in (list(pairs) if isinstance(pairs, list) else []):
                try:
                    qq, aa = str(q), str(a)
                except Exception:  # noqa: BLE001
                    continue
                if any(m in qq for m in marks) and any(m in query for m in marks):
                    best = aa.strip()
                    break
            if best:
                return best, "qa"
            t2 = first(item.get("tips"))
            if t2:
                return t2, "tips"
            t3 = first(item.get("opinion"))
            if t3:
                return t3, "opinion"
            t4 = first(item.get("def"))
            return (t4, "def") if t4 else (first(item.get("facts")), "fact")
        if qtype == "why":
            if item.get("why"):
                return first(item.get("why")), "why"
            if qa_hit is not None and qa_hit[2] >= 0.45:
                return qa_hit[1], "qa"
            t = first(item.get("def"))
            return (t, "def") if t else (first(item.get("facts")), "fact")
        if qtype == "how":
            steps = [str(s).strip() for s in (item.get("how") or []) if str(s).strip()]
            if steps:
                return _join_steps(steps), "how"
            t = first(item.get("tips"))
            if t:
                return t, "tips"
        if qtype in ("when", "where", "who", "cost"):
            t = first(item.get(qtype))
            if t and (qa_hit is None or qa_hit[2] < 0.62):
                return t, qtype
            if qa_hit is not None:
                return qa_hit[1], "qa"
            if t:
                return t, qtype
        if qtype == "opinion":
            if qa_hit is not None and qa_hit[2] >= 0.5:
                return qa_hit[1], "qa"
            t = first(item.get("opinion"))
            if t:
                return t, "opinion"
        if qtype == "count":
            t = first(item.get("cost")) or first(item.get("when"))
            if t:
                return t, "typed"
        if trouble:
            t = first(item.get("tips")) or first(item.get("why"))
            if t:
                return t, "tips"
        if any(m in query for m in _REC_MARKS):
            # 「おすすめは?」「何がいい?」には定義でなく、好みやコツを返す
            t = first(item.get("opinion")) or first(item.get("tips"))
            if t:
                return t, "opinion" if item.get("opinion") else "tips"
        if qtype == "how":
            t = first(item.get("tips"))
            if t:
                return t, "tips"
        if qa_hit is not None:
            return qa_hit[1], "qa"
        t = first(item.get("def"))
        if t:
            return t, "def"
        t = first(item.get("facts"))
        return (t, "fact") if t else ("", "none")

    def answer(self, query: str, min_score: float = 0.30, min_cov: float = 0.34) -> dict | None:
        """発話に合う材料があれば返し、無ければ None（= 知らない）。"""
        if not self.items:
            return None
        # 純粋な数字・記号だけ (「67」「あ」) は話題特定しない。
        # 定義マーカー (とは/って何/意味) がある場合のみ、数字ミーム等へ通す。
        _stripped = re.sub(r"[\s。、！？!?・…「」『』()（）:：;；〜~\-_]", "", str(query or ""))
        if _stripped and re.fullmatch(r"[0-9０-９.,．，]+", _stripped):
            if not re.search(r"(とは|って何|意味|定義)", str(query or "")):
                return None
        # 複合質問の短い常識は、単語 BM25 だけに任せると「夜」か「挨拶」の
        # どちらか一方へ寄る。両方の語が揃った場合は意味を確定させる。
        normalized_query = normalize(query)
        if "夜" in normalized_query and any(x in normalized_query for x in ("挨拶", "あいさつ")):
            greeting = next((it for it in self.items if str(it.get("topic")) == "挨拶"), None)
            if greeting is not None:
                chosen = {"item": greeting, "index": self.items.index(greeting), "score": 1.0,
                          "coverage": 1.0, "covered_n": 2, "word_hits": ["夜", "挨拶"],
                          "topic_hit": True, "via": "semantic_shortcut"}
                return self._material(
                    chosen, question_type(query), query,
                    ("夜の挨拶", "夜の挨拶は「こんばんは」です。", 1.0),
                )
        words, word_score, word_hits = self.match_words(query)
        # 被覆率は *話題の内容語* だけで数える。「について」「詳しく」のような
        # 助詞句・副詞・依頼語が残ると、被覆率が下がり正しい話題が落ちる
        # （「日本について詳しく教えてください」→ 日本 だけが残ることが重要）。
        content = {w for w in words
                   if w not in _FUNC_WORDS
                   and not w.endswith("て")
                   and not w.endswith(("ください", "ましょう", "ます", "です"))
                   and not (len(w) >= 3 and w.endswith("く"))}
        # 問いの *形* を決める語（値段・いくら・いつ・どこ）は話題の内容ではなく
        # 質問の型です（「電球 平均値段」→ 話題は 電球 だけ）。
        _Q_SHAPE = {"値段", "価格", "いくら", "いくらか", "費用", "料金", "コスト", "何円",
                    "いつ", "どこ", "誰", "だれ", "どこで", "理由", "仕組み", "作り方", "やり方"}
        content_narrow = {w for w in content if w not in _Q_SHAPE}
        if len(content_narrow) < len(content):
            content = content_narrow
        qtype = question_type(query)
        hits = self.search(query, top_k=4)

        # --- 1) 発話全体が 1 話題に一致 --------------------------------- #
        exact = self.exact_topic(query)
        if exact is not None:
            chosen = {"item": exact, "index": self.items.index(exact), "score": 1.0, "coverage": 1.0,
                      "covered_n": len(content) or 1, "word_hits": sorted(content)[:8],
                      "topic_hit": True, "via": "exact"}
            qa_hit = _qa3(self.match_qa(query, only={chosen["index"]}))
            return self._material(chosen, qtype, query, qa_hit)

        # --- 1.5) 発話が KB の「よくある問い」とほぼ同じ形 ---------------- #
        #   「気分が悪い」のように、別の話題の alias が引っ張っても、
        #   問いそのものが KB にあればそちらが正解です。
        qa0 = self.match_qa(query)
        if qa0 is not None and float(qa0[3]) >= 0.72:
            i0, question0, answer0, sim0 = qa0
            chosen = {"item": self.items[i0], "index": i0, "score": round(0.6 + 0.4 * sim0, 4),
                      "coverage": round(sim0, 4), "covered_n": max(1, len(content)),
                      "word_hits": sorted(content)[:8], "matched": sorted(content)[:12],
                      "topic_hit": True, "kind": "alias", "via": "qa_strong"}
            return self._material(chosen, qtype, query, (question0, answer0, sim0))

        # --- 2) 話題名 / alias が実際に発話に出た ------------------------ #
        strong_hit = None
        for h in hits:
            if h.get("topic_hit") and (float(h["score"]) >= min_score):
                cov = float(h.get("coverage", 0.0))
                covered_n = int(h.get("covered_n", 0))
                item0 = h.get("item") or {}
                # 話題名が実際に発話に出ていて、かつ発話の内容語の多くがその話題のもの
                # であること。未知語（ゾルタクス等）が混ざる発話を別話題の定義で
                # 答えてしまわないよう、内容語が多いときは被覆率を必ず見ます。
                short = len(content) <= 2 and covered_n >= 1 and float(h["score"]) >= 0.6
                if covered_n >= 1 and (cov >= min_cov or short):
                    strong_hit = h
                    break
        if strong_hit is None:
            for h in hits:
                if int(h.get("covered_n", 0)) >= 2 and float(h.get("coverage", 0.0)) >= min_cov:
                    strong_hit = h
                    break
        if strong_hit is None and _QUESTIONISH.search(normalized_query):
            # 質問の形で *話題名そのもの* を踏んでいるなら、被覆率が低くてもその話題の質問です
            # （「WebAssembly をブラウザで動かすメリット」は WebAssembly の話）。長い名前を優先するので、
            # 「観葉植物」と「植物」が競合ときは詳しい側を選びます。
            best: tuple[int, dict] | None = None
            for h in hits:
                item0 = h.get("item") or {}
                name = str(item0.get("topic") or "")
                if len(name) < 3 or not h.get("topic_hit"):
                    continue
                if float(h.get("score") or 0.0) < 0.5:
                    continue
                if name.lower() not in normalized_query.lower():
                    continue
                if best is None or len(name) > best[0]:
                    best = (len(name), h)
            if best is not None:
                strong_hit = best[1]
        if strong_hit is None:
            # 名詞句の問い: 4 文字以上の *具体的な索引語*（話題名・alias）が発話の
            # 末尾に立っていれば、疑問詞が無くてもその話題の質問です
            # （「世界で一番小さい言語モデル」→ 末尾の「言語モデル」）。
            # 「世界」「小さい」のような修飾語が被覆率を薄めても、末尾の語が話題を決める。
            t2 = normalized_query.rstrip("。、！？!?…・〜 　")
            if 4 <= len(t2) <= 28:
                tail2 = words[-1] if words else ""
                if tail2 and len(tail2) >= 4:
                    for h in hits:
                        if h.get("topic_hit") and tail2 in (h.get("word_hits") or []):
                            strong_hit = h
                            break
        if strong_hit is not None:
            chosen = dict(strong_hit)
            chosen["via"] = "word"
            qa_hit = _qa3(self.match_qa(query, only={chosen["index"]}))
            return self._material(chosen, qtype, query, qa_hit)

        # --- 3) 話題名は無いが「よくある問い」とほぼ一致 ------------------ #
        qa = self.match_qa(query)
        if qa is not None:
            i, question, answer, sim = qa
            chosen = {"item": self.items[i], "index": i, "score": round(0.55 + 0.45 * sim, 4),
                      "coverage": round(sim, 4), "covered_n": max(1, len(content)),
                      "word_hits": sorted(content)[:8], "topic_hit": True, "via": f"qa:{question}"}
            return self._material(chosen, qtype, query, (question, answer, sim))

        return None

    def _material(self, chosen: dict, qtype: str, query: str,
                  qa_pair: tuple[str, str, float] | None) -> dict | None:
        item = chosen["item"]
        text, used = self._pick_field(item, qtype, query, qa_pair)
        if not text:
            return None
        facts = [str(x).strip() for x in (item.get("facts") or []) if str(x).strip()]
        # 短すぎる答えには、同じ話題の事実を 1 文だけ足す（別の話題の文は混ぜない）
        if used in ("def", "fact", "opinion") and len(text) < 40 and facts:
            extra = next((f for f in facts if f != text), "")
            if extra:
                text = f"{text}{extra}"
        follow = ""
        fus = [str(x).strip() for x in (item.get("followups") or []) if str(x).strip()]
        if fus:
            follow = fus[0] if len(fus) == 1 else fus[len(char_set(query)) % len(fus)]
        cov = float(chosen.get("coverage", 0.0))
        conf = min(0.95, 0.44 + 0.26 * float(chosen["score"]) + 0.28 * min(1.0, cov * 1.5))
        if used in ("def", "how", "why", "qa", "opinion"):
            conf = min(0.97, conf + 0.05)
        if chosen.get("via") == "exact":
            conf = max(conf, 0.93)
        return {
            "text": text,
            "confidence": round(float(conf), 4),
            "topic": item.get("topic"),
            "id": item.get("id"),
            "cat": item.get("cat"),
            "score": chosen["score"],
            "coverage": cov,
            "covered_n": int(chosen.get("covered_n", 0)),
            "word_hits": chosen.get("word_hits", []),
            "via": chosen.get("via"),
            "qtype": qtype,
            "field": used,
            "fact": facts[0] if facts else None,
            "facts": facts,
            "def": item.get("def", ""),
            "opinion": item.get("opinion", ""),
            "how": item.get("how", []),
            "why": item.get("why", []),
            "tips": item.get("tips", []),
            "followup": follow,
            "asked": qa_pair[0] if qa_pair else None,
            "usage": used,
            "related": item.get("related") or [],
            "matched_topic": bool(chosen.get("topic_hit")),
            "item": item,
        }

    # v1 互換
    def material(self, query: str) -> dict | None:
        return self.answer(query)

    def suggest(self, query: str, top_k: int = 3) -> list[str]:
        """採用には至らなかったが近そうな話題名（「知らない」応答の言い換えに使う）。"""
        return [h["item"].get("topic", "") for h in self.search(query, top_k=top_k)]

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        return {
            "topics": len(self.items),
            "facts": sum(len(i.get("facts", [])) for i in self.items),
            "questions": sum(len(i.get("questions", [])) for i in self.items),
            "answers": sum(len(i.get("answers", [])) for i in self.items),
            "qa": len(self.qa_index),
            "how": sum(len(i.get("how", [])) for i in self.items),
            "why": sum(len(i.get("why", [])) for i in self.items),
            "opinions": sum(1 for i in self.items if i.get("opinion")),
            "terms": len(self.postings),
            "words": len(self.index),
            "version": self.version,
        }

    def all_sentences(self) -> list[str]:
        """KB に書かれた全ての文（n-gram LM とニューラルコアの教師データにする）。"""
        out: list[str] = []
        for it in self.items:
            if it.get("def"):
                out.append(str(it["def"]).strip())
            for field in ("facts", "why", "tips", "followups"):
                for s in it.get(field) or []:
                    s = str(s).strip()
                    if s:
                        out.append(s if s.endswith(("。", "！", "？")) else s + "。")
            for s in it.get("how") or []:
                s = str(s).strip()
                if s:
                    out.append(s + "。")
            for k in ("when", "where", "who", "cost", "opinion"):
                v = str(it.get(k) or "").strip()
                if v:
                    out.append(v)
            for pair in it.get("qa") or []:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    q, a = str(pair[0]).strip(), str(pair[1]).strip()
                    if q:
                        out.append(q if q.endswith(("。", "？", "！")) else q + "。")
                    if a:
                        out.append(a)
        return [s for s in out if s]


def _join_steps(steps: list[str], max_steps: int = 4) -> str:
    """手順のリストを 1 文の自然な日本語にまとめる。"""
    steps = [s.rstrip("。") for s in steps if s][:max_steps]
    if not steps:
        return ""
    if len(steps) == 1:
        return steps[0] + "。"
    head, tail = steps[0], steps[-1]
    body = steps[1:-1]
    return "、".join([head] + body + [f"最後に{tail}"]) + "。"


def _qa3(hit) -> tuple[str, str, float] | None:
    """match_qa の戻り値 (i, q, a, dice) を (q, a, dice) に揃える。"""
    if hit is None:
        return None
    if len(hit) == 4:
        return (hit[1], hit[2], hit[3])
    return (hit[0], hit[1], hit[2])
