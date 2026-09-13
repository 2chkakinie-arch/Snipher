"""翻訳（英訳・和訳）— 材料の語を、指示された言語へ写し替える。

方針は 3 つだけです。

1. 材料に無い事実を足さない（語彙の対応と文型の組み替えだけで訳を作る）。
2. 対応表に無い語は *読み*（ローマ字）で残し、`notes` に記録する（黙って落とさない）。
3. 出力は訳文だけ（解説を混ぜない）。翻訳の仕事の成果は訳文です。

仕組みは「文型 → 語彙」の 2 段です。文を節に割り、節ごとに文型（〜は〜です /
〜ので / 〜に〜に行きました …）を当てて英語の語順に並べ替え、中の語を対応表で写します。
対応表に無い語は語彙バンクの読みで残すので、材料の語を落とさずに訳せます。
"""

from __future__ import annotations

import re

from ..lang import lex

# --------------------------------------------------------------------------- #
# 語彙対応（日→英）
# --------------------------------------------------------------------------- #
_TIME: dict[str, str] = {
    "今日": "today", "本日": "today", "明日": "tomorrow", "明後日": "the day after tomorrow",
    "昨日": "yesterday", "一昨日": "the day before yesterday", "今": "now", "今朝": "this morning",
    "今夜": "tonight", "今週": "this week", "来週": "next week", "先週": "last week",
    "今月": "this month", "来月": "next month", "先月": "last month", "今年": "this year",
    "来年": "next year", "去年": "last year", "午前": "in the morning", "午後": "in the afternoon",
    "朝": "in the morning", "昼": "at noon", "夕方": "in the evening", "夜": "at night",
    "週末": "on the weekend", "毎日": "every day", "毎朝": "every morning",
    "毎晩": "every evening", "毎週": "every week", "毎月": "every month", "毎年": "every year",
    "いつも": "always", "時々": "sometimes",
    "よく": "often", "あまり": "not much", "とても": "very", "少し": "a little",
    "もう": "already", "まだ": "yet", "すぐ": "soon", "初めて": "for the first time",
    "約": "about", "およそ": "about", "大体": "roughly", "だいたい": "roughly",
    "ちょうど": "exactly", "ほぼ": "almost", "毎朝": "every morning", "毎晩": "every evening",
    "先週": "last week", "先日": "the other day", "昨晩": "last night",
}

#: 数 + 助数詞（2時間 / 3日 / 120円）→ 英語の数量
_COUNTER = {
    "時間": "hours", "分": "minutes", "秒": "seconds", "日": "days", "週間": "weeks",
    "週": "weeks", "か月": "months", "ヶ月": "months", "月": "months", "年": "years",
    "円": "yen", "台": "units", "個": "pieces", "件": "items", "枚": "sheets",
    "本": "bottles", "人": "people", "回": "times", "歳": "years old", "キロ": "kilometers",
    "メートル": "meters", "km": "kilometers", "分間": "minutes", "時間半": "and a half hours",
}
_NUM_COUNTER = re.compile(r"([0-9０-９]+)\s*(" + "|".join(
    sorted((re.escape(k) for k in _COUNTER), key=len, reverse=True)) + r")")

_NOUNS: dict[str, str] = {
    "天気": "the weather", "雨": "rain", "雪": "snow", "風": "the wind", "空": "the sky",
    "台風": "a typhoon", "気温": "the temperature", "湿度": "the humidity",
    "公園": "the park", "庭": "the garden", "山": "the mountain", "川": "the river",
    "海": "the sea", "湖": "the lake", "森": "the forest", "駅": "the station",
    "空港": "the airport", "学校": "school", "大学": "the university", "会社": "the office",
    "病院": "the hospital", "図書館": "the library", "店": "the shop", "お店": "the shop",
    "コンビニ": "the convenience store", "スーパー": "the supermarket",
    "レストラン": "the restaurant", "カフェ": "the cafe", "家": "home", "部屋": "the room",
    "台所": "the kitchen", "玄関": "the entrance", "映画館": "the cinema",
    "美術館": "the art museum", "博物館": "the museum", "神社": "the shrine", "寺": "the temple",
    "都市": "a city", "街": "the town", "村": "the village", "国": "the country",
    "東京": "Tokyo", "大阪": "Osaka", "京都": "Kyoto", "日本": "Japan",
    "アメリカ": "America", "イギリス": "Britain", "中国": "China", "韓国": "Korea",
    "散歩": "a walk", "買い物": "shopping", "食事": "a meal", "昼食": "lunch",
    "朝食": "breakfast", "夕食": "dinner", "勉強": "studying", "仕事": "work",
    "会議": "a meeting", "旅行": "a trip", "出張": "a business trip", "運動": "exercise",
    "水": "water", "お茶": "tea", "緑茶": "green tea", "コーヒー": "coffee",
    "紅茶": "black tea", "牛乳": "milk", "ビール": "beer", "ご飯": "rice", "ごはん": "rice",
    "パン": "bread", "肉": "meat", "魚": "fish", "野菜": "vegetables", "果物": "fruit",
    "本": "a book", "新聞": "the newspaper", "雑誌": "a magazine", "手紙": "a letter",
    "メール": "an email", "電話": "a phone call", "電車": "the train", "新幹線": "the bullet train",
    "バス": "the bus", "車": "the car", "自転車": "a bicycle", "飛行機": "the plane",
    "映画": "a movie", "音楽": "music", "写真": "a photo", "料理": "cooking",
    "スポーツ": "sports", "犬": "a dog", "猫": "a cat", "鳥": "a bird", "花": "a flower",
    "桜": "cherry blossoms", "木": "a tree", "日本語": "Japanese", "英語": "English",
    "言葉": "words", "意味": "the meaning", "問題": "a problem", "答え": "the answer",
    "質問": "a question", "計画": "a plan", "予定": "the plan", "予約": "a reservation",
    "席": "a seat", "料金": "the fare", "値段": "the price", "お金": "money", "円": "yen",
    "人": "a person", "名前": "the name", "時間": "time", "場所": "the place",
    "物": "a thing", "もの": "a thing", "こと": "things", "友達": "a friend",
    "友人": "a friend", "家族": "family", "子供": "a child", "先生": "the teacher",
    "学生": "a student", "医者": "a doctor", "母": "my mother", "父": "my father",
}

_PRONOUN: dict[str, str] = {
    "私": "I", "わたし": "I", "僕": "I", "ぼく": "I", "俺": "I", "あなた": "you",
    "彼": "he", "彼女": "she", "私たち": "we", "わたしたち": "we", "彼ら": "they",
    "みんな": "everyone", "兄": "my older brother", "姉": "my older sister",
    "弟": "my younger brother", "妹": "my younger sister",
}

_ADJ: dict[str, str] = {
    "良い": "good", "よい": "good", "いい": "good", "悪い": "bad", "暑い": "hot",
    "寒い": "cold", "暖かい": "warm", "涼しい": "cool", "高い": "high", "安い": "cheap",
    "大きい": "big", "小さな": "small", "小さい": "small", "新しい": "new", "古い": "old",
    "忙しい": "busy", "暇": "free", "楽しい": "fun", "面白い": "interesting",
    "難しい": "difficult", "易しい": "easy", "簡単": "easy", "美しい": "beautiful",
    "綺麗": "beautiful", "おいしい": "delicious", "美味しい": "delicious",
    "近い": "close", "遠い": "far", "早い": "early", "遅い": "late", "長い": "long",
    "短い": "short", "強い": "strong", "弱い": "weak", "明るい": "bright", "暗い": "dark",
    "静か": "quiet", "賑やか": "lively", "元気": "energetic", "好き": "like",
    "嫌い": "dislike", "大切": "important", "必要": "necessary", "大丈夫": "all right",
    "素敵": "wonderful", "大変": "hard", "同じ": "the same", "違う": "different",
    "多い": "many", "少ない": "few",
}

# 動詞: 辞書形 → (現在, 過去)
_VERBS: dict[str, tuple[str, str]] = {
    "行く": ("go", "went"), "来る": ("come", "came"), "帰る": ("return", "returned"),
    "出る": ("leave", "left"), "入る": ("enter", "entered"), "着く": ("arrive", "arrived"),
    "会う": ("meet", "met"), "食べる": ("eat", "ate"), "飲む": ("drink", "drank"),
    "見る": ("see", "saw"), "聞く": ("listen to", "listened to"), "話す": ("talk", "talked"),
    "言う": ("say", "said"), "読む": ("read", "read"), "書く": ("write", "wrote"),
    "買う": ("buy", "bought"), "売る": ("sell", "sold"), "作る": ("make", "made"),
    "する": ("do", "did"), "遊ぶ": ("play", "played"), "働く": ("work", "worked"),
    "歩く": ("walk", "walked"), "走る": ("run", "ran"), "泳ぐ": ("swim", "swam"),
    "待つ": ("wait", "waited"), "起きる": ("get up", "got up"), "寝る": ("sleep", "slept"),
    "休む": ("rest", "rested"), "始める": ("start", "started"), "終わる": ("finish", "finished"),
    "教える": ("teach", "taught"), "習う": ("learn", "learned"), "思う": ("think", "thought"),
    "考える": ("think about", "thought about"), "持つ": ("have", "had"),
    "できる": ("can do", "could do"), "知る": ("know", "knew"),
    "分かる": ("understand", "understood"), "使う": ("use", "used"),
    "乗る": ("ride", "rode"), "降りる": ("get off", "got off"), "撮る": ("take", "took"),
    "歌う": ("sing", "sang"), "踊る": ("dance", "danced"), "手伝う": ("help", "helped"),
    "調べる": ("look up", "looked up"), "予約する": ("reserve", "reserved"),
    "かかる": ("take", "took"), "訪れる": ("visit", "visited"), "過ごす": ("spend", "spent"),
    "楽しむ": ("enjoy", "enjoyed"), "泊まる": ("stay", "stayed"), "出発する": ("depart", "departed"),
    "到着する": ("arrive", "arrived"), "戻る": ("return", "returned"), "向かう": ("head", "headed"),
    "過ごす": ("spend", "spent"), "見物する": ("sightsee", "went sightseeing"),
    "食べる": ("eat", "ate"), "注文する": ("order", "ordered"),
    "勉強する": ("study", "studied"), "散歩する": ("take a walk", "took a walk"),
    "買い物する": ("go shopping", "went shopping"),
}

_SA_SURU = ("勉強", "散歩", "買い物", "予約", "運動", "仕事", "食事", "料理", "掃除", "洗濯",
            "出発", "到着", "注文", "見物", "観光", "連絡", "準備", "説明")

_PARTICLES: dict[str, str] = {
    "は": "", "が": "", "を": "", "に": "to", "へ": "to", "で": "at", "と": "with",
    "から": "from", "まで": "until", "も": "also", "の": "of", "や": "and",
    "など": "and so on", "より": "than",
}

_CONJ: dict[str, str] = {
    "ので": "so", "しかし": "however", "でも": "but", "そして": "and", "それから": "then",
    "また": "also", "もし": "if", "とき": "when", "前に": "before", "後に": "after",
    "ために": "for", "のに": "although",
}

_JA_WORDS: dict[str, str] = {**_TIME, **_PRONOUN, **_NOUNS, **_ADJ, **_CONJ}
for _k, (_a, _b) in _VERBS.items():
    _JA_WORDS.setdefault(_k, _a)

# 英→日（上の表を反転 + 文型で使う機能語）
_EN2JA: dict[str, str] = {}
for _ja, _en in {**_TIME, **_PRONOUN, **_NOUNS, **_ADJ, **_CONJ}.items():
    _EN2JA.setdefault(_en.lower().strip(" .,"), _ja)
for _ja, (_now, _past) in _VERBS.items():
    _EN2JA.setdefault(_past.lower(), _ja)
    _EN2JA.setdefault(_now.lower(), _ja)
_EN2JA.update({
    "is": "です", "was": "でした", "are": "です", "were": "でした", "am": "です",
    "the": "", "a": "", "an": "", "to": "に", "at": "で", "in": "に", "on": "に",
    "for": "に", "because": "ので", "so": "ので", "and": "そして", "but": "しかし",
    "went": "行きました", "go": "行きます", "walk": "散歩", "park": "公園",
    "weather": "天気", "very": "とても", "not": "ない", "with": "と", "of": "の",
    "today": "今日", "yesterday": "昨日", "tomorrow": "明日", "good": "良い",
    "i": "私", "you": "あなた", "movie": "映画", "friend": "友達",
})

_SENT_SPLIT = re.compile(r"(?<=[。！？!?])\s*")
_CLAUSE_SPLIT = re.compile(r"[、,]\s*")
_JP_CHAR = re.compile(r"[ぁ-んァ-ヶー一-龯]")


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #
_REQUEST = re.compile(r"翻訳|英訳|和訳|訳して|訳し|に訳す|translate|translation|"
                      r"into (?:english|japanese)|in (?:english|japanese)", re.IGNORECASE)


def is_translation_request(text: str) -> bool:
    """指示が翻訳の仕事かどうか（構造だけで見る: 訳すという語があるか）。"""
    return bool(_REQUEST.search(str(text or "")))


def detect_target(text: str, *, source: str = "") -> str:
    """訳先の言語（`"en"` / `"ja"`）。指示に書いてあればそれ、無ければ材料の言語の逆にします。"""
    t = str(text or "")
    if re.search(r"英訳|英語に|英語へ|into english|in english|to english", t, re.IGNORECASE):
        return "en"
    if re.search(r"和訳|日本語に|日本語へ|into japanese|in japanese|to japanese", t, re.IGNORECASE):
        return "ja"
    src = str(source or "")
    if src and not _JP_CHAR.search(src):
        return "ja"                      # 材料が英語 → 和訳
    return "en"

# --------------------------------------------------------------------------- #
# 分かち書き（語 → 役割）
# --------------------------------------------------------------------------- #
_TOK: dict[str, str] = {}
for _w in _TIME:
    _TOK[_w] = "time"
for _w in _PRONOUN:
    _TOK[_w] = "pron"
for _w in _NOUNS:
    _TOK[_w] = "noun"
for _w in _ADJ:
    _TOK[_w] = "adj"
for _w in _VERBS:
    _TOK[_w] = "verb"
for _w in _PARTICLES:
    _TOK[_w] = "part"
for _w in _CONJ:
    _TOK[_w] = "conj"
_TOK.update({"です": "cop", "だ": "cop", "でした": "cop_past", "だった": "cop_past",
             "ではありません": "neg_cop", "ません": "neg_cop", "ない": "neg",
             "なかった": "neg_past"})

_ROLE = {"は": "topic", "が": "subject", "を": "object", "に": "dest", "へ": "dest",
         "で": "loc", "と": "comp", "から": "from", "まで": "until", "も": "topic",
         "の": "mod", "や": "and", "など": "etc", "より": "than"}

_VERB_END = re.compile(r"(?:ませんでした|ませんでした|ました|ません|ます|た|て|だ|ない|なかった)$")
_ICHIDAN = {"見る", "食べる", "起きる", "寝る", "教える", "考える", "出る", "入る", "始める",
            "調べる", "着る", "開ける", "閉める", "覚える", "忘れる", "降りる", "いる"}
_IRREG = {"する": ("し", "し"), "来る": ("来", "来"), "できる": ("でき", "でき")}
_GODAN = {"く": "き", "ぐ": "ぎ", "す": "し", "つ": "ち", "ぬ": "に", "ぶ": "び",
           "む": "み", "う": "い", "る": "り"}
_SURU_END = ("し", "して", "した", "します", "しました", "しない", "せず", "していません")


def _stems_of(dictform: str) -> tuple[str, ...]:
    """辞書形から活用の語幹を作る（飲む → 飲み, 見る → 見, する → し）。"""
    if dictform in _IRREG:
        return (_IRREG[dictform][0],)
    if dictform in _ICHIDAN:
        return (dictform[:-1],)
    return (dictform[:-1] + _GODAN.get(dictform[-1], "り"),)


_STEM2VERB: dict[str, str] = {}
for _v in _VERBS:
    _STEM2VERB.setdefault(_v, _v)
    for _st in _stems_of(_v):
        _STEM2VERB.setdefault(_st, _v)

# 天気の話（主語が無い文でも自然な英語になるように）
_WEATHER_PRED = {"rain": "it will rain", "snow": "it will snow"}


def _romaji(word: str) -> str:
    """語彙バンクの読み（ローマ字）。対応表に無い語を落とさないための保険です。"""
    try:
        w = lex.bank().entry(word)
        if w is not None:
            return (w.romaji() or "").replace("'", "")
    except Exception:  # noqa: BLE001
        pass
    return ""


def _verb_forms(surface: str) -> tuple[str, bool, bool]:
    """動詞の語幹 → (辞書形, 過去, 否定)。`見ました` → (`見る`, True, False)。"""
    s = str(surface or "").strip("。、 ")
    past = bool(re.search(r"(?:ました|だった|でした|た)$", s))
    neg = bool(re.search(r"(?:ませんでした|ません|なかった|ない)$", s))
    stem = re.sub(r"(?:ませんでした|ません|ました|ます|でした|です|だ|た|て|ない|なかった)$",
                  "", s).strip("。、 ")
    if not stem:
        return "", past, neg
    if stem in _VERBS:
        return stem, past, neg
    if stem in _STEM2VERB:                    # 飲み → 飲む / 見 → 見る / 行き → 行く
        return _STEM2VERB[stem], past, neg
    for base in _SA_SURU:                     # 散歩し → 散歩する（末尾が サ変の活用のときだけ）
        if stem.startswith(base) and stem[len(base):] in _SURU_END:
            return base + "する", past, neg
    hits = [k for k in _VERBS if k.startswith(stem) and len(k) - len(stem) <= 2]
    if hits:
        hits.sort(key=len)
        return hits[0], past, neg
    return stem, past, neg


def _ja_polite(dictform: str, *, past: bool) -> str:
    """辞書形 → ます/ました（和訳の述語用）。"""
    v = str(dictform or "").strip()
    if not v:
        return "しました" if past else "します"
    if v in _IRREG:
        return _IRREG[v][0] + ("ました" if past else "ます")
    if v in _ICHIDAN:
        return v[:-1] + ("ました" if past else "ます")
    return v[:-1] + _GODAN.get(v[-1], "り") + ("ました" if past else "ます")


def _en_of(surface: str, kind: str) -> str:
    if kind == "time":
        return _TIME.get(surface, surface)
    if kind == "pron":
        return _PRONOUN.get(surface, surface)
    if kind == "noun":
        return _NOUNS.get(surface, surface)
    if kind == "adj":
        return _ADJ.get(surface, surface)
    if kind == "conj":
        return _CONJ.get(surface, surface)
    if kind == "verb":
        stem, past, _neg = _verb_forms(surface)
        pair = _VERBS.get(stem)
        if pair:
            return pair[1] if past else pair[0]
        return stem
    if kind == "part":
        return _PARTICLES.get(surface, "")
    if kind in ("cop", "cop_past"):
        return "was" if kind == "cop_past" else "is"
    if re.fullmatch(r"[A-Za-z0-9 ,.'\-]+", surface):
        return surface
    return _romaji(surface) or surface


def _next_token(src: str, i: int, unknown: list[str]) -> dict:
    n = len(src)
    mq = _NUM_COUNTER.match(src, i)
    if mq:                                     # 2時間 → 2 hours の形（数はそのまま）
        num, unit = mq.group(1), mq.group(2)
        en = f"{num} {_COUNTER[unit]}"
        return {"surface": mq.group(0), "kind": "quantity", "en": en,
                "len": mq.end() - mq.start()}
    for size in range(min(8, n - i), 0, -1):   # 1 文字の助詞（は・が・を・に）もここで拾う
        cand = src[i:i + size]
        if cand in _TOK:
            kind = _TOK[cand]
            tok = {"surface": cand, "kind": kind, "en": _en_of(cand, kind), "len": size}
            if kind == "verb":
                stem, past, neg = _verb_forms(cand)
                tok.update(stem=stem, past=past, neg=neg)
            return tok
        if _VERB_END.search(cand):
            stem, past, neg = _verb_forms(cand)
            if stem in _VERBS:
                pair = _VERBS[stem]
                return {"surface": cand, "kind": "verb", "stem": stem, "past": past,
                        "neg": neg, "en": pair[1] if past else pair[0], "len": size}
    for size in range(min(6, n - i), 1, -1):     # 対応表に無い語 → 読みで残す
        cand = src[i:i + size]
        rom = _romaji(cand)
        if rom:
            unknown.append(cand)
            return {"surface": cand, "kind": "unk", "en": rom, "len": size}
    # 対応表にも語彙バンクにも無い連続 → 次の既知語の手前までを 1 語として残す
    run_end = min(n, i + 8)
    for j in range(i + 1, min(n, i + 9)):
        if any(src[j:j + k] in _TOK for k in range(min(6, n - j), 1, -1)):
            run_end = j
            break
    chunk = src[i:run_end].strip(" 　。、！？!?")
    if chunk:
        unknown.append(chunk)
        rom = _romaji(chunk)
        return {"surface": chunk, "kind": "unk", "en": rom or chunk, "len": run_end - i}
    return {"surface": src[i], "kind": "punct", "en": "", "len": 1}


def _tokens(clause: str, unknown: list[str]) -> list[dict]:
    src = str(clause or "").strip("。、！？!? 　")
    out: list[dict] = []
    i, n = 0, len(src)
    while i < n:
        tok = _next_token(src, i, unknown)
        i += max(1, int(tok.get("len") or 1))
        if tok["kind"] != "punct":
            out.append(tok)
    return out


def _frame(tokens: list[dict]) -> dict:
    """役割の枠組み（topic / subject / object / dest / loc / comp / time / verb / cop）。"""
    fr: dict = {"topic": [], "subject": [], "object": [], "dest": [], "loc": [], "comp": [],
                "from": [], "until": [], "mod": [], "time": [], "verb": None, "cop": None,
                "neg": False, "tail": [], "conj": None}
    buf: list[dict] = []
    for t in tokens:
        k = t["kind"]
        if k == "part":
            role = _ROLE.get(t["surface"], "topic")
            if role == "mod":
                fr["mod"].extend(buf)
            elif role == "dest":
                fr["dest"].append(list(buf))
            elif role in fr:
                fr[role].extend(buf)
            else:
                fr["tail"].extend(buf)
            buf = []
        elif k == "verb":
            fr["verb"] = t
            fr["time"].extend([x for x in buf if x["kind"] == "time"])
            fr["tail"].extend([x for x in buf if x["kind"] != "time"])
            buf = []
        elif k in ("cop", "cop_past", "neg_cop", "neg", "neg_past"):
            fr["cop"] = t
            fr["neg"] = fr["neg"] or k.startswith("neg")
            fr["time"].extend([x for x in buf if x["kind"] == "time"])
            fr["tail"].extend([x for x in buf if x["kind"] != "time"])
            buf = []
        elif k == "conj":
            fr["conj"] = t
            fr["tail"].extend(buf)
            buf = []
        else:
            buf.append(t)
    fr["tail"].extend(buf)
    for key in ("subject", "tail", "object"):
        times = [x for x in fr[key] if x["kind"] == "time"]
        if times:
            fr["time"].extend(times)
            fr[key] = [x for x in fr[key] if x["kind"] != "time"]
    return fr


def _np(tokens: list[dict]) -> str:
    """名詞句を英語に（形容詞 + 名詞 → a big city のように冠詞を前へ）。"""
    words = [str(t.get("en") or "").strip() for t in tokens if t.get("kind") != "part"]
    words = [w for w in words if w]
    if not words:
        return ""
    text = " ".join(words)
    m = re.search(r"\b(a|an|the)\b", text)
    if m and not text.lower().startswith((m.group(1).lower(),)):
        art = m.group(1)
        text = re.sub(r"\b(?:a|an|the)\s+", "", text, count=1).strip()
        text = f"{art} {text}"
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# 日→英
# --------------------------------------------------------------------------- #
def _clause_en(clause: str, ctx: dict, unknown: list[str]) -> str:
    toks = _tokens(clause, unknown)
    if not toks:
        return ""
    if re.search(r"(?:て|で)$", str(clause).strip("。、 ")) and any(
            t["kind"] == "verb" for t in toks):
        ctx["te"] = True                       # 〜て、＝動作の連続（and でつなぐ）
    fr = _frame(toks)

    # 主題が時間の語だけ（今日は / 明日は）→ 副詞として前に出す
    topic_time = [t for t in fr["topic"] if t["kind"] == "time"]
    subject_toks = [t for t in fr["topic"] if t["kind"] != "time"] or fr["subject"]
    times = list(fr["time"]) + topic_time
    time_txt = " ".join(dict.fromkeys(str(t.get("en") or "") for t in times)).strip()

    subj = _np(subject_toks)
    if subj and any(t["kind"] == "pron" for t in subject_toks):
        ctx["subject"] = subj                # 引き継ぐのは「私は」のような代名詞だけ
    subj = subj or (ctx.get("subject") or "")

    pred_toks = fr["mod"] + fr["tail"]
    if fr["verb"] and (fr["verb"].get("stem") or "") == "かかる":
        qty = [t for t in toks if t["kind"] == "quantity"]
        by = _np(fr["loc"]) or (_np(fr["dest"][0]) if fr["dest"] else "")
        amount = " ".join(str(t.get("en") or "") for t in qty).strip()
        line = f"It took {amount}" if amount else "It took some time"
        if by:
            line += f" by {by}"
        return line

    if fr["verb"]:
        v = fr["verb"]
        stem = v.get("stem") or ""
        now, past_form = _VERBS.get(stem, (v.get("en") or "", v.get("en") or ""))
        vp = (past_form if v.get("past") else now) or (v.get("en") or "")
        if v.get("neg"):
            vp = f"did not {now}"
        pieces = [vp]
        if fr["object"]:
            pieces.append(_np(fr["object"]))
        elif [t for t in toks if t["kind"] == "quantity"] and stem in ("過ごす", "待つ", "食べる"):
            pieces.append(" ".join(str(t.get("en") or "") for t in toks
                                   if t["kind"] == "quantity"))
        if fr["dest"]:
            if len(fr["dest"]) >= 2 and stem == "行く":
                pieces.append(f"to {_np(fr['dest'][0])} for {_np(fr['dest'][1])}")
            else:
                pieces.extend(f"to {_np(dsp)}" for dsp in fr["dest"] if dsp)
        if fr["loc"]:
            pieces.append(f"at {_np(fr['loc'])}")
        if fr["comp"]:
            pieces.append(f"with {_np(fr['comp'])}")
        if fr["from"]:
            pieces.append(f"from {_np(fr['from'])}")
        if fr["until"]:
            pieces.append(f"until {_np(fr['until'])}")
        body = " ".join(p for p in pieces if p)
        who = subj or ("I" if re.search(r"(?:ました|ます|た|て|ない)", clause) else "")
        line = f"{who} {body}".strip() if who else body
        if who:
            ctx["subject"] = who
    elif not pred_toks and not fr["cop"] and not fr["neg"]:
        if time_txt:
            ctx.setdefault("times", []).append(time_txt)
        if subj:
            ctx.setdefault("subject", subj)
        return ""
    elif pred_toks or fr["neg"]:
        complement = _np(pred_toks)
        was = bool(fr["cop"] and fr["cop"]["kind"] in ("cop_past", "neg_past"))
        neg = bool(fr["neg"])
        if not complement and not neg:
            line = subj
        elif complement in _WEATHER_PRED and not subj:
            line = _WEATHER_PRED[complement]
        else:
            be = "was" if was else "is"
            if neg:
                be = "was not" if was else "is not"
            line = f"{subj} {be} {complement}".strip() if subj else f"It {be} {complement}".strip()
    else:
        line = subj

    if not line:
        if time_txt:
            ctx.setdefault("times", []).append(time_txt)
        if subj and subj != ctx.get("subject"):
            ctx["subject"] = subj
        return ""
    if time_txt:
        line = f"{time_txt}, {line}"
    return line.strip().strip(",")


def to_english(source: str) -> tuple[str, list[str]]:
    """日本語の材料を英語に訳す。返るのは (訳文, notes)。"""
    unknown: list[str] = []
    out: list[str] = []
    for sent in [s.strip("「」\"' 　") for s in _SENT_SPLIT.split(str(source or "")) if s.strip()]:
        parts = [p.strip() for p in _CLAUSE_SPLIT.split(sent) if p.strip()]
        if not parts:
            continue
        ctx: dict = {"subject": "", "times": []}
        pieces: list[str] = []
        i = 0
        while i < len(parts):
            p = parts[i]
            m = re.match(r"^(.+?)ので$", p)                  # Aので、B → A, so B
            if m:
                reason = _clause_en(m.group(1).strip(), ctx, unknown)
                rest = [_clause_en(q, ctx, unknown) for q in parts[i + 1:]]
                tail = ", and ".join(x for x in rest if x)
                if reason and tail:
                    pieces.append(("", f"{reason}, so {tail}"))
                elif reason or tail:
                    pieces.append(("", reason or tail))
                break
            ctx.pop("te", None)
            piece = _clause_en(p, ctx, unknown)
            if piece:
                pieces.append(("and" if ctx.get("te") else "", piece))
            i += 1
        front = " ".join(ctx.get("times") or [])
        joined: list[str] = []
        for _conj, piece in pieces:
            if not piece:
                continue
            if _conj == "and" and joined:
                joined[-1] = joined[-1] + ", and " + piece
            else:
                joined.append(piece)
        line = ", ".join(joined)
        line = re.sub(r"\s+", " ", f"{front}, {line}" if front and line else (line or front)).strip()
        line = line.strip(", ").strip()
        if not line:
            continue
        if not line.endswith("."):
            line = line[0].upper() + line[1:] + "."
        out.append(line)
    notes = [f"訳: 英訳 {len(out)}文"]
    if unknown:
        notes.append("読みで残した語: " + "、".join(list(dict.fromkeys(unknown))[:6]))
    return " ".join(out), notes


# --------------------------------------------------------------------------- #
# 英→日
# --------------------------------------------------------------------------- #
_EN_PREP = {"to": "dest", "at": "loc", "in": "loc", "on": "loc", "with": "comp",
            "for": "purpose", "from": "from", "until": "until", "by": "loc",
            "of": "mod", "about": "about"}
_EN_COP = {"is": "です", "was": "でした", "are": "です", "were": "でした", "am": "です"}
_EN_ART = {"the", "a", "an", "some", "any"}
_EN_DUMMY = {"it", "there", "will", "would", "shall", "may", "might", "do", "does"}
_EN_PAST = {"went", "ate", "saw", "did", "had", "met", "bought", "read", "wrote", "made",
            "came", "took", "got", "left", "entered", "arrived", "listened", "talked",
            "said", "sold", "played", "worked", "walked", "ran", "swam", "waited",
            "slept", "rested", "started", "finished", "taught", "learned", "thought",
            "knew", "understood", "used", "rode", "sang", "danced", "helped", "returned",
            "studied", "could", "drank"}
def _rev(table: dict[str, str]) -> dict[str, str]:
    """日→英の対応表を逆引きにする（`the weather` → `weather` でも引けるように）。"""
    out: dict[str, str] = {}
    for ja, en in table.items():
        for key in {str(en).lower(), re.sub(r"^(?:a|an|the)\s+", "", str(en).lower())}:
            key = key.strip(" .,")
            if key:
                out.setdefault(key, ja)
    return out


_EN_VERB: dict[str, tuple[str, bool]] = {}
for _ja, (_now, _past) in _VERBS.items():
    for _form, _past_flag in ((_past, True), (_now, False)):
        _form = _form.lower().strip()
        if _form and " " not in _form:        # 句（take a walk）は語としては登録しない
            _EN_VERB.setdefault(_form, (_ja, _past_flag))
_EN_TIME = _rev(_TIME)
_EN_NOUN = _rev(_NOUNS)
_EN_ADJ = _rev(_ADJ)
_EN_PRON = _rev(_PRONOUN)
for _k in [k for k in _EN_VERB if k in _EN_NOUN and k not in _EN_PAST]:
    _EN_VERB.pop(_k, None)                    # walk → 散歩（名詞）/ walked → 歩く（動詞）

# 2 語以上のまとまり（every morning / a walk / the park）は 1 つの語として扱う
_EN_MULTI: dict[str, dict] = {}
for _table, _kind in ((_EN_TIME, "time"), (_EN_NOUN, "noun"), (_EN_ADJ, "adj"),
                      (_EN_PRON, "pron")):
    for _en, _ja in _table.items():
        if " " in _en:
            _EN_MULTI.setdefault(_en, {"kind": _kind, "ja": _ja})
for _en, (_ja, _past) in _EN_VERB.items():
    if " " in _en:
        _EN_MULTI.setdefault(_en, {"kind": "verb", "ja": _ja, "past": _past})
for _ja, (_now, _past) in _VERBS.items():     # 句の動詞（take a walk）も登録
    for _form, _flag in ((_past, True), (_now, False)):
        if " " in _form:
            _EN_MULTI.setdefault(_form.lower(), {"kind": "verb", "ja": _ja, "past": _flag})
_EN_MULTI_KEYS = sorted(_EN_MULTI, key=lambda x: -len(x.split()))


def _clean_en(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9'\-]", "", str(raw or "")).lower()


def _en_tokens(s: str, unknown: list[str]) -> list[dict]:
    out: list[dict] = []
    raws = [x for x in re.split(r"[\s,]+", str(s or "").strip()) if x.strip()]
    idx = 0
    while idx < len(raws):                    # 複数語のまとまりを先に拾う
        hit = None
        for size in range(min(4, len(raws) - idx), 1, -1):
            cand = " ".join(_clean_en(x) for x in raws[idx:idx + size])
            cand = re.sub(r"^(?:a|an|the)\s+", "", cand)
            for key in _EN_MULTI_KEYS:
                if key == cand or re.sub(r"^(?:a|an|the)\s+", "", key) == cand:
                    hit = (size, dict(_EN_MULTI[key]))
                    break
            if hit:
                break
        if hit:
            size, tok = hit
            tok["surface"] = " ".join(raws[idx:idx + size])
            out.append(tok)
            idx += size
            continue
        idx += 1
        raw = raws[idx - 1]
        w = _clean_en(raw)
        if not w:
            continue
        if w in _EN_ART or w in _EN_DUMMY:
            continue
        if w in _EN_PRON:
            out.append({"kind": "pron", "ja": _EN_PRON[w], "surface": raw})
        elif w in _EN_TIME:
            out.append({"kind": "time", "ja": _EN_TIME[w], "surface": raw})
        elif w in _EN_COP:
            out.append({"kind": "cop", "ja": _EN_COP[w], "surface": raw})
        elif w in _EN_PREP:
            out.append({"kind": "prep", "role": _EN_PREP[w], "surface": raw})
        elif w in ("so", "because", "and", "but"):
            out.append({"kind": "conj", "ja": _CONJ.get(w) or _EN2JA.get(w, ""), "surface": raw})
        elif w in _EN_VERB:
            ja, past = _EN_VERB[w]
            if w in _EN_PAST:
                past = True
            out.append({"kind": "verb", "ja": ja, "past": past, "surface": raw})
        elif w in _EN_ADJ:
            out.append({"kind": "adj", "ja": _EN_ADJ[w], "surface": raw})
        elif w in _EN_NOUN:
            out.append({"kind": "noun", "ja": _EN_NOUN[w], "surface": raw})
        else:
            unknown.append(raw.strip(".,!?"))
            out.append({"kind": "unk", "ja": raw.strip(".,!?"), "surface": raw})
    return out


def _ja_phrase(toks: list[dict]) -> str:
    return "".join(str(t.get("ja") or "") for t in toks if t.get("ja"))


def _en_clause(s: str, unknown: list[str]) -> str:
    toks = _en_tokens(s, unknown)
    if not toks:
        return ""
    times = [t for t in toks if t["kind"] == "time"]
    vi = next((i for i, t in enumerate(toks) if t["kind"] in ("verb", "cop")), -1)
    head = toks[:vi] if vi > 0 else []
    subj = [t for t in head if t["kind"] in ("pron", "noun", "adj", "unk") and t not in times]
    rest = toks[vi + 1:] if vi >= 0 else toks
    front = "、".join(_ja_phrase([t]) for t in times if t.get("ja"))
    front = "、".join(x for x in (_ja_phrase([t]) for t in times if t.get("ja")) if x)

    if vi >= 0 and toks[vi]["kind"] == "cop":
        comp = [t for t in rest if t["kind"] != "time"]
        body = f"{_ja_phrase(subj)}は{_ja_phrase(comp)}{toks[vi]['ja']}"
        return f"{front}、{body}" if front else body

    if vi >= 0:
        v = toks[vi]
        verb = _ja_polite(str(v.get("ja") or ""), past=bool(v.get("past")))
        roles: dict[str, list[dict]] = {}
        buf: list[dict] = []
        cur_role = ""
        for t in rest:
            if t["kind"] == "time":
                continue
            if t["kind"] == "prep":
                if buf and not cur_role:
                    roles.setdefault("object", []).extend(buf)
                buf = []
                cur_role = t["role"]
                continue
            if cur_role:
                roles.setdefault(cur_role, []).append(t)
            else:
                buf.append(t)
        if buf and not cur_role:
            roles.setdefault("object", []).extend(buf)
        subj_ja = _ja_phrase(subj) or "私"
        comp = _ja_phrase(roles.get("comp", []))
        obj = _ja_phrase(roles.get("object", []))
        dest = _ja_phrase(roles.get("dest", []))
        purpose = _ja_phrase(roles.get("purpose", []))
        loc = _ja_phrase(roles.get("loc", []))
        src = _ja_phrase(roles.get("from", []))
        mid = "".join(x for x in (
            f"{comp}と" if comp else "",
            f"{obj}を" if obj else "",
            f"{src}から" if src else "",
            f"{dest}に" if dest else "",
            f"{purpose}に" if purpose else "",
            f"{loc}で" if loc else "",
        ) if x)
        body = f"{subj_ja}は{mid}{verb}"
        return f"{front}、{body}" if front else body

    body = _ja_phrase([t for t in toks if t["kind"] != "time"])
    if body and not re.search(r"(?:ます|ました|です|だ)$", body):
        body += "です"                        # 動詞も copula も無い文 → 名詞述語としてですを付ける
    return f"{front}、{body}" if front else body


def to_japanese(source: str) -> tuple[str, list[str]]:
    """英語の材料を日本語に訳す。返るのは (訳文, notes)。"""
    unknown: list[str] = []
    out: list[str] = []
    for sent in [s.strip("\"'「」 　") for s in re.split(r"(?<=[.!?])\s+", str(source or "").strip())
                 if s.strip()]:
        s = sent.strip()
        m = re.match(r"^(?:because\s+)?(.+?),\s*so\s+(.+)$", s, re.IGNORECASE)
        if not m:
            m = re.match(r"^because\s+(.+?),\s*(.+)$", s, re.IGNORECASE)
        if m:
            a, b = _en_clause(m.group(1), unknown), _en_clause(m.group(2), unknown)
            a = re.sub(r"(?:です|でした)$", "", a.strip("。、 "))   # 良いです + ので → 良いので
            out.append(f"{a}ので、{b}。")
            continue
        out.append(_en_clause(s, unknown) + "。")
    body = "".join(x for x in out if x.strip("。"))
    notes = [f"訳: 和訳 {len(out)}文"]
    if unknown:
        notes.append("対応表に無い語: " + "、".join(list(dict.fromkeys(unknown))[:6]))
    return body, notes


def translate(source: str, target: str = "en") -> tuple[str, list[str]]:
    """材料を `target`（`"en"` / `"ja"`）の言語に訳す。"""
    src = str(source or "").strip().strip("「」\"'")
    if not src:
        return "", ["訳: 材料がありません"]
    tgt = str(target or "").lower()
    if tgt.startswith("ja") or "日本語" in tgt:
        return to_japanese(src)
    if tgt.startswith("en") or "英語" in tgt:
        return to_english(src) if _JP_CHAR.search(src) else to_japanese(src)
    return to_japanese(src) if not _JP_CHAR.search(src) else to_english(src)


__all__ = ["translate", "to_english", "to_japanese", "is_translation_request", "detect_target"]
