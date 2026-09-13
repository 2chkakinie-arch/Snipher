"""発話 → 意味フレーム。Snipher の「見る」層。

キーワード一致で話題を決めるのではなく、**発話の形**（語気・問いの種類・格・制約・
指示語・文字種）を先に決めます。ここで決まったことが、証拠集め（ground）と
組み立て（voice）の順序を決める。
"""

from __future__ import annotations

import re

from ..lang.lex import bank
from ..lang.phonetics import (
    ascii_ratio,
    char_count,
    is_all_kana,
    mora_count,
    normalize,
    to_hiragana,
)
from .frame import Entity, Frame
from .rules import compile_rules
from .state import ConversationState, turn_word

_Q_END = re.compile(r"(か|かな|かしら|でしょう|ですか|ますか|ませんか|だっけ|の)\s*[?？!！。]*$")
_ASCII_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_+.\\-]*")
_DIGITS = re.compile(r"[0-9０-９]+(?:%|割|円|分|秒|時間|日|年|回)?")

# 問いの型（longer / more specific first）
ASK_KINDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # 自己紹介は「名前・役割・相手に何が起きるか」の 3 点だけを選ぶ別の道にします。
    # 「できること」という語を能力質問より先に拾わないと、道具一覧を返してしまいます。
    ("self_intro", ("名前とできること", "名前と役割", "名前教えて", "自己紹介して", "自己是")),
    ("capability", ("何ができる", "なにができる", "何できますか", "なにができますか", "何ができるの",
                    "できること", "機能", "何のアプリ", "何が得意", "どこまで", "何が苦手",
                    "誰が作った", "モデルは", "どんなことできる", "何がusable")),
    ("identity", ("あなたは何", "何者", "名前は何", "だれですか", "誰ですか", "自己紹介",
                  "你是谁", "snipher は", "スニファーは")), 


    ("now", ("今は", "いまは", "今何時", "今何日", "今日は何日", "今日は何日", "何曜日",
             "西暦何年", "何年ですか", "現在は", "本日は", "unix", " UNIX")),
    ("definition", ("とは", "って何", "とは何", "どういうもの", "どんなもの", "何者", "定義",
                    "意味は何", "どういう意味", "どんな意味", "何のこと", "いったい何")),
    ("reading", ("読み", "なんて読む", "何と読む", "よみ", "ふりがな", "読み方", "なんという字",
                 "何という字", "なんていう字")),
    ("word_property", ("何文字", "なんもじ", "何音", "拍数", "画数", "部首", "品詞", "活用形",
                       "ローマ字で", "スペル", "綴り")),
    ("synonym", ("類義語", "言い換え", "似た言葉", "同義語", "別の言い方", "別の表現",
                 "かたい表現", "やわらかい表現")),
    ("antonym", ("対義語", "反対語", "逆の意味", "オポジット")),
    ("example", ("例文", "使い方を教えて", "用例", "どんな文", "文例", "使い方")),
    ("translation", ("英語で", "日本語で", "訳して", "翻訳", "中国語で", "韓国語で", "フランス語で",
                     "ドイツ語で", "スペイン語で", "ローマ字に", "英語に", "日本語に", "中国語に",
                     "中文に", "韓国語に", "フランス語に", "ドイツ語に", "スペイン語に", "イタリア語に",
                     "ロシア語に", "外国語に", "訳し", "言い換えて")),
    ("word_list", ("で始まる word", "で始まる言葉", "で始まる単語", "しりとり", "何がある",
                   "どんな言葉", "列举して", "列挙して", "言葉を教えて")),
    ("reason", ("なぜ", "どうして", "なんで", "理由", "しくみ", "仕組み", "どうして?", "訳は")),
    ("procedure", ("作り方", "やり方", "手順", "方法", "どうすれば", "どうしたら", "コツ",
                   "こつ", "ポイント", "始め方", "覚え方", "使い方", "選び方", "書き方",
                   "過ごし方", "対策", "直し方", "設定方法", "インストール", "始めたい")),
    ("comparison", ("違い", "比較", "どちら", "vs", "Against", "優劣", "どっちが", "長所と短所")),
    ("count", ("いくつ", "何個", "何匹", "何本", "何人", "何枚", "何文字", "何歳", "何点")),
    ("price", ("いくら", "値段", "価格", "費用", "料金", "コスト")),
    ("when", ("いつ", "何時", "時期", "何月", "季節", "旬", "タイミング", "何日", "何年",
              "どれくらい", "どのくらい", "何分", "何時間")),
    ("where", ("どこ", "何処", "場所", "どのあたり", "どちらの国")),
    ("who", ("だれ", "誰", "どなた", "作者は", "発明者は", "何人")),
    ("manner", ("どうやって", "どのように", "どうする", "やり方を", "進め方")),
    ("recommend", ("おすすめ", "お勧め", "何がいい", "なにがいい", "どのへんが", "選びたい",
                   "ランキング", "良いのは")),
    ("opinion", ("どう思う", "感想", "好き", "嫌い", "気になる", "どうですかね", "いかが")),
    ("trouble", ("できない", "動かない", "壊れ", "失敗", "つらい", "困っ", "エラー", "直したい",
                 "治らない", "うまくいかない", "迷っ")),
    ("yesno", ("かどうか", "だと思うけど", "合ってますか", "合っている", "あっていますか",
               "本当に", "可能ですか", "できますか")),
    ("summarize", ("要約", "まとめて", "短くして", "3 行で", "三行で", "要点")),
    ("extract", ("抜き出して", "抽出して", "リスト化", "数えて", "件数は")),
    ("transform", ("変換して", "変えて", "逆にして", "並び替えて", "整列して", "大文字に",
                   "小文字に", "半角に", "全角に", "ソート", "重複を", "個数を数えて")),
    ("compute", ("計算", "式", "方程式", "解いて", "何になりますか", "いくらになる", "確率",
                 "掛ける", "かける", "足して", "たして", "引き算", "割り算", "何割", "たす", "ひく",
                 "わる",
                 "平均", "合計", "割", "％", "パーセント", "ルート", "二乗", "素因数",
                 "約数", "最小公倍数", "最大公約数", "微分", "積分", "展開", "因数分解")),
    ("code", ("コード", "プログラム", "実装", "スクリプト", "関数を書いて", "クラス", "API",
              "python", "javascript", "typescript", "html", "css", "sql", "bash", "regex",
              "デバッグ", "バグ", "エラーを直", "ゲームを作って", "アプリを作って")),
    ("write", ("書いて", "作って", "創って", "考えて", "作文", "小説", "詩", "短歌", "俳句",
               "川柳", "メール", "文案", "キャッチコピー", "挨拶文", "お礼の文", "スピーチ",
               "台本", "歌詞", "童話", "ミステリ", "題名")),
    ("estimate", ("およそ", "概算", "どれくらい?", "どのくらい?", "どれぐらい", "どのぐらい",
                  "どれくらいある", "何グラム", "何キロ", "何ミリ")),
)


_MARKERS = {
    "greet": ("こんにちは", "こんばんは", "おはよう", "はじめまして", "やあ", "もしもし",
              "hello", "hi", "hey", "hiya"),
    "thanks": ("ありがとう", "ありがと", "感謝", "サンキュー", "助かった", "おかげ",
               "thanks", "thank you", "ty"),
    "apology": ("ごめん", "すみません", "申し訳", "失礼", "sorry", "apolog"),
    "farewell": ("さようなら", "さよなら", "またね", "おやすみ", "じゃあね", "バイバイ",
                 "bye", "お先に"),
    "praise": ("すごい", "えらい", "上手", "うまい", "素晴らしい", "天才", "さすが", "完璧",
               "いいね", "好き"),
    "agree": ("その通り", "たしかに", "なるほど", "同感", "賛成", "そうだね", "合ってる"),
    "disagree": ("違う", "ちがう", "間違っ", "そうではない", "反対", "それは違う", "違うと思う"),
}

_INVITE = re.compile(r"(しよう|しよ|やろう|やろ|遊ぼ|あそぼ|しませんか|したい|つきあって|"
                     r"付き合って|play|let'?s|やりましょ)")
_CURRENT = ("今日", "今", "最新", "ニュース", "天気", "為替", "株価", "現在", "いま", "最新の",
            "この前", "昨日の", "明日の", "速報")
_PRONOUNS = ("それ", "これ", "あれ", "そのこと", "この前", "さっきの", "さきほどの", "同じ",
             "続き", "つづき", "ついでに", "あとで", "さっき")


def _kind_from_end(tail: str) -> str:
    if re.search(r"[A-Za-z]", tail) and not re.search(r"[ぁ-ん一-龯]", tail):
        return "ascii"
    if is_all_kana(tail):
        return "kana"
    if re.fullmatch(r"[0-9０-９%.]+", tail or ""):
        return "digits"
    return "mixed"


def classify_ask(text: str) -> str:
    t = normalize(text).lower()
    for kind, marks in ASK_KINDS:
        for m in marks:
            if m in t:
                return kind
    return ""


_GOOD_POS = {"名詞/普通名詞", "名詞/一般", "名詞/固有名詞", "名詞/普通名詞/サ変可能", "名詞/数"}
_BAD_POS = ("形容動詞語幹", "接頭詞", "非自立", "補助記題", "未知語")


def rank_entities(ents: list[Entity], text: str) -> list[Entity]:
    """「何について聞かれているか」の一等席を決める。

    知識ベースに当たりる語 ＞ 名詞（一般・固有名） ＞ 長い語、の順。
    「好きな食べ物は？」で 好き より 食べ物 を前に出すための整列です。
    """
    head = re.match(r"\s*[「『]?([^」』\nと]{1,16}?)[」』]?\s*(?:とは|って何|とは何)", normalize(text))
    head_word = normalize(head.group(1)) if head else ""

    def key(e: Entity) -> tuple[int, int, int, int]:
        score = 0
        if e.surface == head_word:
            score += 40
        if e.known_kb:
            score += 14
        if e.pos.split("/")[0] == "名詞":
            score += 8
        if any(b in e.pos for b in _BAD_POS):
            score -= 9
        if e.pos in _GOOD_POS:
            score += 3
        score += min(6, len(e.surface))
        if e.kind == "digits":
            score -= 4
        # 手元に無い語（＝調べないと答えられない語）が「話題の一等席」。
        # 逆に *長い語の断片* としてかな 2 文字が引けてしまうと、话题がそこにすり替わる
        # （「ブラウザ」の中の「ブラ」が主語になっていた事故）ので、強く下げる。
        if e.web_needed and e.kind in ("ascii", "mixed"):
            score += 12 + min(8, len(e.surface))
        if e.kind == "kana" and len(e.surface) <= 3 and not e.known_kb:
            score -= 12
        return (-score, len(e.surface), 0, 0)

    return sorted(ents, key=key)


def detect_act(text: str, *, ask: str = "") -> str:
    t = normalize(text)
    low = t.lower()
    tail = t.rstrip()
    for act, words in (("greet", _MARKERS["greet"]), ("thanks", _MARKERS["thanks"]),
                       ("apology", _MARKERS["apology"]), ("farewell", _MARKERS["farewell"]),
                       ("disagree", _MARKERS["disagree"]), ("agree", _MARKERS["agree"]),
                       ("praise", _MARKERS["praise"])):
        if any(w in low for w in words) and not ask:
            return act
    if _INVITE.search(low):
        return "invite"
    if re.search(r"(てくれ|てくれる|てください|てほしい|してよ|しなさい|教えろ|出しなさい|作って)", t):
        return "command"
    if tail.endswith(("?", "？")) or _Q_END.search(tail) or ask:
        return "ask"
    if re.search(r"(たい|ほしい|したい|願う| hoped)", low):
        return "wish"
    return "declare"


def extract_entities(text: str, *, kb=None) -> list[Entity]:
    """内容語・英数字列・数字の塊を「対象」として集める（辞書照会つき）。"""
    t = normalize(text)
    out: list[Entity] = []
    seen: set[str] = set()
    b = bank()

    def push(surface: str) -> None:
        s = str(surface or "").strip("。、,.!?！？::「」『』()（）【】・~〜 ")
        if not s or len(s) > 16 or s in seen:
            return
        if s in STOP:
            return
        seen.add(s)
        w = b.entry(s)
        kb_topic = ""
        if kb is not None:
            try:
                ids = kb.index.topics_of(to_hiragana(s)) | kb.index.topics_of(s)
                if ids:
                    kb_topic = str(kb.items[sorted(ids)[0]].get("topic") or "")
            except Exception:  # noqa: BLE001
                kb_topic = ""
        kind = _kind_from_end(s)
        ent = Entity(
            surface=s, kind=kind,
            reading=(w.reading if w else to_hiragana(s) if kind != "ascii" else ""),
            pos=(w.pos if w else ""), morae=mora_count(w.reading if w else s),
            known_dict=w is not None, known_kb=kb_topic,
        )
        # 手元に無い語（辞書にも KB にも無い、かつ固有名らし）は Web 裏取りが必要
        rare = (w is None and kb_topic == "" and len(s) >= 2
                and not re.fullmatch(r"[ぁ-ん]{1,2}", s))
        if kind == "ascii":
            rare = True
        ent.web_needed = bool(rare) and kind in ("ascii", "mixed", "kana") and len(s) >= 2
        out.append(ent)

    for m in _ASCII_TOKEN.finditer(t):
        push(m.group())
    for seg, pos in b.segment(t):
        if pos.split("/")[0] in {"名詞", "動詞", "形容詞", "副詞"} or pos in ("未知語", "数詞"):
            push(seg)
    for m in re.finditer(r"[一-龯ぁ-んァ-ヶー]{2,8}(?=[をがはにでとはの])", t):
        push(m.group())
    # 「X とは」型の主語を最優先で先頭に寄せる
    head = re.match(r"\s*[「『]?([^」』\nと]{1,16}?)[」』]?\s*(?:とは|って何|って何)", t)
    if head:
        push(head.group(1))
        hit = next((e for e in out if e.surface == normalize(head.group(1))), None)
        if hit is not None:
            out.remove(hit)
            out.insert(0, hit)
    for e in out:
        if e.kind == "digits":
            e.web_needed = False
    return out[:8]


STOP = {
    "こと", "もの", "ため", "よう", "とき", "ほど", "だけ", "など", "これ", "それ", "あれ",
    "ここ", "そこ", "あそこ", "なに", "何", "誰", "いつ", "どう", "なぜ", "いくつ", "いくら",
    "どんな", "どの", "かどうか", "あなた", "私", "僕", "俺", "うち", "向こう", "こちら",
    "ちょっと", "少し", "いろいろ", "なんか", "なにか", "何か", "たくさん", "ずっと",
    "もう", "まだ", "すぐ", "たぶん", "もちろん", "たしかに", "なるほど", "そう", "それでは",
    "では", "ちなみに", "例えば", "たとえば", "って", "とは", "について", "に関して", "に対して",
    "方法", "意味", "場合", "時点", "あたり", "くらい", "ぐらい", "はず", "つもり", "わけ",
    "はず", "もの",
}


def opaque_reason(text: str) -> str:
    """内容が読み取れない入力（数字のみ・文字化け・1 文字）の理由を返す（無ければ ""）。"""
    t = normalize(text)
    if not t:
        return "empty"
    body = re.sub(r"[\s。、！？!?・…「」『』()（）]+", "", t)
    if not body:
        return "punct_only"
    if len(body) == 1:
        try:
            from ..lang.lex import bank as _bank

            ent = _bank().entry(body)
            if ent is not None and ent.pos.split("/")[0] == "名詞":
                return ""           # 「猫」など 1 語で話題になる入力は opaque ではない
        except Exception:  # noqa: BLE001
            return "single_char"
        return "single_char"
    if re.fullmatch(r"[0-9０-９.%]+", body):
        return "digits_only"
    if (re.fullmatch(r"[A-Za-z0-9_+.,:;/%-]+", body) and ascii_ratio(body) > 0.92
            and not any(x in body.lower() for x in ("hi", "no", "ok", "id", "ai", "tv"))):
        # 辞書にも KB にも無い欧文の塊は「読めない入力」として扱う（でたらめを返さない）
        known = False
        try:
            from ..lang.lex import bank as _bank

            known = _bank().has(body) or bool(re.search(
                r"(code|python|sql|html|css|json|api|git|npm|node|linux|cuda)$", body.lower()))
        except Exception:  # noqa: BLE001
            known = False
        return "" if known else "latin_noise"
    if re.search(r"[\ufffd\u00c3\u00e3\u00e5\u00ef\u00be\u0092]", t):
        return "mojibake"
    return ""


_ABOUT_USER_RE = re.compile(
    r"(?:私の|わたしの|僕の名前|ぼくの|俺の名前|ボクの|自分の|私のこと|私の事を|私の名)")


def build_frame(text: str, *, history: list[dict] | None = None, kb=None,
                state: ConversationState | None = None) -> Frame:
    """発話 1 つを Frame に分解する（この 1 本道だけが、下のパイプラインに材料を渡す）。"""
    raw = str(text or "")
    t = normalize(raw)
    state = state or ConversationState(history, kb=kb)
    f = Frame(raw=raw, norm=t, register=state.register, language=state.language)
    f.numbers = [normalize(x) for x in _DIGITS.findall(t)][:6]
    ask = classify_ask(t)
    # 「私の名前は？」のように一人称に掛かった問いは、こちらの自己紹介ではない。
    # 会話の記憶（ユーザーが先に話した事実）を読む仕事として分ける。
    if ask == "identity" and _ABOUT_USER_RE.search(t):
        ask = "identity_user"
    f.act = detect_act(t, ask=ask)
    f.ask = ask
    f.entities = rank_entities(extract_entities(t, kb=kb), t)
    f.rules = compile_rules(t, previous_word=state.previous_token())
    f.mood = _mood(t)
    f.flags["opaque"] = opaque_reason(t)
    f.flags["invite"] = f.act == "invite"
    f.flags["continuation"] = state.expects_continuation(t)
    f.flags["pronoun"] = any(p in t for p in _PRONOUNS)
    f.flags["current"] = any(c in t for c in _CURRENT)
    f.is_followup = bool(f.flags["continuation"] or (f.flags["pronoun"] and len(t) <= 14))

    # 話題文字列: KB 検索には「相手が発話した語」をそのまま渡す（推測で書き換えない）
    if f.entities:
        f.topic = f.entities[0].surface
    else:
        f.topic = re.sub(r"[。、！？!?・…]+$", "", t)[:18]

    # 手元に無い語への定義要求は、検索の判定材料にする（＝推測で答えない）
    unknown = [e for e in f.entities if e.web_needed]
    if f.ask in ("definition", "reason", "procedure", "comparison", "recommend", "when",
                 "price", "who", "where", "count") and unknown:
        f.needs_web = True
        f.flags["unknown_terms"] = [e.surface for e in unknown][:4]
    if f.ask == "translation":
        # 対応語は手元の単語辞書には無い。*実際に用例が取れるのは検索側*なので、
        # 翻訳の依頼は常に裏取り対象にする（取れないとき用に読み・品詞は別途返す）。
        f.needs_web = True
    if f.ask == "code" or f.ask == "compute":
        f.needs_web = False
    if f.act == "invite" and state.activity is None:
        f.flags["propose_activity"] = True
    if f.flags.get("opaque") and f.act not in ("greet", "thanks", "apology", "farewell"):
        f.ask = "opaque"
    if f.ask == "now":
        f.needs_web = False
    f.flags["ask_source"] = "pattern"
    if f.entities:
        f.flags["first_entity"] = f.entities[0].as_dict()
    if state.activity is not None:
        f.activity = state.activity
    return f


_MOOD_NEG = ("つらい", "辛い", "疲れた", "寂しい", "不安", "心配", "怖い", "痛い", "苦しい",
             "忙しい", "眠い", "暑い", "寒い", "しんどい", "だるい", "余裕がない", "しんど",
             "眠れない", "嫌", "失敗", "怒", "困っ", "大変", "しんど", "落ち込", "無理",
             "つまらない", "退屈", "できない", "分からない", "迷っ", "どうしよう", "最悪",
             "だるい", "悲しい", "悔しい", "イライラ", "腹が立")
_MOOD_POS = ("嬉しい", "うれしい", "楽しい", "たのしい", "良かった", "好き", "幸せ", "すごい",
             "やった", "成功", "楽しみ", "おいし", "美味", "面白", "素晴らし", "素敵", "感謝",
             "ありがとう", "うれしい", "最高", "うまくい", "晴れ晴れ")


def _mood(t: str) -> str:
    if any(x in t for x in _MOOD_NEG):
        return "negative"
    if any(x in t for x in _MOOD_POS):
        return "positive"
    if re.search(r"(たい|ほしい|したい|しよう)$", t.rstrip("。！？")):
        return "positive"
    return "neutral"


# --------------------------------------------------------------------------- #
# 「語そのもの」に関する質問か（辞書情報を使って良いかの門番）
# --------------------------------------------------------------------------- #
_WORD_INFO_ASK = ("reading", "word_property", "synonym", "antonym", "word_list", "example")
_WORD_INFO_MARK = re.compile(
    r"(とは|って何|ってどんな|読み方|怎么|何と読む|なんて読む|よみ|ふりがな|読みは|読みを|"
    r"何文字|なんもじ|何音|拍数|何拍|画数|品詞|活用|活用形|スペル|綴り|ローマ字|"
    r"意味は|語義|ことばの意味|どんな語|どんな言葉)")
_ANSWERED_BY_LEXICON = re.compile(r"(拍|文字数|読み|意味|語源|品詞|活用)")


def wants_word_info(text: str, frame=None) -> bool:
    """発話が *語そのもの* について聞いているか。

    知識ベースに話題が無いとき、v3 は「読み・拍・品詞」を返して誤魔化していました。
    語を素材として尋ねている場合だけ辞書を使い、それ以外は別の経路（検索・生成）に
    回すための判定です。
    """
    t = normalize(str(text or ""))
    if frame is not None and str(getattr(frame, "ask", "") or "") in _WORD_INFO_ASK:
        return True
    head = t.strip("。！？!?、 ")
    if len(head) > 34:            # 長い文は語彙の質問ではなく *内容* の質問
        return False
    if not _WORD_INFO_MARK.search(head):
        return False
    # 「天気とは」のように語が既 knowledge に在る場合は、そちらを先に使う（呼び出し側の役目）
    return bool(_ANSWERED_BY_LEXICON.search(head) or re.search(r"(とは|って何|読み|拍|品詞|活用)", head))


def short(text: str, limit: int = 24) -> str:
    t = normalize(text)
    return t if len(t) <= limit else t[:limit] + "…"


__all__ = ["build_frame", "classify_ask", "detect_act", "extract_entities", "opaque_reason",
           "short", "ASK_KINDS", "turn_word", "wants_word_info"]
