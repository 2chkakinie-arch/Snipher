"""文章を 0 から組み立てる作曲器（Composer）。

「定型文を並べてお茶を濁す」のをやめるための層です。Snipher は次の順で応答を作ります。

    1. 発話を解析する（内容語 / 問いの型 / 意図 / 気分 / 数字 / 自分への質問か）
    2. 知識ベースに材料があるか確かめる（無ければ「知らない」と言う）
    3. **文を組み立てる**（材料 + 発話の語をスロットに入れて、文法の枠で文を作る）
    4. 検証する（文末・重複・長さ・未展開のスロット・文体の一貫性）
    5. 通らなかった枠は捨てて、次の枠で作る（必ず何かを返す）

どの応答にも「相手の発話から取った語」か「知識ベースの根拠」のどちらかが入るので、
話題と無関係な文や、毎回同じ相づちが出ることはありません。

枠（frame）は同じ計画にいくつも用意してあり、**会話の回数で回す**ので、
同じ質問をされても前の応答と同じ文にはなりません。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .knowledge import GENERIC_ALIASES, KnowledgeBase, content_words, question_type
from .polisher import Polisher

# ---------------------------------------------------------------------- #
# 発話の解析
# ---------------------------------------------------------------------- #
_NEG_MARKS = ("つらい", "辛い", "疲れた", "疲れ", "悲しい", "かなしい", "寂しい", "さみしい",
              "不安", "心配", "怖い", "こわい", "痛い", "苦しい", "くるしい",
              "眠れない", "寝られない", "嫌", "いや", "失敗", "怒", "イライラ", "いらいら",
              "困っ", "困る", "大変", "しんどい", "落ち込", "無理", "飽き", "つまらない",
              "退屈", "足りない", "できない", "分からない", "わからない", "迷う", "迷っ",
              # 過去形・語幹（つらかっ / 悲しかっ / だるい …）も拾う
              "つらかっ", "辛かっ", "悲しかっ", "痛かっ", "苦しかっ", "怖かっ", "こわかっ",
              "だるい", "しんど", "まずい", "不味", "最悪", "嫌だ", "いやだ", "眠れなかっ",
              "寂しかっ", "さみしかっ", "怒っ", "腹が立つ", "疲れる", "壊れ", "失敗し")
_POS_MARKS = ("嬉しい", "うれしい", "楽しい", "たのしい", "良かった", "よかった", "好き",
              "ありがとう", "幸せ", "しあわせ", "上手", "すごい", "やった", "成功", "楽しみ",
              # 語幹（楽し / 嬉し / おいし …）で過去形・連用形も拾う
              "楽し", "嬉し", "うれし", "おいし", "美味", "面白", "よかっ", "素晴らし",
              "すばらし", "素敵", "すてき", "きれい", "綺麗", "感動", "ありがたい", "うまくい",
              "上手に", "成功し", "褒め", "ほめ")

_GREETING = ("こんにちは", "こんばんは", "おはよう", "はじめまして", "初めまして", "どうも",
             "ごきげんよう", "やあ", "もしもし", "おつかれ", "お疲れ")
_THANKS = ("ありがとう", "ありがと", "感謝", "サンキュー", "さんきゅー", "助かり", "たすかり",
           "おかげ", "ありがとう")
_APOLOGY = ("ごめん", "すみません", "申し訳", "もうしわけ", "失礼", "しつれい", "許して", "詫び")
_FAREWELL = ("さようなら", "さよなら", "またね", "バイバイ", "ばいばい", "おやすみ", "じゃあね",
             "また明日", "失礼します", "お先に")
_PRAISE = ("すごい", "凄い", "えらい", "偉い", "上手", "うまい", "素晴らしい", "すばらしい",
           "いいね", "良いね", "天才", "さすが", "完璧", "助かる", "便利")
_AGREE = ("その通り", "たしかに", "確か", "わかる", "分かる", "賛成", "そうそう", "なるほど",
          "そうだね", "ですね", "同感", "まったく")
_DISAGREE = ("違う", "ちがう", "間違っ", "まちがっ", "そうではない", "反対", "ちがう", "誤り",
             "おかしい", "納得できない")
_IDENTITY = ("あなた", "お前", "君は", "きみは", "何者", "なにもん", "誰ですか", "だれですか",
             "名前", "snipher", "スニファー", "aiなの", "人間", "ロボットなの", "bot")
_CAPABILITY = ("何ができる", "なにができる", "できること", "手伝って", "手伝い", "助けて",
               "使い方", "何をすれば", "機能")
_QUESTION_WORDS = ("何", "なに", "なぜ", "どうして", "いつ", "どこ", "だれ", "誰", "どう",
                   "どれ", "いくつ", "いくら", "どんな", "どの", "かどうか", "か？", "ですか")

_NUM = re.compile(r"[0-9０-９]+")
_ASCII_ONLY = re.compile(r"^[a-zA-Z0-9\s\W_]+$")
_KANA_RUN = re.compile(r"[ぁ-ん]{1,2}")

_SELF_WORDS = {"あなた", "snipher", "スニファー", "きみ", "君", "お前", "自分"}

# 内容語が取れなくても意味が分かる意図（＝「読み取れない入力」扱いしない）
import re as _re_mod

_VERBISH = _re_mod.compile(r"[るいうくぐすつぬぶむえけせてねべめれ]")
_KANA_CHARS = _re_mod.compile(r"[ぁ-ん]")
# 述語（動詞・形容詞・助動詞）で終わる日本語
_PRED_END = _re_mod.compile(
    r"(た|だ|い|ない|る|う|く|ぐ|す|つ|ぬ|ぶ|む|えた|かった|ている|ています|てます|"
    r"です|ます|たい|たく|たいです|たいです。|よう|らしい|みたい|そう|ちたい|ましょう|たいの|"
    r"ほしい|欲しい|こわい|怖い|つらい|辛い|いたい|痛い)$")

# 述語だけの発話（「疲れた」「眠い」）への受け方。気分に合わせる。
PRED_FRAMES_NEG = [
    "{p}んですね。少し休んでください。何があったのか、話せる範囲で教えてください。",
    "{p}のはつらいですね。無理はしないでください。いつ頃からですか。",
    "{p}んですね。そういうときもあります。今日はもう切り上げますか。",
]
PRED_FRAMES_POS = [
    "{p}んですね。よかったです。どんなところが良かったですか。",
    "{p}のは素敵ですね。もう少し聞かせてください。",
    "{p}んですね。それは嬉しいです。どうしてそう思いましたか。",
]
PRED_FRAMES_NEUTRAL = [
    "{p}んですね。もう少し詳しく教えてください。",
    "{p}んですか。その話、興味があります。どんな様子でしたか。",
    "{p}んですね。そこから何が変わりましたか。",
]

# 応答に「相手の言葉」として埋め込むには相応しくない語
# （助詞・形式名詞・相槌語。知識ベースの別名に入っていることがあるので明示的に弾く）
_ECHO_SKIP = frozenset({
    "こと", "もの", "ため", "よう", "まま", "ほど", "ずつ", "など", "しか", "だけ",
    "ばかり", "みたい", "らしい", "そう", "という", "について", "に関して", "に対して",
    "から", "より", "まで", "とき", "場合", "ところ", "わけ", "はず", "つもり", "ほう",
    "これ", "それ", "あれ", "この", "その", "あの", "ここ", "そこ", "あそこ", "どこ",
    "なに", "何", "誰", "いつ", "どう", "なぜ", "話", "話し", "話して", "教えて",
    "いう", "言う", "思う", "思うけど", "すごい", "いい", "良い", "だめ", "ダメ",
    "って何", "とは", "って", "のが", "のは", "って何？", "についてのこと",
})

# 分かち書きの断片に助詞がくっ付いてしまうことがある（「が好き」「って何」）
_PARTICLE_HEAD = re.compile(r"^(が|は|を|に|で|と|も|へ|の|や|って|から|まで|より)")
# 助詞を落としても助詞句のままの語（主語にはできない）
_PARTICLE_PHRASES = frozenset({
    "について", "ついて", "に対して", "に関して", "によって", "にとって", "としても",
    "という", "なのか", "のか", "たい", "よう", "ので", "のに", "けど", "から",
    "まで", "より", "しか", "さえ", "ほど", "くらい", "ぐらい", "など", "なり",
})


_SOCIAL_INTENTS = frozenset({"greeting", "thanks", "apology", "praise", "farewell",
                             "identity", "capability", "agreement", "disagreement"})


@dataclass
class Utterance:
    """発話の解析結果（composer の入力）。"""

    text: str = ""
    norm: str = ""
    words: list[str] = field(default_factory=list)      # 内容語（辞書の語 + 未知の断片）
    kb_words: list[str] = field(default_factory=list)   # 知識ベースに当たった語
    numbers: list[str] = field(default_factory=list)
    ascii_words: list[str] = field(default_factory=list)
    qtype: str = "general"
    intent: str = "statement"
    mood: str = "neutral"                               # neutral / negative / positive
    is_question: bool = False
    is_opaque: bool = False                             # 内容語が 1 つも無い（数字・記号だけ等）
    self_ref: bool = False                              # 自分（Snipher）への質問
    echo: str = ""                                      # 応答に埋め込む「相手の語」
    echo2: str = ""
    pred: str = ""                                      # 述語だけの発話（「疲れた」等）
    length: int = 0

    def subject(self) -> str:
        if self.echo:
            return self.echo
        for w in self.words:
            if w not in _ECHO_SKIP and len(w) >= 2:
                return w
        for w in self.words:
            if w not in _ECHO_SKIP:
                return w
        return ""


def detect_intent(text: str) -> str:
    t = text.lower()
    flat = re.sub(r"[\s。、！？!?・…]", "", t)
    for kw in _GREETING:
        if kw.lower() in flat:
            return "greeting"
    for kw in _THANKS:
        if kw.lower() in flat:
            return "thanks"
    for kw in _APOLOGY:
        if kw.lower() in flat:
            return "apology"
    for kw in _FAREWELL:
        if kw.lower() in flat:
            return "farewell"
    for kw in _DISAGREE:
        if kw.lower() in flat:
            return "disagreement"
    for kw in _PRAISE:
        if kw.lower() in flat:
            return "praise"
    for kw in _CAPABILITY:
        if kw.lower() in flat:
            return "capability"
    for kw in _IDENTITY:
        if kw.lower() in flat:
            return "identity"
    for kw in _AGREE:
        if kw.lower() in flat:
            return "agreement"
    return "statement"


# 願望（〜たい / 〜よう / 〜ほしい）は、気分的には前向きな発話
_DESIRE_END = re.compile(r"(たい|たかった|よう|ようかな|ほしい|欲しい|てほしい|たいな|たいです)$")


def detect_mood(text: str) -> str:
    t = (text or "").strip().rstrip("。！？!?.,、 ")
    if any(m in t for m in _NEG_MARKS):
        return "negative"
    if any(m in t for m in _POS_MARKS):
        return "positive"
    if _DESIRE_END.search(t):
        return "positive"
    return "neutral"


# ---------------------------------------------------------------------- #
# 枠（frame）の在庫
# ---------------------------------------------------------------------- #
# どの枠も「相手の発話から取った語」か「知識ベースの材料」を必ず 1 つ含む。
UNKNOWN_FRAMES = (
    "「{w}」について、確かだと言える情報を私は持っていません。{ask}",
    "「{w}」は私の知識ベースに無い言葉です。{ask}",
    "{w}のことですね。手元に確かな材料が無いので、断定は避けます。{ask}",
    "「{w}」には答えを出せませんでした。{ask}",
    "{w}について、根拠のあることが一つも言えません。{ask}",
    "「{w}」は聞いたことがありますが、確かだと断言できる材料がありません。{ask}",
)
UNKNOWN_ASKS = (
    "それが何なのか、一言で教えてもらえますか。",
    "どの部分を知りたいのか教えてください。",
    "どんな文脈で出てきた言葉ですか。",
    "もう少し文で書いてもらえれば、そこから一緒に整理します。",
    "何を知りたいのかを、具体的に書いてもらえますか。",
)
UNKNOWN_ASKS_NEG = (
    "いま一番困っていることを、一言で書いてもらえますか。",
    "何がつらいのか教えてもらえれば、そこから考えます。",
    "状況を一行でいいので教えてください。",
)

OPAQUE_FRAMES = (
    "「{raw}」だけでは、何を指しているのか分かりませんでした。{ask}",
    "送ってもらった「{raw}」の意味を読み取れませんでした。{ask}",
    "「{raw}」に対応する話題が見つかりませんでした。{ask}",
)
OPAQUE_NUM_FRAMES = (
    "「{raw}」は数字だけのようです。年齢や数量、計算の途中でしょうか。{ask}",
    "「{raw}」という数字を受け取りました。それが何の数なのか教えてください。{ask}",
    "数字の「{raw}」だけでは、何を答えるべきか分かりませんでした。{ask}",
)
OPAQUE_LATIN_FRAMES = (
    "「{raw}」は私には読めない文字列です。日本語の文で書いてもらえますか。",
    "「{raw}」の意味が取れませんでした。何を調べたいのか、日本語で教えてください。",
)
OPAQUE_ASKS = (
    "文にしてもらえれば、それに合わせて答えます。",
    "何について知りたいですか。",
    "一言だけ言葉を足してもらえれば、そこから広げます。",
)

ACK_NEG = (
    "それはつらいですね。", "大変でしたね。", "それは困りますね。",
    "無理はしないでください。", "落ち着いて、一つずつ見ましょう。", "気持ちは分かります。",
)
ACK_POS = (
    "それは良かったです。", "いいですね。", "素敵です。", "うらやましいです。",
)
ACK_NEUTRAL = (
    "なるほど。", "ふむふむ。", "その話、もう少し聞かせてください。", "",
    "", "",      # 何も付けない枠を多めにしておく（くどくならないように）
)

STATEMENT_FRAMES = (
    "{w}のことですね。{ask}",
    "{w}について、もう少し教えてください。{ask}",
    "{w}の話は興味があります。{ask}",
)
STATEMENT_GENERIC_FRAMES = (
    "その話、もう少し聞かせてください。{ask}",
    "なるほど。{ask}",
    "その話、興味があります。{ask}",
)
STATEMENT_ASKS = (
    "どんなところが印象に残りましたか。",
    "いつ頃からそうですか。",
    "その中でいちばん大事なのはどこですか。",
    "どうしてそう思ったのですか。",
    "他に気になっていることはありますか。",
)

SELF_FRAMES = (
    "私は Snipher です。日本語の質問に答えたり、文章を組み立てたりしています。{ask}",
    "私は文章を作って応答する AI で、Snipher と呼ばれています。{ask}",
    "Snipher という名前の AI です。知識ベースに根拠がある話だけを答えるようにしています。{ask}",
)
SELF_ASKS = (
    "何を聞いてみたいですか。", "試したい質問があればどうぞ。", "何か手伝いましょうか。",
)

CANNOT_FRAMES = (
    "そこは私には分かりません。{ask}",
    "確かな材料が無いので、答えられません。{ask}",
    "その情報は持っていません。{ask}",
)

THANKS_FRAMES = (
    "どういたしまして。{open}",
    "役に立てたなら嬉しいです。{open}",
    "いえいえ、こちらこそ。{open}",
)
APOLOGY_FRAMES = (
    "気にしないでください。{open}",
    "大丈夫です。{open}",
    "こちらこそ、至らないところがあります。{open}",
)
PRAISE_FRAMES = (
    "ありがとうございます。{open}",
    "そう言ってもらえると、次の応答も丁寧になります。{open}",
    "褒められた部分を、もう少し具体的に教えてもらえると嬉しいです。{open}",
)
FAREWELL_FRAMES = (
    "また話しましょう。{open}",
    "ありがとうございました。{open}",
    "いつでも声をかけてください。{open}",
)
OPEN_QUESTIONS = (
    "他に気になることはありますか。",
    "続きがあればどうぞ。",
    "また何かあれば聞いてください。",
    "",
    "",
)

FOLLOW_CONNECTORS = ("", "", "ちなみに、", "そういえば、", "ひとつ聞きたいのですが、")


# ---------------------------------------------------------------------- #
# 検証
# ---------------------------------------------------------------------- #
_BAD_PATTERNS = (
    "ですです", "ますます", "ましたました", "ませんません", "はは", "がが", "をを",
    "。。", "、、", "？？", "！！", "いますです", "ますです", "ですます", "たです",
    "「」", "{}", "{w}", "{ask}", "{raw}", "{open}", "None", "nan",
)
# 「脂ののった」「風の音」のように、の + のX が正当な場合
_NO_NO_OK = re.compile(r"のの[っりらろれいうえおん]")

_DANGLING_ENDS = ("を", "が", "は", "に", "で", "と", "も", "へ", "の", "や", "から", "まで", "より")
# 助詞で終わっていても、それ自体で complete な定型の挨拶・感嘆
_DANGLING_OK = frozenset({
    "こんにちは", "こんばんは", "おはよう", "さようなら", "もしもし", "では", "それでは",
    "はじめまして", "ありがとう", "すみません", "いただきます", "ごちそうさま", "おかえり",
    "ただいま", "よろしく", "おやすみ", "いってきます", "いってらっしゃい", "なるほど",
    "おかえりなさい", "おつかれさま", "ごめん", "わかった", "了解",
})


def validate(text: str, *, max_len: int = 170, min_len: int = 6) -> tuple[bool, str]:
    """組み立てた文が日本語として成立しているかを検査する。→ (ok, 理由)"""
    t = (text or "").strip()
    if len(t) < min_len:
        return False, "too_short"
    if len(t) > max_len:
        return False, "too_long"
    if not t.endswith(("。", "！", "？", "!", "?")):
        return False, "no_terminal"
    if "のの" in t and not _NO_NO_OK.search(t):
        return False, "bad_pattern:のの"
    for bad in _BAD_PATTERNS:
        if bad in t:
            return False, f"bad_pattern:{bad}"
    # 文の途中が助詞で切れていたら、組み立てに失敗している
    for part in re.split(r"[。！？!?]", t):
        part = part.strip().rstrip("、")
        if not part:
            continue
        if part.endswith(_DANGLING_ENDS) and part not in _DANGLING_OK:
            return False, f"dangling:{part[-4:]}"
    # 3 回以上繰り返す 2-gram はループの兆候
    grams = [t[i:i + 2] for i in range(len(t) - 1)]
    seen: dict[str, int] = {}
    for g in grams:
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= 4:
            return False, f"loop:{g}"
    # 文体の一貫性: です/ます と だ/た が混ざっていないか
    polite = len(re.findall(r"(ます|です|ました|ません|でしょう|ください)", t))
    plain = len(re.findall(r"(だ。|た。|ない。|る。|だ、|た、)", t))
    if polite and plain >= 2:
        return False, "register_mix"
    return True, "ok"


# ---------------------------------------------------------------------- #
# 応答
# ---------------------------------------------------------------------- #
@dataclass
class Reply:
    text: str
    plan: str
    confidence: float = 0.5
    knowledge: dict | None = None
    sentences: list[str] = field(default_factory=list)
    frame: str = ""
    notes: dict = field(default_factory=dict)


class Composer:
    """発話と知識ベースの材料から、日本語の応答を組み立てる。"""

    def __init__(self, kb: KnowledgeBase | None = None, polisher: Polisher | None = None,
                 lm=None, seed: int = 0):
        self.kb = kb if kb is not None else KnowledgeBase.shared()
        self.polisher = polisher or Polisher()
        self.lm = lm                    # 任意: n-gram LM（候補の採点に使う）
        self.seed = seed
        self.turn = 0

    # ------------------------------------------------------------------ #
    def analyze(self, text: str) -> Utterance:
        u = Utterance(text=text)
        raw = str(text or "").strip()
        u.norm = re.sub(r"\s+", " ", raw).lower()
        u.length = len(raw)
        u.numbers = _NUM.findall(raw)
        u.ascii_words = [m.group() for m in re.finditer(r"[A-Za-z][A-Za-z0-9_+#.\-]*", raw)]
        try:
            u.words = [w for w in content_words(raw, self.kb.index) if w not in GENERIC_ALIASES]
        except Exception:  # noqa: BLE001
            u.words = []
        u.qtype = question_type(raw)
        u.intent = detect_intent(raw)
        u.mood = detect_mood(raw)
        u.is_question = raw.rstrip().endswith(("?", "？")) or u.qtype.endswith("_q") \
            or u.qtype in ("def", "why", "how", "when", "where", "who", "cost", "count") \
            or any(w in raw for w in _QUESTION_WORDS)
        u.self_ref = any(w in u.norm for w in ("あなた", "snipher", "スニファー", "きみ", "君は", "お前"))
        # 意味のある語が 1 つも無ければ「読み取れない入力」
        real = [w for w in u.words if len(w) >= 1 and not w.isdigit()]
        u.is_opaque = (not real) and (not u.numbers or len(raw) <= 6) or bool(_ASCII_ONLY.match(raw))
        if not real and not u.ascii_words:
            u.is_opaque = True
        # ただし「疲れた」「眠い」「寒かった」のような短い述語の日本語は読み取れている。
        # 語彙に無くても気持ちとして受けられるので、opaque にはしない。
        if not u.ascii_words and not u.numbers and not u.is_question:
            body = raw.strip().rstrip("。！？!?.,、 ")
            if 2 <= len(body) <= 8 and len(_KANA_CHARS.findall(body)) >= 2 \
                    and _PRED_END.search(body):
                u.pred = body
                u.is_opaque = False
        # 挨拶・礼・謝罪・自分への質問は、内容語が無くても「読めている」
        if u.intent in _SOCIAL_INTENTS or u.self_ref:
            u.is_opaque = False
        # 応答に埋め込む語（相手の言葉をそのまま返すので、話題がずれない）
        # ※ u.words は GENERIC_ALIASES を落とした一覧。echo 候補は「相手の言葉」なので
        #    generic かどうかは関係なく、分かち書きの結果をそのまま使う（歴史 / 好き 等）。
        try:
            seg_words = [w for w in content_words(raw, self.kb.index) if not w.isdigit()]
        except Exception:  # noqa: BLE001
            seg_words = list(real)
        all_cands = [w for w in seg_words if len(w) <= 14] or u.ascii_words
        # 助詞・形式名詞（「について」「こと」など）は主語にできないので避ける
        def _usable(w: str) -> bool:
            return bool(w) and w not in _ECHO_SKIP and w not in _PARTICLE_PHRASES

        cleaned: list[str] = []
        for w in all_cands:
            if not _usable(w):
                continue                      # 「について」「って何」など
            w2 = _PARTICLE_HEAD.sub("", w, count=1)
            if w2 != w and _usable(w2) and len(w2) >= 2:
                cleaned.append(w2)            # 「が好き」→「好き」
            else:
                cleaned.append(w)             # 「雨」のような 1 文字の名詞はそのまま
        cands = cleaned
        kb_words: list[str] = []
        try:
            kb_words = [w for w in cands if self.kb.index.topics_of(w)]
        except Exception:  # noqa: BLE001
            kb_words = []
        u.kb_words = kb_words

        def _rank(w: str) -> tuple[int, int]:
            """名詞らしく、知識ベースが知っている語ほど高く。同点なら文末側を優先。

            「歩いていたら急に雨が」→ 雨（歩いて・急に は動詞/副詞なので低い）
            """
            score = 0
            if w in kb_words:
                score += 4
            if len(w) >= 2:
                score += 1
            if not _VERBISH.match(w[-1]) and not w.endswith(("て", "た", "です", "ます")):
                score += 2
            return (score, -cands.index(w))

        if cands:
            ordered = sorted(cands, key=_rank, reverse=True)
            u.echo = ordered[0]
            u.echo2 = next((w for w in ordered[1:] if w != u.echo), "")
        else:
            u.echo = u.echo2 = ""
        if u.pred and u.echo and (u.echo == u.pred or u.echo in u.pred):
            u.echo = u.echo2 = ""
        return u

    # ------------------------------------------------------------------ #
    def compose(self, text: str, *, material: dict | None = None,
                history: list[dict] | None = None, turn: int | None = None) -> Reply:
        """応答を組み立てる。material が無ければ自分で知識ベースを引く。"""
        self.turn = int(turn if turn is not None else self.turn + 1)
        u = self.analyze(text)
        history = history or []
        prev = _previous_assistant_texts(history)

        if material is None:
            try:
                material = self.kb.answer(text)
            except Exception:  # noqa: BLE001
                material = None
        if material is not None:
            u.is_opaque = False        # 材料が引けたなら「読み取れている」

        # ---- 計画を選ぶ ------------------------------------------------ #
        actionable = bool(material) and str(material.get("usage") or "") in {
            "qa", "how", "why", "tips", "fact"}
        # 「疲れた」「うれしい」= 知識ではなく気持ちの発話。名詞が当たっていなければ
        # 定義を読み上げるより、まず受け止める。
        feelings = bool(u.pred) and not u.kb_words and not actionable
        if feelings:
            plan = (self._plan_predicate(u, prev)
                    or (self._plan_knowledge(u, material, prev) if material else None)
                    or self._plan_statement(u, prev))
        elif material is not None:
            plan = self._plan_knowledge(u, material, prev)
        elif u.is_opaque:
            plan = self._plan_opaque(u, prev)
        elif u.intent in ("thanks", "apology", "praise", "farewell"):
            plan = self._plan_social(u, prev)
        elif u.intent == "identity" or (u.self_ref and u.is_question):
            plan = self._plan_self(u, prev)
        elif u.intent == "greeting":
            plan = self._plan_greeting(u, prev)
        elif u.is_question:
            plan = self._plan_unknown(u, prev)
        else:
            plan = self._plan_statement(u, prev)

        if plan is None:                                  # 最後の安全網
            plan = self._plan_unknown(u, prev) or Reply(
                text="もう少し言葉を足してもらえますか。", plan="safety", confidence=0.2)
        plan.notes.update({"intent": u.intent, "qtype": u.qtype, "mood": u.mood,
                           "opaque": u.is_opaque, "words": u.words[:6]})
        return plan

    # ------------------------------------------------------------------ #
    # 各計画
    # ------------------------------------------------------------------ #
    def _plan_knowledge(self, u: Utterance, m: dict, prev: list[str]) -> Reply | None:
        body = str(m.get("text") or "").strip()
        if not body:
            return None
        topic = str(m.get("topic") or "")
        ack = ""
        social = u.intent in _SOCIAL_INTENTS
        if not social:
            if u.mood == "negative" and m.get("qtype") not in ("def",):
                ack = _pick(ACK_NEG, self.turn, prev)
            elif u.mood == "positive" and u.qtype in ("general", "general_q"):
                ack = _pick(ACK_POS, self.turn, prev)
        follow = str(m.get("followup") or "").strip()
        sentences = [s for s in (ack, body) if s]
        # 本文にすでに問いがあるなら、問いを二重に付けない
        has_question = bool(re.search(r"(か|かな|かしら|でしょう)[。！？!?…]", body)) or \
            body.rstrip().endswith(("?", "？"))
        if follow and not has_question and len(body) + len(follow) < 130 \
                and u.qtype not in ("how",):
            conn = _pick(FOLLOW_CONNECTORS, self.turn + 1, prev)
            sentences.append(f"{conn}{follow}" if conn else follow)
        text = _join(sentences)
        ok, why = validate(text)
        if not ok:                                        # 長すぎる/壊れた → 本体だけに戻す
            text = _join([s for s in (ack, body) if s])
            ok, why = validate(text, max_len=220)
            if not ok:
                text = body
        text = self._polish(text)
        conf = float(m.get("confidence", 0.7))
        return Reply(text=text, plan=f"knowledge:{m.get('field', 'answer')}", confidence=conf,
                     knowledge={"topic": topic, "score": m.get("score"),
                                "coverage": m.get("coverage"), "usage": m.get("usage"),
                                "qtype": m.get("qtype"), "field": m.get("field"),
                                "via": m.get("via"), "confidence": conf},
                     sentences=_split(text),
                     notes={"ack": bool(ack), "followup": bool(follow), "why": why})

    def _plan_unknown(self, u: Utterance, prev: list[str]) -> Reply | None:
        w = u.echo or u.subject() or "そのこと"
        asks = UNKNOWN_ASKS_NEG if u.mood == "negative" else UNKNOWN_ASKS
        for i in range(len(UNKNOWN_FRAMES)):
            frame = _rotate(UNKNOWN_FRAMES, self.turn + i)
            ask = _rotate(asks, self.turn + i * 2)
            text = self._polish(frame.format(w=w, ask=ask))
            ok, _why = validate(text)
            if ok and text not in prev:
                return Reply(text=text, plan="unknown_topic", confidence=0.30,
                             sentences=_split(text), frame=frame,
                             notes={"echo": w})
        # どれも通らなければ最短の正直な応答に落ちる
        return Reply(text=f"「{w}」について、私には確かな情報がありません。"
                          "何を知りたいのか教えてください。",
                     plan="unknown_topic", confidence=0.25, notes={"echo": w})

    def _plan_opaque(self, u: Utterance, prev: list[str]) -> Reply | None:
        raw = u.text.strip()[:24] or "その入力"
        frames = OPAQUE_NUM_FRAMES if u.numbers else (
            OPAQUE_LATIN_FRAMES if u.ascii_words else OPAQUE_FRAMES)
        for i in range(len(frames)):
            frame = _rotate(frames, self.turn + i)
            ask = _rotate(OPAQUE_ASKS, self.turn + i)
            text = self._polish(frame.format(raw=raw, ask=ask))
            ok, _why = validate(text)
            if ok and text not in prev:
                return Reply(text=text, plan="opaque_input", confidence=0.18,
                             sentences=_split(text), frame=frame,
                             notes={"raw": raw, "numbers": u.numbers})
        return Reply(text=f"「{raw}」の意味が分かりませんでした。文で教えてください。",
                     plan="opaque_input", confidence=0.15, notes={"raw": raw})

    def _plan_statement(self, u: Utterance, prev: list[str]) -> Reply | None:
        w = u.echo or u.subject()
        if u.pred and not u.echo:
            return self._plan_predicate(u, prev) or self._plan_opaque(u, prev)
        generic = not w
        if generic:
            # 内容語はあるが主語にできない（機能語だけ）→ 語を埋めずに受ける
            if not u.words:
                return self._plan_opaque(u, prev)
            w = "その話"
        frames = STATEMENT_GENERIC_FRAMES if generic else STATEMENT_FRAMES
        ack = _pick(ACK_NEG, self.turn, prev) if u.mood == "negative" else \
            (_pick(ACK_POS, self.turn, prev) if u.mood == "positive" else _pick(ACK_NEUTRAL, self.turn, prev))
        for i in range(len(frames)):
            frame = _rotate(frames, self.turn + i)
            ask = _rotate(STATEMENT_ASKS, self.turn + i * 3)
            text = self._polish(_join([s for s in (ack, frame.format(w=w, ask=ask)) if s]))
            ok, _why = validate(text)
            if ok and text not in prev:
                return Reply(text=text, plan="statement", confidence=0.34,
                             sentences=_split(text), frame=frame, notes={"echo": w})
        return None

    def _plan_predicate(self, u: Utterance, prev: list[str]) -> Reply | None:
        """述語だけの発話（「疲れた」「眠い」）を、気持ちとして受け取る。"""
        frames = {"negative": PRED_FRAMES_NEG, "positive": PRED_FRAMES_POS}.get(
            u.mood, PRED_FRAMES_NEUTRAL)
        for i in range(len(frames)):
            frame = _rotate(frames, self.turn + i)
            text = self._polish(frame.format(p=u.pred))
            ok, _why = validate(text)
            if ok and text not in prev:
                return Reply(text=text, plan="statement", confidence=0.42,
                             sentences=_split(text), frame=frame, notes={"pred": u.pred})
        return None

    def _plan_self(self, u: Utterance, prev: list[str]) -> Reply | None:
        # 知識ベースに自分自身の項目（自己紹介）があればそれを優先する
        try:
            m = self.kb.answer(u.text) or self.kb.answer("あなたは何")
        except Exception:  # noqa: BLE001
            m = None
        if m and m.get("id") in ("selfintro", "greeting"):
            r = self._plan_knowledge(u, m, prev)
            if r:
                r.plan = "self:" + r.plan
                return r
        for i in range(len(SELF_FRAMES)):
            frame = _rotate(SELF_FRAMES, self.turn + i)
            text = self._polish(frame.format(ask=_rotate(SELF_ASKS, self.turn + i)))
            ok, _why = validate(text)
            if ok and text not in prev:
                return Reply(text=text, plan="self", confidence=0.6,
                             sentences=_split(text), frame=frame)
        return None

    def _plan_greeting(self, u: Utterance, prev: list[str]) -> Reply | None:
        try:
            m = self.kb.answer(u.text)
        except Exception:  # noqa: BLE001
            m = None
        if m:
            r = self._plan_knowledge(u, m, prev)
            if r:
                r.plan = "greeting:" + r.plan
                return r
        base = "こんにちは" if "こんばん" not in u.norm else "こんばんは"
        if "おはよう" in u.norm:
            base = "おはようございます"
        openers = ("今日はどうしましたか。", "何か話したいことはありますか。",
                   "調子はどうですか。", "どんな用件でしょうか。")
        for i in range(len(openers)):
            text = self._polish(f"{base}。{_rotate(openers, self.turn + i)}")
            ok, _why = validate(text)
            if ok and text not in prev:
                return Reply(text=text, plan="greeting", confidence=0.8,
                             sentences=_split(text))
        return Reply(text=f"{base}。", plan="greeting", confidence=0.75)

    def _plan_social(self, u: Utterance, prev: list[str]) -> Reply | None:
        try:
            m = self.kb.answer(u.text)
        except Exception:  # noqa: BLE001
            m = None
        if m and m.get("id") in ("thanks", "apology", "praise", "greeting", "feelings", "worry"):
            r = self._plan_knowledge(u, m, prev)
            if r:
                r.plan = f"{u.intent}:" + r.plan
                return r
        frames = {"thanks": THANKS_FRAMES, "apology": APOLOGY_FRAMES,
                  "praise": PRAISE_FRAMES, "farewell": FAREWELL_FRAMES}[u.intent]
        for i in range(len(frames)):
            frame = _rotate(frames, self.turn + i)
            text = self._polish(frame.format(open=_rotate(OPEN_QUESTIONS, self.turn + i * 2)))
            ok, _why = validate(text)
            if ok and text not in prev:
                return Reply(text=text, plan=u.intent, confidence=0.85,
                             sentences=_split(text), frame=frame)
        return None

    # ------------------------------------------------------------------ #
    def _polish(self, text: str) -> str:
        try:
            return self.polisher.polish(text, register="polite")["text"] or text
        except Exception:  # noqa: BLE001
            return text

    # ------------------------------------------------------------------ #
    def rank(self, candidates: list[str], *, min_len: int = 6) -> list[tuple[float, str]]:
        """候補を (n-gram LM の自然さ + 文法の妥当性) で並べ替える。"""
        out: list[tuple[float, str]] = []
        for c in candidates:
            c = (c or "").strip()
            if len(c) < min_len:
                continue
            ok, _why = validate(c, max_len=220)
            score = 0.5 if ok else 0.0
            if self.lm is not None and ok:
                try:
                    score = float(self.lm.confidence(c))
                except Exception:  # noqa: BLE001
                    pass
            out.append((score, c))
        out.sort(key=lambda t: -t[0])
        return out

    def accept(self, text: str, *, u: Utterance | None = None, min_conf: float = 0.30) -> bool:
        """ニューラルコアが作った文を、応答として採用してよいか判定する。"""
        t = (text or "").strip()
        if not t or len(t) < 6 or len(t) > 200:
            return False
        ok, _why = validate(t, max_len=200)
        if not ok:
            return False
        if self.lm is not None:
            try:
                if float(self.lm.confidence(t)) < min_conf:
                    return False
            except Exception:  # noqa: BLE001
                pass
        return True


# ---------------------------------------------------------------------- #
# ヘルパ
# ---------------------------------------------------------------------- #
def _rotate(seq: tuple[str, ...] | list[str], k: int) -> str:
    if not seq:
        return ""
    return seq[k % len(seq)]


def _pick(seq, k: int, prev: list[str]) -> str:
    """直前の応答と同じ文を避けて 1 つ選ぶ。"""
    n = len(seq)
    for i in range(n):
        cand = seq[(k + i) % n]
        if not cand:
            return cand
        if not any(cand in p for p in prev[-2:]):
            return cand
    return seq[k % n]


def _join(sentences: list[str]) -> str:
    out: list[str] = []
    for s in sentences:
        s = str(s or "").strip()
        if not s:
            continue
        if out and out[-1].endswith(("。", "！", "？")) and not s[0].isupper():
            out.append(s)
        else:
            out.append(s)
    text = "".join(out)
    return re.sub(r"。{2,}", "。", text)


def _split(text: str) -> list[str]:
    return [p for p in re.split(r"(?<=[。！？])", text or "") if p.strip()]


def _previous_assistant_texts(history: list[dict], limit: int = 4) -> list[str]:
    out: list[str] = []
    for m in reversed(history or []):
        if m.get("role") == "assistant":
            out.append(str(m.get("content", "")))
            if len(out) >= limit:
                break
    return list(reversed(out))
