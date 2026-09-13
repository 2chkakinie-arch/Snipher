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

import os
import re
from dataclasses import dataclass, field

from .knowledge import GENERIC_ALIASES, KnowledgeBase, content_words, question_type
from .polisher import Polisher
from .tasks import TaskRouter

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
              "寂しかっ", "さみしかっ", "怒っ", "腹が立つ", "疲れる", "壊れ", "失敗し",
              # 「どうしよう」は願望（〜よう）ではなく困りごとの声
              "どうしよう", "どうしよ", "困った", "どうしたらいい")
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
# 名詞らしい語尾（漢字・カタカナで終わる）。「聞いて」「知らん」は名詞にならない
_NOUN_END = _re_mod.compile(r"[\u4e00-\u9fff\u30a1-\u30f6ー]$")
_KANA_CHARS = _re_mod.compile(r"[ぁ-ん]")
# 述語（動詞・形容詞・助動詞）で終わる日本語
_PRED_END = _re_mod.compile(
    r"(た|だ|い|ない|る|う|く|ぐ|す|つ|ぬ|ぶ|む|えた|かった|ている|ています|てます|"
    r"です|ます|たい|たく|たいです|たいです。|よう|らしい|みたい|そう|ちたい|ましょう|たいの|"
    r"ほしい|欲しい|こわい|怖い|つらい|辛い|いたい|痛い)$")

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
    "ちょっ", "ちょっと", "少し", "少しだけ", "なんか", "なにか", "何か", "いろいろ",
    "やっぱり", "もちろん", "たぶん", "もしかして", "たとえば", "例えば",
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
        """主語にできる語（名詞らしい語）を 1 つ返す。無ければ ""。

        「聞いて」「知らん」のような活用形を主語にすると
        「聞いてについて、もう少し教えてください。」になってしまうので返さない。
        """
        if self.echo:
            return self.echo
        ok = [w for w in self.words
              if w not in _ECHO_SKIP and w not in _PARTICLE_PHRASES
              and not w.endswith(("て", "た", "ん", "ない", "です", "ます"))]
        for w in ok:
            if _NOUN_END.search(w) or w in self.kb_words:
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
# 検証
# ---------------------------------------------------------------------- #
_BAD_PATTERNS = (
    "ですです", "ましたました", "ませんません", "はは", "がが", "をを",
    "。。", "、、", "？？", "！！", "いますです", "ますです", "ですます",
    "「」", "{}", "{w}", "{ask}", "{raw}", "{open}", "None", "nan",
)
# 「良かったです」は正しい日本語。「食べたです」は間違い。
# 「ますます（益々）」も正しい副詞なので、重複は 2-gram のループ検査に任せる。
_BAD_TADESU = "たです"
# 英数字だけの 2-gram（コマンド名や略語の繰り返しは正常）
_ASCII_GRAM = re.compile(r"^[\x20-\x7e]{2}$")
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


# 名詞化語尾。これで終わる節はそれ自体が述語化できるので、吊り下げとは数えない。
_NOMINAL_ENDS = ("こと", "もの", "はず", "ため", "ほど", "わけ", "つもり", "よう", "のみ",
                 "ばかり", "限り", "最中", "場合", "必要", "予定", "確認")


def _mask_quoted(sentence: str) -> str:
    """「」で囲まれた引用部をユニークな印に置き換える。

    語を *説明する文*（「ぬるぬる猿」は「ぬるぬる」と「猿」を合わせた…）では、
    説明対象の語が 2 回以上引用されて 2-gram が繰り返される。それは正当な引用なので、
    ループ検査の前で引用部だけマスクする（各引用に番号を付けるので印自体は重複しない）。
    """
    if "「" not in sentence:
        return sentence
    out: list[str] = []
    i, k = 0, 0
    while i < len(sentence):
        ch = sentence[i]
        if ch == "「":
            j = sentence.find("」", i + 1)
            if j == -1:
                out.append(sentence[i:])
                break
            k += 1
            out.append(f"◆{k}◆")
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def validate(text: str, *, max_len: int = 170, min_len: int = 6) -> tuple[bool, str]:
    """組み立てた文が日本語として成立しているかを検査する。→ (ok, 理由)"""
    t = (text or "").strip()
    if len(t) < min_len:
        return False, "too_short"
    if len(t) > max_len:
        return False, "too_long"
    # コードブロックは *そのまま* 示すものなので、文末の句点検査は文章部分だけを見る。
    # （実行結果を表や配列で返す応答は、句点で閉じないのが正しい形。）
    if "```" not in t and not t.endswith(("。", "！", "？", "!", "?", "」", "）", "：", ":")):
        return False, "no_terminal"
    if "のの" in t and not _NO_NO_OK.search(t):
        return False, "bad_pattern:のの"
    if _BAD_TADESU in t.replace("かったです", ""):
        return False, "bad_pattern:たです"
    for bad in _BAD_PATTERNS:
        if bad in t:
            return False, f"bad_pattern:{bad}"
    # 引用符「」が片方だけ残っている文は、組み立ての途中漏れ
    if t.count("「") != t.count("」"):
        return False, "quote:unbalanced"
    # 文の途中が助詞で切れていたら、組み立てに失敗している
    for part in re.split(r"[。！？!?]", t):
        part = part.strip().rstrip("、")
        if not part:
            continue
        # 「〜とのこと」「どういうこと」は名辞化して成立するので、助詞止めと数えない
        if part.endswith(_DANGLING_ENDS) and part not in _DANGLING_OK \
                and not part.endswith(_NOMINAL_ENDS):
            return False, f"dangling:{part[-4:]}"
    # 3 回以上繰り返す 2-gram はループの兆候。
    # ただし英数字の並び（git init / git add / git commit の "it" など）は
    # 技術系の文で普通に繰り返されるので数えない。
    # 文そのものの反復（コピー＆ペーストで壊れた文）は先に弾く
    sentences = [x.strip() for x in re.split(r"(?<=[。！？!?])", t) if x.strip()]
    counts: dict[str, int] = {}
    for sent in sentences:
        counts[sent] = counts.get(sent, 0) + 1
        if counts[sent] >= 3:
            return False, f"loop:{sent[:2]}×3"
    # 2-gram の反復は文ごとに数える（文をまたいだ相同 2-gram をループと誤検知していた）。
    # 「ぬるぬる猿」のように *説明している語自体* を引用して繰り返す文は、引用部を
    # マスクしてから数える（引用の反復は生成のループではない）。
    for sentence in re.split(r"[。！？!?]", t):
        masked = _mask_quoted(sentence.strip())
        seen: dict[str, int] = {}
        for i in range(len(masked) - 1):
            g = masked[i:i + 2]
            if _ASCII_GRAM.match(g):
                continue
            seen[g] = seen.get(g, 0) + 1
            if seen[g] >= 4:
                return False, f"loop:{g}"
    # 文体の一貫性: です/ます と だ/た が混ざっていないか
    polite = len(re.findall(r"(ます|です|ました|ません|でしょう|ください)", t))
    # 「選びました、「ん」で…」のような丁寧語の連なりを常体と数えないように、
    # ます・です の直後の 「た」 は打ち消さない。
    plain = len(re.findall(r"(だ。|ない。|る。|だ、|(?<!まし)(?<!でし)(?<!ませ)(?<!い)た[。、])", t))
    if polite and plain >= 2:
        return False, "register_mix"
    return True, "ok"


# ---------------------------------------------------------------------- #
# 応答
# ---------------------------------------------------------------------- #
# 文末に立つ述語の種類（活用語終止・断定・終助詞・引用終わり）
_PREDICATE_ENDINGS = (
    "です", "ます", "だ", "である", "じゃない", "ない", "たい", "た", "る", "っている",
    "いる", "ある", "する", "できる", "い", "ね", "よ", "な", "さ", "わ",
    "かな", "かな。", "っす", "で、", "せ", "しろ", "せよ", "よう", "だろう", "と思う", "だと思う",
    "です！", "ます！", "だ！", ":", "：", "…",
)


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
                 lm=None, seed: int = 0, task_router: TaskRouter | None = None):
        self.kb = kb if kb is not None else KnowledgeBase.shared()
        self.polisher = polisher or Polisher()
        self.lm = lm                    # 任意: n-gram LM（候補の採点に使う）
        self.seed = seed
        self.turn = 0
        # 定型文より先に、計算・コード・比較・現在情報を扱う厳密な道具層。
        # TaskRouter は標準ライブラリだけで、必要な質問だけウェブへ出る。
        self.tasks = task_router or TaskRouter()
        self._web = None
        self._think = None

    # ------------------------------------------------------------------ #
    def web_grounding(self):
        """Snipher がインターネットに出る唯一の口（TaskRouter の検索器を共有する）。"""
        if self._web is None:
            try:
                from .ground.web import WebGrounding

                self._web = WebGrounding(getattr(self.tasks, "research", None))
            except Exception:  # noqa: BLE001
                self._web = False
        return self._web or None

    def think(self, text: str, *, history: list[dict] | None = None, turn: int | None = None,
              web: bool | None = None):
        """思考層（mind）を呼ぶ。使えない環境では None を返し、旧経路が引き継ぐ。"""
        if os.environ.get("SNIPHER_MIND", "1") in ("0", "off", "false"):
            return None
        if self._think is None:
            from .mind.think import think as _think

            self._think = _think
        return self._think(text, history=history or [], kb=self.kb, web=self.web_grounding(),
                           lm=self.lm, core=None, polisher=self.polisher, tasks=self.tasks,
                           web_flag=web, turn=turn)

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
                score += 4          # 知識ベースが知っている語がいちばん話題らしい
            if len(w) >= 2:
                score += 1
            if _NOUN_END.search(w):
                score += 2          # 漢字/カタカナ終わり = 名詞らしい
            if w.endswith(("て", "た", "です", "ます", "ん", "ない")):
                score -= 2          # 動詞・形容詞の活用形は主語にできない
            return (score, -cands.index(w))

        # 英字は分かち書きで小文字になるので、元の表記（Docker / Wi-Fi）に戻す
        if cands and u.ascii_words:
            upper = {w.lower(): w for w in u.ascii_words}
            cands = [upper.get(w.lower(), w) if w.isascii() else w for w in cands]
        if cands:
            scored = sorted(((_rank(w), w) for w in cands), key=lambda x: x[0], reverse=True)
            # 名詞らしくない語（「聞いて」「どうし」「らいい」のような活用の断片）は
            # 主語にしない。主語にできる語が無ければ空のまま → 汎用の受け方に落ちる。
            usable = [(w, sc[0]) for sc, w in scored if sc[0] >= 2]
            if usable:
                u.echo = usable[0][0]
                u.echo2 = next((w for w, _s in usable[1:] if w != u.echo), "")
            else:
                u.echo = u.echo2 = ""
        else:
            u.echo = u.echo2 = ""
        if u.pred and u.echo and (u.echo == u.pred or u.echo in u.pred):
            u.echo = u.echo2 = ""
        return u

    # ------------------------------------------------------------------ #
    def compose(self, text: str, *, material: dict | None = None,
                history: list[dict] | None = None, turn: int | None = None,
                web: bool | None = None) -> Reply:
        """応答を組み立てる。

        計算・コード・比較・現在情報は、知識ベースの話題当てより先に
        ``TaskRouter`` を通す。これが無いと、例えば文章題の「鉛筆」を
        買い物の説明へ誤ルーティングしてしまう。
        """
        self.turn = int(turn if turn is not None else self.turn + 1)
        history = history or []
        prev = _previous_assistant_texts(history)

        # ---- 0) 思考層（mind）が主経路。定型文テーブルには戻さない ------- #
        try:
            got = self.think(text, history=history, turn=self.turn, web=web)
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning("mind が失敗（旧経路に降ります）", exc_info=True)
            got = None
        if got is not None:
            rendered, thought = got
            if rendered.text and len(rendered.text) >= 4:
                from .composer import validate as _validate

                ok, _why = _validate(rendered.text, max_len=max(240, len(rendered.text)))
                if ok or rendered.authoritative or "\n" in rendered.text:
                    plan = rendered.plan or "mind"
                    knowledge = dict(thought.knowledge or {})
                    if knowledge:
                        knowledge["mind"] = thought.as_dict()
                    return Reply(
                        text=rendered.text, plan=plan,
                        confidence=float(rendered.confidence), knowledge=knowledge or None,
                        sentences=rendered.sentences or _split(rendered.text),
                        notes={"authoritative": bool(rendered.authoritative),
                               "task": ({"kind": plan, "answer": rendered.text}
                                        if rendered.authoritative else None),
                               "mind": thought.as_dict(),
                               "fixes": rendered.fixes, "lm": rendered.lm,
                               "intent": thought.frame.get("act"), "qtype": thought.frame.get("ask")})

        # 厳密に解ける仕事は、生成モデルの確率や KB の近さで上書きしない。
        try:
            task = self.tasks.answer(text, web=web, history=history)
        except Exception:  # noqa: BLE001
            task = None
        if task is not None:
            return Reply(
                text=task.text,
                plan=task.plan,
                confidence=float(task.confidence),
                knowledge=(
                    {"task": task.kind, **task.metadata}
                    if task.metadata else {"task": task.kind}
                ),
                sentences=_split(task.text) if "```" not in task.text else [task.text],
                notes={"authoritative": task.authoritative, "task": task.as_dict()},
            )

        u = self.analyze(text)

        if material is None:
            try:
                # オフライン指定で「今日」「最新」などを KB の古い材料で答えない。
                # その場合は安全な案内へ落とし、ユーザーが web を許可した次の
                # ターンで ResearchEngine に任せる。
                current_needed, _reason = self.tasks.research.should_research(text, None)
                if web is False and current_needed:
                    material = None
                else:
                    material = self.kb.answer(text)
            except Exception:  # noqa: BLE001
                material = None
        if material is not None:
            u.is_opaque = False        # 材料が引けたなら「読み取れている」

        # ---- 安全網 ------------------------------------------------------ #
        # かつてここには「入力テンプレ → 定型文」の棚があった。同じ入力に同じ文を
        # 返す Bot に見えるため、棚は撤去した。ここから先も定型文は使わず、
        # 根拠（KB の文・語彙の観測・検索で取れた文）だけを文に組み立てる。
        return self._fallback_compose(text, material=material, history=history,
                                      prev=prev, web=web)

    # ------------------------------------------------------------------ #
    # 安全網: mind が例外で落ちたときでも、同じ部品で組み直す
    # ------------------------------------------------------------------ #
    def _fallback_compose(self, text: str, *, material: dict | None = None,
                          history: list[dict] | None = None, prev: list[str] | None = None,
                          web: bool | None = None) -> Reply:
        """定型文にしないための安全網。

        根拠源（知識ベース / 語彙バンク / Web）から証拠を集め直し、同じ合成器
        （``mind.voice``）で文章にする。証拠が 1 つも無いときは、その入力から
        実際に読めた事実（語数・辞書にある語・数字）を述べる。読めた事実は
        入力ごとに違うので、ここでは同じ文が繰り返されない。
        """
        history = history or []
        prev = prev or []
        u = self.analyze(text)
        try:
            from .ground.evidence import gather, suggestion_claims
            from .mind.parse import build_frame
            from .mind.voice import render

            frame = build_frame(text, history=history, kb=self.kb)
            grounding = None if web is False else self.web_grounding()
            dossier = gather(frame, kb=self.kb, web=grounding,
                             history_text=" ".join(str(m.get("content", "")) for m in history)[:400])
            if not dossier.claims:
                dossier.claims.extend(suggestion_claims(text, kb=self.kb))
            if not dossier.claims and material and str(material.get("text") or "").strip():
                from .mind.frame import Claim

                dossier.claims.append(Claim(
                    kind="definition", subject=str(material.get("topic") or u.subject()),
                    content=str(material["text"]).strip(), source="local:kb", weight=0.85))
            out = render(dossier, frame, turn=self.turn, lm=self.lm, core=None,
                         polisher=self.polisher, history=history, validate=validate)
            if out.text and len(out.text) >= 4:
                return Reply(text=out.text, plan=out.plan or "fallback",
                             confidence=float(out.confidence),
                             knowledge={"via": dossier.via, "topic": dossier.topic,
                                        "coverage": round(float(dossier.coverage), 3),
                                        "fallback": True, "sources": dossier.sources[:4]},
                             sentences=out.sentences or _split(out.text),
                             notes={"authoritative": bool(getattr(out, "authoritative", False)),
                                    "fallback": True, "intent": u.intent, "qtype": u.qtype,
                                    "mood": u.mood, "opaque": u.is_opaque,
                                    "words": u.words[:6], "fixes": out.fixes, "lm": out.lm})
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning("安全網の合成も失敗", exc_info=True)

        # 合成器まで失敗した最後: 観測事実だけを、その場で計算して返す。
        facts = self._observations(text, u=u, prev=prev)
        if facts:
            return Reply(text=facts, plan="observations", confidence=0.3,
                         sentences=_split(facts), knowledge={"fallback": True},
                         notes={"intent": u.intent, "opaque": u.is_opaque})
        return Reply(text="", plan="empty", confidence=0.0)

    def _observations(self, text: str, *, u: Utterance | None = None,
                      prev: list[str] | None = None) -> str:
        """最終手段。*観測値（文字数・語彙の状態）は並べず*、読めた語から答えにいきます。

        ここは mind が壊れたときだけ通る道ですが、通ったからといって
        「入力は N 文字」「語彙バンクにある語は…」を画面に出すのは
        応答として許されません（v4 で禁止した辞書引きの形そのものなので）。
        """
        u = u or self.analyze(text)
        raw = (text or "").strip()
        words = [str(w) for w in (u.words or []) if w]
        out: list[str] = []
        for probe in [u.echo or u.subject()] + words[:4]:
            probe = str(probe or "").strip(" 「」『』、。")
            if len(probe) < 2:
                continue
            try:
                got = self.kb.answer(probe) if self.kb is not None else None
            except Exception:  # noqa: BLE001
                got = None
            body = str((got or {}).get("text") or "").strip()
            if body and 10 <= len(body) <= 130 and not re.search(r"(拍|品詞|索引|U\+)", body):
                out.append(body)
                break
        if not out and raw:
            try:
                from .mind.think import shape_line

                out.append(shape_line(raw))
            except Exception:  # noqa: BLE001
                out.append("用件から組み直します。いちばん近い語を一言もらえますか。")
        text_out = "\n".join(out).strip()
        if prev and text_out in prev:
            text_out = f"{text_out}\n別の角度からも見ます。どれを先に決めたいですか。"
        return text_out


    def _word_hint(self, w: str) -> str:
        """未知語の中に既知の話題が含まれていれば、手がかりの一文を返す。"""
        if not w or len(w) < 2:
            return ""
        try:
            topics: list[str] = []
            for cand in sorted({w[i:j] for i in range(len(w)) for j in range(i + 1, len(w) + 1)
                                if j - i >= 1}, key=len, reverse=True):
                if cand in ("こと", "もの", "について", "って", "とは"):
                    continue
                ids = self.kb.index.topics_of(cand) if self.kb is not None else set()
                if ids:
                    for i in sorted(ids)[:1]:
                        t = str(self.kb.items[i].get("topic") or "")
                        if t and t not in topics and t != w:
                            topics.append(t)
                if len(topics) >= 2:
                    break
            if topics:
                joined = "・".join(topics[:2])
                return f"「{w}」の中の{joined}の部分は、手元の知識とつながりそうです。"
        except Exception:
            pass
        # カタカナ語は外来・固有名の可能性を示す
        try:
            import re as _re

            if _re.search(r"[ァ-ヶー]{2,}", w):
                return f"「{w}」はカタカナを含む言葉で、固有名や外来の可能性があります。"
        except Exception:
            pass
        return ""

    def _suggest_topic(self, text: str, word: str) -> str:
        """発話に近い知識ベースの話題名（無ければ ""）。字の重なりが 2 文字以上あるものだけ。"""
        if not word:
            return ""
        try:
            for t in self.kb.suggest(text, top_k=3):
                t = str(t or "")
                if not t:
                    continue
                if word in t or t in word:
                    return t
                if len(set(word) & set(t)) >= 2:
                    return t
        except Exception:  # noqa: BLE001
            return ""
        return ""

    # ------------------------------------------------------------------ #
    def _polish(self, text: str) -> str:
        try:
            out = self.polisher.polish(text, register="polite")["text"] or text
        except Exception:  # noqa: BLE001
            out = text
        return balance_quotes(out)

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

    def accept(self, text: str, *, u: Utterance | None = None, min_conf: float = 0.4) -> bool:
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
        # 文末が述語（活用語・断定・終助詞）になっていない列は採用しない。
        # 「きょはいいんてきすあ。」のように語だけ並んだ文を弾くための最低関門。
        stem = re.sub(r"[。！？!?\.\s]+$", "", t)
        if stem and not stem.endswith(_PREDICATE_ENDINGS):
            return False
        # 語彙に 1 つも当たらない列（「きょはいいんてきすあ」等）は日本語として採用しない
        try:
            from .lang.lex import bank as _bank

            known = [w for w, pos in _bank().segment(t)
                     if pos.split("/")[0] in {"名詞", "動詞", "形容詞", "副詞"} and len(w) >= 2]
            if not known:
                return False
        except Exception:  # noqa: BLE001
            pass
        return True


# ---------------------------------------------------------------------- #
# ヘルパ
# ---------------------------------------------------------------------- #
def balance_quotes(text: str) -> str:
    """「」の対応を揃える。開け閉めを一方だけ残した文を、文章として成立させる。

    組み立てた文の見た目の傷として最も多いのがこれなので、
    採用前に必ず通す（句読点の位置も一緒に整える）。
    """
    t = str(text or "")
    if "「" not in t and "」" not in t:
        return t
    out: list[str] = []
    depth = 0
    for ch in t:
        if ch == "「":
            if depth:                                   # 二重に開いたら閉じてから開く
                out.append("」")
                depth -= 1
            out.append(ch)
            depth += 1
        elif ch == "」":
            if not depth:                                # 対応の無い閉じ括弧は落とす
                continue
            out.append(ch)
            depth -= 1
        else:
            if depth and ch in "。！？":                  # 文中で開いたまま句点に来たら閉じる
                out.append("」")
                depth -= 1
            out.append(ch)
    if depth:                                            # 末尾で開いたままなら閉じる
        for i in range(len(out) - 1, -1, -1):
            if out[i] in "。！？":
                out.insert(i, "」")
                break
        else:
            out.append("」")
    return "".join(out)





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
