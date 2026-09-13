"""日常会話の組み立て — 相手の *中身* を読んで、反応・材料・次の一手を並べる層。

v3 までの `statement` 経路は「相手の文をそのまま引用 → 語を一語ください」という
受け取り方しか出来ませんでした。会話は出来ていても、返答は *その文章について何も考えていない*。
ここでは発話を (1) 内容語 (2) 発話行為 (3) 抜けている情報 の 3 点に分解し、知識ベースから
**その話題について実際に言えること** を引いてから文を組みます。

    反応   … 発話行為と気分に応じた一文（相手の語を *ばらして* 使い、全文の複写はしない）
    材料   … 知識ベースがその話題について持っている事実（source: local:kb）
    手立て … 計画・困りごとには、KB の手順やコツを 1 つだけ添える
    隙     … 抜けている情報（時間・場所・相手・目的）を名指しで聞く。KB の followup を優先する

材料が無い手は捨てます。総当たりで文を足すことはしません。
"""

from __future__ import annotations

import hashlib
import re

from ..lang import lex
from ..lang.phonetics import normalize
from .frame import Claim

# --------------------------------------------------------------------------- #
# 発話の観察
# --------------------------------------------------------------------------- #
_PLAN = re.compile(r"(たい|たいです|たいな|つもり|予定|おうかな|ようかな|に行こう|へ行きます|行こう|"
                   r"やろう|しよう|買おう|始めよう|と思ってます|と思っています|ことにします)")
_DONE = re.compile(r"(ました。|ちゃった|しまった|終わった|済んだ|食べてきた|行ってきた|作った|買った|"
                   r"観た|見たよ|読んでみた)")
_TROUBLE = re.compile(r"(つらい|辛い|しんどい|疲れた|疲れ|眠れない|不安|心配|怖い|痛い|凹ん|落ち込|"
                      r"イライラ|腹が立|きつい|キツ|無理|分からん|わからない|分からない|困った|どうしよう|"
                      r"嫌だ|いやだ|最悪|失敗)")
_FEEL = re.compile(r"(嬉しい|うれしい|楽しい|たのしい|好き|おいしい|美味しい|美味|最高|よかった|"
                   r"ありがた|幸せ|落ち着く|気が済んだ|さっぱり)")
_OPINION = re.compile(r"(と思う|って思う|だと思います|な気がする|気がする|思うけど|どうかな|どう思う)")
_INVITE = re.compile(r"(しよ|しよう|しませんか|ません？|一緒に|話そ|語ろ|遊ぼ|付き合って|教えてよ|"
                     r"聞かせて|しゃべろ|話そう)")
_QUESTION = re.compile(r"(？|\?|ですか|ますか|かな$|どれ|いつ|どこ|なぜ|どうして|いくら|何回)")

# 発話から読み取れる *情報の枠*。無い枠だけを名指しで聞く（「語を一語ください」は使わない）
_TIME_WORDS = ("今日", "昨日", "明日", "一緒", "いま", "今")
# 文の骨格だけを表す語を主語にすると「始めを終えられたんですね」になるので避ける
_FORM_NOUNS = ("始め", "ため", "こと", "もの", "ところ", "ほう", "はず", "わけ", "つもり",
               "時点", "全体", "中", "上", "下", "前", "後", "度", "予定", "話", "最悪",
               "一緒", "了解", "思い", "ついで", "はずみ", "わけ")


_SLOTS: tuple[tuple[str, re.Pattern[str], tuple[str, ...]], ...] = (
    ("time", re.compile(r"(今日|明日|昨日|明後日|今週|来週|週末|来月|\d+日|\d+月|[0-9]{1,2}時|"
                         r"朝|昼|夜|夕|午前|午後|夜勤)"),
     ("何時頃を予定していますか。", "時間帯はもう決めてありますか。")),
    ("place", re.compile(r"(公園|駅|家|職場|学校|店|ホテル|温泉|海|山|川|図書館|カフェ|病院|"
                         r"どこ|こっち|あっち|自宅|近所|実家)"),
     ("場所はもう決めてありますか。", "どこまで出る予定ですか。")),
    ("person", re.compile(r"(一人|誰か|友達|家族|恋人|上司|同僚|子供|親|先輩|後輩|相手|仲間)"),
     ("一人で行きますか、誰かと行きますか。", "誰に見せる予定ですか。")),
    ("purpose", re.compile(r"(ため|ために|目的|気分転換|リフレッシュ|発散|確認|勉強|作業|仕事|備忘)"),
     ("何が整えば良い一日になりますか。", "何を一番得たいですか。")),
    ("manner", re.compile(r"(どうやって|やり方|手順|方法|コツ|どこから|何をすれば)"),
     ("手順のどこで止まっていますか。", "何から着手する予定ですか。")),
)

# 内容語として数えない語（形式名詞・副詞的な汎用語）
_NOISE = frozenset({
    "こと", "もの", "ため", "ところ", "ほう", "わけ", "はず", "つもり", "まま", "ほど",
    "ちょっと", "少し", "いろいろ", "なんか", "やはり", "もちろん", "たぶん", "これ", "それ",
    "あれ", "ここ", "そこ", "方面", "時点", "状況", "予定", "予定", "今回", "毎回",
})

_SELF_TARGET = re.compile(r"(お前|てめえ|貴様|ゴミ|クズ|屑|使えない|使えん|遅い|のろま|バカ|阿保|"
                          r"何言ってん|分かってない|わかってない|聞いてない|返事が変|変な返事|"
                          r"最低|嫌だ|嫌い|ムカカ|ムカカつく|カス)")

# 気分が前向きのときの受け取り方（「〜に行こうと思う」等に同じ相槌を返さない）
_REACT_POS: dict[str, tuple[str, ...]] = {
    "plan": ("{n}の予定が立ったんですね。前の日に用意ができると、当日は動くだけになります。",
             "{n}、いい日を選びましたね。天気と出る時間まで確認できると崩れません。",
             "{n}に行けると良いですね。出る時間を先に決めると、あとの流れが楽になります。"),
    "report": ("{n}を終えられたんですね。終わったあとに何が残ったかが、次の役に立ちます。",
               "{n}、片付いてよかったです。次は同じ手間を減らせるかだけ見ておきましょう。",
               "{n}が終わったなら、1 手で済む形に直せるかを見る番です。"),
}


# 主語が読めない発話でも、質問だけ投げ返さないための受け取り口
_REACT_PLAIN: dict[str, tuple[str, ...]] = {
    "plan": ("予定として立てているところなんですね。", "やると決めた段階なんですね。",
             "計画の途中なんですね。"),
    "report": ("ここまで進んだんですね。", "一区切りついたんですね。", "終わったところなんですね。"),
    "trouble": ("困ったことになったんですね。", "厄介なところに当たったんですね。",
                "手が止まってしまう場面なんですね。"),
    "feel": ("今の気分を聞かせてもらえました。", "その状態なんですね。", "気分が動いたんですね。"),
    "opinion": ("その見方なんですね。", "そう感じるんですね。", "評価はそこにあるんですね。"),
    "invite": ("一緒に進める話なんですね。", "声をかけてもらえました。", "声をかけてもらえたので、進め方を決めます。"),
    "ask": ("質問として受け取りました。", "そこを数えたいんですね。", "決め手を知りたいんですね。"),
    "declare": ("話をもらえました。", "その件、受け取りました。", "聞きました。"),
}


# 述語だけの発話（疲れた・うれしい・終わった）は、その述語を使った受け取り方を返す
_REACT_PRED: dict[str, tuple[str, ...]] = {
    "trouble": ("{p}んですね。今日は動く範囲を絞ったほうが、あとが楽です。",
                "{p}のは今日だけですか、それとも続いていますか。",
                "{p}のはつらいですね。いちばん効いているのはどこですか。"),
    "feel": ("{p}とのこと。きっかけを一言もらえますか。",
             "{p}のはいいですね。一度きりでしたか、続いていますか。",
             "{p}のは一番です。次も同じ調子でいけそうですか。"),
    "declare": ("{p}んですね。どこまで進んだかだけ教えてください。",
                "{p}の話、受け取りました。いちばん近かった点を一言もらえますか。",
                "{p}とのこと。次に関わってくるのはどんな部分ですか。"),
    "report": ("{p}んですね。やってみて何が残りましたか。",
               "{p}のは良かったです。次に同じことがあれば変えたい所はありますか。",
               "{p}なら、次は手間を減らせるかを見てみましょう。"),
    "plan": ("{p}んですね。いつ頃から動く予定ですか。",
             "{p}の準備で、いま決まっている部分是ですか。",
             "{p}なら、先に順序だけ決めておくと楽です。"),
    "opinion": ("{p}という見方なんですね。どこがいちばん大きく効きましたか。",
                "{p}とのこと。同じように感じたのはいつ頃からですか。",
                "{p}の話を、もう少し聞かせてもらえますか。"),
}


_REACT: dict[str, tuple[str, ...]] = {
    "plan": ("{n}の計画、いいですね。", "{n}なら動ける日を選べましたね。",
             "{v}を先に決めておくのが、続きやすい作り方です。"),
    "report": ("ちゃんと {v}できましたね。", "{n}、良い結果になりそうです。",
               "やった分が {n} に出ていますね。"),
    "trouble": ("それは重たい {n} ですね。", "{n}が続くと堪えますね。",
                "今日は動く範囲を絞ったほうが、あとが楽です。"),
    "feel": ("{n}、それは良かったです。", "{n}を良いと思えるのが一番です。",
             "{n}の続きを聞かせてもらえますか。"),
    "opinion": ("{n}はその通りですね。", "{n}についての見立て、分かります。",
                "{n}はそういう読み方もあります。"),
    "invite": ("一緒にやりますよ。", "こちらこそ、進めましょう。", "会話の相手なら務まります。"),
    "ask": ("{n}の話ですね。", "{n}、そこから数えます。", "{n}の件、引き受けます。"),
    "declare": ("{n}の話をもらえました。", "{n}の件、受け取りました。", "了解、{n}の話ですね。"),
}

_ACT_ASK: dict[str, tuple[str, ...]] = {
    "plan": ("実行するのは今日ですか、次回ですか。", "やめる条件も決めておきますか。"),
    "report": ("一番効いたのはどの部分でしたか。", "次に同じことがあれば変えたい所はありますか。"),
    "trouble": ("いちばん効くのは何時ごろですか。", "今日は何を減らせそうですか。"),
    "feel": ("一番良かったのはどこでしたか。", "どのへんが一番効きましたか。"),
    "opinion": ("そう思うに至った出来事はありますか。", "逆に考えた例は見たことがありますか。"),
    "invite": ("どんな進め方にしますか。", "何から始めましょうか。"),
    "ask": ("数字と手順、どちらが要りますか。", "どんな場面で使う予定ですか。"),
    "declare": ("それはどんな場面の話ですか。", "何がきっかけでしたか。"),
}


def _rot(*, salt: str, pool, turn: int = 0) -> str:
    """内容のハッシュで回す。同じ言い方が続かず、ターンでも変わる。"""
    pool = [x for x in (pool or ()) if x]
    if not pool:
        return ""
    if len(pool) == 1:
        return pool[0]
    h = int(hashlib.sha1(str(salt).encode("utf-8", "ignore")).hexdigest(), 16)
    return pool[(h + int(turn) * 7) % len(pool)]


def observe(text: str) -> dict:
    """発話行為・気分・内容語・抜けている枠を読む。"""
    body = re.sub(r"\s+", " ", normalize(text or "")).strip()
    act = "declare"
    if _TROUBLE.search(body):
        act = "trouble"
    elif _INVITE.search(body):
        act = "invite"
    elif _PLAN.search(body):
        act = "plan"
    elif _OPINION.search(body):
        act = "opinion"
    elif _FEEL.search(body):
        act = "feel"
    elif _QUESTION.search(body):
        act = "ask"
    elif re.search(r"(たい$|たく|したい|するつもり|やりたい)", body):
        act = "plan"
    elif _DONE.search(body):
        act = "report"
    mood = "neutral"
    if act == "trouble":
        mood = "negative"
    elif act in ("plan", "feel", "report"):
        mood = "positive"

    bank = lex.bank()
    nouns: list[str] = []
    verbs: list[str] = []
    adjs: list[str] = []
    for tok, pos in bank.segment(body):
        head = str(pos or "").split("/")[0]
        tok = tok.strip("。、!?！？・…「」『』()（） ")
        if len(tok) < 2 or tok in _NOISE:
            continue
        if head == "名詞" and tok not in nouns:
            nouns.append(tok)
        elif head == "動詞" and tok not in verbs:
            verbs.append(tok)
        elif head == "形容詞" and tok not in adjs:
            adjs.append(tok)
    marker = re.search(r"(予定|しよ|たい|行く|いく|行こ|来た|きた|始めた|終わ|でき|った|もらった|くれた)",
                       body)
    mp = marker.start() if marker else len(body) // 2
    focus = ""
    best = 10 ** 6
    for cand in nouns:
        if cand in _FORM_NOUNS or cand in _TIME_WORDS:
            continue
        pos = body.find(cand)
        if pos < 0:
            continue
        if abs(pos - mp) < best:
            best, focus = abs(pos - mp), cand
    pred = re.sub(r"[。、！？!?・…\s]+$", "", body)
    predicate = pred if 1 < len(pred) <= 12 and not nouns else ""
    have = {name for name, pat, _ in _SLOTS if pat.search(body)}
    missing = [name for name, _pat, _ask in _SLOTS if name not in have]
    casual = bool(re.search(r"(だ$|だよ|だよね|じゃん|やん|てやんで|すね|る$|た$|て$|や$|ね$|よ$|か$)",
                            body.rstrip("。！？!? ")))
    return {"text": body, "act": act, "mood": mood, "predicate": predicate,
            "nouns": nouns[:8], "verbs": verbs[:6],
            "adjs": adjs[:4], "missing": missing, "have": sorted(have), "casual": casual,
            "focus": focus or next((n for n in nouns
                                    if n not in _FORM_NOUNS and n not in _TIME_WORDS), ""),
            "known_nouns": [n for n in nouns if bank.has(n)]}


# --------------------------------------------------------------------------- #
# 知識ベースからの手
# --------------------------------------------------------------------------- #
def topic_items(text: str, *, kb, limit: int = 3, min_score: float = 0.42) -> list[dict]:
    """発話に *実際に引っかかった* 話題だけ返す（スコアが低い物は使わない）。"""
    if kb is None:
        return []
    try:
        hits = kb.search(text, top_k=limit) or []
    except Exception:  # noqa: BLE001
        return []
    out: list[dict] = []
    for h in hits:
        if float(h.get("score") or 0.0) < min_score:
            continue
        # スコアだけだと「雨降り → 火山」のように薄い取り違えが起きます。
        # *発話に出てきた語* を実際に踏んだ話題だけを材料にします。
        if not (h.get("topic_hit") or (h.get("word_hits") or [])):
            continue
        item = h.get("item") or {}
        if item.get("def") or item.get("facts"):
            out.append(item)
    return out


_LEX_TRIVIA = re.compile(r"(拍|索引|U\+|文字数|読み仮名|品詞)")


def _usable(line: str) -> bool:
    """会話に混ぜてよい 1 文か。辞書の語彙情報・壊れた文はここで落とします。"""
    line = str(line or "")
    if not (10 <= len(line) <= 110):
        return False
    if _LEX_TRIVIA.search(line):
        return False
    if re.search(r"(软件|数据集|高频|请查询|时候)", line):
        return False
    return line.endswith(("。", "！", "？", "?", "!", "、"))


def _fact_lines(item: dict, *, want: int = 2, turn: int = 0) -> list[str]:
    """1 話題から、順番に違う文を引く（同じ受け答を繰り返さないための回転）。"""
    pool = [str(item.get("def") or "").strip()]
    pool += [str(x).strip() for x in (item.get("facts") or [])]
    pool += [str(x).strip() for x in (item.get("why") or [])]
    pool += [str(x).strip() for x in (item.get("tips") or [])]
    pool = [x for x in pool if len(x) >= 10 and _usable(x)]
    if not pool:
        return []
    start = int(turn) % len(pool)
    ordered = pool[start:] + pool[:start]
    return ordered[: max(1, want)]


def _kb_followup(item: dict, *, turn: int = 0) -> str:
    pool = [str(x).strip() for x in (item.get("followups") or []) if str(x or "").strip()]
    if not pool:
        return ""
    return pool[(int(turn) + len(str(item.get("topic") or ""))) % len(pool)]


def _ask_line(obs: dict, items: list[dict], *, turn: int = 0) -> str:
    """抜けている枠を 1 つだけ名指しする。KB の followup があればそれを優先する。"""
    salt = f"{obs['text'][:48]}|{turn}"
    act_ask_first = _ACT_ASK.get(obs["act"], ())
    if obs["act"] in ("trouble", "feel") and act_ask_first:
        # 困りごと・気持ちには、時刻や場所を掘らず、相手の選びやすい問いだけ返す
        return _rot(salt=salt, pool=act_ask_first, turn=turn)
    for it in items:
        kb_ask = _kb_followup(it, turn=turn)
        if kb_ask:
            return kb_ask
    slot_ask = ""
    for name, _pat, pool in _SLOTS:
        if name in obs["missing"]:
            slot_ask = _rot(salt=f"{salt}:{name}", pool=pool, turn=turn)
            break
    act_ask = _rot(salt=salt, pool=_ACT_ASK.get(obs["act"], ()), turn=turn)
    return slot_ask or act_ask


def _verb_ok(v: str) -> bool:
    """反応文に使える動詞か（「行こ」のような崩れた語幹は使わない）。"""
    v = str(v or "").strip()
    if len(v) < 2 or re.search(r"(っ|ゃ|ゅ|ょ|ー)$", v):
        return False
    return not re.search(r"(い|き|し|ち|に|ひ|み|り|ぎ|じ|び|ぴ)$", v)


def _react_line(obs: dict, items: list[dict], *, turn: int = 0) -> str:
    """相手の語を *分解して* 使う反応文。全文複写は検証側で弾く（voice の重複検査）。"""
    n = str(obs.get("focus") or "")
    if not n:
        n = next((x for x in obs["nouns"] if x not in _NOISE and x not in _TIME_WORDS
                  and x not in _FORM_NOUNS), "")
    if not n and items:
        n = str(items[0].get("topic") or "")
    pool = _REACT.get(obs["act"], _REACT["declare"])
    if obs.get("mood") == "positive" and n and obs["act"] in _REACT_POS:
        pool = _REACT_POS[obs["act"]]
    v = re.sub(r"(ました|ません|たい|です|ます|て|た|よう)$", "", obs["verbs"][0] if obs["verbs"] else "")
    if not _verb_ok(v):
        v = ""
    p_ = str(obs.get("predicate") or "")
    if not n and p_:
        pool = list(_REACT_PRED.get(obs["act"], ()))
    usable = [x for x in pool if (("{n}" not in x) or n) and (("{v}" not in x) or v)
              and (("{p}" not in x) or p_)]
    if not usable:
        # 主語を読めない発話でも、質問だけを投げ返さない（受け取り口は必ず置く）
        pool = _REACT_PLAIN.get(obs["act"], _REACT_PLAIN["declare"])
        usable = [x for x in pool if "{n}" not in x and "{v}" not in x]
    if not usable:
        return ""
    line = _rot(salt=f"{obs['text'][:32]}|react|{obs['act']}", pool=usable, turn=turn)
    line = line.replace("{n}", n or p_ or "その話").replace("{v}", v or n or "やる")
    line = line.replace("{p}", p_ or "その話")
    if n and line.count(n) > 1:            # 「天気、いい日…」の後で同じ語を繰り返さない
        head, sep, tail = line.partition(n)
        line = head + sep + tail.replace(n, "その日", 1)
    return re.sub(r"\s+", "", line)


# --------------------------------------------------------------------------- #
# 特殊な受け取り方（こちらへの不満・読めない短文）
# --------------------------------------------------------------------------- #
def self_claims(obs: dict, *, items: list[dict], turn: int = 0) -> list[Claim]:
    """こちらへの不満には、言い訳ではなく *直し方の具体案* を返す。"""
    pool = ("前の返しが取り留めがなかったのは其のとおりです。話題そのものを見て組み直します。",
            "答えの形がずれていました。用件だけ受け取って、もう一回組みます。",
            "的を外していました。何が欲しかったかに絞って組み直します。")
    out = [Claim(kind="correction", content=_rot(salt=obs["text"], pool=pool, turn=turn),
                 subject="自分", source="local:meta", weight=0.66)]
    ask = _rot(salt=obs["text"] + "|ask",
               pool=("どこが気に入らなかったか、一言だけもらえますか。",
                     "出力の形（表・コード・一文）を指定してもらえますか。",
                     "どの話題から片付ければ良いですか。"),
               turn=turn)
    out.append(Claim(kind="question", content=ask, source="local:meta", weight=0.55))
    return out


def repair_claims(obs: dict, *, last_topic: str = "", items: list[dict] | None = None,
                  turn: int = 0) -> list[Claim]:
    """「は？」「え」など、短くて意味が読めない発話の修復。"""
    pool = ("今の話を、別の角度から言い直します。",
            "前の返しが短すぎました。中身から組み直します。",
            "読み取りが浅かったです。用件から数え直します。")
    out = [Claim(kind="correction", content=_rot(salt=obs["text"], pool=pool, turn=turn),
                 source="local:meta", weight=0.6)]
    topic = last_topic or (str(items[0].get("topic")) if items else "")
    if items:
        for line in _fact_lines(items[0], want=1, turn=turn):
            out.append(Claim(kind="fact", content=line, subject=topic, source="local:kb",
                             weight=0.8, extra={"topic": topic}))
    elif topic and items:
        # 話題の*中身*が引めるときだけ、何が言えるかを数える（空案内は出さない）
        head = str(items[0].get("def") or "").strip()
        if head and 10 <= len(head) <= 120:
            out.append(Claim(kind="fact", content=head if head.endswith("。") else head + "。",
                             subject=topic, source="local:kb", weight=0.7,
                             extra={"topic": topic}))
    ask = _rot(salt=obs["text"] + "|ask",
               pool=("どの欄が欲しかったですか。", "何を先に決めたいですか。",
                     "数字・手順・比較のどれが必要ですか。"), turn=turn)
    out.append(Claim(kind="question", content=ask, source="local:meta", weight=0.5))
    return out


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #
def claims_for(text: str, *, frame=None, kb=None, history: list[dict] | None = None,
               turn: int = 0) -> list[Claim]:
    """日常発話に対する主張の列。材料が全く無いときは空 list を返す（上位が別の手を選ぶ）。"""
    obs = observe(text)
    items = topic_items(obs["text"], kb=kb, limit=3)
    said = " ".join(str(m.get("content") or "") for m in (history or [])
                    if m.get("role") == "assistant")

    if _SELF_TARGET.search(obs["text"]):
        return self_claims(obs, items=items, turn=turn)
    # 修復に回すのは「読める内容語が 1 つも無い極短入力」だけ。
    # 「疲れた」「うれしい」のような形容詞・動詞だけの発話には、相手の語を使った受け取り方をします。
    opaque_short = (len(obs["text"]) <= 4 and not obs["nouns"] and not obs["verbs"]
                    and not obs["adjs"])
    if opaque_short:
        topic = str(getattr(frame, "topic", "") or "") if frame is not None else ""
        return repair_claims(obs, last_topic=topic, items=items, turn=turn)

    if obs["act"] == "invite" and not items:
        return [Claim(kind="answer",
                      content="一緒にやることなら、雑談・語を並べる遊び・手元の計算や下書きまで続けられます。",
                      source="local:chat", weight=0.6),
                Claim(kind="question",
                      content=_rot(salt=f"{obs['text']}|inv", turn=turn,
                                   pool=("何を一緒にやりたいですか。",
                                         "話す題材を一言もらえますか。",
                                         "遊びますか、手を動かす用事ですか。")),
                      source="local:chat", weight=0.6)]

    if re.fullmatch(r"(了解|りかい|承知|ok|okay|なるほど|わかった|分かる)[。.!!?…\s]*",
                    obs["text"].lower()):
        return [Claim(kind="answer",
                      content=_rot(salt=obs["text"], turn=turn,
                                   pool=("はい、次の準備はできています。",
                                         "承知しました。続けるなら、どれから進めますか。",
                                         "ありがとうございます。次は数字と手順、どちらから観ますか。")),
                      source="local:chat", weight=0.58)]

    claims: list[Claim] = []
    react = _react_line(obs, items, turn=turn)
    if react and len(react) >= 6:
        claims.append(Claim(kind="answer", content=react,
                            subject=obs["nouns"][0] if obs["nouns"] else "",
                            source="local:chat", weight=0.62))
    # 事実を積むとき、*最初の話題から順に* 取ります（別話題に流れていくのを防ぐ）
    lines: list[tuple[str, dict]] = []
    for it in items:
        for line in _fact_lines(it, want=2, turn=turn):
            if line and line not in said:
                lines.append((line, it))
    for line, it in lines[:2]:
        claims.append(Claim(kind="fact", content=line, subject=str(it.get("topic") or ""),
                            source="local:kb", weight=0.82,
                            extra={"topic": it.get("topic"), "id": it.get("id")}))
    if obs["act"] in ("plan", "trouble"):
        for it in items[:1]:
            steps = [str(x).strip() for x in (it.get("how") or []) if str(x or "").strip()]
            tips = [str(x).strip() for x in (it.get("tips") or []) if str(x or "").strip()]
            pick = (steps[int(turn) % len(steps)] if steps else "") or \
                   (tips[int(turn) % len(tips)] if tips else "")
            if len(pick) >= 6:
                body = pick if pick.endswith(("。", "！", "？")) else f"{pick}、の順で進めると早いです。"
                claims.append(Claim(kind="advice", content=body, subject=str(it.get("topic") or ""),
                                    source="local:kb", weight=0.74))
                break
    ask = _ask_line(obs, items, turn=turn)
    if obs["act"] in ("trouble", "feel") and ask and re.search(r"(何時|どこ|何日|何時間|どのくらい)", ask):
        ask = ""                      # しんどい話に時刻・場所の掘り方はしない
    if ask and len(claims) >= 4:
        ask = ""                      # 質問は 1 応答に 1 つまで
    if ask:
        claims.append(Claim(kind="question", content=ask, source="local:chat", weight=0.52))
    return claims


__all__ = ["observe", "claims_for", "topic_items", "self_claims", "repair_claims"]
