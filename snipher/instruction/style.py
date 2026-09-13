"""文体の実装 — 指示された口調・文体・長さに、語の活用を組み替えて合わせる。

「〜だよ、〜だね で」と言われたら、語尾の文字を差し替えるのではなく
(1) 文末の述語を `lang.morph` で辞書形に引き直し (2) 常体に組み直し (3) その上に
指定の語尾（だよ／だね／である／です）を載せます。だから「速いです → 速いんだよ」、
「削減できます → 削減できるんだよ」という *文法的に正しい* 変換になります。

長さも同じで、文を途中で切らずに「入る文だけを選ぶ」＋「節（、）の単位で詰める」
ので、200 文字程度という指定に対して壊れた文を出しません。詰め方が述語を失うなら
その詰め方は *使わない*（空を返して呼び出し側に別の文を選ばせる）のが要点です。
"""

from __future__ import annotations

import re

from ..lang import morph
from ..lang.phonetics import normalize
from ..mind.frame import split_sentences

# 口調ごとの語尾（回転させて同じ語尾の連打を避ける）
_VOWEL_STEM = "いきしちにひみりえけせてねべめれおこそとのほぼもよを"      # 一段動詞のます幹の末尾（母音・エ段）
_CONSONANT_STEM = "かきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"      # 五段動詞のます幹の末尾（子音）

TAILS = {
    "friendly": ("だよ", "だね", "んだよ"),
    "friendly_professional": ("だよ", "だね", "んだよ"),
    "plain": ("だ", "である", ""),
    "professional": ("だ", "である", ""),
    "polite": ("です", "ます", "です"),
}

_PROTECTED = re.compile(r"```.*?```", re.S)
_LIST_HEAD = re.compile(r"^(?:[・\-*•●○▪]|\d+[.)、．]|[a-zA-Z][.)]|\{|\}|\||出典|>)")

# --------------------------------------------------------------------------- #
# 述語で終わっているか（名詞止め・助詞止めに語尾を載せないため）
# --------------------------------------------------------------------------- #
_PRED_1CHAR = "いただぬるすつうくぐむぶよねわ"
_PRED_MULTI = (
    "ない", "たい", "できる", "いる", "ある", "なる", "する", "くる", "だった", "ほしい",
    "らしい", "そうだ", "ようだ", "みたい", "でしょう", "だろう", "です", "ます", "ている",
    "てある", "られる", "させる", "える", "かわる", "おわる", "である", "となる", "となる",
    "ません", "れない", "えない", "きれる", "つづく", "おわる", "はじまる", "であり",
    "である", "でない", "ではない", "ならない", "あろう", "であろう", "ましょう",
)
_PRED_RE = re.compile(
    "(?:[" + _PRED_1CHAR + "]$|"
    + "|".join(re.escape(x) + "$" for x in sorted(set(_PRED_MULTI), key=len, reverse=True))
    + ")"
)


def is_predicate_end(text: str) -> bool:
    """文末が述語（終止形）か。名詞・助詞で終わっていたら False。

    判定は 2 段です。(1) 文末の 1 語を実辞書で引き、品詞が動詞／形容詞／形容動詞語幹／
    助動詞なら述語。(2) 引けなかったら *述語にしか現れない末尾*（できる・ている・ない・
    たい・である …）で見る。文字種だけで見ると「ことで」のような助詞止めを述語と
    間違えるので、そこは辞書に決めてもらいます。
    """
    s = normalize(str(text or "")).rstrip("。！？!?…、 　")
    if not s or len(s) < 2:
        return False
    if re.search(r"(?:こと|もの|ため|ところ|ほど|まま|うえで|上で)$", s):
        return False                       # 名詞止め（〜すること。は述語ではない）
    try:
        from ..lang import lex

        _prefix, head = morph.split_predicate(s)
        head = (head or "").strip()
        if head:
            pos = str(lex.bank().pos(head) or "")
            if pos.startswith(("動詞", "形容詞", "助動詞")):
                return True
            if "形容動詞語幹" in pos:
                return True
    except Exception:  # noqa: BLE001
        pass
    return bool(_PRED_RE.search(s))


def _is_protected(line: str) -> bool:
    """口調を載せてはいけない行（コード・箇条書き・引用・出典）。"""
    s = line.strip()
    return (not s) or bool(_LIST_HEAD.match(s)) or s.startswith(("```", '"', "<"))


def ends_with_tail(sentence: str, tails: tuple[str, ...] | list[str]) -> bool:
    s = normalize(sentence).rstrip("。！？!?…")
    return any(s.endswith(t) for t in tails if t)


# --------------------------------------------------------------------------- #
# 常体化
# --------------------------------------------------------------------------- #
#: 文末に敬体が残っていないかの判定（か・ね・よ が付いていても拾う）
_POLITE_LEFT = re.compile(r"(?:ます|です|ましょう|ください|いたします)(?:か|ね|よ|かしら)?$")
_POLITE_DROP = re.compile(
    r"(?:でしょうか|ですか|ましょうか|ますか|でしたか|ですね|ですよ|です|ました|でした|ません|"
    r"でしょう|いたします|くださいませ|ください|ます)$")
#: 丁寧な否定（〜ありません）は morph に渡すと壊れやすいので先に常体へ
_NEG_POLITE = (
    ("くありません", "くない"), ("ぎありません", "ぎない"), ("しません", "しない"),
    ("ちません", "ちない"), ("みません", "みない"), ("りません", "りない"),
    ("いません", "いない"), ("えません", "えない"), ("れません", "れない"),
    ("べません", "べない"), ("ぜません", "ぜない"), ("でありません", "でない"),
    ("ありません", "ない"), ("ません", "ない"),
)


def _plain_sentence(sentence: str) -> str:
    """1 文を常体にする（述語を辞書形に引き直してから組み直す）。"""
    s = str(sentence or "").strip()
    if not s:
        return s
    marks = "".join(re.findall(r"[。！？!?…]+$", s))
    body = s[: len(s) - len(marks)] if marks else s
    # 語尾を規則で落とすと *疑問の「か」* まで消えて「〜あり。」になります。
    # 疑問は形として保ち、後で 「？」 に組み替えます。
    asked = bool(re.search(r"(?:か|のか|かな)$", body))

    cand = ""
    for src, dst in _NEG_POLITE:
        if body.endswith(src) and len(body) > len(src) + 1:
            cand = body[: -len(src)] + dst
            break

    got = ""
    try:
        plain = morph.to_plain(body).rstrip("。")
        if plain and not _POLITE_LEFT.search(plain):
            got = plain
    except Exception:  # noqa: BLE001
        got = ""
    repaired = _plain_from_masu(body, got)
    if repaired:
        # 活用を組み直せたので、規則で語尾を落とした形（します→しる）は使わない
        got, cand = repaired, ""
    elif not got:
        got = _POLITE_DROP.sub("", body)
    if cand:
        # 辞書の引きが正しい終止形を作れたならそれを優先し、
        # 作れなかったときだけ規則で落とした形を使う。
        if not got or not is_predicate_end(got) or _POLITE_LEFT.search(got):
            got = cand
    got = got or body
    if asked and not str(got).rstrip("。").endswith(("か", "かな", "ですか", "ますか")):
        got = str(got).rstrip("。") + "か"
    if asked and marks in ("。", ""):
        marks = "？"
    return got + (marks or "")


def _plain_from_masu(body: str, got: str) -> str:
    """ます形の文末が *壊れた終止形* になったとき、辞書で確かめて組み直す。

    ``morph.to_plain`` は語幹の母音を落として ``します → しる``、
    ``組み立てます → 組み立る`` のような形を作ることがあります。語尾の差し替えで
    ごまかさず、*述語 1 語を切り出して活用を組み直し、辞書に当てる* のがここです
    （当てられなければ ``to_plain`` の形をそのまま信じます）。

        話をします   → 話をする      （サ変）
        組み立てます → 組み立てる    （一段）
        行きます     → 行く         （五段は壊れないので何もしない）
        増やせます   → 増やせる      （可能形も to_plain が正しく作れる）
    """
    m = re.search(r"ます$", body)
    if not m:
        return ""
    stem = body[: m.start()]
    if not stem:
        return ""
    got = got or ""
    if got == stem + "る" and _is_verb(got):
        return ""                                   # 既に正しい終止形（食べます → 食べる）
    # サ変: 「しる」は「する」の取り違え（出します → 出す は正しいので触らない）
    if stem.endswith("し") and _in_dict("する") and not _is_verb(got):
        return stem[:-1] + "する"                   # 話をしる → 話をする
    # 一段動詞: ます幹の *文末の 1 語* + る が動詞として辞書に立つなら、それが終止形。
    # 語を切り出してから引くので、「その欄を狙って組み立てます」のような文でも
    # 助詞を巻き込まずに 組み立てる へ直せます。
    if stem[-1] in _VOWEL_STEM:
        try:
            from ..lang.morph import _split_last_word

            prefix, head = _split_last_word(stem)
        except Exception:  # noqa: BLE001
            return ""
        if head and _is_verb(head + "る"):
            return prefix + head + "る"             # 組み立る → 組み立てる
    return ""


def _is_verb(word: str) -> bool:
    """その表記が語彙バンクに *動詞* として立つか。

    「しる」のように別の品詞で見出しが立っている語があるので、終止形を確かめるときは
    品詞まで見ます（品詞が引けない語は「ある」とみなさない）。
    """
    try:
        from ..lang import lex

        return str(lex.bank().pos(word) or "").startswith("動詞")
    except Exception:  # noqa: BLE001
        return False


def _in_dict(word: str) -> bool:
    """その表記が語彙バンクに見出しとして立つか（述語を確かめるときの裏取り）。"""
    try:
        from ..lang import lex

        return bool(lex.bank().has(word))
    except Exception:  # noqa: BLE001
        return False


def _polite_sentence(sentence: str) -> str:
    s = str(sentence or "").strip()
    if not s:
        return s
    body = s.rstrip("。")
    if re.search(r"(?:です|ます|ました|でした|ません|でしょう)$", body):
        return s if s.endswith("。") else s + "。"
    try:
        got = morph.to_polite(body)
        # 敬体へ組み直した結果が *本当に敬体の述語* になっているかを確かめる。
        # 「組み立て → 組み立ます」のように活用を取り違えた形は採用しない（元の文を残す）。
        if got and re.search(r"(?:ます|です|ました|でした|ません|でしょう|ましょう)[。！？!?]?$",
                             got.strip()):
            return got
    except Exception:  # noqa: BLE001
        pass
    return s if s.endswith("。") else s + "。"


def _friendly_tail(body: str, seq: list[str]) -> str | None:
    """砕いた口調で *付けてはいけない* 語尾だけを弾きます。None なら繰り回しに任せます。

    問い（〜たいですか。）に だよ/だね は乗せられないので、そこだけ空を返します。
    平文で だよ／だね を交互に置くのは、口調指定（両方を使って、という指示）を守るためです。
    """
    probe = str(body or "").rstrip("。！!… ").strip()
    if not probe:
        return ""
    if re.search(r"(?:か|ですか|ますか)\s*$", probe) or re.search(r"[？?]$", probe):
        return ""
    return None


def retime(sentence: str, tone: str, *, index: int = 0,
           tail_override: str | None = None) -> str:
    """1 文を指定の口調に組み替える（語尾の文字差し替えではなく活用の組み直し）。"""
    s = _plain_sentence(sentence).strip()
    if not s or _is_protected(s):
        return s
    tails = TAILS.get(tone, TAILS["plain"])
    # "" は「語尾を載せない」という指示です。None (未指定) と混ぜると、問いに
    # 繰り回しで選んだ語尾が乗って「したいんだね」のようになります。
    tail = tails[index % len(tails)] if tail_override is None else tail_override
    marks = "".join(re.findall(r"[。！？!?…]+$", s))
    body = s[: len(s) - len(marks)] if marks else s
    asked = bool(re.search(r"(?:か|ますか|ですか|でしょうか)$", body)) or marks in ("？", "?")
    if marks in ("？", "?"):
        return body + "？"
    if marks in ("！", "!"):
        return body + ("！" if tone not in ("friendly", "friendly_professional") else "よ！")
    if asked and not tail:
        if tone in ("friendly", "friendly_professional"):
            return re.sub(r"(?:ますか|ですか|か)\s*$", "", body) + "か？"
    if not tail:
        if tone in ("friendly", "friendly_professional"):
            # 問いには だよ/だね を乗せられないので、*疑問の形のまま* 砕きます。
            # 「〜したいですか。」→「〜したい？」。「〜したい。」と語気を落とすのは別文です。
            soft = re.sub(r"(?:でしょうか|ですか|ますか)\s*$", "", body)
            if soft != body and soft:
                return soft + "？"
        return body + "。"
    if ends_with_tail(body, tails):
        return body + "。"
    if not is_predicate_end(body):
        # 名詞止め（箇条書きの語）には語尾を載せない。「アルゴリズムだよ」のような
        # 意味のずれた文になるため。
        return body + "。"
    if re.search(r"だ$", body):
        if tail.startswith("だ"):
            return body[:-1] + tail + "。"          # 静かだ + だよ → 静かだよ
        return body + tail + "。"
    if tail.startswith("だ"):
        # 動詞・イ形容詞には の を挟む（削減できる + んだよ）
        return body + "ん" + tail + "。"
    return body + tail + "。"


def restyle(text: str, *, tone: str = "", register: str = "", tails: list[str] | None = None,
            bullets: bool = False) -> tuple[str, list[str]]:
    """文全体を口調・文体に合わせる。返るのは (本文, 行った修正)。"""
    fixes: list[str] = []
    src = str(text or "")
    if not src.strip():
        return src, fixes
    protected: list[str] = []

    def _hold(m: re.Match) -> str:
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    work = _PROTECTED.sub(_hold, src)
    out: list[str] = []
    idx = 0
    for line in [x for x in re.split(r"\n", work) if x.strip()]:
        if _is_protected(line) or not (tone or register):
            out.append(line)
            continue
        pieces = [p for p in split_sentences(line) if p.strip()] or [line]
        rebuilt: list[str] = []
        for p in pieces:
            if tone in ("friendly", "friendly_professional"):
                seq = tails or list(TAILS[tone])
                if re.search(r"(?:か|ですか|ますか)\s*[。？?]?$", p.strip()):
                    got = retime(p, tone, index=idx, tail_override="")
                    rebuilt.append(got.strip())
                    continue
                # 語尾は *その文の形* で決めるので、繰り回し（idx）で壊れません。
                got = retime(p, tone, index=idx, tail_override=_friendly_tail(p, seq))
                idx += 1
            elif bullets and not is_predicate_end(p.rstrip("。！？!?")):
                got = p                            # 名詞止めの箇条書きはそのまま
            elif register == "polite" or tone == "polite":
                got = _polite_sentence(p)
            else:
                got = _plain_sentence(p)
                if not re.search(r"[。！？!?…]$", got):
                    got += "。"
            rebuilt.append(got.strip())
        out.append("".join(rebuilt))
    text_out = "\n".join(out)
    for i, block in enumerate(protected):
        text_out = text_out.replace(f"\x00{i}\x00", block)
    if tone or register:
        fixes.append(f"style:{tone or register}")
    return text_out, fixes


# --------------------------------------------------------------------------- #
# 長さ
# --------------------------------------------------------------------------- #
def count_chars(text: str) -> int:
    """指示の「文字数」と同じ数え方（空白と改行は数えない）。"""
    return len(re.sub(r"\s+", "", str(text or "")))


def clause_trim(sentence: str, budget: int) -> str:
    """文を節（、）の単位で budget に収める。述語を失う詰め方はしない（空を返す）。"""
    s = str(sentence or "").strip()
    if count_chars(s) <= budget or budget <= 8:
        return s
    marks = "".join(re.findall(r"[。！？!?…]+$", s))
    body = s[: len(s) - len(marks)] if marks else s
    parts = [p for p in re.split(r"(、|；|;|，)", body) if p]
    merged: list[str] = []
    i = 0
    while i < len(parts):
        chunk = parts[i]
        if i + 1 < len(parts) and re.fullmatch(r"[、；;，]", parts[i + 1]):
            chunk += parts[i + 1]
            i += 2
        else:
            i += 1
        merged.append(chunk)
    keep: list[str] = []
    used = 0
    for chunk in merged:
        if used + count_chars(chunk) > budget and keep:
            break
        keep.append(chunk)
        used += count_chars(chunk)
    if not keep or len(keep) == len(merged):
        return "".join(keep) + (marks or "")
    got = "".join(keep).rstrip("、；;， ")
    if not is_predicate_end(got):
        return ""                                  # 述語を失う＝文として壊れるので使わない
    return got + (marks or "。")


def fit_length(sentences: list[str], *, target: int = 0, hard_max: int = 0,
               weights: list[float] | None = None) -> tuple[list[str], list[str]]:
    """目標の長さに収まるように *文を選ぶ*（切らない）。重みの高い文を先に詰める。"""
    fixes: list[str] = []
    items = [s for s in sentences if s and s.strip()]
    if not items:
        return items, fixes
    weights = list(weights or [1.0] * len(items))

    def _select(limit: int) -> tuple[list[str], list[str]]:
        order = sorted(range(len(items)), key=lambda i: -float(weights[i]))
        chosen: list[int] = []
        used = 0
        for i in order:
            n = count_chars(items[i])
            if used + n > limit and chosen:
                continue
            chosen.append(i)
            used += n
        chosen.sort()
        return [items[i] for i in chosen], [weights[i] for i in chosen]

    if hard_max and sum(count_chars(s) for s in items) > hard_max:
        items, weights = _select(hard_max)
        fixes.append(f"length:{hard_max}字に収めた")
    if target and items:
        total = sum(count_chars(s) for s in items)
        if total > target * 1.55 and len(items) > 2:
            items, weights = _select(int(target * 1.25))
            fixes.append(f"fit:{target}字")
        elif total < target * 0.5:
            fixes.append("short_of_target")
    return items, fixes


__all__ = ["restyle", "retime", "fit_length", "clause_trim", "count_chars", "ends_with_tail",
           "is_predicate_end", "TAILS"]
