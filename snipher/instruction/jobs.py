"""「言葉の仕事」の型を読む — 材料の形ではっきり決まる依頼だけを拾う。

v3 までの指示層は、指示を 7 つのタスク（extract / summarize / code / answer /
transform / list / write）に振り分けていました。ところが実際に投げられた依頼には
そのどれでもない *言葉の仕事* が混ざります。

    空欄補充    「次の文の空欄に入る最も適切な助詞を1文字で」   → 助詞を選ぶ
    選択        「次の単語の中から「果物」だけを選んでください」 → 仲間だけを残す
    語の関係    「「嬉しい」と同じような意味／反対の語は？」     → 語義で引く
    論理        「人間は必ず息をします。太郎は人間です。…ますか？」→ 全称命題の適用
    規則の適用  「赤信号では止まれ。いま信号は赤です。どうすれば？」→ 条件文の適用
    語の写し    「犬と猫を英語に訳して」                        → 語彙の対応表
    礼          「「ありがとう」を使わずに感謝を表す返答を」      → 禁止語つきの定型

どれも *材料の形* で決まります（「空欄」という字がある、「はい」か「いいえ」を
求めている、引用符の語が並んでいる…）。だから分類器の確信度に任せず、ここで
構造として確定させます。会話の 1 通（「こんにちは」「今日はいい天気だね」）が
ここを通り抜けないよう、各型は「材料 + 操作 + 形の指定」を要求します。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..lang.phonetics import normalize


@dataclass
class Job:
    """読めた仕事の型（task と、実行に必要な手がかり）。"""

    task: str
    reason: str
    word: str = ""                 # 対象の語（引用符から読む）
    items: list[str] = field(default_factory=list)
    category: str = ""
    forbidden: list[str] = field(default_factory=list)
    language: str = ""

    def as_signal(self) -> str:
        return f"job:{self.task}"


# --------------------------------------------------------------------------- #
# 空欄補充
# --------------------------------------------------------------------------- #
_FILL_WORD = re.compile(r"空欄|ブランク|（\s*）|\(\s*\)|＿{2,}|_{2,}")
_FILL_ASK = re.compile(r"に入る|を埋め|補充|当てはまる|入る(?:語|言葉|助詞)")
_FILL_TARGET = re.compile(r"助詞|助動詞|接続詞|言葉|語|文字")
_ONE_CHAR = re.compile(r"(?:1|一|ひと)\s*文字|一字|1\s*字")

# --------------------------------------------------------------------------- #
# 選択（カテゴリに属するものだけを残す）
# --------------------------------------------------------------------------- #
_SELECT_ASK = re.compile(r"(?:だけ|のみ|に属する(?:もの)?)\s*(?:を)?\s*"
                         r"(?:選ん|選び|選べ|抜き出|拾っ|抽出|挙げ)")
_FROM_LIST = re.compile(r"(?:の中から|から|のうち)")
_BRACKET_ITEMS = re.compile(r"[\[［]([^\]］]{2,200})[\]］]")
_QUOTED = re.compile(r"[「『\"']([^「」『』\"']{1,40})[」』\"']")

# --------------------------------------------------------------------------- #
# 語の関係
# --------------------------------------------------------------------------- #
_SYN = re.compile(r"類義語|同義語|同じ(?:ような)?意味|似た(?:意味|言葉|語)|別の言い方|言い換え")
_ANT = re.compile(r"対義語|反対(?:の)?(?:意味|語)|逆の意味|反意語|オポジット")

# --------------------------------------------------------------------------- #
# かな書き（実辞書の読み）
# --------------------------------------------------------------------------- #
_KANA_ASK = re.compile(r"(?:を|は)?\s*(カタカナ|かたかな|ひらがな|平仮名|ローマ字)\s*"
                       r"(?:に|へ)\s*(?:変換|直し|して|に|表記)")

# --------------------------------------------------------------------------- #
# 語の写し（語彙の対応表）
# --------------------------------------------------------------------------- #
_GLOSS_ASK = re.compile(r"(?:を|は)?\s*(?:英語|英単語|えいご)\s*(?:に|へ)\s*"
                        r"(?:訳して|訳す|変換して|変えて|直して|して)")
_GLOSS_WORDS = re.compile(r"(?:次の)?\s*(?:単語|言葉|語|日本語)\s*[:：]")

# --------------------------------------------------------------------------- #
# 礼（禁止語つき）
# --------------------------------------------------------------------------- #
_THANKS_ART = re.compile(r"感謝|お礼|御礼|ありがたく|感謝の気持ち")
_THANKS_BAN = re.compile(r"[「『\"']([^「」『』\"']{1,12})[」』\"']\s*(?:を|は)?\s*"
                         r"(?:使わずに|使わないで|使わず|抜きで|無しで|なしで)")
_NEG_ASK = re.compile(r"使わずに|使わないで|使わず|抜きで|無しで|なしで|以外で|を除いて")

# --------------------------------------------------------------------------- #
# 指示文と入力データの点検（指示適用の検査 + 修正版）
# --------------------------------------------------------------------------- #
_AUDIT_ROLE = re.compile(r"指示文|指示(?:\s*\(|（)?Instruction|Instruction\b|システムプロンプト|"
                         r"入力データ|入力(?:\s*\(|（)?Input|Input\b|プロンプト構造")
_AUDIT_ASK = re.compile(r"判定|点検|検査|確認|検証|評価")
_AUDIT_FIX = re.compile(r"修正版|改善版|直した版|修正して|書き直して|整えた版|分離して")
_TAG = re.compile(r"<\s*(instruction|input|指示|入力)\s*>(.*?)<\s*/\s*\1\s*>",
                  re.IGNORECASE | re.DOTALL)
_LABEL = re.compile(r"(?:^|\n)\s*(?:指示文?|Instruction)\s*[:：]\s*(?P<ins>.+?)"
                    r"(?:\n\s*(?:入力データ?|Input)\s*[:：]\s*(?P<inp>.+))?$",
                    re.IGNORECASE | re.DOTALL)

# --------------------------------------------------------------------------- #
# 判定の本体
# --------------------------------------------------------------------------- #
def detect(raw: str) -> Job | None:
    """1 通の入力を読み、言葉の仕事なら `Job` を返す（会話なら None）。"""
    text = normalize(str(raw or "")).strip()
    if not text or len(text) < 6:
        return None
    t = text

    # --- 空欄補充 ---------------------------------------------------------- #
    if _FILL_WORD.search(t) and _FILL_ASK.search(t) and _FILL_TARGET.search(t):
        if _ONE_CHAR.search(t) or "助詞" in t:
            return Job("fill", "空欄 + 補充の指示 + 1 文字の指定")

    # --- 選択（材料が並んでいる） ------------------------------------------ #
    if _SELECT_ASK.search(t) and _FROM_LIST.search(t):
        m_list = _BRACKET_ITEMS.search(t)
        items: list[str] = []
        if m_list:
            items = [x.strip(" 　「」『』") for x in re.split(r"\s*[,、，]\s*", m_list.group(1))]
        else:
            quoted = [m.group(1).strip() for m in _QUOTED.finditer(t)]
            # カテゴリの語を除いた、2 語以上の引用は材料の並び
            q = [x for x in quoted if x not in ("選んで", "選び")]
            if len(q) >= 3:
                items = q[1:]
        items = [x for x in items if x and len(x) <= 12][:8]
        if len(items) >= 2:
            cat = ""
            m_cat = re.search(r"[「『]([^」』]{1,12})[」』]\s*(?:だけ|のみ|を|に)", t)
            if m_cat:
                cat = m_cat.group(1).strip()
            return Job("select", f"「〜の中から◯◯だけ」の形（材料 {len(items)} 件）",
                       items=items, category=cat)

    # --- 語の関係 ---------------------------------------------------------- #
    if _SYN.search(t) or _ANT.search(t):
        word = ""
        m = _QUOTED.search(t)
        if m:
            word = m.group(1).strip()
        if not word:
            m2 = re.search(r"([^\s、。]{1,12}?)(?:の)?(?:類義語|同義語|対義語|反対語)", t)
            word = m2.group(1) if m2 else ""
        if word:
            kind = "antonym" if _ANT.search(t) else "synonym"
            return Job("relations", f"{'対義語' if kind == 'antonym' else '類義語'}の依頼",
                       word=word)

    # --- かな書き（山羊 → やぎ／りんご → リンゴ） -------------------------- #
    m_kana = _KANA_ASK.search(t)
    if m_kana:
        word = ""
        m = _QUOTED.search(t)
        if m:
            word = m.group(1).strip()
        else:
            m2 = re.match(r"^(.{1,16}?)(?:を|は|の)", t)
            word = (m2.group(1).strip() if m2 else "")
        if word and not re.search(r"カタカナ|ひらがな|ローマ字|変換|ください", word):
            return Job("kana", f"{m_kana.group(1)} への書き換え（実辞書の読みを使う）", word=word)

    # --- 語の写し（英語へ） ------------------------------------------------- #
    if _GLOSS_ASK.search(t) or (_GLOSS_WORDS.search(t) and "英語" in t):
        words = [m.group(1).strip() for m in _QUOTED.finditer(t)]
        if not words:
            m_body = re.search(r"[:：]\s*([^。\n]+)", t)
            if m_body:
                words = [x.strip(" 　") for x in re.split(r"[,、，]", m_body.group(1))]
        if not words:
            words = [x for x in re.split(r"(?:と|、|,)", re.sub(r"(?:を|は).*$", "", t))
                     if x.strip(" 　")]
        words = [w for w in words if w and len(w) <= 12][:6]
        if words:
            return Job("gloss", f"語の写し（{len(words)} 語）", items=words, language="english")

    # --- 礼（禁止語つき） -------------------------------------------------- #
    if _THANKS_ART.search(t) and _NEG_ASK.search(t):
        banned = [m.group(1).strip() for m in _THANKS_BAN.finditer(t)]
        if banned:
            return Job("thanks", f"礼の依頼 + 禁止語 {banned}", forbidden=banned)
        # 「感謝を表す短い返答をください」のように禁止語が引用符で無い場合
        return Job("thanks", "礼の依頼（言い換えの要求）")

    # --- 指示文と入力データの点検 ------------------------------------------ #
    if _AUDIT_ROLE.search(t) and _AUDIT_ASK.search(t) and _AUDIT_FIX.search(t):
        return Job("audit", "指示文と入力データの点検（判定 + 修正版の依頼）")

    # --- 論理（全称命題・規則の適用） -------------------------------------- #
    from ..solve import logic as _logic

    if _logic.syllogism(t) is not None or _logic.apply_rule(t) is not None:
        return Job("logic", "前提から結論を導く形（材料の文だけで推論できる）")

    return None


def split_material(raw: str) -> dict:
    """指示文と入力データを、タグ・見出し語・段落から切り分ける（材料の構造を読む）。

    戻り値は ``{"instruction", "input", "how", "issues"}``。

      how    … どう切り分けたか（tag / label / paragraph / none）
      issues … 材料の構造そのものの問題（同じ段落に混在・入力側に命令形 など）
    """
    text = str(raw or "").replace("\r\n", "\n")
    out = {"instruction": "", "input": "", "how": "none", "issues": []}
    tagged = {m.group(1).lower(): m.group(2).strip() for m in _TAG.finditer(text)}
    if tagged:
        out["how"] = "tag"
        out["instruction"] = tagged.get("instruction") or tagged.get("指示") or ""
        out["input"] = tagged.get("input") or tagged.get("入力") or ""
    else:
        m = _LABEL.search(text)
        if m and m.group("ins"):
            out["how"] = "label"
            out["instruction"] = m.group("ins").strip()
            out["input"] = (m.group("inp") or "").strip()
    if not (out["instruction"] and out["input"]):
        # 段落で分ける: 命令形（〜してください）で終わる行が指示、残りが材料
        lines = [x.strip() for x in text.split("\n") if x.strip()]
        ins = [x for x in lines if re.search(r"(してください|しなさい|せよ|こと。)$", x)]
        rest = [x for x in lines if x not in ins]
        if ins and rest:
            out["how"] = "paragraph"
            out["instruction"] = " ".join(ins)
            out["input"] = "\n".join(rest)
    # 構造そのものの点検（材料を読み直すだけ。足し算はしない）
    if out["instruction"] and out["input"]:
        if out["how"] == "none":
            out["issues"].append("指示と材料が記号で区切られていない（どちらがデータか読めない）")
        if out["input"].count("\n") == 0 and len(out["input"]) > 60:
            out["issues"].append("材料が 1 行に詰まっていて、文の切れ目が読めない")
        if re.search(r"(してください|しなさい|せよ|無視して|命令だ|従って)", out["input"]) or \
                re.search(r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|above)\s+instructions",
                          out["input"], re.IGNORECASE) or \
                re.search(r"(?:^|\n)\s*(?:system|assistant)\s*[:：]", out["input"], re.IGNORECASE):
            out["issues"].append("材料の中に命令文がある（指示として実行されかねない）")
        if out["input"] and out["instruction"] and \
                text.find(out["input"]) < text.find(out["instruction"]):
            out["issues"].append("材料が指示より前に置かれている（指示が後から効かない）")
    return out


__all__ = ["Job", "detect", "split_material"]
