"""指示のコンパイラ — 「何をさせたいか」と「何を材料にするか」を分離する。

v3 までの応答は *発話全体を 1 つの話題* として扱っていました。だから

    「次のテキストから情報を抽出し、JSON 形式のみで出力してください。
      テキスト: 「東京から京都まで新幹線で約2時間15分。」」

という 1 通の中では「テキスト」という語がいちばん強く引けてしまい、
辞書の読み（てきすと・4 拍・名詞）が返っていました。これは知識の不足ではなく
**指示文とデータを分離する層が無い**ことの症状です。

ここで行うのは 3 つだけです。

    1. 指示部（ imperative / 役割 / 出力仕様 ）と材料部（ payload / question ）を切り分ける
    2. タスクを 1 つに決める（extract / summarize / code / answer / transform / list / write）
    3. 出力仕様（JSON スキーマ・箇条書きの数・文字数・文体・禁止事項）を *機械的に検証できる形* にする

3 が肝です。仕様が決定的なデータになれば、生成した応答を `run.verify()` で
「指示通りか」に照らして検査し、外れていたら組み直せます（＝指示追従が測れる）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..lang.phonetics import normalize
from ..mind.frame import Rule
from ..mind.rules import compile_rules, to_int

# --------------------------------------------------------------------------- #
# 材料（payload）の目印
# --------------------------------------------------------------------------- #
#: 「テキスト:」「文章：」「本文」「原文」「入力」「対象」「データ」「資料」「課題」「引用」…
#: 指示文が *データを差し出す* ときに使う語。この後ろにあるのが材料で、話題ではない。
PAYLOAD_MARKERS = (
    "テキスト", "文章", "本文", "原文", "入力", "対象", "データ", "資料", "課題", "引用",
    "抜粋", "要約する文", "内容", "項目", "リスト", "一覧", "候補", "要素", "リスト",
    "次の文", "以下の文", "この文", "次の文章", "以下の文章",
    "以下のテキスト", "下記", "以下", "次", "この文章", "その文章", "問題文", "説明文",
    "質問", "問い", "単語", "単語リスト", "キー", "数値", "値",
    "text", "input", "content", "passage", "data", "document", "prompt", "output",
    "question", "query",
)

_LABELS = "|".join(sorted(PAYLOAD_MARKERS, key=len, reverse=True))
_PAYLOAD_LABEL = re.compile(
    rf"(?:次の|以下の|下記の?|この|以下の|下記の?)?\s*(?:{_LABELS})\s*"
    rf"(?:は|って|という|と呼ばれる)?\s*[：:]\s*",
    re.IGNORECASE,
)
#: 「テキスト: 「…」」のように引用符で材料を括る書き方
_QUOTED_AFTER_LABEL = re.compile(
    rf"(?:{_LABELS})\s*(?:は)?\s*[：:]\s*(?:[「『]|\"\"\")(.+?)(?:[」』]|\"\"\")",
    re.S | re.IGNORECASE,
)
#: 全体を「」で括った 12 文字以上の塊（文末が述語なら文＝材料とみなす）
_QUOTE_BLOCK = re.compile(r"[「『]([^」』]{12,})[」』]", re.S)
#: ``` … ``` フェンス
_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.S)
#: 「質問：〜」「問：〜」「Q: 〜」（行末まで）
_QUESTION = re.compile(
    r"(?:質問|問い|問|設問|課題|クエスチョン|依頼|リクエスト|お願い|q)\s*[：:]\s*([^\n]+)",
    re.IGNORECASE,
)
#: 文の終わり（材料が「文」として立っているかの判定に使う）
_SENTENCE_END = re.compile(r"(?:。|です|ました|である|だ|た|る|い|な|よ|ね|？|\?)\s*$")
#: 「〜してください。」の後ろに続く文が、まだ *指示* である印（v8）。
#: 「他の文字は含めないでください」「改行や説明は不要です」は材料ではない。
_INSTRUCTION_TAIL = re.compile(
    r"(?:ください|下さい|しなさい|せよ|しろ|なさい|ちょうだい|くれ|してほしい|"
    r"不要|いらない|なし|無し|禁止|使わず|抜きで|だけで|のみ|"
    r"添えて|添えない|しないで|やめて|避けて|除いて|除き)"
)
#: 問いかけの印（疑問詞 + 文末）。「言葉の仕事」の型が読めた 1 通を後押しする。
_WH_QUESTION = re.compile(
    r"(?:どちら|どれ|どっち|いずれ|何|いつ|どこ|だれ|誰|なぜ|なんで|どう|"
    r"いくつ|いくら|どの).{0,8}(?:[?？]|ですか|ますか|でしょうか|なの|のか|こと)"
)
#: 「〜してください。」「〜を出力せよ。」のような命令の終わり
_IMPERATIVE_SENT = re.compile(
    r"[^\n。]{0,60}?(?:してください|して下さい|てください|て下さい|ください|下さい|"
    r"くださいます?か|いただけないでしょうか|お願いします|"
    r"してちょうだい|してくれ|しなさい|せよ|しろ|して|応じて|従って|沿って|まとめて|要約して|"
    r"抽出して|抜き出して|出力して|返して|書いて|作成して|作って|実装して|答えて|教えて|述べよ|"
    r"列挙して|説明して|変換して|直して|修正して|評価して|比較して|分類して|翻訳して|訳して)"
    r"[。！!]?",
)
_IMPERATIVE_EN = re.compile(
    r"\b(?:please\s+)?(answer|explain|summarize|summarise|list|write|extract|translate|"
    r"describe|compare|convert|generate|create|output|return|provide|give|rewrite|count)\b",
    re.IGNORECASE)
_SENTENCES = re.compile(r"(?:([0-9０-９]+|[一二三四五六七八九十]+)\s*文(?:で|以内|以下|程度|くらい|に)|"
                        r"in\s+([0-9]+)\s+sentences?|([0-9]+)\s*sentence\s+(?:answer|summary|reply))",
                        re.IGNORECASE)
_ARTIFACT = re.compile(r"(?:メール|電子メール|記事|レポート|報告書|文案|コピー|手紙|案内文|お詫び|"
                       r"説明文|議事録|スピーチ|プレゼン|ポエム|詩|短文|作文|スローガン|見出し|タイトル|"
                       r"挨拶|あいさつ|自己紹介|メッセージ|返事|"
                       r"essay|article|email|report|memo|paragraph|headline|slogan|speech)",
                       re.IGNORECASE)

# 成果物の *形* を指定する語（語尾・長さ・口調）。成果物の名 + 形指定 が揃うと
# 「X の挨拶」という名詞句でも、それは指示（生成依頼）です。
_STYLE_SPEC = re.compile(
    r"(語尾|つける|suffix|終わる|で終わ|[0-9０-９]+\s*(?:文字|字|文|語)|一言|ひとこと|"
    r"である調|ですます|です・ます|カジュアル|フレンドリー|専門的|フォーマル|口調)")
# 形指定に *具体値* が無いと artifact+style は加点しない。裸の「語尾」だと
# 「あの挨拶の語尾はどうする？」のような *メタ質問* を生成依頼と誤認する。
# 具体値 = 長さ（N文字）・口調（である調等）・引用符で示された語（「〜ロボ」）。
# 語尾指定の実体値: 「語尾に「X」」「語尾に〜ロボ」(角括弧は必須ではない・〜付き/なし)。
_SUFFIX_SPEC = re.compile(
    r"(?:語尾|語末|末尾)に[「『]?(?:〜|~)?[A-Za-z０-９一-龯ぁ-んァ-ヶー]{1,12}[」』]?")
_STYLE_VALUE = re.compile(
    r"[0-9０-９]+\s*(?:文字|字|文|語)|一言|ひとこと|"
    r"(?:である調|ですます|です・ます|カジュアル|フレンドリー|専門的|フォーマル)|"
    r"(?:語尾|語末|末尾)に[「『]?(?:〜|~)?[A-Za-z０-９一-龯ぁ-んァ-ヶー]{1,12}[」』]?")
_ROLE = re.compile(
    r"(?:あなた|君|きみ|お前|そちら|assistant|ai)\s*(?:は|って|として)\s*"
    r"[「『\"']?([^」』\"'。\n]{2,40})[」』\"']?\s*(?:という|の)?\s*"
    r"(?:役|役割|ロール|キャラ|キャラクター|設定|人|です|として|になりきって|になって)",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# 出力仕様の目印
# --------------------------------------------------------------------------- #
_JSON_ONLY = re.compile(
    r"(?:json|JSON|Json)\s*(?:形式|フォーマット|記法|だけ|のみ|のみで|で)?\s*"
    r"(?:のみ|だけ|以外|なし|無し)?\s*(?:で)?\s*(?:出力|表示|返|書|出|answer|respond|print)?",
)
_JSON_WORD = re.compile(r"\bjson\b|ｊｓｏｎ|ジェイソン", re.IGNORECASE)
_JSON_AS_OUTPUT = re.compile(r"json\s*(?:形式|フォーマット|記法|で|に|として|のみ|だけ)", re.IGNORECASE)
_CSV_WORD = re.compile(r"(?<![A-Za-z])csv(?![A-Za-z])|カンマ区切り|コンマ区切り", re.IGNORECASE)
_TABLE_WORD = re.compile(r"表形式|テーブルで|markdown\s*の?表|マークダウンの表|(?<![A-Za-z])table(?![A-Za-z])", re.IGNORECASE)
_KEYVALUE_WORD = re.compile(r"キーと値|key\s*[:：\-]?\s*value|keys?\s*and\s*values?|"
                            r"項目\s*[：:]\s*値|k\s*[:：]\s*v|ラベルと値", re.IGNORECASE)

_N = r"(?:[0-9０-９]+|[一二三四五六七八九十]+)"
_BULLETS = re.compile(
    rf"(?:(?P<a>{_N})\s*(?:つの|個の|本の|点の|件の)?\s*"
    rf"(?:箇条書き|ポイント|要点|項目|ブルレット|bullet ?points?|リスト)|"
    rf"箇条書き\s*(?:で|にして)?\s*(?P<b>{_N})\s*(?:つ|個|本|点)|"
    rf"(?P<c>{_N})\s*(?:行|つ|個|ポイント|要点)\s*(?:で|以内に)?\s*(?:短く)?\s*"
    rf"(?:まとめて|要約して|まとめてください|書く|列挙して|挙げて|あげて)|"
    rf"(?P<e>{_N})\s*(?:つ|個|本|点|行|件)\s*(?:を|に)?\s*"
    rf"(?:箇条書き|ポイント|要点|リスト|ブルレット)?\s*(?:で|にして|に)?\s*"
    rf"(?:列挙|挙げ|あげ|書|出|まとめ|要約|並べ|書き出し)|"
    rf"(?:ポイント|要点|項目|箇条書き)\s*(?:を|は)?\s*(?P<d>{_N})\s*(?:つ|個|本|点|つほど)"
    rf"(?:\s*(?:で|に)?\s*(?:短く)?\s*(?:まとめて|要約して|挙げ|あげ|列挙|書|出))?)",
    re.IGNORECASE,
)
_BULLET_WORD = re.compile(r"箇条書き|箇条書|ポイントで|要点を|リストで|リスト化|bullet", re.IGNORECASE)
_NUMBERED_WORD = re.compile(r"番号付き|番号を振っ|ナンバリング|numbered", re.IGNORECASE)

_LEN_PAT = r"([0-9０-９]+|[一二三四五六七八九十百]+)\s*(?:文字|字|chars?|characters?|文字数)"
_LEN_KIND = re.compile(
    rf"{_LEN_PAT}\s*(?:程度|くらい|ぐらい|前後|以内|以下|くらいで|で|ほど|を目安に|を目標に)?")
_SHORT_WORDS = re.compile(r"(?:短く|簡潔に|簡潔な|シンプルに|端的に|ひとことで|一言で|1文で|一文で|手短に)")
_LONG_WORDS = re.compile(r"(?:詳しく|詳細に|丁寧に|たっぷり|長めに|掘り下げて|網羅的に)")

_TONE_FRIENDLY = re.compile(
    r"(?:親しみやすい|フレンドリー|カジュアル|くだけた|タメ口|ため口|やわらかい|柔らかい|"
    r"やさしい|優しい|口調|〜?だよ|〜?だね|だよね|口ぶりで|フランク)")
_TONE_PRO = re.compile(r"(?:専門的|プロフェッショナル|技術的|エキスパート|ベテラン(?![一-龯A-Za-z])|論理的|的確)")
_PRO_WORD = re.compile(r"ベテラン|[一-龯]{2,}(?:的|家|者)")
_TONE_POLITE = re.compile(r"(?:ですます|です・ます|敬体|丁寧語|丁寧な|敬語|ビジネスメール|フォーマル)")
_TONE_PLAIN = re.compile(r"(?:だ・である|である調|常体|論文調|レポート調)")

_NO_GREETING = re.compile(r"(?:挨拶|前置き|導入|説明|解説|余計|不要| unnecessary|no preamble)")
_ONLY_OUTPUT = re.compile(r"(?:のみ|だけ|以外は何も|余計な|他に何も|without any)")

_CODE_WORDS = re.compile(
    r"(?:コード|プログラム|スクリプト|関数|クラス|メソッド|実装|snippet|code|function|class|"
    r"script|program|メソッドを書いて)", re.IGNORECASE)
# 日本語の中では `\b` が効かない（「Pythonで」の n と で はどちらも単語構成文字）。
# 前後を「英数字・_・.・-・+ が続かない」で見る。
_LB = r"(?<![A-Za-z0-9_])"
_RB = r"(?![A-Za-z0-9_])"
_LANG_NAMES = (
    "python|javascript|typescript|node\.js|next\.js|react|vue|deno|bun|java(?!script)|golang|"
    "ruby|php|kotlin|swift|scala|perl|lua|racket|haskell|elixir|erlang|clojure|dart|rust|"
    "objective-c|c\+\+|c#|f#|visual basic|assembly|sql|html|css|sass|scss|bash|zsh|shell|"
    "powershell|yaml|toml|dockerfile|makefile|solidity|terraform"
)
_LANG_WORDS = re.compile(rf"{_LB}({_LANG_NAMES}){_RB}", re.IGNORECASE)
#: 2〜3 文字の言語名は *単独の語* として立っているときだけ（誤爆防止）
_LANG_SHORT = re.compile(
    rf"{_LB}(js|ts|go|c|r|jsx|tsx|py|rb|cs)(?:{_RB}|(?=[言語でにとを]))", re.IGNORECASE)
_LANG_ALIAS = {"js": "javascript", "ts": "typescript", "py": "python", "rb": "ruby",
               "cs": "c#", "node.js": "javascript", "jsx": "javascript", "tsx": "typescript",
               "golang": "go", "shell": "bash", "zsh": "bash", "powershell": "bash",
               "next.js": "javascript", "react": "javascript", "vue": "javascript",
               "deno": "javascript", "bun": "javascript", "scss": "css", "sass": "css"}
#: マークアップ／データ記述言語（「コードを書いて」の明示か成果物の名が要る）
_MARKUP_LANGS = {"html", "css", "yaml", "toml", "sql", "dockerfile", "makefile", "scss", "sass",
                 "json", "xml", "regex"}
_CODE_ARTIFACT = re.compile(
    r"(?:関数|クラス|メソッド|スクリプト|プログラム|コード|ライブラリ|モジュール|テスト|"
    r"アルゴリズム|api|sdk|cli|regex|正規表現|html|css|sql|エンドポイント|サーバー|フォーム|ページ)",
    re.IGNORECASE)
_FUNC_NAME = re.compile(r"`\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:\(([^)`]*)\))?\s*`")
_FUNC_NAME2 = re.compile(r"(?:関数|function|メソッド)\s*`?([A-Za-z_$][A-Za-z0-9_$]*)`?\s*(?:\(([^)]*)\))?")

_EXTRACT_WORDS = re.compile(
    r"(?:抽出|抜き出し|取り出し|情報を取り|エンティティ|構造化|フィールド|項目を埋め|情報を整理し|"
    r"extract|parse out|pull out|structured)", re.IGNORECASE)
_SUMMARY_WORDS = re.compile(
    r"(?:要約|まとめ(?:て|る)|要点|サマリ|サマリー|概括|要旨|summar|tl;dr| abstract )", re.IGNORECASE)
_LIST_WORDS = re.compile(r"(?:列挙|列举|リストアップ|挙げて|あげて|並べて|書き出して|enumerate)", re.IGNORECASE)
_TRANSLATE_WORDS = re.compile(r"(?:翻訳|訳して|訳し|に翻訳|translate)", re.IGNORECASE)
_CLASSIFY_WORDS = re.compile(r"(?:分類|仕分け|カテゴリー分け|カテゴリ分け|類別|classify|分類して|仕分けて)", re.IGNORECASE)
_ANSWER_WORDS = re.compile(
    r"(?:答えて|答えよ|回答して|回答せよ|応えて|解説して|説明して|教えて|述べて|説明せよ|"
    r"answer|explain|describe|define|compare|discuss|elaborate)",
    re.IGNORECASE)
_WRITE_WORDS = re.compile(
    r"(?:を[^。\n]{0,24}(?:書いて|書き|作成して|作って|したためて|執筆し|まとめて)|"
    r"(?:書いて|作成して|作って)(?:ください|下さい|くれ|ほしい|頂く|いただく)|"
    r"を書いて|を書いてください|を書いて下さい|作成して|作文|エッセイ|記事を書いて|文案|"
    r"コピーを考えて|write|compose|draft)", re.IGNORECASE)

_KANJI_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
              "十": 10}


def _num(value: str) -> int | None:
    v = normalize(str(value or "")).replace(",", "")
    if v.isdigit():
        return int(v)
    got = to_int(v)
    if got:
        return got
    return None


@dataclass
class FormatSpec:
    """指示が求める *出力の形*。ここが決定的なので、生成後に機械検査できる。"""

    kind: str = ""                       # json / csv / table / keyvalue / ""（指定なし）
    strict: bool = False                 # 「JSON のみ」「余計な挨拶は不要」＝飾りを一切付けない
    only_output: bool = False            # 指定の形だけを返す
    no_greeting: bool = False
    no_explanation: bool = False
    schema_fields: list[tuple[str, str]] = field(default_factory=list)   # [(key, hint)]
    schema_template: str = ""            # JSON テンプレートそのもの（入れ子の形を保つ）
    indent: int = 2
    bullets: int = 0
    sentences: int = 0                   # 「2文で」「in 2 sentences」
    bullet_char: str = "・"
    numbered: bool = False
    lines: int = 0
    max_chars: int = 0
    target_chars: int = 0                # 「200文字程度」の目標
    length_kind: str = ""                # approx / max / min
    brief: bool = False                  # 「短く」だけ（文字数の指定は無い）
    tone: str = ""                       # friendly / professional / polite / plain
    tone_words: list[str] = field(default_factory=list)
    language: str = ""                   # 出力に指定された自然言語 / プログラミング言語
    register: str = ""                   # polite / plain
    extra_rules: list[Rule] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"kind": self.kind or None, "strict": self.strict,
                "schema": [k for k, _ in self.schema_fields],
                "schema_template": self.schema_template or None,
                "bullets": self.bullets or None, "sentences": self.sentences or None,
                "numbered": self.numbered or None,
                "target_chars": self.target_chars or None, "length_kind": self.length_kind or None,
                "max_chars": self.max_chars or None, "tone": self.tone or None,
                "tone_words": self.tone_words or None, "no_greeting": self.no_greeting or None,
                "no_explanation": self.no_explanation or None, "language": self.language or None,
                "brief": self.brief or None}


@dataclass
class Directive:
    """1 通の指示の読み取り結果。"""

    task: str = ""                       # extract / summarize / code / answer / transform / list / write
    raw: str = ""
    instruction: str = ""                # 指示部（材料を抜いた文）
    payload: str = ""                    # 材料（＝処理対象のデータ）
    question: str = ""                   # 「質問：〜」で差し出された問い
    fmt: FormatSpec = field(default_factory=FormatSpec)
    role: str = ""                       # 「あなたは〜です」で与えられた役割
    job: object | None = None             # jobs.detect が読んだ「言葉の仕事」の型
    confidence: float = 0.0
    signals: list[str] = field(default_factory=list)
    markers: list[str] = field(default_factory=list)   # 根拠にした指示語（UI 表示用）
    rules: list[Rule] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def strict(self) -> bool:
        return bool(self.fmt.strict)

    @property
    def subject(self) -> str:
        """答え／処理の対象になる文（問いがあればそれが主役）。"""
        return (self.question or self.payload or self.instruction or self.raw).strip()

    def as_dict(self) -> dict:
        return {"task": self.task, "confidence": round(self.confidence, 3),
                "role": self.role or None, "question": self.question or None,
                "payload_chars": len(self.payload or ""), "signals": self.signals[:8],
                "markers": self.markers[:8], "format": self.fmt.as_dict(),
                "strict": self.strict}


# --------------------------------------------------------------------------- #
# JSON スキーマの読み取り
# --------------------------------------------------------------------------- #
def find_json_templates(text: str) -> list[tuple[str, int, int]]:
    """文中の JSON 型テンプレートを波括弧の対応を取って抜き出す（span 付き）。"""
    out: list[tuple[str, int, int]] = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth, j, quote = 0, i, ""
        while j < len(text):
            c = text[j]
            if quote:
                if c == "\\":
                    j += 2
                    continue
                if c == quote:
                    quote = ""
            elif c in "\"'":
                quote = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth == 0 and j > i:
            body = text[i:j + 1]
            if re.search(r'"[^"\n]{1,30}"\s*:', body) and len(body) <= 4000:
                out.append((body, i, j + 1))
            i = j
    # 入れ子（外側が内側を含む）は外側だけ残す
    kept: list[tuple[str, int, int]] = []
    for body, a, b in out:
        # ほかのテンプレートに *含まれている* ものは内側 → 捨てる
        if any(a2 <= a and b <= b2 and (a, b) != (a2, b2) for _x, a2, b2 in out):
            continue
        kept.append((body, a, b))
    return kept


def schema_fields(template: str) -> list[tuple[str, str]]:
    """{"origin": "出発地", ...} → [("origin", "出発地"), ...]（順序を保つ）。

    入れ子（`"customer": {"name": "顧客名"}`）は *葉の欄* を返します。形そのものは
    `FormatSpec.schema_template` に残してあるので、出力は入れ子のまま組み直せます。
    """
    try:
        obj = json.loads(template)
    except Exception:  # noqa: BLE001
        obj = None
    if isinstance(obj, (dict, list)):
        leaves: list[tuple[str, str]] = []

        def walk(node) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(v, (dict, list)):
                        walk(v)
                    elif k not in [x for x, _ in leaves]:
                        leaves.append((str(k), str(v) if isinstance(v, str) else ""))
            elif isinstance(node, list):
                for it in node:
                    walk(it)

        walk(obj)
        if leaves:
            return leaves

    out: list[tuple[str, str]] = []
    for m in re.finditer(r'"([^"\n]{1,40})"\s*:\s*(?:"([^"\n]{0,80})"|(\[[^\]\n]*\])|(\{[^}\n]*\})|([^,\n}]+))',
                         template):
        key = m.group(1).strip()
        val = (m.group(2) if m.group(2) is not None else
               m.group(3) or m.group(4) or m.group(5) or "").strip()
        if key and key not in [k for k, _ in out]:
            out.append((key, val))
    for m in re.finditer(r'\"([^\"\n]{1,40})\"\s*:\s*(?=(?:,|\n|}|$))', template):
        key = m.group(1).strip()
        if key and key not in [k for k, _ in out]:
            out.append((key, ""))
    if len(out) < 2:
        keys = re.findall(r'\"([A-Za-z_][A-Za-z0-9_]*)\"\s*:', template)
        for k in keys:
            if k not in [x for x, _ in out]:
                out.append((k, ""))
    return out


_CSV_HEADER_LABEL = re.compile(
    r"(?:csv|カンマ区切り|コンマ区切り)\s*(?:フォーマット|形式|ヘッダー|の形|で)?\s*[：:]\s*([^\n]+)",
    re.IGNORECASE)
_FIELD_LIST_LABEL = re.compile(
    r"(?:項目|欄|キー|フィールド|columns?|fields?)\s*[：:]\s*([^\n]+)", re.IGNORECASE)
_LATIN_HEADER = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)+)\s*$")
_EXPLICIT_KEY = re.compile(
    r"キー(?:名)?(?:は|を|が)?\s*[「『\"']?"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\s*[、，,と・]\s*[A-Za-z_][A-Za-z0-9_]*)*)"
    r"[」』\"']?\s*(?:に|で|と|として)")


def _is_sentence_list(body: str) -> bool:
    """`項目：` の後ろが *欄名の列* ではなく *文* かどうか（材料をスキーマにしないため）。"""
    src = str(body or "")
    if re.search(r"(?:。|です|ます|だった|である|有名|完成|開業|開通|生まれた|住んで)", src):
        return True
    return len(src) > 60


def _split_field_names(body: str) -> list[str]:
    """欄名の列を割る（`name,age,city` / `名前と値段` / `名前・値段`）。"""
    src = str(body or "")
    # 日本語の「名前と値段」は空白が無いので、*名詞をつなぐ と* で割る
    src = re.sub(r"(?<=[一-龯ァ-ヶーA-Za-z0-9])と(?=[一-龯ァ-ヶーA-Za-z])", "、", src)
    parts = [x.strip(" 　。、,.・/") for x in re.split(r"[,、・]", src)]
    return [p for p in parts if p and len(p) <= 24][:12]


def parse_schema(text: str) -> tuple[list[tuple[str, str]], list[tuple[int, int]]]:
    """出力スキーマ（キーと日本語ヒント）を読む。

    読む順番は (1) JSON テンプレート (2) `CSVフォーマット: name,age,city` のヘッダ行
    (3) `項目: 名前と値段` の欄名リスト (4) 単独の欧文ヘッダ行。
    どれも *指示文に書かれていた形* そのままなので、欄名をでっち上げません。
    """
    t = str(text or "")
    spans: list[tuple[int, int]] = []
    fields_: list[tuple[str, str]] = []
    for body, a, b in find_json_templates(t):
        got = schema_fields(body)
        if not got:
            continue
        spans.append((a, b))
        for k, v in got:
            if k not in [x for x, _ in fields_]:
                fields_.append((k, v))
    if fields_:
        return fields_, spans

    m = _CSV_HEADER_LABEL.search(t)
    if m:
        names = _split_field_names(m.group(1))
        if len(names) >= 2:
            spans.append((m.start(1), m.end(1)))
            return [(n, "") for n in names], spans
    m = _FIELD_LIST_LABEL.search(t)
    if m and not _is_sentence_list(m.group(1)):
        names = _split_field_names(m.group(1))
        if names:
            spans.append((m.start(1), m.end(1)))
            return [(n, n) for n in names], spans
    # (3.5) 「キーを user_name にした JSON」のようにキーを名指しした指定（v8）。
    # 出力の欄名はここで確定する（値のラベルと違ってもよい — 突き合わせは概念で行う）。
    m = _EXPLICIT_KEY.search(t)
    if m:
        names = [x.strip().strip("「」『』\"'") for x in re.split(r"[、，,と・]", m.group(1))]
        names = [n for n in names if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n or "")]
        if names:
            spans.append((m.start(1), m.end(1)))
            return [(n, "") for n in names[:8]], spans
    for line in t.split("\n"):
        mline = _LATIN_HEADER.match(line)
        if mline:
            names = [x.strip() for x in mline.group(1).split(",")]
            if len(names) >= 2:
                return [(n, "") for n in names], spans
    return fields_, spans


# --------------------------------------------------------------------------- #
# 出力仕様の読み取り
# --------------------------------------------------------------------------- #
def parse_format(text: str, *, schema: list[tuple[str, str]] | None = None) -> FormatSpec:
    """指示文から出力仕様を作る。`schema` は parse_schema の結果（無い場合はここで探す）。"""
    t = str(text or "")
    low = t.lower()
    f = FormatSpec()
    if schema:
        f.schema_fields = list(schema)
    if not f.schema_template:
        tmpl = find_json_templates(t)
        if tmpl:
            f.schema_template = tmpl[0][0]

    # ---- 形式 ---- #
    # 明示された形式（CSV / 表 / key: value）を *先に* 読みます。JSON テンプレートが
    # 無い指示でもスキーマ欄は読めるので、ここで JSON 扱いにすると出力が化けます。
    has_json_template = bool(f.schema_fields) and bool(_JSON_WORD.search(t)) \
        or bool(_JSON_AS_OUTPUT.search(t)) or bool(_JSON_ONLY.search(t))
    if _CSV_WORD.search(t):
        f.kind = "csv"
    elif _TABLE_WORD.search(t):
        f.kind = "table"
    elif _KEYVALUE_WORD.search(t):
        f.kind = "keyvalue"
    elif bool(_JSON_WORD.search(t)) or has_json_template or bool(f.schema_fields):
        f.kind = "json"

    # 「JSON 形式のみで出力」「余計な挨拶や解説は不要」→ strict
    only = bool(_ONLY_OUTPUT.search(t))
    needless = bool(re.search(r"(?:不要です|不要|いらない|無しで|なしで|without|省略して|省いて)", t))
    if f.kind and (bool(_JSON_AS_OUTPUT.search(t)) or needless
                   or re.search(r"(?:のみ|だけ)", t)
                   or re.search(r"json\s*(?:形式|フォーマット)?\s*(?:のみ|だけ)", low)):
        f.strict = True
        f.only_output = True
    if re.search(r"(?:余計な|他の|ほかの|それ以外の)?\s*(?:挨拶|前置き|解説|説明|注釈)\s*(?:は|も)?\s*(?:不要|いらない|なし|無し|しないで|添えない)", t):
        f.no_greeting = True
        f.no_explanation = True
        f.strict = f.strict or bool(f.kind)
    if only and f.kind:
        f.only_output = True
    if f.no_explanation and f.kind:
        f.strict = True

    # ---- 箇条書き ---- #
    m = _BULLETS.search(t)
    if m:
        raw = next((g for g in m.groups() if g), "")
        n = _num(raw)
        if n and 1 <= n <= 30:
            f.bullets = n
    msent = _SENTENCES.search(t)
    if msent and not f.bullets:
        n = _num(next((g for g in msent.groups() if g), ""))
        if n and 1 <= n <= 20:
            f.sentences = n
    if _NUMBERED_WORD.search(t):
        f.numbered = True
    mline = re.search(r"([0-9０-９]+|[一二三四五六七八九十]+)\s*行\s*(?:で|以内|に)", t)
    if mline:
        n = _num(mline.group(1))
        if n:
            f.lines = n
            f.bullets = f.bullets or n

    # ---- 長さ ---- #
    # 「1 文字以上 2 文字以内」のように複数あるときは、*上限（以内/以下/まで）* を
    # 優先して max_chars にする（v8: 先に読んだ「1 文字」を目標 1 字にしない）。
    _max_ns: list[int] = []
    _approx_n = 0
    for m in _LEN_KIND.finditer(t):
        n = _num(m.group(1))
        if not n or n > 100000:
            continue
        tail = t[m.end():m.end() + 4]
        # `_LEN_KIND` が「以内」まで飲み込むので、*一致した文字列自体* でも判定する
        if re.search(r"(?:以内|以下|まで|を超えない|以内に収め)", m.group(0)) \
                or re.match(r"\s*(?:以内|以下)", tail):
            _max_ns.append(n)
        elif re.match(r"\s*(?:程度|くらい|ぐらい|前後|ほど|を目安|を目標)", tail):
            _approx_n = _approx_n or n
        elif re.match(r"\s*(?:以上|から| over)", tail):
            continue            # 「1 文字以上」は下限（目標ではない）
        else:
            _approx_n = _approx_n or n
    if _max_ns:
        f.length_kind, f.max_chars = "max", min(_max_ns)
    if _approx_n:
        if not f.length_kind:
            f.length_kind = "approx"
        f.target_chars = _approx_n
    if f.max_chars and not f.target_chars:
        f.target_chars = int(f.max_chars * 0.85)
    if f.target_chars and not f.max_chars:
        f.max_chars = int(f.target_chars * 1.4) + 20
    if _SHORT_WORDS.search(t) and not f.target_chars and not f.max_chars:
        f.brief = True
    if _LONG_WORDS.search(t) and not f.target_chars:
        f.length_kind, f.target_chars = "approx", 500

    # ---- 文体・口調 ---- #
    tones: list[str] = []
    for pat, name in ((_TONE_POLITE, "polite"), (_TONE_PLAIN, "plain"),
                      (_TONE_FRIENDLY, "friendly"), (_TONE_PRO, "professional")):
        m = pat.search(t)
        if m:
            word = m.group(0)
            tail = t[m.end():m.end() + 14]
            if name == "polite" and re.search(
                    r"^(?:[」』\"”]\s*)?(?:調|口調)?\s*(?:は|も|を)?\s*"
                    r"(?:使わ|使うな|使っては|禁止|ダメ|だめ|避け|なく|不要|抜き|やめ)", tail):
                tones.append("plain")     # 「「です・ます」は使わない」＝常体で書く指示
                continue
            tones.append(name)
            # 「ベテランエンジニアのアシスタント」のような *名詞の一部* は口調の指定ではない
            if _PRO_WORD.fullmatch(word) and not re.search(r"(?:な|の|に|だ|である|口調|風|さ)", t[m.end():m.end() + 3]):
                continue
            if word not in f.tone_words:
                f.tone_words.append(word)
    if "friendly" in tones or "professional" in tones:
        f.tone = "friendly_professional" if len(tones) > 1 else (
            "friendly" if "friendly" in tones else "professional")
    elif "polite" in tones:
        f.tone = "polite"
    elif "plain" in tones:
        f.tone = "plain"
    if f.tone in ("friendly", "friendly_professional", "plain"):
        f.register = "plain"
    elif f.tone == "polite":
        f.register = "polite"
    # 「〜だよ、〜だね」のような *語尾そのもの* の指定は最優先で残す
    ends = re.findall(r"[〜~]?\s*(だよ|だね|だよね|です|ます|だ|である|ね|よ)", t)
    tail_spec = re.findall(r"(?:口調|語尾|文末|ですます|だ・である)\s*[（(]\s*([^)）]{1,24})\s*[)）]", t)
    if tail_spec:
        f.tone_words.extend(x.strip() for x in tail_spec[0].split("、") if x.strip())
    elif ends and f.tone.startswith("friendly"):
        for e in ends:
            if e in ("だよ", "だね", "だよね") and e not in f.tone_words:
                f.tone_words.append(e)

    # ---- 言語 ---- #
    m = re.search(r"(英語|日本語|中国語|韓国語|フランス語|ドイツ語|スペイン語|イタリア語|ロシア語)\s*(?:で|に)\s*(?:書いて|出力して|答えて|返して|訳して|説明して|まとめて)", t)
    if m:
        f.language = m.group(1)
    else:
        m2 = re.search(r"\bin\s+(english|japanese|chinese|korean|french|german|spanish|"
                       r"italian|russian)\b", t, re.IGNORECASE)
        if m2:
            f.language = {"english": "英語", "japanese": "日本語", "chinese": "中国語",
                          "korean": "韓国語", "french": "フランス語", "german": "ドイツ語",
                          "spanish": "スペイン語", "italian": "イタリア語",
                          "russian": "ロシア語"}[m2.group(1).lower()]

    # ---- 出力ルール: 指示層の厳密な読みを先に、既存コンパイラは形の話だけ ---- #
    # `compile_rules` の require/forbid は緩く拾うので（「〜を要約してください」まで
    # 含めてしまう）、ここでは *引用符基準* の読みを優先し、向こうからは
    # start / end / charset / lang だけ取り込みます。
    rules = read_output_rules(t)
    try:
        rules += [r for r in compile_rules(t) if r.kind in ("start", "end", "charset", "lang")]
    except Exception:  # noqa: BLE001
        pass
    seen: set[tuple[str, str]] = set()
    f.extra_rules = []
    for r in rules:
        key = (r.kind, str(r.value))
        if key in seen:
            continue
        seen.add(key)
        f.extra_rules.append(r)
    return f


# --------------------------------------------------------------------------- #
# 出力ルール（「必ず「X」を含めて」「Xは書かないこと」）
# --------------------------------------------------------------------------- #
#: 含める指定。引用符の中身をそのまま値にする（引用符が無ければ「Xという語」の X）
_REQUIRE = re.compile(
    r"(?:必ず|かならず|絶対|絶対に|must)?\s*"
    r"(?:[「『\"']([^」』\"'\n]{1,40})[」』\"']"
    r"|([^「」『』\"'\s\n]{1,24}?)(?:という語|という単語|の語|という文字列))\s*"
    r"(?:という(?:語|単語|文字列))?\s*(?:を|は|も)?\s*(?:含め|入れ|使い|使用し|記載|書く|入れよ)")
#: 含めない指定
_FORBID = re.compile(
    r"(?:[「『\"']([^」』\"'\n]{1,40})[」』\"']"
    r"|([^「」『』\"'\s\n]{1,24}?)(?:という語|という単語|の語))\s*"
    r"(?:という(?:語|単語|文字列))?\s*(?:を|は|も)?\s*"
    r"(?:含めな|入れな|使わ|使用しな|書か|書くな|禁じ|なしで|無しで|不要|"
    r"入れない|含めない)")


_Q = r"[「『\"\'“][^「」『』\"\'“”\n]{1,40}[」』\"\'”]"
_QUOTE_RUN = re.compile(_Q + r"(?:\s*(?:[とや、,]\s*)?" + _Q + r")*")
_REQUIRE_TAIL = re.compile(r"^(?:という(?:語|単語|文字列|言葉)?\s*)?(?:を|は|も)?\s*"
                           r"(?:含め|入れ|使い|使用し|記載|書い|書く|入れよ|残し|残せ)")
_FORBID_TAIL = re.compile(r"^(?:という(?:語|単語|文字列|言葉)?\s*)?(?:を|は|も)?\s*"
                          r"(?:含めな|入れな|使わ|使用しな|書か|禁じ|なしで|無しで|不要|"
                          r"入れない|含めない|避け|やめ|出すな|述べるな)")


def read_output_rules(text: str) -> list[Rule]:
    """「必ず「X」を含めて」「Xは書かない」を *引用符の中身そのもの* として読む。

    `mind.rules.compile_rules` は遊びの条件（文字数・音数・語の連鎖）用に緩く拾うので、
    「〜を要約してください」のような指示の一部まで require に入れてしまいます。
    指示層では引用符を基準に読み直し、検証（`run.verify`）で実際に照合します。
    """
    t = str(text or "")
    out: list[Rule] = []

    def add(kind: str, val: str, raw: str) -> None:
        val = str(val or "").strip(" 　。、,.!?！？")
        if not val or any(r.kind == kind and r.value == val for r in out):
            return
        out.append(Rule(kind=kind, value=val, raw=raw, hard=True))

    # 引用符の *並び*（「すごい」「素晴らしい」という言葉は使わない）→ 中の語を全部拾う
    for m in _QUOTE_RUN.finditer(t):
        tail = t[m.end():m.end() + 30]
        kind = "require" if _REQUIRE_TAIL.match(tail) else (
            "forbid" if _FORBID_TAIL.match(tail) else "")
        if not kind:
            continue
        for q in re.findall(r"[「『\"\'“]([^「」『』\"\'“”\n]{1,40})[」』\"\'”]", m.group(0)):
            add(kind, q, m.group(0))

    # 引用符の無い形（`αという語を含めて`）と、1 引用 + 動詞の形
    for pat, kind in ((_REQUIRE, "require"), (_FORBID, "forbid")):
        for m in pat.finditer(t):
            add(kind, (m.group(1) or m.group(2) or ""), m.group(0))
    return out


# --------------------------------------------------------------------------- #
# 材料（payload）の切り出し
# --------------------------------------------------------------------------- #
def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    out, prev = [], 0
    for a, b in sorted(spans):
        out.append(text[prev:a])
        prev = b
    out.append(text[prev:])
    return "".join(out)


def split_payload(text: str) -> tuple[str, str, list[tuple[int, int]], list[str]]:
    """(指示部, 材料部, 取り出した span, 根拠にした目印) を返す。"""
    t = str(text or "")
    marks: list[str] = []
    spans: list[tuple[int, int]] = []
    payload = ""

    # 1) フェンス（``` … ```）は最優先で材料
    for m in _FENCE.finditer(t):
        body = m.group(1).strip()
        if body and len(body) > len(payload):
            payload = body
        spans.append((m.start(), m.end()))
        marks.append("code fence")

    # 2) 「テキスト: 「…」」のように *ラベル + 引用* で差し出された材料
    if not payload:
        m = _QUOTED_AFTER_LABEL.search(t)
        if m:
            payload = m.group(1).strip()
            spans.append((m.start(), m.end()))
            marks.append("label+quote")

    # 3) 「テキスト: 〜」ラベルの後ろ（行をまたいでも良い）
    if not payload:
        best: tuple[int, int, str] | None = None
        for m in _PAYLOAD_LABEL.finditer(t):
            rest = t[m.end():].strip()
            if not rest:
                continue
            # ラベル直後の引用符は外す
            quote = ""
            if rest[:1] in "「『":
                close = "」" if rest[0] == "「" else "』"
                end = rest.find(close)
                if end > 0:
                    quote, rest = close, rest[1:end]
            elif rest[:3] == '"""':
                end = rest.find('"""', 3)
                if end > 0:
                    rest = rest[3:end]
            elif rest[:1] in "\"'":
                # 「テキスト: "hello"」のように半角引用符で括る書き方
                close = rest[0]
                end = rest.find(close, 1)
                if end > 0:
                    quote, rest = close, rest[1:end]
            body = rest.strip()
            # JSON テンプレート（=出力仕様）は材料にしない
            body = re.sub(r"\{[^{}]*\"[^{}]*\}", "", body).strip()
            # 短い材料（「猫は寝る。」）も *文として立っていれば* 材料として読む。
            # ここで捨てると「3 つの箇条書きで要約」の指示だけが残って、材料なしになる。
            if len(body) < 8 and not (len(body) >= 4 and _SENTENCE_END.search(body)) \
                    and not (len(body) >= 3 and re.fullmatch(r"[A-Za-z0-9 ,.;:'\-_+*/=%\[\]{}()]+", body)):
                continue            # 文でも欧文のデータでもない短い塊は材料にしない
            if best is None or len(body) > len(best[2]):
                best = (m.start(), m.end() + (len(t[m.end():]) - len(rest)), body)
        if best is not None:
            payload = best[2]
            spans.append((best[0], best[1]))
            marks.append("label")

    # 4) 長い引用（文末が述語なら「文＝材料」とみなす）
    if not payload:
        for m in _QUOTE_BLOCK.finditer(t):
            body = m.group(1).strip()
            if body in spans:
                continue
            if re.search(r"(?:。|です|ました|である|だ|た|る|い|な|よ|ね|？|\?)\s*$", body) \
                    and len(body) >= 12:
                if len(body) > len(payload):
                    payload = body
                    marks.append("quote")
            if not any(a <= m.start() and m.end() <= b for a, b in spans):
                spans.append((m.start(), m.end()))
    if not payload and re.search(r"続き", t):
        for m in re.finditer(r"[「『]([^」』]+)[」』]", t):
            body = m.group(1).strip()
            if body and len(body) >= 4:
                payload = body
                marks.append("continuation-quote")
                spans.append((m.start(), m.end()))
                break
    if not payload and re.search(r"続き", t):
        # 引用なしの「X、の続きを1文で」→ 「の続き」より前が材料
        m2 = re.search(r"^(.{4,}?)[、,]\s*の続き", t)
        if m2:
            body = m2.group(1).strip()
            if body:
                payload = body
                marks.append("continuation-prefix")
                spans.append((m2.start(1), m2.end(1) + 1))
    if not payload:
        m = re.search(r"(?:してください|して下さい|してください。|お願いします|せよ|しろ)[。！!?]?\s*", t)
        if m:
            rest = t[m.end():].strip()
            # 「〜してください。他の文字は含めないでください」の後半は *指示* であって
            # 材料ではない（v8: ここを材料にすると指示文をそのまま返す事故になる）。
            still_instruction = bool(_INSTRUCTION_TAIL.search(rest)) if rest else False
            if rest and not still_instruction and ("\n" in rest or ":" in rest or "・" in rest or re.search(r"[A-Za-z0-9_{}\[]", rest) or len(rest) >= 6):
                if not re.search(r"\{[^{}]*\"[^{}]*\}", rest) or len(rest) > 30:
                    payload = rest
                    marks.append("fallback-tail")
                    spans.append((m.end(), len(t)))

    instruction = _strip_spans(t, spans) if spans else t
    instruction = re.sub(r"[ \t]{2,}", " ", instruction).strip()
    return instruction, _unquote(payload), sorted(spans), marks


_QUOTE_PAIRS = (("「", "」"), ("『", "』"), ('"""', '"""'), ('"', '"'), ("'", "'"))


def _unquote(text: str) -> str:
    """材料を包んでいる引用符を外す（値そのものを答えに使うため）。"""
    body = str(text or "").strip()
    for open_q, close_q in _QUOTE_PAIRS:
        if body.startswith(open_q) and body.endswith(close_q) and len(body) > len(open_q) + len(close_q):
            return body[len(open_q): len(body) - len(close_q)].strip()
    return body


def parse_role(text: str) -> str:
    """「あなたは〜です」「〜として答えて」の役割を読む。"""
    t = str(text or "")
    m = _ROLE.search(t)
    if m:
        return m.group(1).strip("「」『』\"' 、")
    # fallback for embedded quotes like 語尾に「〜ロボ」をつけるロボット
    m2 = re.search(r"(?:あなた|君|きみ|お前|そちら|assistant|ai)\s*(?:は|って|として)\s*(.+?)(?:です|である|だよ|だね)\s*[。！？!?]?", t)
    if m2:
        cand = m2.group(1).strip("「」『』\"' 、")[:40]
        if len(cand) >= 4 and not re.search(r"(?:してください|教えて|答えて)", cand):
            return cand
    m = re.search(r"[「『]([^」』]{2,30})[」』]\s*(?:という|の)?\s*(?:役|役割|ロール|キャラ|キャラクター|設定)", t)
    if m:
        return m.group(1).strip()
    m = re.search(r"([一-龯ァ-ヶーA-Za-z0-9_・]{2,24})\s*(?:として|になりきって|になって)\s*(?:答えて|応じて|書いて|振る舞って|話して)", t)
    if m:
        return m.group(1).strip("、。 ")
    return ""
#: ラベルの無い問い（「〜とは何ですか？」/ `What is …?`）も行として拾う
_BARE_QUESTION_LINE = re.compile(r"([^\n]{4,80}?(?:とは何ですか|とはなんで|は何ですか|って何ですか|"
                                 r"とは何ですか|ですか|でしょうか|[?？]))")
_EN_QUESTION_LINE = re.compile(
    r"((?:what|why|how|who|when|where|which|is|are|can|could|does|do|did|will|should)\b"
    r"[^\n]{2,90}\?)", re.IGNORECASE)


def parse_question(text: str) -> str:
    """「質問：〜」の問いを読む。ラベルが無ければ問いの 1 文そのものを読む。"""
    t = str(text or "")
    m = _QUESTION.search(t)
    if m:
        got = m.group(1).strip("「」『』 　。")
        # 「質問：X とは？200文字程度で。」→ 出力指定は問いの *外* に置かれたものです。
        for _ in range(3):        # 「200文字程度、である調で。」のように *重ねて* 書かれるので回します
            cut = re.sub(r"\s*(?:次の|上記の)?\s*[0-9０-９]+\s*(?:文字|字|語)\s*(?:程度|ぐらい|くらい)?\s*"
                         r"(?:で|に|にて|で答えて|で出力して|にまとめて)?\s*(?:ください|下さい|ね)?\s*[。.、,]*\s*$", "", got)
            cut = re.sub(r"\s*(?:である調|ですます調|です・ます調|カジュアルな口調|口調で?)"
                         r"\s*(?:で)?\s*[。.、,]*\s*$", "", cut)
            if cut == got:
                break
            got = cut
        return got.strip("「」『』 　。、:")
    m_about = re.search(r"([^\n。]{2,30}?)(?:について|に関して|に関しての|についての)\s*(?:詳しく)?\s*(?:教えて|説明して|知りたい|とは)", t)
    if m_about:
        got = m_about.group(1).strip("「」『』 　、、。")
        got = re.sub(r"^(?:指示|以下|上記|次の|この|その).*?[：:]\s*", "", got).strip()
        if got and len(got) >= 1 and not re.search(r"(?:してください|指示)", got):
            return got.strip("「」『』 　。、:") + "について教えて"
    # 指示文の中に *問いの形* の 1 文があれば、それが答え的对象
    for line in [x.strip() for x in t.split("\n") if x.strip()]:
        m2 = _BARE_QUESTION_LINE.search(line)
        if m2 and not re.search(r"(?:してください|して下さい|ください|せよ|しろ)$", m2.group(1)):
            return m2.group(1).strip("「」『』 　。、?？")
    # 英語の 1 通（`What is photosynthesis? Answer in English in 2 sentences.`）
    for line in [x.strip() for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]:
        m3 = _EN_QUESTION_LINE.match(line)
        if m3:
            return m3.group(1).strip()
    return ""


def detect_language(text: str) -> str:
    """指示文中のプログラミング／マークアップ言語名を読む（日本語の中でも効く）。"""
    t = str(text or "")
    if re.search(r"正規表現|regular expression", t, re.IGNORECASE):
        return "regex"
    m = _LANG_WORDS.search(t)
    if m is None:
        m = _LANG_SHORT.search(t)
    if not m:
        # 「パイソン」「ジャバスクリプト」のようなカナ表記
        kana = {"パイソン": "python", "ジャバスクリプト": "javascript", "タイプスクリプト": "typescript",
                "パール": "perl", "ルビー": "ruby", "ラスト": "rust", "シーシャープ": "c#"}
        for k, v in kana.items():
            if k in t:
                return v
        return ""
    got = m.group(1).lower()
    return _LANG_ALIAS.get(got, got)


def function_spec(text: str) -> tuple[str, list[str]]:
    """`uniqueSort(arr)` のような *作るべき関数の名前と引数* を読む。"""
    for pat in (_FUNC_NAME, _FUNC_NAME2):
        m = pat.search(text or "")
        if m:
            name = m.group(1)
            if name and name.lower() not in ("json", "code", "function", "class", "text", "html"):
                args = [a.strip() for a in re.split(r"[,、]", m.group(2) or "") if a.strip()]
                return name, args
    return "", []


# --------------------------------------------------------------------------- #
# タスク判定
# --------------------------------------------------------------------------- #
# *変換の操作語*（成果物が「材料を加工した文字列」であることを決める語）。
# これらが *最後の命令* のとき、成果物は変換結果であり、文書・リスト・回答にならない。
_TRANSFORM_OPS = re.compile(
    r"(?:取り除|重複|逆順|並び替|ソート|変換|大文字|小文字|ローマ字|カタカナ|ひらがな|"
    r"全角|半角|置き換え|整形|文字数|翻訳|訳して|英訳|和訳|translate|reverse|sort|dedupe|"
    r"uniq|uppercase|lowercase|format|convert)(?![a-z])", re.IGNORECASE)
# コード成果物の形（関数名・言語+関数・コードブロック）。これがある変換語は *コードの話*。
_CODE_ARTIFACT_HARD = re.compile(
    r"```|function\s+\w+\s*\(|def\s+\w+\s*\(|class\s+\w+|関数|メソッド|実装|プログラム|コード",
    re.IGNORECASE)


def _last_verb_is_transform(instruction: str, *, payload: str) -> bool:
    """指示の *最後の命令* が変換の操作語なら True（「リストの重複を取り除いて」= 変換）。

    「次のリストを…してください」のように *材料を指す語*（リスト・テキスト）が命令語より
    前にあるだけで文書・列挙タスクに読まれないための構造規則。
    """
    t = str(instruction or "")
    if not str(payload or "").strip():
        return False
    if _CODE_ARTIFACT_HARD.search(t) or function_spec(t)[0]:
        return False
    m = _TRANSFORM_OPS.search(t)
    if not m:
        return False
    # 変換語より後に、抽出・要約・列挙・回答の命令が来ていたらそちらが成果物
    later = t[m.end():]
    if re.search(r"(?:抽出|抜出し?て|要約|まとめ|列挙|リストアップ|回答|答えて|書いて|作成して)",
                 later):
        return False
    return True


def classify_task(instruction: str, fmt: FormatSpec, *, payload: str = "", question: str = "") -> str:
    # 構造が明確な変換依頼は、学習分類器より先に確定させる
    # （「次のリストの重複を取り除いてください」をメール文書に読ませない）
    if _last_verb_is_transform(instruction, payload=payload):
        return "transform"
    # 高次元分類器を最優先で試す（全文脈で判定し、単一キーワードに引っ張られない）
    try:
        from .highdim_classifier import classify_full as _hd_classify
        task_hd, conf_hd, scores_hd = _hd_classify("", instruction=instruction, payload=payload, question=question)
        # 高い確信度で、かつ構造との整合が取れればそれを採用
        if task_hd and conf_hd >= 0.62:
            # 構造化データ（JSON等）があるときは extract を優先する整合チェック
            structured_hd = fmt.kind in ("json", "csv", "table", "keyvalue")
            if task_hd == "extract" and not (structured_hd or fmt.schema_fields):
                # 構造が無ければ extract の高確信は疑う（単語「抽出」だけで決めない）
                pass
            elif task_hd == "code" and not str(instruction or "").strip():
                pass
            else:
                return task_hd
        # 中確信度でも、キーワードだけの判定より全文脈の判定を優先（偏り防止）
        if task_hd and conf_hd >= 0.55:
            # extract は構造（JSON/CSV等）が無ければ採用しない（単語「抽出」だけで決めない）
            if task_hd == "extract" and not (fmt.kind in ("json", "csv", "table", "keyvalue") or fmt.schema_fields):
                pass
            elif task_hd in ("classify", "list") and "分類" in str(instruction or "") and payload:
                return task_hd
            elif task_hd not in ("", "answer") or conf_hd >= 0.60:
                # ただし extract のように構造が必要なものは除外済み
                if task_hd != "extract":
                    return task_hd
                # extract は上ですでに除外したのでここには来ない
                return task_hd
    except Exception:
        pass
    t = str(instruction or "")
    structured = fmt.kind in ("json", "csv", "table", "keyvalue")

    def _last(pat) -> int:
        hits = list(pat.finditer(t))
        return hits[-1].start() if hits else -1

    # 「都市名を抽出し、それを箇条書きで列挙して」→ 成果物は *最後* の動詞の形（列挙）
    p_ex, p_list, p_sum = _last(_EXTRACT_WORDS), _last(_LIST_WORDS), _last(_SUMMARY_WORDS)
    if p_list >= 0 and p_list > max(p_ex, p_sum) and (payload or question or fmt.bullets) \
            and not (structured and fmt.schema_fields):
        return "list"
    if fmt.schema_fields and (structured or _EXTRACT_WORDS.search(t) or _JSON_WORD.search(t)):
        return "extract"
    if _EXTRACT_WORDS.search(t) and (structured or payload):
        return "extract"
    if _SUMMARY_WORDS.search(t) and not (structured and fmt.schema_fields):
        return "summarize"
    if structured and (payload or (question and len(question) >= 20)):
        return "extract"
    if _SUMMARY_WORDS.search(t):
        return "summarize"
    lang = detect_language(t)
    make = re.search(
        r"(?:書(?:い|き|こ|か|け)|作(?:っ|り|る|成)|実装|作成|つくっ|直し|直して|修正|リファクタ|"
        r"動かし|組み立て|create|write|implement|fix|refactor|build)", t, re.IGNORECASE)
    named = function_spec(t)[0]
    if named and (lang or _CODE_WORDS.search(t)):
        return "code"
    if lang and make and (_CODE_WORDS.search(t) or named):
        return "code"
    if make and (lang or _CODE_WORDS.search(t)) and (
            _CODE_WORDS.search(t) or _CODE_ARTIFACT.search(t) or named
            or (lang and lang not in _MARKUP_LANGS)):
        return "code"
    if _LIST_WORDS.search(t):
        return "list"
    if _TRANSLATE_WORDS.search(t):
        return "transform"
    if question or _ANSWER_WORDS.search(t):
        return "answer"
    if payload and (fmt.bullets or fmt.max_chars or fmt.target_chars or fmt.tone):
        return "summarize"
    if _WRITE_WORDS.search(t):
        return "write"
    if _CLASSIFY_WORDS.search(t):
        return "classify"
    if re.search(r"続きを(?:書いて|作成して|作って)", t):
        return "write"
    if payload and re.search(r"(?:変換|変えて|直して|置き換え|整形|逆順|逆から|並び替|大文字|小文字|"
                             r"ローマ字|カタカナ|ひらがな|全角|半角|frequency|頻度|文字数|重複|取り除|"
                             r"ソート|format|convert|reverse|uppercase|lowercase|dedupe|uniq|sort)",
                             t, re.IGNORECASE):
        return "transform"
    if re.search(r"(?:一言|一語|ひとこと)で", t):
        return "answer"
    if re.search(r"(?:挨拶|あいさつ|自己紹介)", t):
        return "answer"
    return ""


def _imperatives(text: str) -> list[str]:
    seen: list[str] = []
    for m in _IMPERATIVE_EN.finditer(text or ""):      # 英語の命令文（Answer in English …）
        v = m.group(1).lower()
        if v not in seen:
            seen.append(v)
    for m in _IMPERATIVE_SENT.finditer(text or ""):
        body = m.group(0).strip("。！! ")
        tail = body[-6:]
        for verb in ("してください", "てください", "して", "ください", "せよ", "しろ", "まとめる",
                     "要約", "抽出", "出力", "書いて", "作成", "作って", "実装", "答えて",
                     "教えて", "列挙", "変換", "取り除"):
            if verb in tail or verb in body[-10:]:
                if verb not in seen:
                    seen.append(verb)
    return seen


# 「仕事を名指しする動詞」。これがある 1 通は *依頼* であって感想ではない。
_TASK_VERBS = ("要約", "まとめ", "抽出", "出力", "書いて", "作成", "作って", "実装", "答えて",
               "挨拶", "あいさつ",
               "分類", "仕分け",
               "答えよ", "列挙", "変換", "翻訳", "整形", "比較し", "説明し", "解説し", "直して",
               "逆順", "並び替", "大文字", "小文字", "ローマ字", "カタカナ", "ひらがな",
               "取り除", "除い", "置き換え", "整形",
               "全角", "半角", "文字数",
               "summarize", "extract", "output", "write", "implement", "create", "list",
               "translate", "convert", "reverse", "answer", "explain", "describe", "compare",
               "generate", "provide", "rewrite", "define", "discuss", "elaborate")


def _task_verbs(text: str) -> list[str]:
    t = str(text or "").lower()
    return [v for v in _TASK_VERBS if v.lower() in t]


def parse(text: str, *, min_score: float = 0.55) -> Directive | None:
    """1 通の指示を読んで `Directive` にする。指示でなければ None。

    判定は *形* で行います（語の意味引きではありません）:
        命令の述語があるか / 材料が差し出されているか / 出力仕様が書かれているか /
        役割が与えられているか — の得点が閾値を超えたときだけ「指示」とみなす。
    """
    raw = str(text or "")
    if not raw.strip():
        return None
    instruction, payload, spans, marks = split_payload(raw)
    schema, _schema_spans = parse_schema(raw)
    # スキーマ（出力仕様）は指示部からも材料からも外す
    if _schema_spans:
        instruction = _strip_spans(instruction, [(a, b) for a, b in _schema_spans
                                                 if not any(a2 <= a and b <= b2 for a2, b2 in spans)])
        if not payload:
            payload = ""
        marks.append("json schema")
    fmt = parse_format(raw, schema=schema)
    question = parse_question(raw) or (parse_question(instruction) if instruction else "")
    role = parse_role(raw)
    verbs = _imperatives(instruction or raw)
    task = classify_task(instruction, fmt, payload=payload, question=question)
    # 言葉の仕事（空欄補充・選択・語の関係・論理・語の写し・礼）は *材料の形* で
    # 決まるので、分類器の判断より先に確定させる。ここを分類器に任せると
    # 「同じような意味」が write（文書の作成）に化ける。
    job = None
    try:
        from .jobs import detect as _detect_job

        job = _detect_job(raw)
    except Exception:  # noqa: BLE001
        job = None
    if job is not None:
        task = job.task
        # 引用符で差し出された語・並びも *材料*（指示文と混ぜない）
        if not payload:
            if job.items:
                payload = "、".join(job.items)
            elif job.word:
                payload = job.word
    if not task and role:
        # role-specified greeting like 「語尾に〜ロボをつけるロボットです。挨拶を」
        task = "answer"
    if not task:
        return None

    score = 0.0
    signals: list[str] = []
    if job is not None:
        score += 0.34
        signals.append(job.as_signal())
        signals.append(f"job-reason:{job.reason[:24]}")
    if job is not None and _WH_QUESTION.search(raw):
        # 「言葉の仕事」の型 + 問いかけ（どちら・何・どう…）は依頼の形
        # （v8: 「犬が好きで猫は嫌い。好きなのはどちら？」のような 1 通を拾う）
        score += 0.25
        signals.append("job+question")
    if payload and len(payload) >= 8:
        score += 0.30
        signals.append(f"payload:{len(payload)}字")
    elif payload or any(m in ("label", "quote") for m in marks):
        # 材料が短くても「文章：」「テキスト:」で *差し出されていれば* 指示の形
        score += 0.22
        signals.append(f"payload-marker:{len(payload or '')}字")
    if question:
        score += 0.22
        signals.append("question")
    if fmt.schema_fields:
        score += 0.35
        signals.append(f"schema:{len(fmt.schema_fields)}項目")
    if fmt.kind:
        score += 0.18
        signals.append(f"format:{fmt.kind}")
    if fmt.strict:
        score += 0.12
        signals.append("strict")
    if verbs:
        score += 0.12 + min(0.06, 0.03 * (len(verbs) - 1))
        signals.append("imperative:" + "/".join(verbs[:3]))
    named_verbs = _task_verbs(instruction or raw)
    if named_verbs:
        score += 0.14
        signals.append("task-verb:" + "/".join(named_verbs[:2]))
    if fmt.bullets or fmt.lines:
        score += 0.12
        signals.append(f"bullets:{fmt.bullets or fmt.lines}")
    if fmt.target_chars or fmt.max_chars:
        score += 0.12
        signals.append(f"length:{fmt.target_chars or fmt.max_chars}")
    if fmt.brief:
        score += 0.10
        signals.append("brief")
    if fmt.no_greeting or fmt.no_explanation:
        score += 0.10
        signals.append("no_extra")
    if (fmt.target_chars or fmt.max_chars or fmt.bullets or fmt.lines or fmt.tone
            or fmt.register or fmt.brief or fmt.no_greeting) and (question or payload or _ANSWER_WORDS.search(instruction or raw) or _TASK_VERBS):
        # 出力の形を *数字や口調で指定している* のが指示の本質です。問いだけの場合より
        # 強くします（v3 はここで閾値に届かず、知識ベースの引き当てに流れていました）。
        score += 0.24
        signals.append("spec+question")
    if fmt.extra_rules:
        kinds = sorted({str(r.kind) for r in fmt.extra_rules if getattr(r, "value", "")})
        if kinds:
            score += 0.16
            signals.append("rules:" + "/".join(kinds))
    if fmt.tone:
        score += 0.10
        signals.append(f"tone:{fmt.tone}")
    if role:
        score += 0.12
        signals.append(f"role:{role}")
        if verbs:
            score += 0.20
            signals.append("role+imperative")
    if task in ("classify", "list"):
        # 項目（2 語以上）＋ 操作（分類・列挙）は指示の形（「りんご、トマト、バナナを分類して」）
        n_items = len(re.findall(r"[ぁ-んァ-ヶ一-龯A-Za-z0-9]{2,6}", str(instruction or "")))
        if n_items >= 2:
            score += 0.30
            signals.append(f"items:{n_items}")
    if _ARTIFACT.search(instruction or raw):
        score += 0.12
        signals.append("artifact")
    if _ARTIFACT.search(instruction or raw) and _STYLE_VALUE.search(instruction or raw):
        # 「成果物の名 + 形の具体値（語尾「X」・N文字・口調）」が揃うと名詞句でも生成依頼
        # （「語尾に〜ロボの挨拶」＝ 挨拶の形を指定した依頼）
        score += 0.30
        signals.append("artifact+style")
    if _SUFFIX_SPEC.search(instruction or raw):
        # 語尾に具体語（「X」/ 〜X）を指定しているのは明確な形指定
        score += 0.25
        signals.append("suffix-spec")
    if task == "write":
        score += 0.20                       # 成果物（メール/記事/報告書）を名指しした依頼
        signals.append("write:deliverable")
    if fmt.sentences:
        score += 0.10
        signals.append(f"sentences:{fmt.sentences}")
    if fmt.language and task in ("answer", "summarize", "write", "list", "transform"):
        score += 0.12                       # 「英語で答えて」＝出力言語の指定
        signals.append(f"out-lang:{fmt.language}")
    if verbs and (fmt.bullets or fmt.sentences or fmt.lines or fmt.max_chars
                  or fmt.target_chars or fmt.kind or fmt.language):
        score += 0.10                       # 命令 + 出力仕様が揃っている形
        signals.append("imperative+spec")
    if task == "transform" and payload:
        # 「材料 + 操作」の形が揃っていれば、それは指示です（会話ではない）
        score += 0.14
        signals.append("transform:payload+op")
    if task == "code":
        score += 0.30
        signals.append("code")
        lang = detect_language(raw)
        if lang:
            fmt.language = lang
            score += 0.12
            signals.append(f"lang:{lang}")
        name, _args = function_spec(raw)
        if name:
            score += 0.14
            signals.append(f"function:{name}")
        if lang and verbs:
            score += 0.08          # 「言語名 + 作れ」の指示は形がはっきりしている
            signals.append("lang+imperative")
    if len(raw) >= 60:
        score += 0.05
    score = round(min(0.98, score), 3)

    if score < min_score:
        return None
    d = Directive(task=task, raw=raw, instruction=instruction.strip(), payload=payload.strip(),
                  question=question.strip(), fmt=fmt, role=role.strip(), confidence=score,
                  signals=signals, markers=[m for m in marks if m],
                  rules=list(fmt.extra_rules), job=job,
                  notes=[f"指示部 {len(instruction)} 字 / 材料部 {len(payload)} 字",
                         *([f"型: {job.task}（{job.reason}）"] if job is not None else [])])
    return d


def explain(d: Directive | None) -> str:
    """読み取り結果を 1 行の説明にする（UI の根拠表示用）。"""
    if d is None:
        return "指示としては読めませんでした（通常の会話として扱います）"
    bits = [f"タスク={d.task}", f"確信度={d.confidence:.2f}"]
    if d.fmt.kind:
        bits.append(f"形式={d.fmt.kind}")
    if d.fmt.strict:
        bits.append("装飾なし")
    if d.payload:
        bits.append(f"材料={len(d.payload)}字")
    if d.question:
        bits.append("問いあり")
    if d.role:
        bits.append(f"役割={d.role}")
    return " / ".join(bits)


__all__ = ["Directive", "FormatSpec", "parse", "parse_format", "parse_schema", "split_payload",
           "classify_task", "detect_language", "function_spec", "find_json_templates",
           "schema_fields", "parse_role", "parse_question", "explain", "PAYLOAD_MARKERS",
           "read_output_rules"]
