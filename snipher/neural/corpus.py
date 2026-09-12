"""内蔵ニューラルコアの学習コーパスを、Snipher の語彙テーブルから自動構築する。

目的は「LFM2.5-1.2B-JP が担う 3 つの仕事」だけを過不足なく学ばせること:

    1. 確率的に不安な部分の文章生成 … 話題から自然な 1〜2 文
    2. 助動詞の補い                  … 断片文の文末・助動詞の復元
    3. 会話を続ける                  … 相手の発話への短い応答

データは外部ファイル不要。語彙テーブル（名詞/動詞/形容詞/助動詞/副詞/文型/
コーパス）・`responses.json` の対話テーブル・`kb.json` の知識ベースから、
**文法チェックを通った文だけ**を決定的（seed 固定）に生成するので再現ビルドできます。
"""

from __future__ import annotations

import random

from ..lexicon import Lexicon
from ..morphology import Morphology

SUBJECTS = ["私", "あなた", "彼", "彼女", "子供", "友達", "先生", "家族", "兄", "姉"]
TIME_WORDS = ["今日", "昨日", "明日", "今朝", "今夜", "毎朝", "毎晩", "週末", "来週", "先週",
              "昼休み", "仕事後", "寝る前"]
PLACE_WORDS = ["家", "学校", "会社", "公園", "駅", "図書館", "店", "部屋", "カフェ", "病院",
               "温泉", "空港", "劇場", "台所", "庭"]
# 「に」を採らない時間語（今日は ✓ / 今日に ✗）
TIME_BARE = set(TIME_WORDS)
TIME_NI = ["3時", "6時", "9時", "午後", "深夜", "早朝", "金曜日", "月曜日", "来月", "来年",
           "夏", "冬", "春", "秋", "昼", "夜", "朝"]
CATEGORIES = ["もの", "こと", "場所", "時間", "話", "予定", "趣味", "仕事", "勉強", "生活",
              "食べ物", "飲み物"]

# 補完タスクのプロンプト接頭辞（core.complete() と必ずそろうこと）
FRAGMENT_PROMPT = "続き: "

# 主語・述語に置きにくい代名詞/指示語
AVOID_NOUNS = {"何", "これ", "それ", "あれ", "ここ", "そこ", "あそこ", "どう", "なぜ", "いつ"}

# 動詞タグ → 取りうる目的語の名詞カテゴリ
OBJECT_CATS = {
    "食事": ("食べ物", "飲み物"), "料理": ("食べ物", "飲み物"), "買い物": ("物", "食べ物"),
    "勉強": ("抽象", "物"), "仕事": ("抽象", "物"), "IT": ("物", "抽象"),
    "芸術": ("物", "抽象"), "娯楽": ("物", "抽象"), "家事": ("物",),
    "生活": ("物", "抽象"), "読書": ("物", "抽象"), "会話": ("抽象", "人"),
    "言語": ("抽象", "物"), "運動": ("物", "場所"), "健康": ("物", "抽象"),
    "旅行": ("場所", "物"), "交通": ("乗り物", "場所"), "移動": ("場所", "乗り物"),
    "人間関係": ("人", "抽象"), "感情": ("抽象", "人"), "思考": ("抽象",),
    "行動": ("物", "場所"), "学校": ("物", "抽象"), "お金": ("抽象", "物"),
    "防災": ("物", "抽象"), "植物": ("植物",), "動物": ("動物",), "人生": ("抽象",),
    "自然": ("場所", "自然"), "科学": ("抽象", "物"), "数学": ("抽象",),
    "文化": ("抽象", "物"), "祝い": ("物", "抽象"), "趣味": ("物", "抽象"),
    "時間": ("時間", "抽象"), "場所": ("場所",), "知覚": ("物", "抽象"),
    "存在": ("物", "場所"), "天気": ("自然", "場所"),
}

# 手書きの核となる文（品質の錨）
SEED_SENTENCES = [
    "今日はいい天気ですね。", "明日は朝から雨が降るそうです。", "私は猫が好きです。",
    "彼は毎日日本語を勉強します。", "昨日は映画を見て、とても楽しかったです。",
    "週末は家族と公園を散歩しました。", "もう少し詳しく教えてください。",
    "どういたしまして。また何でも聞いてください。", "お疲れさまです。今日は早めに休みましょう。",
    "私は毎朝コーヒーを飲んでから仕事を始めます。", "この本は読みやすいので、おすすめです。",
    "電車が遅れたので、約束の時間に間に合いませんでした。",
    "あなたはいつも丁寧に答えてくれるので助かります。",
    "寒い日は温かいスープが恋しくなります。", "練習を続ければ、必ず上手になります。",
    "スマホを長く使うと目が疲れるので、時々休ませます。",
    "料理を作るのは面倒ですが、完成すると嬉しいです。",
    "新しいことを覚えると、世界が少し広がります。", "静かな部屋で本を読む時間が一番好きです。",
    "友達に心配事を話したら、気持ちが軽くなりました。",
    "駅前の新しいパン屋さんは行列ができるほど人気です。",
    "春になると公園の花が一斉に咲きます。",
    "寝る前にストレッチをすると、よく眠れるようになりました。",
    "困っている人がいたら、声をかけられる人でいたいと思います。",
    "予定が変わった場合は、早めに教えてください。",
    "写真を見るだけで、あの日の記憶がよみがえります。",
    "難しい問題こそ、手順を分けて考えるのが近道です。",
    "日本語の助動詞は、文末の意味を決める大切な役割があります。",
    "週末は家でゆっくり過ごすつもりです。",
    "彼は走るのが速いので、すぐに見えなくなりました。",
    "図書館で借りた本を、今日中に読み終わります。",
    "会議の資料は、先に送っておきます。",
    "明日の朝、また考えさせてください。",
]

SHORT_REPLIES = [
    "なるほど、それは面白いですね。", "うんうん、わかります。", "たしかに。",
    "そうでしたね。", "ありがとうございます、助かりました。", "いえいえ、こちらこそ。",
    "それは大変でしたね。", "良かったです、続きをどうぞ。", "少し考えてみます。",
    "もっと詳しく聞かせてください。", "今日はそのへんにしましょう。", "了解です、すぐやります。",
    "無理をしないでください。", "いいですね、それ。", "気をつけていってらっしゃい。",
    "お疲れさまでした、ゆっくり休んでください。", "また何かあれば声をかけてください。",
]

BAD_SUBSTRINGS = ("ますます", "ですです", "ましたます", "ますない", "のの", "はは", "がが", "をを",
                  "ます。です", "いいて", "なな", "ますました", "たいます", "ですます。です")


def _clean(s: str) -> str | None:
    """生成文の品質チェック（壊れた文は学習させない）。"""
    s = str(s or "").strip()
    if not (6 <= len(s) <= 46):
        return None
    if s.count("。") > 2 or s.count("、") > 2:
        return None
    if any(b in s for b in BAD_SUBSTRINGS):
        return None
    if s.endswith(("を。", "が。", "は。", "に。", "で。", "と。", "の。", "も。", "を、", "ますた")):
        return None
    return s


class CorpusBuilder:
    """語彙テーブルから学習文・会話文を生成する。"""

    def __init__(self, lexicon: Lexicon | None = None, seed: int = 13):
        self.lex = lexicon or Lexicon()
        self.morph = Morphology()
        self.rng = random.Random(seed)
        self.verbs = [v for v in self.lex.verbs if v.get("c") in ("五段", "一段", "サ変", "カ変")]
        self.trans_verbs = [v for v in self.verbs if v.get("tr") == "他動詞"]
        self.intrans_verbs = [v for v in self.verbs if v.get("tr") == "自動詞"] or self.verbs
        self.i_adj = [a for a in self.lex.adjectives if a.get("c") == "い形容詞"] or self.lex.adjectives
        self.na_adj = [a for a in self.lex.adjectives if a.get("c") == "な形容詞"]
        self.nouns = self.lex.nouns or [{"s": "もの", "c": "物", "t": []}]
        self.advs = [a["s"] for a in self.lex.adverbs] or ["とても"]
        self.conjs = [c["s"] for c in self.lex.conjunctions] or ["そして"]

    # ------------------------------------------------------------------ #
    def _pick(self, seq):
        return self.rng.choice(list(seq))

    def _object_for(self, verb: dict) -> dict:
        cats: tuple[str, ...] = ()
        for t in verb.get("t", []):
            cats += OBJECT_CATS.get(t, ())
        tg = set(verb.get("t", []))
        same_tag = [n for n in self.nouns if tg & set(n.get("t", []))]
        pool = [n for n in same_tag if not cats or n.get("c") in cats] or same_tag
        if not pool:
            pool = [n for n in self.nouns if n.get("c") in (cats or ("物", "抽象"))]
        return self._pick(pool or self.nouns)

    def _v(self, verb: dict, form: str) -> str:
        return self.morph.inflect_verb(verb, form)

    # ------------------------------------------------------------------ #
    # 1 文生成
    # ------------------------------------------------------------------ #
    KINDS = ("v_polite", "v_past", "v_neg", "v_q", "v_tai", "v_teiru", "v_request",
             "v_suggest", "v_think", "v_reason", "v_contrast", "v_must", "v_can",
             "adj_i", "adj_i_past", "adj_na", "adj_q", "nominal", "hobby", "place",
             "short_reply")

    def sentence(self) -> str | None:
        kind = self.rng.choice(self.KINDS)
        fn = getattr(self, "_" + kind, None)
        if fn is None:
            return None
        try:
            return _clean(fn() or "")
        except Exception:  # noqa: BLE001
            return None

    # ---- 動詞 ---------------------------------------------------------- #
    def _v_polite(self):
        v = self._pick(self.trans_verbs)
        n = self._object_for(v)
        adv = self._pick(self.advs) if self.rng.random() < 0.3 else ""
        return f"{self._pick(SUBJECTS)}は{n['s']}を{adv}{self._v(v, 'masu')}ます。"

    def _v_past(self):
        v = self._pick(self.trans_verbs)
        n = self._object_for(v)
        return f"{self._pick(TIME_WORDS)}、{self._pick(SUBJECTS)}は{n['s']}を{self._v(v, 'ta')}。"

    def _v_neg(self):
        v = self._pick(self.trans_verbs)
        n = self._object_for(v)
        return f"{self._pick(SUBJECTS)}は{n['s']}を{self._v(v, 'masu')}ません。"

    def _v_q(self):
        v = self._pick(self.trans_verbs)
        n = self._object_for(v)
        return f"{self._pick(SUBJECTS)}は{n['s']}を{self._v(v, 'masu')}ますか。"

    def _v_tai(self):
        v = self._pick(self.trans_verbs)
        n = self._object_for(v)
        return f"{self._pick(SUBJECTS)}は{n['s']}を{self._v(v, 'masu')}たいです。"

    def _v_teiru(self):
        v = self._pick(self.trans_verbs)
        n = self._object_for(v)
        return f"今、{self._pick(SUBJECTS)}は{n['s']}を{self._v(v, 'te')}います。"

    def _v_request(self):
        v = self._pick(self.trans_verbs)
        n = self._object_for(v)
        return f"{self._pick(SUBJECTS)}は{n['s']}を{self._v(v, 'te')}ください。"

    def _v_suggest(self):
        v = self._pick(self.trans_verbs)
        n = self._object_for(v)
        return f"{self._tw()}{n['s']}を{self._v(v, 'masu')}ましょう。"

    def _v_think(self):
        adj = self._pick(self.i_adj)
        n = self._object_for(self._pick(self.trans_verbs))
        return f"{n['s']}は{adj['s']}と{self._pick(SUBJECTS)}は思います。"

    def _v_reason(self):
        v = self._pick(self.trans_verbs)
        n = self._object_for(v)
        iv = self._pick(self.intrans_verbs)
        return f"{n['s']}を{self._v(v, 'dict')}ので、{self._pick(SUBJECTS)}は{self._v(iv, 'masu')}ます。"

    def _v_contrast(self):
        v = self._pick(self.trans_verbs)
        n = self._object_for(v)
        return f"{n['s']}を{self._v(v, 'ta')}ですが、{self._pick(SUBJECTS)}は後悔していません。"

    def _v_must(self):
        v = self._pick(self.trans_verbs)
        n = self._object_for(v)
        return f"{self._pick(SUBJECTS)}は{n['s']}を{self._v(v, 'masu')}なければなりません。"

    def _v_can(self):
        v = self._pick(self.trans_verbs)
        n = self._object_for(v)
        return f"{self._pick(SUBJECTS)}は{n['s']}を{self._v(v, 'dict')}ことができます。"

    # ---- 形容詞 -------------------------------------------------------- #
    def _adj_i(self):
        a = self._pick(self.i_adj)
        n = self._plain_noun()
        adv = self._pick(self.advs) if self.rng.random() < 0.4 else ""
        return f"{n['s']}は{adv}{a['s']}です。"

    def _adj_i_past(self):
        a = self._pick(self.i_adj)
        n = self._plain_noun()
        stem = a["s"][:-1] if a["s"].endswith("い") else a["s"]
        return f"{self._pick(TIME_WORDS)}の{n['s']}は{stem}かったです。"

    def _adj_na(self):
        if not self.na_adj:
            return self._adj_i()
        a = self._pick(self.na_adj)
        n = self._plain_noun()
        tail = self.rng.choice(["です。", "ですね。", "でした。", "と思います。"])
        return f"{n['s']}は{a['s']}{tail}"

    def _adj_q(self):
        a = self._pick(self.i_adj)
        n = self._plain_noun()
        return f"{n['s']}は{a['s']}ですか。"

    # ---- 名詞述語 ------------------------------------------------------ #
    def _tw(self) -> str:
        """時間表現 + 正しい助詞（は / に）。"""
        if self.rng.random() < 0.65:
            t = self._pick(TIME_WORDS)
            return f"{t}は"
        return f"{self._pick(TIME_NI)}に"

    def _plain_noun(self):
        pool = [n for n in self.nouns if n["s"] not in AVOID_NOUNS] or self.nouns
        return self._pick(pool)

    def _nominal(self):
        s = self._pick(SUBJECTS)
        n = self._plain_noun()
        return f"{s}は{n['s']}です。"

    def _hobby(self):
        n = self._plain_noun()
        v = self._pick(self.trans_verbs)
        c = self._pick(CATEGORIES)
        if self.rng.random() < 0.5:
            return f"私の好きな{c}は{n['s']}です。"
        return f"{self._pick(SUBJECTS)}の趣味は{n['s']}を{self._v(v, 'dict')}ことです。"

    def _place(self):
        p = self._pick(PLACE_WORDS)
        v = self._pick(self.intrans_verbs)
        return f"{self._tw()}{p}で{self._v(v, 'masu')}ました。"

    def _short_reply(self):
        return self._pick(SHORT_REPLIES)

    # ================================================================== #
    # 文書（会話・補完）の生成
    # ================================================================== #
    def build(self, n: int = 12000, with_dialogue: bool = True) -> list[str]:
        """教師文書のリスト。内部に <user>/<asst> マーカーを含む文書もある。"""
        out: list[str] = list(SEED_SENTENCES)
        for s in self.lex.corpus:
            if isinstance(s, dict) and s.get("src"):
                out.append(str(s["src"]))
        seen = set(out)
        tries = 0
        while len(out) < n and tries < n * 30:
            tries += 1
            sent = self.sentence()
            if sent and sent not in seen:
                seen.add(sent)
                out.append(sent)
        if with_dialogue:
            for doc in self.dialogue_docs() + self.completion_docs(out) + self.kb_docs():
                if doc and doc not in seen:
                    seen.add(doc)
                    out.append(doc)
        self.rng.shuffle(out)
        return out

    # ------------------------------------------------------------------ #
    def dialogue_docs(self) -> list[str]:
        """responses.json / kb.json から (発話 → 応答) 文書を作る。"""
        import json
        from pathlib import Path

        docs: list[str] = []
        # 対話テーブル
        try:
            tables = json.loads((Path(self.lex.data_dir) / "responses.json").read_text(encoding="utf-8"))
        except Exception:
            tables = {}
        for intent in tables.get("intents", []):
            kws = [k for k in intent.get("keywords", []) if k]
            frames = intent.get("replies", [])
            for kw in kws[:6]:
                for frame in frames[:5]:
                    reply = "".join(frame).replace("{topic}", "").strip()
                    if 4 <= len(reply) <= 46:
                        docs.append(f"<user>{kw}\n<asst>{reply}")
        # 知識ベースの Q&A
        for doc in self.kb_docs(pairs_only=True):
            docs.append(doc)
        # テーブル生成の会話（発話を 1 文にして応答を付ける）
        for _ in range(min(2600, max(400, len(docs)))):
            v = self._pick(self.trans_verbs)
            n = self._object_for(v)
            q = self.rng.choice([
                f"{n['s']}を{self._v(v, 'masu')}ましたか。",
                f"{n['s']}はどうする？",
                f"{n['s']}が好きです。",
                f"{n['s']}について教えて。",
                f"{n['s']}が{self._v(v, 'masu')}ません。",
            ])
            a = self.rng.choice([
                f"{n['s']}のことですね。{self._pick(SHORT_REPLIES)}",
                f"はい、{n['s']}は{self._pick(self.i_adj)['s']}ですよ。",
                f"それなら{n['s']}を{self._v(v, 'masu')}ましょう。",
                f"もう少し{n['s']}の話を聞かせてください。",
            ])
            docs.append(f"<user>{q}\n<asst>{a}")
        return docs

    # ------------------------------------------------------------------ #
    def completion_docs(self, base: list[str], n: int = 6000) -> list[str]:
        """「断片 → 全文（続き）」の教師。助動詞・文末の補いを専門に学ぶ。

        文末尾の 1〜8 文字を落とじた断片を渡し、全文を復元させる。
        語尾（ます/です/ました/ません/たい/ください …）が欠けるケースを
        意図的に厚くする。
        """
        docs: list[str] = []
        seen: set[str] = set()
        aux_tails = ("ます。", "です。", "ました。", "ません。", "たいです。", "います。",
                     "ください。", "ましょう。", "だと思います。", "必要があります。",
                     "ことができます。", "ですね。", "でした。", "ないです。")
        pool = [str(x) for x in base if "\n" not in str(x) and str(x).endswith("。")]
        pool = [x for x in pool if 10 <= len(x) <= 46]
        # 助動詞で終わる文を優先（断然たくさん使う）
        aux_pool = [x for x in pool if any(x.endswith(t) for t in aux_tails)]
        ordered = (aux_pool + pool) if aux_pool else pool
        for k in (1, 2, 3, 4, 5, 6, 8):
            for sent in ordered:
                if len(docs) >= n:
                    break
                if len(sent) - k < 7:
                    continue
                frag = sent[: len(sent) - k]
                key = frag + "\x00" + sent
                if key in seen:
                    continue
                seen.add(key)
                docs.append(f"<asst>続き: {frag}\n{sent}")
        self.rng.shuffle(docs)
        return docs[:n]

    # ------------------------------------------------------------------ #
    def kb_docs(self, pairs_only: bool = False) -> list[str]:
        """知識ベースを教師にも使う（事実文 + Q&A ペア）。"""
        import json
        from pathlib import Path

        path = Path(self.lex.data_dir) / "kb.json"
        if not path.exists():
            return []
        try:
            kb = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        docs: list[str] = []
        for item in kb.get("items", []):
            topic = str(item.get("topic", "")).strip()
            facts = item.get("facts", [])
            if not pairs_only:
                for f in facts:
                    if 6 <= len(str(f)) <= 60:
                        docs.append(str(f))
                for i, q in enumerate(item.get("questions", [])):
                    a = (item.get("answers") or [None] * 99)[i]
                    if a and 4 <= len(str(a)) <= 60:
                        docs.append(f"<asst>{topic}のことですね。{a}")
            qs = item.get("questions", [])
            ans = item.get("answers", [])
            for i, q in enumerate(qs):
                a = ans[i] if i < len(ans) else (facts[0] if facts else None)
                if not a:
                    continue
                q = str(q).rstrip("。").strip()
                a = str(a).strip()
                if 2 <= len(q) <= 30 and 4 <= len(a) <= 60:
                    docs.append(f"<user>{q}\n<asst>{a}")
        return docs
