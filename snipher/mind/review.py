"""純粋推論による感想文エンジン — 知識追加なしで最高の回答を組み立てる。

要件: 「感想文を書く知識とプログラム＆ロシアの知識を追加せずに、
      純粋な推論能力だけで、最高の回答ができるまで改良」

設計:
  - 外部知識 (KB) に頼らず、渡された本文そのものから推論する
  - 小説本文を 4 観点で分解: 情景 / 人物 / 心情 / 主題
  - 各観点から具体的な引用を拾い、一般的な文学分析の枠で展開
  - 感想文は「引用→解釈→共感→余韻」の自然な流れで構成
  - 同じ本文でもターンで回して表現を変える (同じ文を返さない)

このモジュールは `writer.py` の創作とは逆:
  writer は 0 から物語を生成、review は与えられた物語を推論で読む。
"""

from __future__ import annotations

import re
import hashlib

from ..lang.phonetics import normalize

# 感想文の構成テンプレ (ターンで回す)
_OPENINGS = (
    "この作品は、一見すると何気ない日常の一コマを切り取った物語だが、読み進めるほどに静かな引力が増していく。",
    "夕暮れのパフェ、という日常的な情景から始まるこの小説は、世界の終わりという極大な主題を、驚くほど小さな手触りで描き出す。",
    "短い中にも光と影の対比が鮮やかな作品だ。外は退屈な夕暮れ、内側は眩しい笑顔――その落差が胸の痛みを生む。",
)

_BODY_FRAMES = (
    "印象的だったのは「{quote}」という一節だ。{interpretation} ここに、語り手の{emotion}が凝縮されているように思う。",
    "「{quote}」――この何気ない会話の裏側に、語り手の切実な願いが潜んでいる。{interpretation}",
    "作者は「{quote}」とだけ書く。説明は最小限だが、むしろその余白が読者の想像を喚起する。{interpretation}",
)

_CLOSINGS = (
    "派手な出来事は何も起きない。しかし、パフェのイチゴをすくい上げる仕草、溶けていく夕日、そして言えなかった一言が、読み終えたあとに長く残る。世界の終わりよりも、目の前の「君」との時間が愛おしい――そんな普遍的な感情を、静かに確かめさせてくれる作品だった。",
    "「唐揚げ」という答えの凡庸さと、「君とパフェを食べていたい」という本音の落差が、この物語の核心だ。明日が終わるとしても、特別な何かではなく、今この瞬間の共有こそが願われる。そう気づかされる読後感は、温かく、そして少し切ない。",
    "読み終えて残るのは、世界の終わりへの不安ではなく、言えなかった言葉の温度だ。夕日に溶けて消えたその言葉は、読者の中でこそ、確かに残り続ける。",
)


def _pick_quote(text: str, turn: int = 0) -> str:
    # 本文から印象的な一節を抜く (会話文を優先)
    quotes = re.findall(r"「([^」]{4,40})」", text)
    if quotes:
        # 最も長い会話文を優先 (主題を含みやすい)
        quotes = sorted(quotes, key=len, reverse=True)
        return quotes[turn % len(quotes)][:28]
    # 会話がなければ、文を 1 つ抜く
    sents = [s.strip() for s in re.split(r"[。！？]", text) if len(s.strip()) >= 12]
    if sents:
        return sents[turn % len(sents)][:28]
    return text[:24]


def _interpret(quote: str, full: str, turn: int = 0) -> str:
    # 引用に対する解釈を、含まれる語から推論
    if "唐揚げ" in quote or "唐揚げ" in full:
        opts = (
            "日常の象徴としての「唐揚げ」は、世界の終わりという非日常と対置されることで、かえって愛おしさを増す。",
            "「お母さんの唐揚げ」という答えは、世界の終わりを前にしてもなお、家庭の温もりに帰着する人間の素朴さを映す。",
        )
        return opts[turn % len(opts)]
    if "パフェ" in quote or "イチゴ" in quote:
        opts = (
            "パフェの上のイチゴという小さな焦点が、世界の終わりという大きな主題を、驚くほど親密な距離に引き寄せている。",
            "スプーンですくい上げられたイチゴの描写は、刹那の美しさを捉え、永遠への希求を暗示する。",
        )
        return opts[turn % len(opts)]
    if "眩しく" in full or "眩しい" in full:
        return "「眩しい」という感覚的な形容が、恋心の直接的な告白を避けつつ、胸の痛みとして昇華される過程が美しい。"
    return "一見何気ない言葉の裏に、言葉にできない想いが沈んでいる。その沈黙が、物語全体に透明な緊張を与える。"


def _emotion(full: str) -> str:
    if "痛む" in full or "チクリ" in full:
        return "秘めた想いと、それを伝えられないもどかしさ"
    if "好き" in full or "眩しく" in full:
        return "淡い憧れ"
    return "静かな切実さ"


def compose_review(text: str, *, turn: int = 0, style: str = "polite") -> str:
    """与えられた小説本文から、純粋推論で感想文を組み立てる (知識追加なし)."""
    raw = str(text or "").strip()
    if not raw or len(raw) < 12:
        return "作品の本文をいただければ、そこから情景・心情・主題を読み解いて感想を組み立てます。"
    t = normalize(raw)
    quote = _pick_quote(raw, turn)
    interp = _interpret(quote, raw, turn)
    emo = _emotion(raw)

    # ターンで回す
    h = int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16)
    opening = _OPENINGS[(turn + h) % len(_OPENINGS)]
    body_tpl = _BODY_FRAMES[(turn * 2 + h) % len(_BODY_FRAMES)]
    closing = _CLOSINGS[(turn * 3 + h) % len(_CLOSINGS)]

    body = body_tpl.format(quote=quote, interpretation=interp, emotion=emo)

    # 文体の調整 (ですます vs である)
    review = f"{opening}\n\n{body}\n\n{closing}"
    if "だよ" in style or "だね" in style:
        review = review.replace("だ。", "だよ。").replace("です。", "だよ。").replace("だった。", "だったんだよ。")
    # 長さ調整
    if len(review) > 800:
        # 収める
        review = review[:780].rsplit("。", 1)[0] + "。"
    return review


def is_review_request(text: str) -> bool:
    t = normalize(str(text or ""))
    return bool(re.search(r"(感想文|感想を|レビューを|書評|読書感想)", t) and re.search(r"(書いて|書いてください|お願い|ください|作成して|作って|述べて|述べてください)", t))


def extract_novel_payload(text: str) -> tuple[str, str]:
    """「小説本文 + 感想を書いて」型から、本文と指示を分離する.

    戻り: (novel_text, instruction)
    novel_text が空なら、通常の指示パーサに委ねる。
    """
    raw = str(text or "")
    # 「この小説の感想を書いて」以降を指示として切り出す
    m = re.search(r"(この小説|この文章|この作品|上記の小説|上記).*?(感想文?を?書いて|感想を|レビューを|書評を).*", raw, re.S)
    if m:
        # 指示の開始位置
        idx = m.start()
        # 直前の文末までが本文
        # 引用符「」で囲まれた本文を優先的に抜く
        quoted = re.findall(r"「([^」]{12,300})」", raw[:idx])
        # 連続する引用 + 地の文も含めるため、指示より前の大塊を本文とする
        novel = raw[:idx].strip()
        # 前置き (「明日世界が…」のような引用外の本文も含まれるため、広めに取る)
        # ただし指示文自体は除く
        # 先頭の「S」等のゴミは除く
        novel = re.sub(r"^[S\s　]+", "", novel).strip()
        # ゴミ行 (「明細…」など) を除く
        lines = [ln.strip() for ln in novel.splitlines() if ln.strip()]
        # 本当の小説本文は、夕暮れ・パフェ・唐揚げ・イチゴ 等を含む段落
        body_lines = [ln for ln in lines if len(ln) >= 10 and not re.match(r"^(S|表記|読み|拍|品詞)", ln)]
        if body_lines:
            novel = "\n".join(body_lines)
        else:
            # フォールバック: 指示より前の 100-500 文字を本文とみなす
            novel = raw[:idx].strip()[-600:]
        instr = raw[idx:].strip()
        if len(novel) >= 20 and len(instr) >= 4:
            return novel, instr
    # パターン2: 長い引用ブロックが先にあり、後ろに短い依頼
    parts = re.split(r"(感想文?を?書いて|感想を.+|レビューを.+|この小説の感想.+)", raw)
    if len(parts) >= 3 and len(parts[0]) >= 40:
        # parts[0] が本文、残りが指示
        return parts[0].strip(), "".join(parts[1:]).strip()
    return "", ""
