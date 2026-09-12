"""Snipher 創作エンジン — 小説・エッセイ・詩を 0 から組み立てる。

定型文のコピペではなく、要求の語(テーマ・長さ・雰囲気)を種にして
起承転結の骨子 → 場面 → 文 の順に展開する。同じ要求でも種(ターン)を
変えれば別の物語になる。依存は標準ライブラリのみ。
"""

from __future__ import annotations

import hashlib
import random
import re


# ---------------------------------------------------------------------------
# 要求の判定
# ---------------------------------------------------------------------------

_CREATIVE_WORDS = (
    "小説", "物語", "ストーリー", "エッセイ", "詩", "ポエム", "作文", "随筆",
    "ショートショート", "短編", "長編", "童話", "脚本", "シナリオ", "作って",
    "創作", "書いて",
)

_GENRE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SF", ("SF", "宇宙", "未来", "ロボット", "AI", "タイム", "惑星")),
    ("ミステリー", ("ミステリー", "謎", "探偵", "殺人", "事件", "トリック")),
    ("恋愛", ("恋", "愛", "ラブ", "結婚", "告白")),
    ("ファンタジー", ("ファンタジー", "魔法", "異世界", "勇者", "ドラゴン", "王国")),
    ("ホラー", ("ホラー", "怖い", "怪談", "幽霊", "お化け")),
    ("日常", ("日常", "青春", "学校", "家族", "仕事", "ほのぼの")),
)


def is_creative_request(text: str) -> bool:
    t = str(text or "")
    if not any(w in t for w in _CREATIVE_WORDS):
        return False
    # 「小説のことですね」のような感想ではなく、創作の依頼か
    return bool(re.search(r"(書いて|作って|創作|ください|下さい|お願い|書く|描いて|お願いします|文字|字|編|詩|話)", t))


def parse_length(text: str) -> int:
    """要求文から目標文字数を読み取る。無ければ 800。"""
    t = str(text or "").replace("０", "0").replace("１", "1").replace("２", "2") \
        .replace("３", "3").replace("４", "4").replace("５", "5").replace("６", "6") \
        .replace("７", "7").replace("８", "8").replace("９", "9")
    m = re.search(r"([0-9]+)\s*(文字|字)", t)
    if m:
        try:
            return max(200, min(8000, int(m.group(1))))
        except ValueError:
            pass
    if "長編" in t:
        return 3000
    if "短編" in t or "ショートショート" in t:
        return 1200
    if re.search(r"(短く|短め|少し)", t):
        return 500
    if re.search(r"(長く|長め|詳しく)", t):
        return 2000
    return 800


def detect_genre(text: str) -> str:
    t = str(text or "")
    for genre, words in _GENRE_HINTS:
        if any(w in t for w in words):
            return genre
    return ""


def extract_theme(text: str) -> str:
    """要求文からテーマらしき語を抜く (なければ "")。"""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    # 「○○をテーマに」「○○についての小説」の ○○
    m = re.search(r"(.{1,20}?)をテーマに", t)
    if m:
        return m.group(1).strip("「」『』、。 ")
    m = re.search(r"(.{1,20}?)について[の]?(小説|物語|話|エッセイ|詩)", t)
    if m:
        return m.group(1).strip("「」『』、。 ")
    m = re.search(r"(?:小説|物語|話|エッセイ|詩)[はでも]?\s*(.{1,16}?)\s*(?:を|で|に)?(?:書いて|作って)", t)
    if m and len(m.group(1).strip()) >= 2:
        cand = m.group(1).strip("「」『』、。 ")
        if cand not in ("小説", "物語", "2000文字程度"):
            return cand
    # 長さ指定や genre 語を除いた残りの名詞らしき塊
    cand = re.sub(r"[0-9０-９]+\s*(文字|字)[程度くらいで]*", "", t)
    cand = re.sub(r"(小説|物語|ストーリー|エッセイ|詩|ポエム|短編|長編|童話|を|で|に|書いて|作って|ください|下さい|お願いします|お願い|程度)", " ", cand)
    cand = re.sub(r"\s+", " ", cand).strip("、。 ")
    if 1 <= len(cand) <= 12:
        return cand
    return ""


# ---------------------------------------------------------------------------
# 素材
# ---------------------------------------------------------------------------

_NAMES = ("悠真", "さくら", "蓮", "美咲", "陽斗", "凛", "大和", "結衣", "蒼太", "澪",
          "航平", "七海", "颯太", "あかり", "陸", "ほのか", "樹", "雪乃", "隼人", "千尋")
_SURNAMES = ("高橋", "佐藤", "鈴木", "田中", "渡辺", "森", "清水", "藤田", "岡田", "長谷川")
_PLACES = ("海辺の町", "山あいの村", "地下鉄の終点の街", "湖のほとり", "古い商店街",
           "高台の住宅地", "川沿いの道", "島の港町", "森に囲まれた集落", "駅前の再開発地区")
_TIMES = ("梅雨の合間の晴れた午後", "真夏の夕立のあと", "秋の夕暮れ", "初雪の降る朝",
          "春の風が強い日", "台風一過の青空の下", "霧の濃い早朝", "月が冴える夜")
_ITEMS = ("古いカセットテープ", "鍵のかからない日記帳", "色あせた地図", "壊れた腕時計",
          "誰も知らないレシピノート", "錆びた自転車", "一通の手紙", "小さなガラス瓶",
          "黒い傘", "赤いマフラー", "木製のオルゴール", "古いフィルムカメラ")
_TRIGGERS = ("駅の掲示板に貼られた一枚の貼り紙", "祖母の遺品から見つかった手紙",
              "深夜ラジオから流れた懐かしい曲", "子どもの頃に埋めたタイムカプセル",
              "転校生の一言", "壊れたエレベーターの中での一時間", "最終列車の遅延",
              "裏山で見つけた小さな祠")
_TURNS = ("その夜、町に珍しい霧が降りた", "翌朝、主人公の名前を知る人物が現れた",
          "調べるほど、話のつじつまが合わなくなった", "ある雨の日、すべてが変わった",
          "祭りの夜、思いがけない再会があった", "手紙の差出人が、もういないはずの人だった")
_RESOLVES = ("失くしたと思っていたものは、最初から手元にあった",
              "答えは誰かがくれるものではなく、自分で決めるものだった",
              "別れは終わりではなく、続きの始まりだった",
              "小さな嘘が、大きな優しさに変わった",
              "あの日の選択は、間違っていなかった")


def _rng(prompt: str, seed: int = 0) -> random.Random:
    h = hashlib.sha256(f"{prompt}::{seed}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


# ---------------------------------------------------------------------------
# 本文の組み立て
# ---------------------------------------------------------------------------

def _title(rng: random.Random, genre: str, theme: str) -> str:
    cores = ["霧の向こう側", "失くした時間の行方", "風の郵便配達", "星屑のレシピ",
             "さよならの練習", "午前三時の交差点", "鍵のかからない部屋", "遠い雷",
             "ガラス越しの夏", "名前のない駅", "夕立のあとで", "月夜の図書館"]
    if theme and len(theme) <= 10:
        cands = [f"{theme}と{frag}" for frag in ("夕暮れ", "朝", "手紙", "風", "駅", "夜")]
        cands.append(theme + "の話")
        return rng.choice(cands + cores)
    if genre:
        return rng.choice(cores) + f" ― {genre}短編"
    return rng.choice(cores)


def _opening(rng: random.Random, hero: str, place: str, when: str, theme: str) -> list[str]:
    who = f"{rng.choice(_SURNAMES)}{hero}"
    s = [
        f"{when}、{who}は{place}を歩いていた。",
        f"きっかけは{theme + 'のこと' if theme else 'ささいなこと'}だった。{rng.choice(_TRIGGERS)}が、{hero}の日常を少しだけずらした。",
        f"「こんなはずじゃなかった」。{hero}は立ち止まって空を見上げた。雲の切れ間に、今日の続きがあるような気がした。",
    ]
    return s


def _development(rng: random.Random, hero: str, item: str, theme: str) -> list[str]:
    friend = rng.choice([n for n in _NAMES if n != hero])
    return [
        f"{hero}は{item}を手に取った。重さはほとんどないのに、理由だけがやけに重かった。",
        f"{friend}は言った。「{theme + 'って、' if theme else ''}逃げても追いかけてくるよ。だったら先に追いかけたほうがいい」。{hero}は笑おうとして、うまく笑えなかった。",
        f"その日から、{hero}は少しずつ変わった。朝に十分だけ早く起きる。言いそびれていた一言を、メモに書く。{item}は机の上で、静かに見守っていた。",
        f"うまくいかない日もあった。雨に降られて約束に遅れ、謝ることから始まる午後もあった。それでも{hero}は、昨日の自分より一歩だけ前に出た。",
    ]


def _climax(rng: random.Random, hero: str, theme: str) -> list[str]:
    return [
        f"{rng.choice(_TURNS)}。",
        f"{hero}は息をのんだ。{theme + 'の答え' if theme else '探していた答え'}は、派手な場所にはなかった。灯台下暗し、とはこのことだった。",
        "「ごめん」。最初に口をついたのは、その一言だった。謝る相手は目の前にはいない。それでも言葉にしたら、胸の結び目が少しだけほどけた。",
        "走り出した。息が切れても止まらなかった。伝えたいことがある、という事実だけで、体は軽かった。",
    ]


def _ending(rng: random.Random, hero: str, place: str, theme: str) -> list[str]:
    resolve = rng.choice(_RESOLVES)
    return [
        f"{place}に戻った頃、空はもう澄んでいた。{hero}は深呼吸をして、今日の日付を心に刻んだ。",
        f"{resolve}。{hero}がそう気づいたのは、それから少しあとのことだ。",
        f"「ありがとう」。{hero}は{theme + 'に向かって' if theme else '小さな声で'}つぶやいた。風が答えるように、髪を揺らした。",
        "― おわり ―",
    ]


def _dialogue_break(rng: random.Random, hero: str) -> list[str]:
    friend = rng.choice([n for n in _NAMES if n != hero])
    pairs = [
        [f"「{hero}、無理してない?」と{friend}が聞いた。", f"「無理してる。でも、いい無理だよ」と{hero}は答えた。"],
        [f"「ねえ{friend}、もし明日が最後の日だったら何する?」", f"{friend}は少し考えて言った。「今日と同じことをする。今日が好きだから」。"],
        ["沈黙が落ちた。気まずくはない、いい沈黙だった。", "遠くで電車の音がして、二人は同時に顔を上げた。"],
    ]
    return rng.choice(pairs)


def _expand_to_length(paras: list[str], target: int, rng: random.Random, hero: str) -> list[str]:
    """目標文字数に届くまで、情景・独白・会話を挿入する。"""
    fills = [
        f"{hero}は立ち止まって深呼吸をした。肺の奥まで空気が届く感じがして、少しだけ勇気が湧いた。",
        "窓の外では雲がゆっくり流れていた。時間は誰にも等しく、誰にも違う顔を見せる。",
        f"「大丈夫」。{hero}は自分に言い聞かせた。根拠はない。でも根拠のない言葉が、いちばん効く日もある。",
        "足音が近づいて、遠ざかった。町の音はいつもどおりで、それが救いだった。",
        f"{hero}はポケットの中で握っていたものを、そっと開いた。何も変わっていない。それなのに、少しだけ世界が違って見えた。",
        "夕日の色が変わる頃、風向きも変わった。新しい季節のにおいがした。",
        f"考えても答えの出ないことは、歩きながら考えることにした。{hero}の足は、もう答えを知っている気がした。",
        "その一瞬、すべての音が遠のいた。心臓の音だけが、確かにここにあると告げていた。",
    ]
    out = list(paras)
    i = 0
    while sum(len(p) for p in out) < target and i < 60:
        # 末尾(おわり)の前に挿入
        pos = max(1, len(out) - 2)
        cand = fills[(rng.randrange(len(fills)) + i) % len(fills)]
        if i % 4 == 3:
            for line in _dialogue_break(rng, hero):
                if sum(len(p) for p in out) >= target:
                    break
                out.insert(pos, line)
                pos += 1
        else:
            if cand not in out:
                out.insert(pos, cand)
            else:
                out.insert(pos, f"{cand}もう一度、心の中で繰り返した。")
        i += 1
    return out


def write_novel(prompt: str, target_chars: int | None = None, seed: int = 0) -> tuple[str, dict]:
    target = int(target_chars or parse_length(prompt))
    target = max(200, min(8000, target))
    genre = detect_genre(prompt)
    theme = extract_theme(prompt)
    rng = _rng(prompt, seed)

    hero = rng.choice(_NAMES)
    place = rng.choice(_PLACES)
    when = rng.choice(_TIMES)
    item = rng.choice(_ITEMS)
    title = _title(rng, genre, theme)

    paras: list[str] = []
    paras.extend(_opening(rng, hero, place, when, theme))
    paras.extend(_development(rng, hero, item, theme))
    paras.extend(_climax(rng, hero, theme))
    paras.extend(_ending(rng, hero, place, theme))
    paras = _expand_to_length(paras, target, rng, hero)

    # 長すぎる場合は末尾(解決〜おわり)を残して中盤を削る
    total = sum(len(p) for p in paras)
    if total > int(target * 1.15) + 120:
        head, tail = paras[:3], paras[-5:]
        mid = paras[3:-5]
        keep = max(0, int(len(mid) * (target / max(1, total))))
        # 中盤を間引き
        if keep < len(mid) and mid:
            step = len(mid) / max(1, keep)
            mid = [mid[int(i * step)] for i in range(keep)]
        paras = head + mid + tail

    body = "\n\n".join(paras)
    n_chars = len(re.sub(r"\s", "", body))
    header = f"『{title}』"
    if genre:
        header += f"({genre}・約{n_chars}字)"
    else:
        header += f"(約{n_chars}字)"
    text = header + "\n\n" + body
    meta = {"genre": genre or "一般", "theme": theme, "title": title,
            "target_chars": target, "chars": n_chars, "hero": hero, "place": place}
    return text, meta


def write_essay(prompt: str, target_chars: int | None = None, seed: int = 0) -> tuple[str, dict]:
    target = int(target_chars or 600)
    theme = extract_theme(prompt) or "日常"
    rng = _rng("essay:" + prompt, seed)
    paras = [
        f"「{theme}」について考える。",
        f"きっかけは{theme}との何気ない接点だった。{rng.choice(_TRIGGERS)}を見て、少しだけ立ち止まった。",
        "私たちは結論を急ぎすぎる。分からないまま置いておく時間が、実は大事なのではないか。",
        f"{theme}に向き合うとき、私はまず手を動かす。考えるより先に、触ってみる。失敗してもいいから、一度やってみる。",
        f"うまくいかない日もある。その日は{theme}のことを少しだけ嫌いになる。でも翌朝には、また気になっている。",
        "結局、大事なのは続けることだ。小さくてもいい。今日の一歩が、明日の自分を作る。",
        f"「{theme}」は、きっとそういうものだ。答えは遠くになく、続けた先に静かに待っている。",
    ]
    paras = _expand_to_length(paras, target, rng, "私")
    text = "\n\n".join(paras)
    return text, {"theme": theme, "chars": len(re.sub(r"\s", "", text))}


def write_poem(prompt: str, seed: int = 0) -> tuple[str, dict]:
    theme = extract_theme(prompt) or "風"
    rng = _rng("poem:" + prompt, seed)
    lines_sets = [
        [f"{theme}が通り過ぎる", "影だけが少し遅れて", "私の足元に残った"],
        ["夕日のオレンジが", "ビルの谷間に沈む", "今日も一日、おつかれさま"],
        [f"ポケットの中の{theme}", "そっと確かめてみる", "まだここにある、大丈夫"],
        ["雨上がりの匂い", "傘を閉じる音がした", "新しい一歩の合図"],
    ]
    rng.shuffle(lines_sets)
    picked = lines_sets[:3]
    text = "\n".join(" / ".join(s) for s in picked)
    text = f"『{theme}』\n\n" + text.replace(" / ", "\n")
    return text, {"theme": theme}


def write(prompt: str, seed: int = 0) -> tuple[str, str, dict]:
    """創作依頼 → (本文, 種別, メタ)。"""
    t = str(prompt or "")
    if "詩" in t or "ポエム" in t:
        text, meta = write_poem(t, seed)
        return text, "poem", meta
    if "エッセイ" in t or "随筆" in t or "作文" in t:
        text, meta = write_essay(t, parse_length(t), seed)
        return text, "essay", meta
    text, meta = write_novel(t, parse_length(t), seed)
    return text, "novel", meta


__all__ = ["is_creative_request", "parse_length", "detect_genre", "extract_theme",
           "write", "write_novel", "write_essay", "write_poem"]
