#!/usr/bin/env python3
"""学習コーパス生成器 — 語彙バンク（13 万語）と活用エンジンから自然な日本語文を作る。

`snipher/data/lm.npz`（流暢さの審判）と `snipher/data/neural/core.npz`（生成・補完）は
このコーパスで育つ。手書きの語彙テーブルだけを使っていた v2 までは語の在庫が
名詞 770・動詞 226 で、見たことのある形しか返せなかった。v3 は **実辞書の語** を
**実在する活用** で組み替えて文を作る（`snipher.lang.morph`）。

    python tools/build_corpus.py --sentences 260000
    python tools/build_lm.py --corpus snipher/data/corpus.txt.gz --grammar 0
    python tools/distill_neural.py --corpus snipher/data/corpus.txt.gz --profile v3

出力: `snipher/data/corpus.txt.gz`（1 行 1 文）と `corpus.meta.json`。
動詞の他動/自動は別リストで管理し、「空が食べる」型の不整合を出さない。
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snipher.lang import lex, morph                     # noqa: E402
from snipher.lang.phonetics import normalize, to_hiragana   # noqa: E402

OUT = ROOT / "snipher" / "data" / "corpus.txt.gz"

# --------------------------------------------------------------------------- #
# 動詞・形容詞の在庫（格要素が揃ったものだけを使う）
# --------------------------------------------------------------------------- #
TRANSITIVE = """
食べる 飲む 見る 読む 書く 買う 作る 持つ 使う 調べる 覚える 忘れる 探す 見つける 選ぶ 決める
調べる 運ぶ 届ける 送る 受け取る 開ける 閉める つける 消す 洗う 乾かす 並べる 片付ける 掃除する
洗たくする 料理する 準備する 確認する 比較する 説明する 紹介する 提案する 相談する 質問する
回答する 解決する 計算する 測定する 記録する 保存する 削除する 複製する 編集する 作成する
試す 確かめる 教える 学ぶ 習う 練習する 復習する 続ける やめる 始める 終わる 変える 直す
治す 建てる 壊す 壊れる 折る 切る 刻む 焼く 茹でる 蒸す 炒める 煮る 味付ける 盛り付ける
運ぶ 持ち上げる 押す 引く 蹴る 投げる 捕まえる 縛る 解く 掛ける 敷く 畳む 着る 脱ぐ 履く
塗る 掃く 集める 分ける 数える 数える 並べ替える 検索する 表示する 隠す 出力する 入力する
読み込む 書き出す 実行する 起動する 停止する 更新する 追加する 削除する 変更する 設定する
乗せる 降ろす 詰める 空ける 満たす 支える 押さえる 引き受ける 引き継ぐ 引き出す 申し込む
断る 承諾する 許可する 禁止する 勧める 誘う 招待する 訪問する 案内する 案内する 案内する
連れて行く 連れて来る 着く 届ける 注文する 予約する 支払う 値切る 交渉する 契約する 解約する
延長する 短縮する 折り返す 折り返す 折り返す 使う
""".split()

INTRANSITIVE = """
行く 来る 帰る 出る 入る 歩く 走る 飛ぶ 泳ぐ 寝る 起きる 座る 立つ 歩く 走る 泳ぐ 飛ぶ
降る 止む 晴れる 曇る 積もる 融ける 咲く 枯れる 実る 収穫される 育つ 成長する 老いる
疲れる 眠る 覚める 笑う 泣く 息をする 咳をする 欠伸をする あくびをする 深呼吸する
暮らす 暮れる 暮らす 通う 勤める 働く 休む 遊ぶ 頑張る 働く 努める 尽力する
続く 途切れる 始まる 終わる 変わる 変わる 増える 減る 広がる 縮む 曲がる 伸びる
沈む 浮かぶ 傾く 揺れる 震える 光る 暗くなる 明るくなる 冷える 温まる ぬるくなる
届く 足りない 足りる 間に合う 遅れる 早まる 集中する 注意する 気をつける 気づく
驚く 喜ぶ 悲しむ 悔しがる 悔やむ 安心する 不安になる 緊張する 興奮する 落ち着く
退屈する 飽きる 慣れる 慣らす 迷う 悩む 苦しむ 苦しむ 耐える 堪える 我慢する
暮らす 住む 越す 引っ越す 移住する 旅行する 出張する 帰宅する 外出する 散歩する
走る 駆ける 跳ぶ 跳ねる 寝返る 転ぶ 滑る 転がる 倒れる 起き上がる
""".split()

ADJECTIVES = """
高い 安い 広い 狭い 大きい 小さい 長い 短い 速い 遅い 重い 軽い 強い 弱い 難しい 易しい
新しい 古い 美しい 綺麗 暑い 寒い 暖かい 冷たい 温かい 美味しい 不味い 楽しい 嬉しい 悲しい
苦しい 忙しい 暇 静か 賑やか 忙しい 元気 疲れた 痛い 痒い 甘い 辛い 酸っぱい 苦い 塩っぱい
柔らかい 硬い 細かい 荒い 濃い 薄い 深い 浅い 遠い 近い 早い 遅い 長い 短い 太い 細い
真っ白 真っ黒 明るい 暗い 重い 軽い 硬い 柔らかい 素直 不器用 器用 上手 下手 偉い 偉い
正直 親切 丁寧 雑 曖昧 鮮明 微妙 重要 緊急 危険 安全 便利 不便 快適 不快 快適 満員
""".split()

NOUN_SUFFIX = ("さん", "ちゃん", "君")
PLACE = ("学校", "会社", "図書館", "公園", "駅", "病院", "銀行", "郵便局", "美術館", "体育館",
         "映画館", "水族館", "動物園", "植物園", "博物館", "工場", "研究室", "教室", "寮",
         "店舗", "事務所", "空港", "港", "公民館", "市民会館", "温泉", "寺院", "神社", "教会")
TIME_WORD = ("今日", "昨日", "明日", "今朝", "昨夜", "先週", "来週", "今月", "来月", "去年",
             "来年", "昼休み", "放課後", "朝", "夕方", "夜", "真夜中", "週末", "祝日", "平日")
MANNER = ("ゆっくり", "丁寧に", "素早く", "静かに", "明るく", "元気よく", "真剣に", "慎重に",
          "試しに", "時々", "いつも", "必ず", "たぶん", "おそらく", "残念ながら", "残念ながら",
          "うれしそうに", "楽しそうに", "慣れた手つきで", "段ボールで", "段取りよく", "時間をかけて")
REASON = ("疲れていたから", "明日が早いから", "時間がないから", "準備が必要だから",
          "天気が良かったから", "約束があったから", "体調がいまひとつだったから",
          "値段が手頃だったから", "興味があったから", "手順を確かめたかったから")
QUOTE = ("今日は忙しい", "明日またやろう", "ゆっくり休みなさい", "準備はできた",
         "一緒にどうですか", "ここは静かですね", "次はいつですか", "難しすぎます")

_BAD_NOUN = re.compile(r"^(?:[a-z]+$|[A-Z]+$|[0-9])")
_KANA_ONLY = re.compile(r"^[ぁ-んァ-ヶー]+$")


def _everyday_chars() -> set[str]:
    """同梱の人手日本語（kb + 対話 + 語彙）に現れる文字だけを集める。
    日常語の名詞を選ぶための、安くて効く基準として使う。"""
    chars: set[str] = set()
    for name in ("kb.json", "dialogues.json", "corpus.json", "nouns.json", "verbs.json",
                 "adjectives.json", "other.json"):
        fp = OUT.parent / "snipher" / "data" / name if False else OUT / name
        if fp.exists():
            chars |= set(fp.read_text(encoding="utf-8"))
    return chars


def common_nouns(limit: int, *, rng: random.Random) -> list[str]:
    """語彙バンクから「よく使う普通名詞」だけを、頻度（成本）順に引く。

    滅多に現れない漢字を含む語（モチノキ等）は外して、日常会話で本当に出てくる
    名詞の在庫に絞る。学習文が「見たことのない語の羅列」になるのを防ぐため。
    """
    everyday = _everyday_chars()
    bank = lex.bank()
    out: list[str] = []
    seen: set[str] = set()
    for surface, reading, pos, cost in _scan(bank):
        if pos.split("/")[0] != "名詞":
            continue
        sub = pos.split("/")[1] if "/" in pos else ""
        if sub in {"固有名詞", "人名", "地域", "団体", "接尾", "非自立", "サ変接続", "数"}:
            continue
        if surface in seen or len(surface) > 6:
            continue
        if any(ch.isdigit() for ch in surface) or _BAD_NOUN.match(surface):
            continue
        if len(surface) > 4:
            continue
        kanji = [ch for ch in surface if "\u4e00" <= ch <= "\u9fff"]
        if kanji and not all(ch in everyday for ch in kanji):
            continue
        seen.add(surface)
        out.append(surface)
        if len(out) >= limit:
            break
    return out


def _scan(bank):
    """(表記, 読み, 品詞, 成本) を成本の小さい順に回す。"""
    ws = sorted(bank._by_surface.values(), key=lambda w: (w.cost, len(w.surface)))
    for w in ws:
        yield w.surface, w.reading, w.pos, w.cost


def verb_forms(verbs: list[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for v in verbs:
        try:
            out[v] = {
                "dictionary": morph.inflect(v, "dictionary"),
                "masu": morph.inflect(v, "masu"),
                "mashita": morph.inflect(v, "mashita"),
                "te": morph.inflect(v, "te"),
                "ta": morph.inflect(v, "ta"),
                "nai": morph.inflect(v, "nai"),
                "renyou": morph.inflect(v, "renyou"),
                "imperative": morph.inflect(v, "imperative"),
                "volitional": morph.inflect(v, "volitional"),
                "conditional": morph.inflect(v, "conditional"),
                "potential": morph.inflect(v, "potential"),
            }
        except Exception:  # noqa: BLE001
            continue
    return out


def adj_forms(adjs: list[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for a in adjs:
        k = to_hiragana(normalize(a))
        head = k[:-1] if k.endswith("い") else k
        na = a in {"暇", "静か", "賑やか", "真っ白", "真っ黒", "重要", "危険", "安全", "便利",
                  "不便", "快適", "不快", "満員", "正直", "親切", "丁寧", "雑", "曖昧", "鮮明",
                  "微妙", "緊急", "上手", "下手", "偉い", "器用", "不器用", "元気"}
        if na:
            out[a] = {"dictionary": a, "desu": a + "です", "datta": a + "だった",
                      "kute": a + "で", "kereba": a + "なら", "ku": a + "に",
                      "nai": a + "ではない", "nakatta": a + "ではなかった", "souda": a + "そうだ",
                      "nara": a + "なら"}
        else:
            out[a] = {"dictionary": k, "desu": head + "いです", "datta": head + "かった",
                      "kute": head + "くて", "kereba": head + "ければ", "ku": head + "く",
                      "nai": head + "くない", "nakatta": head + "くなかった",
                      "souda": head + "さそうだ", "nara": head + "ければ"}
    return out


def make(n: int, seed: int = 20260912) -> tuple[list[str], dict]:
    rng = random.Random(seed)
    bank = lex.bank()
    nouns = common_nouns(6500, rng=rng)
    vt = verb_forms(sorted(set(TRANSITIVE)))
    vi = verb_forms(sorted(set(INTRANSITIVE)))
    adj = adj_forms(sorted(set(ADJECTIVES)))
    subjects = ["彼", "彼女", "私", "田中さん", "山田さん", "先生", "母", "父", "兄", "姉",
                "友人", "同僚", "後輩", "先輩", "客", "店員", "子ども", "学生", "子どもたち"]
    frames: list[tuple[str, int]] = []      # (書式, 重み)

    def f(tmpl: str, w: int = 1) -> None:
        frames.append((tmpl, w))

    # 平叙・疑問・命令・願望・理由・引用・relative 節 … を網羅する
    f("{S}は{P}で{V.masu}。")
    f("{S}は{P}で{V.mashita}。")
    f("{S}は{TIME}に{P}で{V.masu}。")
    f("{S}は{TIME}に{V.te}、{V2.renyou}ました。")
    f("{S}は{MANNER}{V.masu}。")
    f("{S}は{OBJ}を{V.masu}。")
    f("{S}は{OBJ}を{V.mashita}。")
    f("{S}は{OBJ}を{MANNER}{V.masu}。")
    f("{S}は{TIME}{OBJ}を{V.masu}。")
    f("{S}は{OBJ}を{V.nai}。")
    f("{S}は{OBJ}を{V.conditional}、{V2.masu}。")
    f("{S}は{P}へ{V.masu}。")
    f("{S}は{P}から{V.masu}。")
    f("{S}は{P}に{V.masu}。")
    f("{V.imperative}、{OBJ}を{V2.te}ください。")
    f("{S}は{V.volitional}と{V3.dictionary}。")
    f("{S}は{V.potential}。")
    f("{S}は{OBJ}を{V.potential}。")
    f("{S}は{V.masu}ことが{ADJ.desu}。")
    f("{S}が{V.dictionary}とき、{OBJ}は{ADJ.desu}。")
    f("{S}は{OBJ}が{ADJ.nai}と言いました。")
    f("{S}は「{QUOTE}」と言いました。")
    f("{S}は「{QUOTE}」かどうかを{V4.masu}。")
    f("{OBJ}を{V.te}から、{S}は{V2.masu}。")
    f("{TIME}は{ADJ.desu}。")
    f("{TIME}の{P}は{ADJ.desu}。")
    f("{P}の{OBJ}は{ADJ.desu}。")
    f("{OBJ}の{ADJ.ku}ところが{S}の{V4.renyou}ました。")
    f("{S}と{S2}は{P}で{V.masu}。")
    f("{S}より{S2}のほうが{ADJ.desu}。")
    f("{OBJ}も{OBJ2}も{ADJ.desu}。")
    f("{S}は{OBJ}と{OBJ2}を{V.masu}。")
    f("{S}は{ADJ.ku}{V.masu}。")
    f("{ADJ.ku}{V.te}も、{S}は{V2.masu}。")
    f("{S}は{REASON}、{P}で{V.masu}。")
    f("{S}は{REASON_KARA}ので{V.masu}。")
    f("{S}は{V.nai}そうです。")
    f("{S}は{V.te}いるところです。")
    f("{S}は{OBJ}を{V.te}最中です。")
    f("{S}は{V.te}ところでした。")
    f("{S}は{OBJ}を{V.dictionary}つもりです。")
    f("{S}は{OBJ}を{V.te}ばかりです。")
    f("{S}は{OBJ}を{V.ta}きりです。")
    f("{S}は{OBJ}を{V.te}、{V2.nai}ました。")
    f("{S}は{OBJ}を{V3.renyou}やすいです。")
    f("{S}は{OBJ}を{V3.renyou}にくいです。")
    f("{S}が{V.dictionary}のは、{OBJ}が{ADJ.desu}からです。")
    f("{S}が{V.te}のは{OBJ}のためです。")
    f("{S}は{OBJ}を{V.dictionary}必要があります。")
    f("{S}は{OBJ}を{V.nai}ほうが{ADJ.desu}。")
    f("{S}は{V.dictionary}べきですか。")
    f("{S}は{P}で{V.masu}か。")
    f("{S}は{OBJ}を{V.masu}か。")
    f("{S}は{OBJ}を{V.nai}か。")
    f("{S}は{P}に{V.masu}か。")
    f("{S}は{V.masu}でしょうか。")
    f("{S}は{OBJ}を{V.dictionary}といいです。")
    f("{OBJ}は{P}に{V.dictionary}のですか。")
    f("{S}は{V.dictionary}と思います。")
    f("{S}は{V.masu}そうです。")
    f("{V.te}ください。")
    f("{V.conditional}、{OBJ}は{ADJ.desu}。")
    f("{S}は{OBJ}を{N}に{V.te}ました。")
    f("{S}は{N}に{OBJ}を{V.masu}。")
    f("{N}が{OBJ}を{V.te}、{S}は{V2.masu}。")
    f("{OBJ}と{OBJ2}と{N}を{V.masu}。")
    f("{TIME}、{S}は{P}で{OBJ}を{V.masu}。")
    f("{TIME}の{name}は{ADJ.desu}。")
    f("{S}は{ADJ.souda}。")
    f("{S}の{OBJ}は{ADJ.desu}。")
    f("{S}の{OBJ}が{V.ta}ので、{OBJ2}は{ADJ.nai}。")
    f("{OBJ}は{S}が{V.te}{N}です。")
    f("{S}は{N}を{V.masu}人です。")
    f("{P}で{V.te}{OBJ}を{V2.masu}人が{V3.masu}。")
    f("{S}は{OBJ}を{V.masu}が、{S2}は{V2.nai}。")
    f("{OBJ}は{V.potential}ですか。")
    weights = [w for _t, w in frames]

    def pick(seq):
        return rng.choice(list(seq))

    def _fill(line: str, tmpl: str) -> str:
        """{TAG.field} を実際の活用形で埋める（動詞・形容詞）。"""
        for tag in ("V4", "V3", "V2", "V"):
            while "{V" in line:
                m = re.search(r"\{(V\d?)\.([a-z]+)\}", line)
                if not m:
                    break
                key = m.group(1)
                transitive = key in ("V", "V2", "V4")
                needs_obj = "{OBJ" in tmpl
                pool = vt if (transitive or needs_obj) else vi
                word = pick(list(pool.keys()))
                fld = m.group(2)
                val = pool[word].get(fld) or pool[word]["dictionary"]
                line = line[: m.start()] + val + line[m.end():]
        while True:
            m = re.search(r"\{ADJ\.([a-z]+)\}", line)
            if not m:
                break
            fld = m.group(1)
            # 連用（く）・テ形・仮定を使う枠は イ形容詞だけを引く
            pool = [k for k, v in adj.items() if fld in ("ku", "kute", "kereba", "nai")
                    and not k.endswith(("だ", "か", "い")) or k.endswith("い")] or list(adj.keys())
            word = pick(pool)
            line = line[: m.start()] + adj[word][fld] + line[m.end():]
        while "{ADJ}" in line:
            line = line.replace("{ADJ}", pick(list(adj.keys())), 1)
        return line

    subs = {
        "S": subjects, "S2": subjects, "N": subjects + list(NOUN_SUFFIX),
        "NAME": PLACE,
    }
    out_lines: list[str] = []
    seen: set[str] = set()
    tries = 0
    noun_pool = nouns if nouns else list(PLACE)
    while len(out_lines) < n and tries < n * 12:
        tries += 1
        tmpl, _ = rng.choices(frames, weights=weights, k=1)[0]
        line = _fill(tmpl, tmpl)
        for key, pool in (("TIME", TIME_WORD), ("MANNER", MANNER), ("REASON", REASON),
                          ("REASON_KARA", tuple(x[:-2] for x in REASON if x.endswith("から"))),
                          ("QUOTE", QUOTE), ("P", PLACE)):
            while "{" + key + "}" in line:
                line = line.replace("{" + key + "}", pick(pool), 1)
        while "{OBJ2}" in line:
            line = line.replace("{OBJ2}", pick(noun_pool), 1)
        while "{OBJ}" in line:
            line = line.replace("{OBJ}", pick(noun_pool), 1)
        for key in ("S", "S2", "N", "name"):
            while "{" + key + "}" in line:
                pool = subs.get("N" if key == "N" else "S")
                if key == "name":
                    pool = noun_pool
                if key == "S2":
                    pool = subs["S2"]
                line = line.replace("{" + key + "}", pick(pool), 1)
        if "{" in line or "}" in line:
            continue
        line = re.sub(r"\s+", "", line).strip()
        line = line.replace("さんさん", "さん").replace("、、", "、").replace("をを", "を")
        if not (6 <= len(line) <= 60) or line in seen:
            continue
        if re.search(r"(がが|はは|をを|にに|でで|とと)", line):
            continue
        seen.add(line)
        out_lines.append(line)

    meta = {"sentences": len(out_lines), "chars": sum(len(x) for x in out_lines),
            "nouns": len(noun_pool), "transitive": len(vt), "intransitive": len(vi),
            "adjectives": len(adj), "frames": len(frames), "seed": seed}
    return out_lines, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", type=int, default=180_000)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--append-from", type=Path, default=None,
                    help="人が書いた文（kb・対話）を先頭に足す")
    args = ap.parse_args()
    t0 = time.time()
    lines, meta = make(args.sentences, args.seed)
    if args.append_from and args.append_from.exists():
        extra = [x.strip() for x in
                 args.append_from.read_text(encoding="utf-8").splitlines() if x.strip()]
        lines = extra + lines
        meta["extra"] = len(extra)
    with gzip.GzipFile(str(args.out), "wb", compresslevel=6, mtime=0) as fh:
        fh.write("\n".join(lines).encode("utf-8"))
    meta.update({"out": str(args.out), "bytes": args.out.stat().st_size,
                 "seconds": round(time.time() - t0, 1)})
    (args.out.parent / "corpus.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    for x in lines[:12]:
        print(" ", x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
