#!/usr/bin/env python3
"""語彙バンク＋活用テーブルのビルド。

Snipher の「言葉の在庫」を作る唯一の入口。既定は **Janome のシステム辞書**
（Apache-2.0 / 辞書データは IPADIC 系・学術的利用可のライセンス）から全見出しを引き、
次の 3 ファイルに圧縮して同梱する。

    snipher/data/wordbank.bin.gz      surface \t 読み \t 品詞 \t 辞書形 \t 成本 \t 活用型
    snipher/data/wordbank_game.bin.gz しりとり・韻などに使える語（読み順）
    snipher/data/conjugation.bin.gz   辞書形 \t 活用形= Surface …（活用型のグループつき）

`--no-extras` を付けなければ、辞書に無い everyday 語（しりとり・三毛猫・生成AI など）を
EXTRA_WORDS から補う。外部ネットワークは使わない（Janome が無ければ同梱の語彙表で縮退生成）。

    python tools/build_wordbank.py --max-entries 120000
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "snipher" / "data"
KANA = re.compile(r"^[ぁ-ん]+$")

# しりとり・語彙クイズで「よく使う語」として必ず在庫に載せたい語
EXTRA_WORDS: tuple[tuple[str, str, str], ...] = (
    ("しりとり", "しりとり", "名詞/一般"), ("取りしりとり", "とりしりとり", "名詞/一般"),
    ("しりとり遊び", "しりとりあそび", "名詞/一般"), ("お題", "だい", "名詞/一般"),
    ("ことわざ", "ことわざ", "名詞/一般"), ("なぞなぞ", "なぞなぞ", "名詞/一般"),
    ("カルタ", "かるた", "名詞/一般"), ("百人一首", "ひゃくにんいっしゅ", "名詞/一般"),
    ("俳句", "はいく", "名詞/一般"), ("短歌", "たんか", "名詞/一般"),
    ("回文", "かいぶん", "名詞/一般"), ("ダジャレ", "だじゃれ", "名詞/一般"),
    ("アタマ文字", "あたかもじ", "名詞/一般"), ("語尾", "ごび", "名詞/一般"),
    ("語頭", "ごとう", "名詞/一般"), ("音読み", "おんよみ", "名詞/一般"),
    ("訓読み", "くんよみ", "名詞/一般"), ("熟語", "じゅくご", "名詞/一般"),
    ("部首", "ぶしゅ", "名詞/一般"), ("画数", "かくすう", "名詞/一般"),
    ("三毛猫", "みけねこ", "名詞/一般"), ("茶トラ", "ちゃとら", "名詞/一般"),
    ("白猫", "しろねこ", "名詞/一般"), ("黒猫", "くろねこ", "名詞/一般"),
    ("子猫", "こねこ", "名詞/一般"), ("成猫", "せいねこ", "名詞/一般"),
    ("犬", "いぬ", "名詞/一般"), ("子犬", "こいぬ", "名詞/一般"),
    ("兎", "うさぎ", "名詞/一般"), ("馬", "うま", "名詞/一般"), ("牛", "うし", "名詞/一般"),
    ("豚", "ぶた", "名詞/一般"), ("鶏", "にわとり", "名詞/一般"), ("羊", "ひつじ", "名詞/一般"),
    ("山羊", "やぎ", "名詞/一般"), ("猿", "さる", "名詞/一般"), ("狸", "たぬき", "名詞/一般"),
    ("狐", "きつね", "名詞/一般"), ("熊", "くま", "名詞/一般"), ("鹿", "しか", "名詞/一般"),
    ("猪", "いのしし", "名詞/一般"), ("兎狩り", "うさがり", "名詞/一般"),
    ("鰯", "いわし", "名詞/一般"), ("鯖", "さば", "名詞/一般"), ("鯵", "あじ", "名詞/一般"),
    ("鮭", "さけ", "名詞/一般"), ("鰻", "うなぎ", "名詞/一般"), ("鮪", "まぐろ", "名詞/一般"),
    ("海豚", "いるか", "名詞/一般"), ("鯱", "しゃちほこ", "名詞/一般"),
    ("生成AI", "せいせいエーアイ", "名詞/一般"), ("機械学習", "きかいがくしゅう", "名詞/一般"),
    ("深層学習", "しんそうがくしゅう", "名詞/一般"), ("言語模型", "げんごもけい", "名詞/一般"),
    ("大規模言語模型", "きこうもげんごもけい", "名詞/一般"), ("自然言語処理", "しぜんげんごしょり", "名詞/一般"),
    ("画像認識", "がぞうにんしき", "名詞/一般"), ("推論", "すいろん", "名詞/一般"),
    ("検索エンジン", "けんさくえんじん", "名詞/一般"), ("スクレイピング", "すくれいぴんぐ", "名詞/一般"),
    ("量子コンピュータ", "りょうしコンピュータ", "名詞/一般"),
    ("朝ご飯", "あさごはん", "名詞/一般"), ("昼ご飯", "ひるごはん", "名詞/一般"),
    ("夕飯", "ゆうはん", "名詞/一般"), ("間食", "かんしょく", "名詞/一般"),
    ("白湯", "さゆ", "名詞/一般"), ("味噌汁", "みそしる", "名詞/一般"),
    ("目玉焼き", "めだまやき", "名詞/一般"), ("トースト", "とーすと", "名詞/一般"),
    ("牛乳", "ぎゅうにゅう", "名詞/一般"), ("豆乳", "とうにゅう", "名詞/一般"),
    ("緑茶", "りょくちゃ", "名詞/一般"), ("麦茶", "むぎちゃ", "名詞/一般"),
    ("ほうじ茶", "ほうじちゃ", "名詞/一般"), ("ウーロン茶", "うーろんちゃ", "名詞/一般"),
)
# 遊び対象に入れて良い品詞の細目（固有名詞・人名を除外する）
GAME_OK_POS2 = {"一般", "普通名詞", "サ変接続", "接尾", "非自立", "代名詞", "数", "副詞可能"}
GAME_BAD_POS2 = {"固有名詞", "人名", "団体", "地域", "個人名", "サ変非自", "接頭"}
BAD_HINT = re.compile(r"[A-Za-z0-9ー々〆]" )


def kata_to_hira(text: str) -> str:
    out = []
    for ch in str(text or ""):
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        elif ch == "ヶ":
            out.append("け")
        elif ch in "ー":
            out.append("ー")
        elif ch in "・―":
            continue
        else:
            out.append(ch)
    return "".join(out)


def from_janome() -> tuple[list[dict], dict]:
    from janome.sysdic import entries as entries_fn  # type: ignore

    raw = entries_fn(compact=False)
    rows: list[dict] = []
    stats = {"source": "janome", "rows": len(raw)}
    # 活用テーブル（辞書形 → {活用形: 表記}）も同時に集める
    conj: dict[tuple[str, str], dict[str, str]] = {}
    for _idx, item in raw.items():
        try:
            surface = str(item[0])
            cost = int(item[3])
            pos_csv = str(item[4] if len(item) > 4 else "")
            ctype = str(item[5] if len(item) > 5 else "*")
            cform = str(item[6] if len(item) > 6 else "*")
            lemma = str(item[7] if len(item) > 7 else "") or surface
            pron = str(item[8] if len(item) > 8 else "") or str(item[9] if len(item) > 9 else "")
        except Exception:  # noqa: BLE001
            continue
        fields = [f.strip() for f in pos_csv.split(",")]
        pos1 = fields[0] if fields else ""
        pos2 = fields[1] if len(fields) > 1 else ""
        if ctype and ctype != "*" and lemma:
            conj.setdefault((lemma, ctype), {})[cform or "*"] = surface
        if pos1 not in {"名詞", "動詞", "形容詞", "副詞", "接頭詞", "連体詞", "感動詞"}:
            continue
        pron = kata_to_hira(pron or "").strip()
        if not pron or not KANA.match(pron):
            continue
        if len(surface) > 12 or len(pron) > 16:
            continue
        rows.append({"surface": surface, "reading": pron, "pos": f"{pos1}/{pos2}" if pos2 else pos1,
                     "dictform": lemma, "cost": cost, "ctype": ctype if ctype != "*" else ""})
    return rows, {"conj": conj, **stats}


def from_repo_tables() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for fname, pos in (("nouns.json", "名詞/一般"), ("verbs.json", "動詞/自立"),
                       ("adjectives.json", "形容詞/自立"), ("other.json", "副詞")):
        p = OUT / fname
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for key in (data if isinstance(data, dict) else []):
            rows.append({"surface": key, "reading": kata_to_hira(key), "pos": pos,
                         "dictform": key, "cost": 5000, "ctype": ""})
    return rows, {"source": "repo_tables"}


def write_lines(path: Path, lines: list[str]) -> int:
    with gzip.GzipFile(str(path), "wb", compresslevel=9, mtime=0) as fh:
        fh.write("\n".join(lines).encode("utf-8"))
    return path.stat().st_size


def build(max_entries: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    extras = [{"surface": s, "reading": r, "pos": p, "dictform": s, "cost": 3000, "ctype": ""}
              for s, r, p in EXTRA_WORDS if KANA.match(r) or s == "生成AI"]
    try:
        rows, info = from_janome()
        conj = info.pop("conj")
    except Exception as exc:  # noqa: BLE001
        rows, info = from_repo_tables()
        info["reason"] = f"{type(exc).__name__}: {exc}"
        conj = {}
    rows.extend(extras)

    best: dict[str, dict] = {}
    for r in rows:
        cur = best.get(r["surface"])
        if cur is None or r["cost"] < cur["cost"]:
            best[r["surface"]] = r
    uniq = list(best.values())
    uniq.sort(key=lambda r: (r["cost"], len(r["surface"])))
    # 動詞・形容詞の辞書形は「高頻度なのに辞書コストが高い」ため、コスト順の切り捨てで
    # 落ちる（食べる・書く・する が居ない語彙は使えない）。ここは予算と無関係に全数残す。
    heads, nouns, rest = [], [], []
    for r in uniq:
        major = r["pos"].split("/")[0]
        if major in ("動詞", "形容詞") and r["surface"] == r["dictform"]:
            heads.append(r)
        elif major == "名詞":
            nouns.append(r)
        else:
            rest.append(r)
    budget = max(0, max_entries - len(heads))
    noun_take = nouns[:int(budget * 0.74)]
    other_take = rest[:max(0, budget - len(noun_take))]
    keep = heads + noun_take + other_take
    keep.sort(key=lambda r: r["reading"])

    lines = ["\t".join((r["surface"], r["reading"], r["pos"], r["dictform"], str(r["cost"]),
                        r["ctype"])) for r in keep]
    size = write_lines(OUT / "wordbank.bin.gz", lines)

    # 遊び対象（読み・拍が确定で、一般的な語だけ）
    game: list[tuple[str, str, str, int]] = []
    for r in keep:
        pos2 = r["pos"].split("/")[1] if "/" in r["pos"] else ""
        if pos2 in GAME_BAD_POS2:
            continue
        if r["pos"].split("/")[0] not in {"名詞", "動詞", "形容詞", "副詞"}:
            continue
        rd = r["reading"]
        if not rd or rd.endswith(("ん", "ー", "っ")) or len(rd) < 2:
            continue
        if re.search(r"[ぁぃぅぇぉ]", rd):        # 拗音の小さな文字を含む語は既定で外す
            continue
        if r["cost"] > 7200:
            continue
        if r["pos"].split("/")[0] in {"動詞", "形容詞"} and r["surface"] != r["dictform"]:
            continue                    # 活用の途中形（「空い」等）を手遊びの対象にしない
        game.append((rd, r["surface"], r["pos"], r["cost"]))
    seen: set[str] = set()
    glines: list[str] = []
    for rd, surface, pos, cost in sorted(game, key=lambda x: (x[0], x[3], len(x[1]))):
        if surface in seen:
            continue
        seen.add(surface)
        glines.append("\t".join((rd, surface, pos, str(cost))))
    game_size = write_lines(OUT / "wordbank_game.bin.gz", glines)

    # 活用テーブル: lemma -> forms（動詞・形容詞だけ。語彙とは別に持つ）
    clines: list[str] = []
    for (lemma, ctype), forms in sorted(conj.items()):
        if not lemma or len(lemma) > 12:
            continue
        if not any(k in ctype for k in ("五段", "一段", "变格", "変格", "サ変", "カ変", "形容詞")):
            continue
        pairs = ";".join(f"{form}={surf}" for form, surf in sorted(forms.items()) if form)
        if pairs:
            clines.append(f"{lemma}\t{ctype}\t{pairs}")
    conj_size = write_lines(OUT / "conjugation.bin.gz", clines)

    meta = {
        "words": len(keep), "game_words": len(glines), "conjugations": len(clines),
        "bytes": size, "game_bytes": game_size, "conjugation_bytes": conj_size,
        "params_estimate": len(keep) * 4 + len(glines) + len(clines) * 8,
        "stats": info,
        "attribution": ("word forms and readings derived from Janome's system dictionary "
                        "(Apache-2.0); underlying dictionary data: IPADIC family"),
    }
    (OUT / "wordbank.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entries", type=int, default=150_000)
    args = ap.parse_args()
    print(json.dumps(build(args.max_entries), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
