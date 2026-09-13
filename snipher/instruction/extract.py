"""情報抽出 — 指示が差し出した材料から、指示が指定した欄だけを埋める。

ここでの原則は **材料に書いてある文字列をそのまま値にする** ことです。
「約2時間15分」を「2.25時間」に言い換えたり、知識ベースから別の事実を
混ぜたりしません（＝抽出タスクでねつ造が起きない）。

やり方は 3 段:

    1. 材料を文に割り、値になりそうな塊（時刻・金額・日付・割合・数量・地名・人名・組織…）を
       *種類つき* で全部拾う（`candidates`）
    2. 出力スキーマの欄名（origin / 出発地 / fare / 料金 …）を概念に写し、
       種類の合う候補を「ヒント語との距離」で採点して割り当てる（`extract_fields`）
    3. 指定の形式（JSON / CSV / 表 / key: value）に整形する（`render_table`）

欄に合う値が材料に無いときは空文字列で返し、`missing` に記録します。
勝手に別の値で埋めることはありません。
"""

from __future__ import annotations

import csv
import io
import json
import re

from ..lang import lex
from ..lang.phonetics import normalize

# --------------------------------------------------------------------------- #
# 概念の語彙（欄名 → 概念）
# --------------------------------------------------------------------------- #
CONCEPTS: dict[str, tuple[str, ...]] = {
    "place": ("origin", "departure", "from", "source", "start", "begin", "departurepoint",
              "destination", "to", "target", "end", "via", "place", "location", "city",
              "station", "airport", "address", "venue", "spot", "site", "country", "region",
              "prefecture", "area", "where", "route", "line",
              "出発地", "出発", "出典", "始発", "起点", "目的地", "到着地", "終点", "行き先", "到着",
              "経由", "場所", "位置", "都市", "駅", "空港", "住所", "会場", "国", "地域", "県",
              "方面", "路線", "区間", "どこ", "発", "着", "港"),
    "duration": ("duration", "time", "elapsed", "takes", "traveltime", "required", "span",
                 "period", "所要時間", "所要", "時間", "所要时长", "かかる時間", "期間", "时长",
                 "何分", "何時間", "どのくらい", "リードタイム"),
    "clock": ("clock", "attime", "timeofday", "departuretime", "arrivaltime", "start_time",
              "end_time", "時刻", "出発時刻", "到着時刻", "開始時刻", "終了時刻", "何時", "時間"),
    "date": ("date", "day", "when", "year", "month", "deadline", "日付", "日時", "年月日", "日",
             "いつ", "年", "月", "締切", "期限", "曜日", "スケジュール"),
    "money": ("fare", "fee", "price", "cost", "amount", "total", "charge", "budget", "money",
              "salary", "tax", "discount", "料金", "価格", "値段", "金額", "費用", "代金", "運賃",
              "価格帯", "予算", "総額", "合計", "税", "割引", "給与", "いくら", "円", "コスト"),
    "count": ("count", "number", "quantity", "qty", "num", "amount", "pieces", "items", "人数",
              "個数", "数", "数量", "件数", "本数", "台数", "回数", "いくつ", "何人", "何個"),
    "percent": ("percent", "percentage", "ratio", "rate", "share", "割合", "率", "パーセント",
                "割", "占有率", "確率"),
    "person": ("person", "name", "who", "author", "owner", "user", "customer", "contact",
               "client", "fullname", "username", "人物", "名前", "氏名", "名", "作者", "著者",
               "所有者", "担当者", "客", "顧客", "顧客名", "取引先", "クライアント", "誰", "人名"),
    "id": ("id", "orderid", "order_id", "code", "number", "serial", "sku", "reference",
           "注文番号", "番号", "識別子", "管理番号", "会員番号", "伝票番号", "コード"),
    "occupation": ("job", "occupation", "profession", "role", "position", "work", "title",
                   "職業", "職種", "仕事", "役職", "勤務先", "担当"),
    "product": ("product", "item", "goods", "merchandise", "商品", "商品名", "品名", "製品",
                "製品名", "品目", "アイテム"),
    "org": ("org", "organization", "company", "brand", "team", "school", "店", "会社", "組織",
            "団体", "チーム", "学校", "メーカー", "ブランド", "企業"),
    "transport": ("transport", "vehicle", "means", "mode", "train", "線", "交通手段", "手段",
                  "乗り物", "列車", "新幹線", "飛行機", "バス", "車", "路線"),
    "distance": ("distance", "km", "miles", "距離", "キロ", "メートル", "里程"),
    "age": ("age", "年齢", "歳", "何歳"),
    "speed": ("speed", "velocity", "速度", "時速", "毎秒"),
    "weight": ("weight", "mass", "重量", "重さ", "キログラム", "グラム", "kg"),
    "temp": ("temperature", "気温", "温度", "度"),
    "title": ("title", "name", "subject", "タイトル", "題名", "件名", "名称", "名前", "表題"),
    "url": ("url", "link", "href", "サイト", "リンク", "アドレス", "urlアドレス"),
    "contact": ("email", "phone", "tel", "contact", "メール", "電話番号", "電話", "連絡先"),
    "reason": ("reason", "why", "cause", "理由", "原因", "なぜ", "訳"),
    "status": ("status", "state", "result", "outcome", "状態", "結果", "状況", "判定"),
    "note": ("note", "notes", "remark", "remarks", "備考", "注記", "メモ", "特記"),
    "genre": ("genre", "category", "type", "kind", "分類", "ジャンル", "種類", "タイプ", "カテゴリ"),
    "lang": ("language", "lang", "言語", "語"),
}
_KEY2CONCEPT: dict[str, str] = {}
for _c, _keys in CONCEPTS.items():
    for _k in _keys:
        _KEY2CONCEPT.setdefault(_k.lower(), _c)
        _KEY2CONCEPT.setdefault(normalize(_k).replace(" ", ""), _c)

#: 値の種類 → 割り当てられる概念（1 番目が最優先）
KIND2CONCEPT: dict[str, tuple[str, ...]] = {
    "duration": ("duration", "clock", "date"),
    "clock": ("clock", "duration", "date"),
    "date": ("date", "clock", "duration"),
    "money": ("money",),
    "count": ("count", "number"),
    "percent": ("percent",),
    "place": ("place", "transport", "org"),
    "person": ("person", "org"),
    "org": ("org", "place"),
    "transport": ("transport",),
    "distance": ("distance", "count"),
    "age": ("age", "count"),
    "speed": ("speed", "count"),
    "weight": ("weight", "count"),
    "temp": ("temp", "count"),
    "url": ("url", "contact"),
    "contact": ("contact", "url"),
    "title": ("title",),
    "product": ("product", "org", "title", "genre"),
    "person": ("person", "org"),
    "id": ("id", "count"),
    "occupation": ("occupation", "org", "title"),
    "lang": ("lang", "title"),
}

_PLACES = (
    "東京", "京都", "大阪", "名古屋", "札幌", "福岡", "神戸", "横浜", "広島", "仙台", "千葉",
    "さいたま", "新潟", "浜松", "静岡", "岡山", "熊本", "鹿児島", "長崎", "金沢", "松山", "高松",
    "那覇", "名古屋", "品川", "新大阪", "新横浜", "博多", "小倉", "姫路", "米原", "熱海", "宇都宮",
    "大宮", "上野", "八戸", "盛岡", "秋田", "山形", "福島", "水戸", "高崎", "軽井沢", "富山",
    "長野", "甲府", "松本", "岐阜", "津", "奈良", "和歌山", "鳥取", "松江", "山口", "徳島",
    "高知", "佐賀", "大分", "宮崎", "青森", "函館", "旭川", "釧路", "帯広", "小樽", "京都駅",
    "日本", "米国", "アメリカ", "イギリス", "フランス", "ドイツ", "中国", "韓国", "イタリア",
    "スペイン", "カナダ", "オーストラリア", "インド", "シンガポール", "タイ", "台湾", "北海道",
    "東北", "関東", "中部", "近畿", "関西", "中国地方", "四国", "九州", "沖縄",
)
_PLACE_RE = "|".join(re.escape(x) for x in sorted(set(_PLACES), key=len, reverse=True))
_TRANSPORT_WORDS = (
    "新幹線", "電車", "列車", "飛行機", "バス", "自動車", "車", "自転車", "徒歩", "船", "フェリー",
    "地下鉄", "地铁", "タクシー", "バイク", "のぞみ", "ひかり", "こだま", "特急", "急行", "普通",
    "リニア", "新交通", "トラム", "路面電車", "モノレール",
)
_TRANSPORT_RE = "|".join(re.escape(x) for x in sorted(set(_TRANSPORT_WORDS), key=len, reverse=True))
_N = r"[0-9０-９]"
_SEP = r"[〜~～\-–—〜から到至]"
_ROUTE_RE = re.compile(
    rf"([一-龯ァ-ヶーA-Za-z]{{2,12}}?)\s*(?:から|発|を|～|〜)\s*([一-龯ァ-ヶーA-Za-z]{{2,12}}?)\s*"
    rf"(?:まで|行き|着|へ|に)")

# --------------------------------------------------------------------------- #
# 値のパターン（長いもの・具体的なものを先に当てる）
# --------------------------------------------------------------------------- #
PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("duration", re.compile(
        rf"(?:約|およそ|大体|だいたい|概ね|ほぼ)?\s*{_N}+\s*(?:時間|時)(?:\s*{_N}+\s*分(?:鐘)?)?"
        rf"(?:\s*{_N}+\s*秒)?(?:くらい|ぐらい|ほど|前後)?")),
    ("duration", re.compile(
        rf"(?:約|およそ|大体|だいたい|概ね|ほぼ)?\s*{_N}+\s*(?:分(?:鐘)?|秒|日間?|週間|か月|ヶ月|"
        rf"カ月|年(?:間)?|時間|か月分|泊|日分|営業日)(?:くらい|ぐらい|ほど|前後)?")),
    ("duration", re.compile(r"(?:約|およそ)?\s*(?:半日|丸一日|一昼夜|終日|即日|当日中|数時間|数分)")),
    ("clock", re.compile(
        rf"(?:午前|午後|正午|深夜|未明|早朝|朝|昼|夜|夕方)?\s*{_N}+\s*時(?:\s*{_N}+\s*分)?"
        rf"(?:\s*(?:発|着|開始|終了|スタート|集合|出発|到着))?(?:ごろ|頃)?")),
    ("date", re.compile(
        rf"(?:{_N}{{4}}\s*年\s*)?{_N}{{1,2}}\s*月\s*{_N}{{1,2}}\s*日(?:\s*\([月火水木金土日]\))?")),
    ("date", re.compile(rf"{_N}{{4}}\s*年(?:{_N}{{1,2}}\s*月)?|(?:{_N}{{1,2}}\s*月{_N}{{1,2}}\s*日)|"
                        rf"(?:元日|大晦日|年末年始|今日|明日|明後日|昨日|本日|翌日|当日|平日|土日)")),
    ("money", re.compile(
        rf"(?:約|およそ|大体|だいたい)?\s*(?:{_N}+(?:,{_N}{{3}})*(?:\.{_N}+)?|{_N}+(?:\.{_N}+)?)\s*"
        rf"(?:円|万円|億円|ドル|ユーロ|ポンド|元|ウォン|米ドル)(?:程度|くらい|ぐらい|前後|ほど|"
        rf"(?:/|／)\s*(?:人|名|個|本|回|泊|時間))?")),
    ("money", re.compile(r"(?:無料|有料|無償|有償|実費|定価|税込|税抜|税込み|割り勘)")),
    ("percent", re.compile(rf"(?:約)?{_N}+(?:\.{_N}+)?\s*(?:%|％|パーセント|割(?:引き)?|分(?:の1)?)")),
    ("count", re.compile(
        rf"{_N}+(?:,{_N}{{3}})*(?:\.{_N}+)?\s*(?:人|名|個|本|枚|台|匹|頭|羽|冊|軒|棟|回|度|"
        rf"点|つ|足|組|席|室|階|分|食|杯|人前|km|キロ(?:メートル)?|メートル|m|cm|センチ|"
        rf"kg|キログラム|グラム|g|リットル|L|ml|坪|畳|件|項目|語|ページ|字|文字|時間|分)")),
    ("age", re.compile(rf"{_N}+\s*歳")),
    ("distance", re.compile(rf"(?:約)?{_N}+(?:\.{_N}+)?\s*(?:km|キロ(?:メートル)?|メートル)(?![A-Za-z])")),
    ("speed", re.compile(rf"(?:時速|毎秒|分速)\s*{_N}+(?:\.{_N}+)?\s*(?:km|キロ|メートル|m)?")),
    ("weight", re.compile(rf"{_N}+(?:\.{_N}+)?\s*(?:kg|キログラム|グラム|g|トン|t)\b")),
    ("temp", re.compile(rf"[+-]?{_N}+(?:\.{_N}+)?\s*度")),
    ("url", re.compile(r"https?://[^\s、。「」』）)]+|www\.[^\s、。「」』）)]+")),
    ("contact", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
                           rf"0\d{{1,4}}-\d{{1,4}}-\d{{3,4}}")),
    ("transport", re.compile(rf"(?:{_TRANSPORT_RE})(?:号|線|便)?")),
    ("place", re.compile(r"[一-龯ァ-ヶー]{2,8}(?:市|区|町|村|県|都|府)")),
    ("id", re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{1,4}[-_][0-9]{2,8}(?![0-9])")),
    ("id", re.compile(r"(?:注文番号|番号|伝票番号|管理番号|会員番号)\s*[:：]?\s*([A-Za-z0-9\-]{3,16})")),
    ("occupation", re.compile(r"([一-龯ァ-ヶーA-Za-z]{2,12})(?:です|であり|をして|として|で、|"
                              r"で働い|に勤め|をしている)")),
    ("product", re.compile(r"(?:商品|製品|品名|品目)\s*(?:は|:|：)?\s*"
                           r"([一-龯ァ-ヶーA-Za-z0-9]{2,20})")),
    ("person", re.compile(r"([一-龯]{2,6})(?:さん|氏|様)?は\s*(?:[0-9０-９]+\s*[歳才]|.{0,8}(?:の|である))")),
    ("place", re.compile(rf"(?:{_PLACE_RE})")),
    ("place", re.compile(r"[一-龯ァ-ヶーA-Za-z0-9]{2,10}駅")),
)

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])\s*")


def sentences(text: str) -> list[str]:
    out = [s.strip() for s in _SENT_SPLIT.split(str(text or "")) if s.strip()]
    return out or ([str(text).strip()] if str(text).strip() else [])


def concept_of(key: str, hint: str = "") -> str:
    """欄名（origin / 出発地）と日本語ヒントから概念を決める。"""
    for word in (str(key or ""), str(hint or "")):
        w = normalize(word).lower().replace(" ", "").replace("_", "")
        if not w:
            continue
        if w in _KEY2CONCEPT:
            return _KEY2CONCEPT[w]
        for k, c in _KEY2CONCEPT.items():
            if len(k) >= 2 and (k in w or w in k):
                return c
    return ""


def _clean(surface: str) -> str:
    return surface.strip(" 　、。，,.!?！？:：;；・「」『』()（）")


def candidates(text: str) -> list[dict]:
    """材料から *値になりそうな塊* を種類つきで全部拾う（重複と包含を整理する）。"""
    src = str(text or "")
    raw: list[dict] = []
    for kind, pat in PATTERNS:
        for m in pat.finditer(src):
            # 取り出し用の括弧があれば *その中身* が値（`注文番号 A-102` の A-102）
            if m.groups() and m.group(1):
                surface, (a, b) = _clean(m.group(1)), m.span(1)
            else:
                surface, (a, b) = _clean(m.group(0)), m.span()
            if not surface:
                continue
            raw.append({"kind": kind, "surface": surface, "start": a, "end": b})
    # 地名（東京駅 ⊃ 東京）や金額（14,000円 ⊃ 000円）の入れ子を整理する
    kept: list[dict] = []
    for c in sorted(raw, key=lambda x: (-(x["end"] - x["start"]), x["start"])):
        if any(k["start"] <= c["start"] and c["end"] <= k["end"] and k is not c
               and (k["end"] - k["start"]) > (c["end"] - c["start"]) for k in kept):
            continue
        if any(k["start"] == c["start"] and k["end"] == c["end"] and k["kind"] != c["kind"]
               for k in kept):
            continue
        kept.append(c)
    kept.sort(key=lambda x: x["start"])
    # 場所の並び（A から B まで）は route として印を付ける
    for m in _ROUTE_RE.finditer(src):
        for i, c in enumerate(kept):
            if c["kind"] == "place" and m.start() <= c["start"] < m.end():
                c["route"] = 1 if c["surface"] in m.group(1) or c["start"] < m.start(2) else 2
    return kept


def _hint_positions(text: str, hints: list[str]) -> list[int]:
    out: list[int] = []
    for h in hints:
        h = str(h or "").strip()
        if not h or len(h) < 1:
            continue
        for m in re.finditer(re.escape(h), text):
            out.append(m.start())
    return sorted(out)


def _hint_words(key: str, hint: str) -> list[str]:
    words: list[str] = []
    for word in (hint, key):
        w = str(word or "").strip()
        if not w:
            continue
        for piece in re.findall(r"[一-龯ァ-ヶー]{1,6}|[A-Za-z][A-Za-z0-9_]{1,}", w):
            if piece.lower() not in [x.lower() for x in words]:
                words.append(piece)
    return words


def extract_fields(payload: str, schema: list[tuple[str, str]]) -> tuple[dict[str, str], list[str], dict]:
    """スキーマの欄を材料の値で埋める。返るのは (値, 埋まらなかった欄, 診断)。"""
    src = str(payload or "")
    cands = candidates(src)
    values: dict[str, str] = {}
    missing: list[str] = []
    used: set[tuple[int, int]] = set()
    trace: dict = {"candidates": [{"kind": c["kind"], "surface": c["surface"]} for c in cands]}

    places = [c for c in cands if c["kind"] == "place"]
    route = _ROUTE_RE.search(src)
    ordered_places: list[str] = []
    if route:
        ordered_places = [_clean(route.group(1)), _clean(route.group(2))]
    else:
        ordered_places = [p["surface"] for p in places]

    place_i = 0
    for key, hint in schema:
        concept = concept_of(key, hint)
        hints = _hint_words(key, hint)
        positions = _hint_positions(src, hints)
        best: tuple[float, dict] | None = None
        pool = [c for c in cands if (c["start"], c["end"]) not in used]
        if concept == "place" and ordered_places:
            # 出発地 / 目的地 は *材料に出た順番* で決める（AからBまで → A, B）
            want = None
            if key.lower() in ("origin", "departure", "from", "start") or "出発" in (hint or "") \
                    or "始発" in (hint or "") or "起点" in (hint or ""):
                want = ordered_places[0]
            elif key.lower() in ("destination", "to", "arrival", "end") or "目的" in (hint or "") \
                    or "到着" in (hint or "") or "終点" in (hint or "") or "行き先" in (hint or ""):
                want = ordered_places[1] if len(ordered_places) > 1 else ""
            elif place_i < len(ordered_places):
                want = ordered_places[place_i]
            if want:
                hit = next((c for c in pool if c["kind"] == "place" and c["surface"] == want), None)
                if hit is not None:
                    best = (1.0, hit)
            if best is None and pool:
                hit = next((c for c in pool if c["kind"] == "place"), None)
                if hit is not None:
                    best = (0.6, hit)
        else:
            kinds = KIND2CONCEPT.get(concept, ()) if concept else ()
            for rank, kind in enumerate(kinds or tuple({c["kind"] for c in pool})):
                for c in pool:
                    if c["kind"] != kind:
                        continue
                    score = 1.0 - 0.15 * rank
                    if positions:
                        dist = min(abs(c["start"] - p) for p in positions)
                        score += max(0.0, 0.5 - dist / 240.0)
                    if c.get("route"):
                        score -= 0.1
                    score += min(0.12, 0.02 * len(c["surface"]))
                    if best is None or score > best[0]:
                        best = (score, c)
                if best is not None:
                    break
        if best is not None and best[1]["surface"]:
            hit = best[1]
            values[key] = hit["surface"]
            used.add((hit["start"], hit["end"]))
            if hit["kind"] == "place":
                place_i += 1
        else:
            # 材料の語をそのまま引く最後の手段（欄のヒント語が材料に出ていればその節）
            got = _phrase_by_hint(src, hints)
            if got:
                values[key] = got
            else:
                values[key] = ""
                missing.append(key)
    trace["missing"] = missing
    trace["route"] = ordered_places or None
    return values, missing, trace


def _phrase_by_hint(text: str, hints: list[str]) -> str:
    """ヒント語を含む節（、で区切った塊）をそのまま返す（値の取りこぼし対策）。"""
    for h in hints:
        h = str(h or "").strip()
        if len(h) < 1:
            continue
        for sent in sentences(text):
            if h not in sent:
                continue
            for chunk in re.split(r"[、。；;\n]", sent):
                if h in chunk and 2 <= len(chunk.strip()) <= 40:
                    return _clean(chunk)
    return ""


# --------------------------------------------------------------------------- #
# 出力の整形
# --------------------------------------------------------------------------- #
def render_table(values: dict[str, str], schema: list[tuple[str, str]], kind: str, *,
                 indent: int = 2) -> str:
    """欄の値を指定の形式に並べる。欄の順序はスキーマの順を保つ。"""
    keys = [k for k, _ in schema] or list(values)
    kind = (kind or "json").lower()
    if kind == "csv":
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(keys)
        w.writerow([values.get(k, "") for k in keys])
        return buf.getvalue().strip()
    if kind in ("table", "markdown"):
        head = "| " + " | ".join(keys) + " |"
        sep = "| " + " | ".join("---" for _ in keys) + " |"
        row = "| " + " | ".join(str(values.get(k, "")).replace("|", "\\|") for k in keys) + " |"
        return "\n".join([head, sep, row])
    if kind in ("keyvalue", "kv"):
        return "\n".join(f"{k}: {values.get(k, '')}" for k in keys)
    return json.dumps({k: values.get(k, "") for k in keys}, ensure_ascii=False, indent=indent)


def flatten_json(obj) -> dict:
    """入れ子の JSON を葉の欄（key → 値）に畳む。配列は *埋まっている値* を優先する。"""
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                for k2, v2 in flatten_json(v).items():
                    if k2 not in out or (not str(out[k2]).strip() and str(v2).strip()):
                        out[k2] = v2
            elif k not in out:
                out[k] = v
    elif isinstance(obj, list):
        for it in obj:
            for k, v in flatten_json(it).items():
                if k not in out or (not str(out[k]).strip() and str(v).strip()):
                    out[k] = v
    return out


def verify_json(text: str, schema: list[tuple[str, str]]) -> tuple[bool, str, dict | None]:
    """出力が *指定のスキーマ通りの JSON* かを検査する（指示追従の機械判定）。"""
    body = text.strip()
    if body.startswith("```"):
        m = re.search(r"```[a-zA-Z]*\n(.*?)```", body, re.S)
        body = (m.group(1) if m else body.strip("`")).strip()
    try:
        got = json.loads(body)
    except Exception as exc:  # noqa: BLE001
        return False, f"JSON として読めません（{type(exc).__name__}）", None
    if not isinstance(got, dict):
        return False, "JSON のトップがオブジェクトではありません", None
    keys = [k for k, _ in schema]
    if keys and any(k not in got for k in keys) and any(isinstance(v, (dict, list))
                                                        for v in got.values()):
        got = flatten_json(got)              # 入れ子のテンプレート → 葉で照合
    if keys:
        missing = [k for k in keys if k not in got]
        extra = [k for k in got if k not in keys]
        if missing:
            return False, f"欄が足りません: {', '.join(missing)}", got
        if extra:
            return False, f"指定に無い欄があります: {', '.join(extra)}", got
        if list(got) != keys:
            return False, "欄の順が指定と違います", got
    if not all(isinstance(v, (str, int, float, bool, type(None))) for v in got.values()):
        return False, "値の型が指定（文字列）と違います", got
    return True, "ok", got


# --------------------------------------------------------------------------- #
# ラベル付きの値（`氏名: 山田太郎`）と、文ごとの記録（`りんごは1個120円`）
# --------------------------------------------------------------------------- #
#: 材料の中で *ラベルとして立つ* 語（コロンが無くても値が続く）
_LABEL_WORDS = (
    "注文番号", "注文日", "出荷日", "配送日", "納期", "顧客名", "顧客", "取引先", "氏名", "名前",
    "商品名", "商品", "製品名", "製品", "数量", "合計金額", "合計", "金額", "単価", "価格",
    "日付", "日時", "開始", "終了", "年齢", "住所", "都市", "職業", "職種", "電話番号", "電話",
    "メール", "件名", "宛先", "担当者", "部署", "備考", "状態", "会員番号", "伝票番号",
    "管理番号", "住所", "行き先", "出発地", "目的地", "所要時間", "料金",
    "配送", "発送", "発送日", "支払", "支払日", "支払い", "請求", "請求額", "宛名",
    "連絡先", "予定", "期限", "締切", "時間", "場所", "内容", "本文", "タイトル",
    "出発", "到着", "便名", "座席", "席", "人数", "型式", "型番", "色", "サイズ",
)
_LABEL_PHRASE = re.compile(
    "(" + "|".join(re.escape(w) for w in sorted(set(_LABEL_WORDS), key=len, reverse=True)) + ")"
    r"\s*(?:は|が|:|：)?\s*([^、。\n]{1,30})")
_TRAILING_QTY = re.compile(r"\s*([0-9０-９][0-9０-９,]*\s*(?:台|個|件|枚|本|足|人|組|杯|点|泊))$")

_LABEL_KEY = re.compile(r"([一-龯ァ-ヶーぁ-んA-Za-z][一-龯ァ-ヶーぁ-んA-Za-z0-9_ ()（）]{0,14})"
                        r"\s*[:：]")
_SUBJECT_VALUE = re.compile(r"^([一-龯ァ-ヶーぁ-んA-Za-z][一-龯ァ-ヶーぁ-んA-Za-z0-9_]{1,14})"
                            r"(?:は|が|って|とは)\s*(.{1,60}?)\s*(?:です|だ|である)?[。！？!?]?$")


def label_values(text: str) -> dict[str, str]:
    """`氏名: 山田太郎, 年齢: 34` のような *ラベル: 値* をそのまま読む。

    ラベルは指示に書かれた欄名と突き合わせる材料になります（欄名をでっち上げない）。
    値は *次のラベルの手前* まで（区切りのカンマは落としません）。
    """
    src = str(text or "")
    hits = list(_LABEL_KEY.finditer(src))
    out: dict[str, str] = {}
    for i, m in enumerate(hits):
        key = m.group(1).strip(" 　、。,，")
        start = m.end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(src)
        val = src[start:end].strip(" 　、。,，\n「」\"'")
        # 直後の文（別の材料）まで飲み込まない
        val = re.split(r"[。\n]", val)[0].strip(" 　、。,，")
        if key and val and key not in out:
            out[key] = val
    # コロンの無いラベル（`注文番号 A-102` / `顧客は佐藤花子`）も読む
    for m in _LABEL_PHRASE.finditer(src):
        key = m.group(1).strip()
        val = _clean(m.group(2))
        mq = _TRAILING_QTY.search(val)
        if mq and len(val[:mq.start()].strip()) >= 2 and not re.search(r"数量|個数", key):
            val = val[:mq.start()].strip()      # `ノートPC 2台` → 商品名は ノートPC
        if key and val and key not in out:
            out[key] = val
    return out


def subject_values(text: str) -> list[tuple[str, str]]:
    """`開始は10時、終了は17時です。` を [(開始, 10時), (終了, 17時)] と読む。

    欄名が指示に無いとき、材料の主語を欄名にします（field1 のような名前は作りません）。
    """
    out: list[tuple[str, str]] = []
    for sent in sentences(str(text or "")):
        for part in re.split(r"[、,]\s*", sent.strip()):
            m = _SUBJECT_VALUE.match(part.strip())
            if m and m.group(1).strip() and m.group(2).strip():
                out.append((m.group(1).strip(), _clean(m.group(2))))
    # 文を区切りで割った断片も拾う（`開始は10時、終了は17時です。`）
    return out


def extract_records(payload: str, schema: list[tuple[str, str]]) -> list[dict[str, str]]:
    """材料を *文単位の記録* として読む（`りんごは1個120円。みかんは1個80円。` → 2 行）。

    各行はスキーマの欄を材料の文字列で埋めたものです。埋まらない欄は空文字のまま
    （ここで値を作ると、材料に無いことを書いたことになります）。
    """
    src = str(payload or "")
    rows: list[dict[str, str]] = []
    labels = label_values(src)
    if labels and schema:
        # `name: 山田太郎, age: 34` は 1 行の記録
        row: dict[str, str] = {}
        pool = dict(labels)
        for key, hint in schema:
            hit = ""
            for lab, val in list(pool.items()):
                if _same_field(lab, key, hint):
                    hit = val
                    pool.pop(lab, None)
                    break
            row[key] = hit
        if any(row.values()):
            rows.append(row)
        return rows

    keys = [k for k, _ in schema]
    if len(keys) < 2:
        return rows
    for sent in sentences(src):
        m = _SUBJECT_VALUE.match(sent.strip())
        if not m:
            continue
        subject, rest = m.group(1).strip(), m.group(2).strip()
        row = {}
        rest_labels = label_values(rest)
        for key, hint in schema:
            val = ""
            for lab, v in rest_labels.items():
                if _same_field(lab, key, hint):
                    val = v
                    break
            if not val:
                concept = concept_of(key, hint)
                kinds = KIND2CONCEPT.get(concept, ()) if concept else ()
                cands = [c for c in candidates(rest) if not kinds or c["kind"] in kinds]
                if cands:
                    val = cands[0]["surface"]
            row[key] = val
        # 主語を入れる欄（name / 名前 / 品名 …）が空なら、文の主語を使う
        for key, hint in schema:
            if row.get(key) or not _is_subject_field(key, hint):
                continue
            row[key] = subject
        if any(row.values()):
            rows.append(row)
    return rows


def _norm_field(word: str) -> str:
    return normalize(str(word or "")).lower().replace(" ", "").replace("_", "")


_SUBJECT_WORDS = ("name", "名前", "名称", "品名", "商品", "item", "product", "title", "題",
                  "人名", "氏名", "label")
_PRICE_WORDS = ("price", "値段", "価格", "金額", "fare", "cost", "円")


def _is_subject_field(key: str, hint: str) -> bool:
    w = _norm_field(key) or _norm_field(hint)
    return any(x in w for x in _SUBJECT_WORDS)


def _same_field(label: str, key: str, hint: str) -> bool:
    """材料のラベル（氏名）と指示の欄名（name / 名前）が同じ欄かを判定する。"""
    lab = _norm_field(label)
    for w in (_norm_field(key), _norm_field(hint)):
        if not w:
            continue
        if lab == w or lab in w or w in lab:
            return True
    concept = concept_of(key, hint)
    if concept and concept_of(label, label) == concept:
        return True
    pairs = (("顧客", "customer"), ("顧客名", "name"), ("取引先", "customer"),
             ("注文番号", "order_id"), ("注文番号", "id"), ("商品", "product"),
             ("商品名", "product"), ("製品", "product"), ("数量", "qty"), ("数量", "count"),
             ("合計", "total"), ("合計金額", "total"), ("配送", "ship_date"),
             ("配送日", "ship_date"), ("納期", "date"), ("職業", "job"), ("職種", "job"),
             ("職業", "occupation"), ("都市", "city"), ("住所", "address"),
             ("氏名", "name"), ("名前", "name"), ("名称", "name"), ("年齢", "age"),
             ("住所", "city"), ("住所", "address"), ("都道府県", "city"), ("値段", "price"),
             ("価格", "price"), ("金額", "price"), ("料金", "fare"), ("所要", "duration"),
             ("時間", "time"), ("開始", "start"), ("終了", "end"), ("行き先", "destination"),
             ("出発", "origin"), ("品名", "item"))
    for a, b in pairs:
        if (a in lab and b in _norm_field(key)) or (b in lab and a in _norm_field(hint) + _norm_field(key)):
            return True
    return False


def render_rows(rows: list[dict[str, str]], schema: list[tuple[str, str]], kind: str, *,
                indent: int = 2) -> str:
    """複数行の記録を指定の形式に並べる（表 / CSV は行が増やせる、JSON は配列にする）。"""
    keys = [k for k, _ in schema] or list(rows[0] if rows else {})
    kind = (kind or "table").lower()
    if not rows:
        return ""
    if kind == "csv":
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(keys)
        for row in rows:
            w.writerow([row.get(k, "") for k in keys])
        return buf.getvalue().strip()
    if kind in ("table", "markdown", "keyvalue", "kv"):
        if kind in ("keyvalue", "kv") and len(rows) == 1:
            return "\n".join(f"{k}: {rows[0].get(k, '')}" for k in keys)
        head = "| " + " | ".join(keys) + " |"
        sep = "| " + " | ".join("---" for _ in keys) + " |"
        body = ["| " + " | ".join(str(r.get(k, "")).replace("|", "\\|") for k in keys) + " |"
                for r in rows]
        return "\n".join([head, sep, *body])
    if len(rows) == 1:
        return json.dumps({k: rows[0].get(k, "") for k in keys}, ensure_ascii=False, indent=indent)
    return json.dumps([{k: r.get(k, "") for k in keys} for r in rows], ensure_ascii=False,
                      indent=indent)


def fill_template(template_text: str, values: dict, *, rows: list[dict] | None = None,
                  indent: int = 2) -> str | None:
    """JSON テンプレートの *形そのもの* に値を埋める（入れ子・配列も保つ）。

    文字列の葉（`"customer": {"name": "顧客名"}` の `顧客名`）がヒントなので、
    欄名（`name`）で値を引いて差し替えます。材料に無ければ空文字のまま＝ねつ造しません。
    """
    try:
        tmpl = json.loads(str(template_text or ""))
    except Exception:  # noqa: BLE001
        return None

    def fill(node, row: dict):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if isinstance(v, str):
                    out[k] = str(row.get(k) or values.get(k) or "")
                elif isinstance(v, (dict, list)):
                    out[k] = fill(v, row)
                else:
                    out[k] = v
            return out
        if isinstance(node, list):
            if node and isinstance(node[0], (dict, list)):
                srcs = [r for r in (rows or []) if r] or [dict(values)]
                return [fill(node[0], r) for r in srcs]
            return list(node)
        return node

    try:
        return json.dumps(fill(tmpl, dict(values or {})), ensure_ascii=False, indent=indent)
    except Exception:  # noqa: BLE001
        return None


__all__ = ["extract_fields", "candidates", "concept_of", "render_table", "verify_json",
           "sentences", "CONCEPTS", "PATTERNS", "label_values", "subject_values",
           "extract_records", "render_rows", "fill_template"]
