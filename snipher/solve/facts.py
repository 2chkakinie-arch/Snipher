"""今・日付・時刻 — 「今は西暦何年？」に検索なしで即答するための能力。

現在性は *知識ベースでは答えられない* 種類の問いです。一方で機械時計は正確。
そこで日付・時刻・経過・年齢・曜日・週番号・残り日数などを、
システム時刻と暦の計算から組み立てます（タイムゾーンは環境設定に従う）。
"""

from __future__ import annotations

import datetime as _dt
import os
import re

from .math import Solution

WEEKDAYS = ("月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日")
_JP_ERA = [  # 令和以降をカバーする最小の元号表（和暦の照会用）
    (_dt.date(2019, 5, 1), "令和"),
    (_dt.date(1989, 1, 8), "平成"),
    (_dt.date(1926, 12, 25), "昭和"),
    (_dt.date(1912, 7, 30), "大正"),
    (_dt.date(1868, 1, 25), "明治"),
]


def now(*, tz_offset_hours: float | None = None) -> _dt.datetime:
    """現在時刻（SNIPHER_TZ_OFFSET 時間指定があれば適用）。"""
    dt = _dt.datetime.now()
    if tz_offset_hours is None:
        raw = os.environ.get("SNIPHER_TZ_OFFSET", "").strip()
        if raw:
            try:
                tz_offset_hours = float(raw)
            except ValueError:
                tz_offset_hours = None
    if tz_offset_hours is not None:
        dt = dt + _dt.timedelta(hours=tz_offset_hours)
    return dt


def era_of(day: _dt.date) -> tuple[str, int]:
    for start, name in _JP_ERA:
        if day >= start:
            return name, day.year - start.year + 1
    return "", day.year


def describe_now(dt: _dt.datetime | None = None) -> Solution:
    dt = dt or now()
    era, era_year = era_of(dt.date())
    doy = dt.timetuple().tm_yday
    total = _dt.date(dt.year, 12, 31).toordinal() - dt.date().toordinal()
    return Solution(
        answer=f"今は西暦 {dt.year} 年 {dt.month} 月 {dt.day} 日（{WEEKDAYS[dt.weekday()]}）、"
               f"{dt:%H:%M} です",
        steps=[f"和暦では {era}{era_year} 年" if era else "和暦の対応はありません",
               f"年内 {doy} 日目／今年残り {total} 日",
               f"{dt.year} 年第 {dt.isocalendar().week} 週（{WEEKDAYS[dt.weekday()]}）",
               f"UNIX 時刻 {int(dt.timestamp())}"],
        kind="now",
        detail={"iso": dt.isoformat(timespec="minutes"), "year": dt.year, "month": dt.month,
                "day": dt.day, "weekday": WEEKDAYS[dt.weekday()], "era": era,
                "era_year": era_year, "day_of_year": doy, "days_left": total},
    )


def try_elapsed(text: str) -> Solution | None:
    """「◯日から何日経った？」「あと何日？」"""
    t = str(text or "")
    m = re.search(r"(20\d{2}|19\d{2}|平成\d+|令和\d+|昭和\d+)?\s*[年\-/.]?\s*(\d{1,2})\s*月\s*"
                  r"(\d{1,2})\s*日[^。]*?(?:経った|過ぎた|すぎて|たった|経過|何日|あと|経つ)", t)
    if not m:
        return None
    today = now().date()
    year = today.year
    if m.group(1):
        g = m.group(1)
        if g.startswith("令和"):
            year = 2018 + int(g[2:])
        elif g.startswith("平成"):
            year = 1988 + int(g[2:])
        elif g.startswith("昭和"):
            year = 1925 + int(g[2:])
        else:
            year = int(g)
    try:
        day = _dt.date(year, int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
    diff = (today - day).days
    if diff >= 0:
        ans = f"{abs(diff)} 日経過"
        steps = [f"{day.isoformat()} → {today.isoformat()} = {diff} 日"]
    else:
        ans = f"あと {abs(diff)} 日"
        steps = [f"{today.isoformat()} → {day.isoformat()} = {abs(diff)} 日"]
    weeks, rest = divmod(abs(diff), 7)
    steps.append(f"約 {weeks} 週 {rest} 日")
    return Solution(answer=ans, steps=steps, kind="elapsed", detail={"days": diff})


def try_age(text: str) -> Solution | None:
    t = str(text or "")
    m = re.search(r"(20\d{2}|19\d{2}|昭和\d+|平成\d+|令和\d+)[^0-9]{0,4}(\d{1,2})?月?"
                  r"[^0-9]{0,3}(\d{1,2})?日?[^。]*?(?:歳|さい|年齢|いくつ)", t)
    if not m:
        return None
    today = now().date()
    g = m.group(1)
    if g.startswith("令和"):
        year = 2018 + int(g[2:])
    elif g.startswith("平成"):
        year = 1988 + int(g[2:])
    elif g.startswith("昭和"):
        year = 1925 + int(g[2:])
    else:
        year = int(g)
    month = int(m.group(2) or 1)
    day = int(m.group(3) or 1)
    try:
        born = _dt.date(year, month, day)
    except ValueError:
        return None
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    days = (today - born).days
    return Solution(answer=f"{age} 歳",
                    steps=[f"{born.isoformat()} 生まれ → {today.isoformat()} 時点で {days:,} 日目"],
                    kind="age", detail={"age": age, "days": days})


def try_clock_math(text: str) -> Solution | None:
    """「1時間45分は何分」「3日後は何日」「あと2時間で何時」"""
    t = str(text or "")
    m = re.search(r"(\d+)\s*(?:時間|h)\s*(\d+)?\s*(?:分|m)?[^。]*?(?:何分|分はいくつ|分にすると)", t)
    if m:
        mins = int(m.group(1)) * 60 + int(m.group(2) or 0)
        return Solution(answer=f"{mins} 分", steps=[f"{m.group(1)}×60 + {m.group(2) or 0}"], kind="time")
    m = re.search(r"(\d+)\s*(?:日|週間|時間|分|ヶ月|か月|年)[^。]{0,6}(?:後|あと|前|まえ|経つと)", t)
    if m:
        n = int(m.group(1))
        unit = m.group(0)
        delta = None
        today = now()
        if "日" in unit:
            delta = _dt.timedelta(days=n)
        elif "週間" in unit:
            delta = _dt.timedelta(weeks=n)
        elif "時間" in unit:
            delta = _dt.timedelta(hours=n)
        elif "分" in unit:
            delta = _dt.timedelta(minutes=n)
        elif "ヶ月" in unit or "か月" in unit:
            delta = _dt.timedelta(days=30 * n)
        if delta is None:
            return None
        sign = -1 if "前" in unit else 1
        target = today + sign * delta
        before = "後" if sign > 0 else "前"
        return Solution(answer=f"{target.year} 年 {target.month} 月 {target.day} 日"
                               f"（{WEEKDAYS[target.weekday()]}）{target:%H:%M}",
                        steps=[f"{today:%Y-%m-%d %H:%M} に {n}{'日' if delta.days else ''}{'時間' if delta.seconds else ''}"
                               f"{before}"],
                        kind="date_offset", detail={"iso": target.isoformat()})
    return None


def handle(text: str) -> Solution | None:
    """現在・日付の問いへの入口（検索を待たずに時計から答える）。"""
    t = str(text or "")
    if re.search(r"(今は|いまは|今日は何|今は何時|何年ですか|西暦|何曜日|今日の日付|本日|何日|現在|unix|"
                 r"何歳|経過|あと何日|何分|何時)", t):
        if re.search(r" unix ", f" {t.lower()} ") or "UNIX 時刻" in t or "エポック" in t:
            dt = now()
            return Solution(answer=str(int(dt.timestamp())), steps=[f"{dt:%Y-%m-%d %H:%M:%S}"],
                            kind="unix")
        sol = try_age(t) or try_elapsed(t) or try_clock_math(t)
        if sol:
            return sol
        return describe_now()
    return None


__all__ = ["handle", "now", "describe_now", "try_elapsed", "try_age", "try_clock_math", "era_of"]
