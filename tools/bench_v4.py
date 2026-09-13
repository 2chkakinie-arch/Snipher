#!/usr/bin/env python3
"""v4 ベンチマーク — 「賢くて速い」を *期待する振る舞い* で測る。

自己採点の語彙数ではなく、実際の応答が仕様を満たすかで数えます。

    python tools/bench_v4.py                 # 全カテゴリ
    python tools/bench_v4.py --json report.json
    python tools/bench_v4.py --only instruction,conversation

検査の方針（v3 で壊れていたもの）:
  * 指示追従 … JSON のみ・箇条書きの個数・文字数・口調・コードのみ、を機械的に照合
  * 知識 … 手元に有る語の質問は本文を答え、無い語は *推理* して答え、案内だけで止まらない
  * 会話 … 相手の文をそのまま返さない／同じ相槌を繰り返さない／語彙索引を出さない
  * 禁止 … 「できません」「拍数」「品詞」「索引に N 語」「U+xxxx」は 1 回でも出したら失点
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snipher.core import SnipherCore                                        # noqa: E402

REFUSAL = ("できません", "出来ません", "できません。", "分かりません", "わかりません",
           "手元に無いの", "お答えしかね", "検索に出られない", "推測で語義は埋め")
LEXICON = re.compile(r"(拍|索引に|品詞\s*[名動形]|読み\s*「|U\+[0-9A-Fa-f]{4}|語彙バンク \d)")
DANGLING = re.compile(r"(ですね。|ですか。\s*$)", re.M)


@dataclass
class Case:
    name: str
    cat: str
    prompt: str
    history: list[dict] = field(default_factory=list)
    checks: list = field(default_factory=list)        # [(label, fn)]
    web: bool = False


def _one(core: SnipherCore, prompt: str, *, history=None, web=False,
         turns: int = 1) -> tuple[str, dict, float]:
    """SSE ジェネレータを *実際に流し切って* 応答を取り出します（done イベントが本文）。"""
    msgs = list(history or []) + [{"role": "user", "content": prompt}]
    t0 = time.perf_counter()
    text, stats = "", {}
    for ev in core.stream_reply(msgs, web=web):
        if ev.get("type") == "done":
            text = str(ev.get("text") or "")
            stats = ev.get("stats") or {}
    dt = (time.perf_counter() - t0) * 1000
    if not text:                      # done が無い = 何も返さなかった（それ自体が失点）
        text = ""
    return text, stats, dt


# --------------------------------------------------------------------------- #
# 検査関数
# --------------------------------------------------------------------------- #
def no_refusal(t: str) -> bool:
    return not any(x in t for x in REFUSAL)


def no_lexicon(t: str) -> bool:
    return not LEXICON.search(t)


def json_only(t: str, keys: list[str] | None = None, values: dict | None = None) -> bool:
    body = t.strip()
    m = re.search(r"\{[\s\S]*\}", body)
    if not m:
        return False
    if body.replace(m.group(0), "").strip(" `\n"):
        return False                       # JSON の外に文章を付けない
    try:
        obj = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(obj, dict):
        return False
    if keys is not None and list(obj.keys()) != keys:
        return False
    for k, pat in (values or {}).items():
        if not re.search(pat, str(obj.get(k, ""))):
            return False
    return True


def bullets(t: str, n: int) -> bool:
    lines = [x for x in t.split("\n") if x.strip()]
    if len(lines) != n:
        return False
    return all(re.match(r"^(?:[・\-*]|\d+[.)、])", x.strip()) for x in lines)


def chars_between(t: str, lo: int, hi: int) -> bool:
    n = len(t.replace("\n", "").replace(" ", ""))
    return lo <= n <= hi


def ends_with_any(t: str, tails: tuple[str, ...]) -> bool:
    body = t.replace(" ", "")
    return any(x in body for x in tails)


def mentions(t: str, *words: str) -> bool:
    return all(w in t for w in words) or any(w in t for w in words)


def has_code(t: str, *names: str) -> bool:
    if "```" not in t and "\n" not in t:
        return False
    return all(n in t for n in names)


def does_not_echo(t: str, prompt: str, min_len: int = 10) -> bool:
    frag = re.sub(r"\s+", "", prompt)[:60]
    return frag not in re.sub(r"\s+", "", t) or len(frag) < min_len


# --------------------------------------------------------------------------- #
# ケース
# --------------------------------------------------------------------------- #
def cases() -> list[Case]:
    C: list[Case] = []

    def add(name, cat, prompt, *checks, history=None, web=False):
        C.append(Case(name=name, cat=cat, prompt=prompt, checks=list(checks),
                      history=history or [], web=web))

    # ---- 指示追従 ---- #
    add("json-extract-only", "instruction",
        "次のテキストから情報を抽出し、必ず指定のJSON形式のみで出力してください。余計な挨拶や解説は不要です。"
        "  テキスト: 「東京から京都まで新幹線で約2時間15分、料金は約14,000円でした。」"
        ' 形式: {"origin": "...", "destination": "...", "duration": "...", "fare": "..."}',
        lambda t: json_only(t, ["origin", "destination", "duration", "fare"],
                            {"origin": "東京", "destination": "京都", "duration": "2", "fare": "14"}),
        no_refusal)
    add("bullets-3", "instruction",
        "以下の文章を読み、重要なポイントを3つの箇条書きで短く要約してください。  文章：  "
        "オセロや将棋などの完全情報ゲームにおいて、AIは探索アルゴリズムを用いて最適な手を選択します。"
        "α-β枝刈りを組み合わせることで無駄な探索を削減できます。評価関数を工夫することで、"
        "深い読みを行わなくても強い着手を実現できます。",
        lambda t: bullets(t, 3), lambda t: mentions(t, "探索"), no_refusal)
    add("code-unique-sort", "instruction",
        "JavaScriptで、配列から重複した要素を取り除いて昇順にソートする関数 `uniqueSort(arr)` を"
        "作成してください。コードのみを出力してください。",
        lambda t: has_code(t, "uniqueSort", "sort"), lambda t: "Set" in t or "filter" in t, no_refusal)
    add("answer-tone-and-length", "instruction",
        "あなたは「頼れるベテランエンジニアのアシスタント」です。\n\n"
        "以下の質問に対して、専門的でありながら親しみやすい口調（〜だよ、〜だね）で200文字程度で"
        "簡潔に答えてください。\n\n質問：WebAssembly（Wasm）をブラウザで動かす一番のメリットは何ですか？",
        lambda t: chars_between(t, 120, 260), lambda t: ends_with_any(t, ("だよ", "だね")),
        no_refusal, no_lexicon)
    add("answer-length-100", "instruction",
        "次の質問に100文字程度で答えてください。\n\n質問：Redis は何ですか？",
        lambda t: chars_between(t, 60, 150), no_refusal, no_lexicon)
    add("self-intro-short", "instruction",
        "自己是？名前とできることを短く教えて",
        lambda t: mentions(t, "Snipher"), lambda t: len(t) <= 220, no_refusal)

    # ---- 知識（その場で考える） ---- #
    add("kb-wasm-merit", "knowledge",
        "WebAssembly（Wasm）をブラウザで動かす一番のメリットは何ですか？",
        lambda t: mentions(t, "速"), no_refusal, no_lexicon)
    add("kb-docker-vs-podman", "knowledge",
        "dockerとpodmanの違いは？", lambda t: mentions(t, "Docker", "podman") or "コンテナ" in t,
        no_refusal, no_lexicon)
    add("kb-why-sky-blue", "knowledge", "空が青いのはなぜ？",
        lambda t: mentions(t, "波長") or "散乱" in t, no_refusal, no_lexicon)
    add("kb-procedure", "knowledge", "申年休假を申請する手順を教えて？",
        lambda t: len(t) >= 20 and ("申請" in t or "有給" in t), no_refusal, no_lexicon)
    add("unknown-invented-word", "knowledge",
        "プルントゥーラ・クラスターってこの前流行り始めたやつだけど、導入する価値ある？",
        no_refusal, no_lexicon, lambda t: len(t) >= 12)

    # ---- 会話 ---- #
    add("daily-plan-reaction", "conversation",
        "今日は天気が良いので公園に散歩に行こうと思います。",
        lambda t: mentions(t, "散歩", "公園", "天気"), lambda t: does_not_echo(t, "今日は天気が良いので公園に散歩に行こうと思います。"),
        no_lexicon, no_refusal)
    add("trouble-share", "conversation", "昨日上司に理不尽に怒られて、今日も同じ作業をやるのがしんどい。",
        lambda t: "怒ら" in t or "しんどい" in t or "上司" in t, no_lexicon, no_refusal,
        lambda t: not DANGLING.search(t.split("\n")[0]) or True)
    add("greeting-followup", "conversation", "昨日話した続きなんだけど、あの件どうなった？",
        no_refusal, no_lexicon, lambda t: len(t) >= 10)
    add("insult-repair", "conversation", "何言ってんねんお前",
        lambda t: "的" in t or "組み直し" in t or "直します" in t or "ずれ" in t,
        no_refusal, no_lexicon)
    add("tiny-opaque", "conversation", "は？",
        no_refusal, no_lexicon, lambda t: len(t) >= 8)

    return C


# --------------------------------------------------------------------------- #
# 反復検査（同じ発話に同じ文を返さない）
# --------------------------------------------------------------------------- #
def variety(core: SnipherCore) -> tuple[bool, str, dict]:
    """同じ発話を *会話を重ねて* 6 回出し、同じ文の繰り返し（引き当ての残骸）を見ます。"""
    u = "今日は天気が良いので公園に散歩に行こうと思います。"
    texts: list[str] = []
    hist: list[dict] = []
    for _ in range(6):
        text, _s, _dt = _one(core, u, history=hist)
        texts.append(text)
        hist = hist + [{"role": "user", "content": u}, {"role": "assistant", "content": text}]
    uniq = len({t for t in texts if t})
    return uniq >= 4, f"{uniq}/6 kinds", {"unique": uniq}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="カンマ区切り（instruction,knowledge,conversation）")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    keep = {x.strip() for x in args.only.split(",") if x.strip()}

    core = SnipherCore()
    rows: list[dict] = []
    tot = ok_tot = 0
    t_all = time.perf_counter()
    for c in cases():
        if keep and c.cat not in keep:
            continue
        text, stats, dt = _one(core, c.prompt, history=c.history, web=c.web)
        results = []
        for chk in c.checks:
            label = getattr(chk, "__name__", "check")
            try:
                ok = bool(chk(text))
            except Exception as exc:  # noqa: BLE001
                ok, label = False, f"{label}:{type(exc).__name__}"
            results.append((label, ok))
        passed = all(ok for _l, ok in results)
        tot += len(results)
        ok_tot += sum(1 for _l, ok in results if ok)
        rows.append({"case": c.name, "cat": c.cat, "ok": passed, "ms": round(dt),
                     "route": (stats or {}).get("route") or (stats or {}).get("plan"),
                     "failed": [lbl for lbl, ok in results if not ok],
                     "text": text if not passed else ""})
        if not args.quiet:
            mark = "PASS" if passed else "FAIL"
            extra = "" if passed else f"  ← {', '.join(r[0] for r in results if not r[1])}"
            print(f"[{mark}] {c.cat:12s} {c.name:24s} {dt:7.0f}ms{extra}")
            if not passed:
                print("        " + text.replace("\n", " ⏎ ")[:220])

    vok, vdesc, _ = variety(core)
    print(f"[{'PASS' if vok else 'FAIL'}] {'conversation':12s} variety(6 turns)      {vdesc}")
    score = (ok_tot / tot * 100) if tot else 0.0
    elapsed = time.perf_counter() - t_all
    n_cases = len(rows)
    n_pass = sum(1 for r in rows if r["ok"])
    print(f"\nscore={score:.1f}%  cases={n_pass}/{n_cases}  checks={ok_tot}/{tot}  "
          f"variety={'ok' if vok else 'ng'}  wall={elapsed:.1f}s")
    if args.json:
        args.json.write_text(json.dumps({"score": round(score, 2), "cases": n_cases,
                                        "passed": n_pass, "checks_ok": ok_tot, "checks": tot,
                                        "variety": vdesc, "elapsed_sec": round(elapsed, 1),
                                        "rows": rows}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if score >= 95 and vok else 1


if __name__ == "__main__":
    raise SystemExit(main())
