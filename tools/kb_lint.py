"""知識ベースの文章検査 — 日本語として壊れている文をビルドの段階で弾く。

README が言う「完璧な文章」は、応答を組み立てる側だけの話ではありません。材料である
知識ベースの文に欧文の混入・簡体字・途中で切れた文が紛れていれば、それを組んだ応答は
必ず壊れます。v3 まで応答に辞書情報のゴミが混ざった件と同じ構造のバグ、つまり
*材料を信じて検査しなかった* のが原因なので、ビルド時に機械で確かめます。

規則は 3 つ。

  1. 用字 … 日本語に存在しない簡体字形と、中国語の機能語混入を拒否する。
     一覧は「`snipher/data` の実データに 1 回も現れない字だけ」を検証して載せてある。
  2. 欧文 … 技術名として確立した語以外を拒否する（files・manage 等の混入対策）。
  3. 文末 … 文末が閉じていない文を拒否する（切り詰め・組み立て途中の混入対策）。
     手順（how）と問い文（qa の左側）は体言止めが正当なので、検査しない。

    python tools/kb_lint.py                 # snipher/data/kb.json を検査
    python tools/kb_lint.py --from-modules  # tools/kb_data/ の方を検査
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "snipher" / "data"

_SENT_END = "。！？!?…：:"

# 技術名・固有名として日本語文中に書いて良いラテン語（小文字で比較して許す）。
LATIN_OK: frozenset[str] = frozenset("""
web webassembly wasm wasm32 wasm-pack emscripten javascript typescript node nodejs npm pnpm
yarn deno bun python rust golang go java javascriptcore hermes c cpp csharp ruby php kotlin
swift scala elixir haskell html css scss sass tailwind react vue svelte astro next nuxt
api rest graphql grpc json jsonl yaml toml xml csv tsv md sql http https http2 http3 quic
dns tcp udp ip ipv4 ipv6 tls ssl ssh ftp smtp imap cdn edge server serverless cloud
cloudflare pages workers wrangler vercel render heroku netlify github gitlab docker compose
kubernetes k8s pod helm terraform ansible nginx apache redis postgres postgresql mysql sqlite
mongodb influxdb elastic search kibana linux unix macos windows ios android chrome firefox
safari edge chromium webkit git gitflow pr ci cd lint eslint prettier ruff mypy pytest jest
vitest cypress playwright shell bash zsh sh terminal cli gui ui ux spa ssr ssg hydration
utf utf-8 utf8 ascii unicode emoji ansi shift_jis euc-jp
ai ml dl nlp cv ocr llm gpt prompt prompt-engineering tokenizer token tokens embedding
embeddings dataset batch epoch learning gradient tensor pytorch torch numpy numpy2 numpyjs
cpu gpu tpu ram ssd hdd vram cache kv cache-control etag
url uri uuid uri-template email webhook websocket websocket-api
kb mb gb tb ms sec v1 v2 v3 v4 beta alpha rc ok ng abc xyz
frontend backend fullstack devops mlops saas paas iaas
man ls cd grep cat sed awk curl wget tar ssh ping traceroute df du top ps kill chmod chown
make cmake dockerfile gitignore readme license dot-env env localhost
one two three four five six seven eight nine ten
snipher snipher-core
gdp dna rna atp kcal calorie toefl toeic ielts h2o co2 ph
get post put patch delete head build run logs stats init add commit push pull diff switch
status merge clone fetch rebase stash revert bisect hello-world microsoft wi-fi user-agent
cpu gpu ram ssd ntp pop3 mac lan vpn proxy instagram youtube tiktok facebook line discord reddit wikipedia google bing duckduckgo
sns gps conflict reset oneline charset console print pip ghz insert select update create table where join
index.html console.log if else for while def class import return none null true false struct enum
log level radar racecar node.js c++ dom object jit interface type sse xss flex grid webgpu webgl undefined help
chatbot composer responder sentence snippet
""".split())

_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_.+\-#]*")
# 「git commit」のように *名指しで示された* 語は、混入ではなく固有名・コマンド名として扱う
_QUOTED = re.compile("[「『][^」』]{0,80}[」』]")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?])")
_HAS_JA = re.compile(r"[ぁ-んァ-ヶ一-龯]")

# 検査対象の欄（文末の閉塞を要求する / 要求しない）
SENTENCE_FIELDS = ("def", "opinion", "when", "where", "who", "cost", "facts", "why", "tips",
                   "followups")
PHRASE_FIELDS = ("how",)


# ---------------------------------------------------------------------- #
# 用字の検査（簡体字・中国語混入）
# ---------------------------------------------------------------------- #
# 簡体字のなかには日本語の常用漢字と字形が同じ物もあるので、
# 「日本語側には存在しない簡体字形だけ」を列挙して拒否する（誤検知を避ける）。
SIMPLIFIED_ONLY: frozenset[str] = frozenset(
    # 日本語の実データ（コーパス・語彙テーブル・既存 KB・英和語彙）に 1 回も現れない
    # 簡体字形だけを検証して載せた一覧。`tests/test_kb_lint.py` がこの前提を壊さない。
    "亲们关动卢厅历县发变启团图圆场坝垒垫处复宾对导尔尘尝岁岂帅师帐帘带帧帮并广庄庆庐库应庙废开异弃张弯录彩彻忆怀态总恼悬惊惧惯戏户扑执扩扫扬报拟拥择挂据撑撒播敌敛斗时显晓术杀权极构柜标栏树样档梦检椭楼欢歼毁毕气汇汉沟泽洁济润涨渐渔滚满滤滨灭灵灾炉炼烦热爱牵牺犹狮猎玛环现珑电畅疗疯皱监睁瞩矫码确碍离种积秽稀约级纪纳纵纷线练组细织终经结绕绘给络绝继绩续维综绿缓缩网罗罚罢职联肃观视让记许论设访证评识译试误课调谈运还这进远违连那麽"
)
# 中国語の機能語が日本語文中に混ざった形（「美丽的花」「没有问题」等）を検出する
_CN_GRAMMAR = (
    # 「具体的」「統計的」は日本語として正しいので、かなに接続した 的 だけを弾く
    # （「美丽的」「問題有的」のような語順は日本語に存在しない）。
    re.compile(r"[ぁ-ん]的"),
    re.compile(r"没有"),                       # 「没有」は日本語として書かない
    re.compile(r"[がをには]了[。、]"),           # 動詞の直後の「了」は中国語の完了助詞
)


def unknown_kanji(text: str, *, kanji: frozenset[str] | None = None) -> list[str]:
    """日本語の字形に存在しない簡体字・中国語混入の検知。"""
    t = str(text or "")
    seen: list[str] = []
    for c in t:
        if c in SIMPLIFIED_ONLY and c not in seen:
            seen.append(c)
    for pat in _CN_GRAMMAR:
        m = pat.search(t)
        if m:
            tag = f"中国語混入:{m.group(0)}"
            if tag not in seen:
                seen.append(tag)
    return seen


def stray_latin(text: str) -> list[str]:
    """技術名として確立していないラテン語の混入。"""
    if not _HAS_JA.search(text):
        return []                              # 欧文のみの文は検査しない
    out: list[str] = []
    body = _QUOTED.sub(" ", text)               # 引用符の中は検査しない
    for raw in _LATIN.findall(body):
        tok = raw.lower().strip(".,;:()（）「」")
        if not tok or len(tok) <= 2 or tok in LATIN_OK or tok.isdigit():
            continue
        if any(p in LATIN_OK for p in tok.split("-")) and "-" in tok:
            continue                           # wasm-pack のような連結語
        if tok not in out:
            out.append(tok)
    return out


def open_sentences(text: str) -> list[str]:
    """文末が閉じていない断片（途中で切れた文の検知）。"""
    out: list[str] = []
    for part in _SENT_SPLIT.split(text):
        piece = part.strip()
        if len(piece) < 6:
            continue
        if piece[-1] not in _SENT_END and not piece.endswith(("」", "）", "】")):
            out.append(piece[-14:])
    return out


def lint_text(text: str, *, sentence: bool = True) -> list[str]:
    """1 文に対する検査 → 問題のリスト（空 = OK）。"""
    t = str(text or "").strip()
    if not t:
        return []
    problems: list[str] = []
    bad = unknown_kanji(t)
    if bad:
        problems.append(f"用字の問題: {''.join(bad[:8])}")
    stray = stray_latin(t)
    if stray:
        problems.append(f"未登録の欧文混入: {', '.join(stray[:6])}")
    if sentence:
        cut = open_sentences(t)
        if cut:
            problems.append(f"文末が閉じていない: {cut[:3]}")
    return problems


def lint_item(item: dict) -> list[str]:
    """KB 1 項目を検査する。"""
    out: list[str] = []
    rid = str(item.get("id") or item.get("topic") or "?")
    for field in SENTENCE_FIELDS:
        for p in lint_text(item.get(field) or ""):
            out.append(f"{rid}.{field}: {p}")
    for field in PHRASE_FIELDS:
        for p in lint_text(" ".join(str(s) for s in (item.get(field) or [])), sentence=False):
            out.append(f"{rid}.{field}: {p}")
    for i, pair in enumerate(item.get("qa") or []):
        for side, s in enumerate(pair):
            # 問い文（左）は体言止めが正当。答え（右）だけ文末を見る。
            for p in lint_text(str(s), sentence=side == 1):
                out.append(f"{rid}.qa[{i}][{'q' if side == 0 else 'a'}]: {p}")
    return out


def load_items(from_modules: bool) -> list[dict]:
    if from_modules:
        sys.path.insert(0, str(ROOT / "tools"))
        from kb_data import all_items  # type: ignore

        return all_items()
    return json.loads((DATA / "kb.json").read_text(encoding="utf-8"))["items"]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-modules", action="store_true", help="tools/kb_data/ の方を検査する")
    ap.add_argument("--limit", type=int, default=120)
    args = ap.parse_args(argv)

    items = load_items(args.from_modules)
    problems: list[str] = []
    for it in items:
        problems.extend(lint_item(it))
    if problems:
        print(f"KB 検査: {len(problems)} 件 / {len(items)} 話題")
        for p in problems[: args.limit]:
            print("  -", p)
        return 1
    print(f"KB 検査: OK（{len(items)} 話題・全文）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
