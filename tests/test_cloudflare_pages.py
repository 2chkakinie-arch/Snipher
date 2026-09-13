"""Cloudflare Pages で *デプロイして動くか* をビルド产物から検証する。

Pages は静的アセット + JS/Wasm の Functions を動かす場所なので、Snipher は
Python を載せず同梱の JS エンジン（public/engine.mjs）で答え、本体 API が
使える場所だけ `SNIPHER_API_ORIGIN` に転送します。ここでは
 1) ビルド产物が最新であること
 2) Pages Function が UI の契約（/api/status・/api/chat の SSE）を満たすこと
 3) JS エンジンが実際に同じ答えを返すこと（node があれば実測）
を見ます。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
FUNCS = ROOT / "functions"


def _node() -> str | None:
    return shutil.which("node")


# --------------------------------------------------------------------------- #
# ビルド产物
# --------------------------------------------------------------------------- #
def test_public_build_is_up_to_date() -> None:
    """`public/` と `functions/` がビルドスクリプトと一致している（古い成果物を配らない）。"""
    r = subprocess.run([sys.executable, "tools/build_public.py", "--check"], cwd=ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_public_assets_exist_and_are_small() -> None:
    for name in ("index.html", "engine.mjs", "kb.json", "kb.meta.json", "_headers"):
        assert (PUBLIC / name).exists(), name
    kb = PUBLIC / "kb.json"
    # Pages Functions は 1 アセット/バンドルに収める（Workers の上限を意識して小さく）
    assert kb.stat().st_size <= 4 * 1024 * 1024, kb.stat().st_size


def test_public_kb_shape() -> None:
    data = json.loads((PUBLIC / "kb.json").read_text(encoding="utf-8"))
    assert data["meta"]["topics"] >= 250, data["meta"]
    assert len(data["items"]) == data["meta"]["topics"]
    rows = [it for it in data["items"] if it.get("topic")]
    assert len(rows) >= 250
    # JS が読む欄だけが入っている（辞書の語彙ファイルや生コーパスを晒さない）
    assert {"id", "topic", "def"} <= set(rows[0].keys())
    assert all("sentences" not in it for it in rows)


def test_index_html_falls_back_to_the_bundled_engine() -> None:
    html = (PUBLIC / "index.html").read_text(encoding="utf-8")
    assert "./engine.mjs" in html, "API の無い場所でも同じ答えが返るフォールバックが必要"
    assert "/api/chat" in html


# --------------------------------------------------------------------------- #
# Pages Function
# --------------------------------------------------------------------------- #
def test_pages_function_covers_the_api_contract() -> None:
    fn = FUNCS / "api" / "[[path]].js"
    assert fn.exists(), "functions/api/[[path]].js がありません"
    src = fn.read_text(encoding="utf-8")
    for needle in ("/api/status", "/api/chat", 'type: "start"', 'type: "delta"', 'type: "done"',
                   "SNIPHER_API_ORIGIN", "env.ASSETS.fetch"):
        assert needle in src, needle
    assert "_engine/engine.mjs" in src
    assert (FUNCS / "_engine" / "engine.mjs").exists()


def test_pages_function_does_not_call_localhost() -> None:
    """ブラウザ側（静的 UI）が localhost を叩くとプレビューで壊れます。"""
    for path in [PUBLIC / "index.html", PUBLIC / "engine.mjs", FUNCS / "api" / "[[path]].js"]:
        src = path.read_text(encoding="utf-8")
        assert "127.0.0.1" not in src and "localhost:" not in src, path


# --------------------------------------------------------------------------- #
# JS エンジンの実測（node があれば）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(_node() is None, reason="node が無い環境では JS 実測を省略")
def test_pages_engine_answers_like_the_python_one() -> None:
    r = subprocess.run([_node(), "--test", "tests/js/engine.test.mjs"], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-4000:] + "\n" + r.stderr[-2000:]


@pytest.mark.skipif(_node() is None, reason="node が無い環境では JS 実測を省略")
def test_pages_engine_never_refuses() -> None:
    """Pages 版でも「できません」で止めない（本体と同じ約束）。"""
    script = """
import { readFileSync } from "node:fs";
import { Index, answer } from "./public/engine.mjs";
const index = new Index(JSON.parse(readFileSync("./public/kb.json", "utf8")));
const cases = ["三毛猫とは", "量子饅頭の作り方", "は？", "今日のニュースは？",
               "GLM9.7 とは何ですか？", "昨日はもう終わ"];
for (const text of cases) {
  const r = answer(text, { index, turn: 2 });
  console.log(JSON.stringify({ text, out: r.text }));
}
"""
    r = subprocess.run([_node(), "--input-type=module", "-e", script], cwd=ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    for line in r.stdout.strip().splitlines():
        got = json.loads(line)
        out = got["out"]
        assert out.strip(), got
        for banned in ("できません", "出来ません", "分かりません", "わかりません", "手元に無いので",
                       "拍", "品詞", "索引", "U+"):
            assert banned not in out, (got["text"], out, banned)
