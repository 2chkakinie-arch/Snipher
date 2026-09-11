"""LFM ニューラルエンジンのテスト（テンプレート / 未知文字学習 / 生成）。"""

from __future__ import annotations

import json
import time

import pytest

from tests.conftest import fresh_engine

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from snipher.lfm.template import (  # noqa: E402
    MODE_BUILTIN,
    MODE_NATIVE,
    MODE_RAW,
    TemplateManager,
)
from transformers import AutoTokenizer  # noqa: E402

RARE = "\U00016787"  # 𠮷 (U+16787, 4バイト文字 → バイト断片に分解される)


# ---------------------------------------------------------------------- #
# テンプレート
# ---------------------------------------------------------------------- #
def test_builtin_template_when_no_native(tiny_model):
    tok = AutoTokenizer.from_pretrained(tiny_model["model_dir"])
    assert not tok.chat_template
    tm = TemplateManager(tok)
    text, mode = tm.apply([{"role": "user", "content": "こんにちは"}])
    assert mode == MODE_BUILTIN
    assert text.startswith("<|startoftext|>")
    assert "<|im_start|>user\nこんにちは<|im_end|>" in text
    assert text.endswith("<|im_start|>assistant\n")


def test_native_template_when_present(tiny_model):
    tok = AutoTokenizer.from_pretrained(tiny_model["native_dir"])
    assert tok.chat_template
    tm = TemplateManager(tok)
    text, mode = tm.apply([{"role": "user", "content": "こんにちは"}])
    assert mode == MODE_NATIVE
    assert "<|im_start|>user" in text and text.endswith("<|im_start|>assistant\n")


def test_raw_mode_no_template(tiny_model):
    tok = AutoTokenizer.from_pretrained(tiny_model["model_dir"])
    tm = TemplateManager(tok)
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "そのまま続けて1"},
        {"role": "assistant", "content": "つづき"},
        {"role": "user", "content": "そのまま続けて2"},
    ]
    text, mode = tm.apply(msgs, use_template=False)
    assert mode == MODE_RAW
    assert "im_start" not in text
    assert "SYS" in text  # 内容は残る（テンプレート無しでもテキストはそのまま）


# ---------------------------------------------------------------------- #
# エンジン: ロード → 生成
# ---------------------------------------------------------------------- #
@pytest.fixture()
def loaded_engine(lfm_env):
    eng = fresh_engine()
    eng.load()
    assert eng.is_ready
    return eng


def _collect(gen, limit=100):
    events = []
    for i, ev in enumerate(gen):
        events.append(ev)
        if ev.get("type") in ("done", "error") or i > limit:
            break
    return events


def test_engine_stream_chat(loaded_engine):
    eng = loaded_engine
    events = _collect(
        eng.stream_chat(
            [{"role": "user", "content": "こんにちは"}],
            max_new_tokens=8,
            min_new_tokens=6,
            temperature=0.5,
        )
    )
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start"
    assert "delta" in kinds
    assert kinds[-1] == "done"
    done = events[-1]
    assert done["stats"]["new_tokens"] >= 6
    assert done["stats"]["tokens_per_second"]


def test_engine_raw_generation(loaded_engine):
    eng = loaded_engine
    events = _collect(
        eng.stream_chat(
            [{"role": "user", "content": "テンプレートなし"}],
            use_template=False,
            max_new_tokens=6,
            min_new_tokens=4,
            temperature=0.5,
        )
    )
    assert events[-1]["type"] == "done"
    assert events[0]["template_mode"] == MODE_RAW


# ---------------------------------------------------------------------- #
# 未知文字の検出と学習
# ---------------------------------------------------------------------- #
def test_scan_detects_unknown(loaded_engine):
    scan = loaded_engine.scan_unknown(f"これは{RARE}です。")
    assert scan["counts"]["unknown"] >= 1
    rep = scan["unknown"][0]
    assert rep["char"] == RARE
    assert rep["status"] in ("unk", "bytes", "fragmented")


def test_instant_learn_assigns_reserved_token(loaded_engine):
    eng = loaded_engine
    res = eng.learn_instant([RARE])
    r = res["results"][0]
    assert r["token_id"] >= eng.base_vocab
    assert not r["already"]
    # 2回目は already
    res2 = eng.learn_instant([RARE])
    assert res2["results"][0]["already"]

    # 埋め込み行が非ゼロ（事前学習済み断片の平均で初期化されている）
    with torch.no_grad():
        row = eng.model.get_input_embeddings().weight[r["token_id"]]
    assert row.abs().sum().item() > 0

    # 永続化されている
    assert eng.store.json_path.exists()
    data = json.loads(eng.store.json_path.read_text(encoding="utf-8"))
    assert RARE in data


def test_deep_learn_updates_rows_and_survives_reload(lfm_env):
    eng = fresh_engine()
    eng.load()
    res = eng.learn_deep([RARE], examples=[f"これは{RARE}です。", f"{RARE}が好きです。"], steps=2)
    assert res["mode"] == "deep"
    assert res["steps"] == 2
    assert res["loss_last"] is not None
    # 予約トークンへの写像が効き、損失が実際に動いている（勾配が行に届いている）
    assert res["loss_last"] != res["loss_first"]
    assert eng.is_ready  # 再ロード済み

    # 行が永続化され、再ロード後も復元される
    eng2 = fresh_engine()
    eng2.load()
    tid = int(eng2.store.chars[RARE]["token_id"])
    with torch.no_grad():
        row = eng2.model.get_input_embeddings().weight[tid]
    assert row.abs().sum().item() > 0


def test_learned_char_mapped_in_prompt(loaded_engine):
    eng = loaded_engine
    eng.learn_instant([RARE])
    tid = int(eng.store.chars[RARE]["token_id"])
    # プロンプト中の断片列が予約トークンに写像される（断片→1トークンなので短くなる）
    ids = eng.tokenizer(f"これは{RARE}です", add_special_tokens=False)["input_ids"]
    mapped = eng.learner.map_ids(eng.tokenizer, list(ids))
    assert tid in mapped
    assert len(mapped) < len(ids)


def test_reserved_suppression(loaded_engine):
    eng = loaded_engine
    from snipher.lfm.engine import _SuppressReserved

    allowed = eng.store.used_ids()
    proc = _SuppressReserved(eng.base_vocab, eng.total_vocab, allowed)
    scores = torch.zeros(1, eng.total_vocab)
    out = proc(None, scores)
    free = [i for i in range(eng.base_vocab, eng.total_vocab) if i not in allowed]
    assert out[0, free[0]].item() == float("-inf")
    assert out[0, 0].item() == 0.0


# ---------------------------------------------------------------------- #
# API（FastAPI TestClient 経由・環境変数は lfm_env を使用）
# ---------------------------------------------------------------------- #
def test_api_end_to_end(lfm_env, monkeypatch):
    from fastapi.testclient import TestClient

    import snipher.api as api_mod

    api_mod._lfm_engine = None  # シングルトンを環境変数付きで作り直す
    client = TestClient(api_mod.app)

    # status: ready まで待つ（バックグラウンドロード）
    deadline = time.time() + 120
    state = None
    while time.time() < deadline:
        st = client.get("/api/status").json()["lfm"]
        state = st["state"]
        if state == "ready":
            break
        if state == "failed":
            pytest.fail(f"モデルロード失敗: {st['error']}")
        time.sleep(0.3)
    assert state == "ready"

    # 未知文字チェック
    scan = client.post("/api/vocab/check", json={"text": f"今日は{RARE}の日です"}).json()
    assert scan["ok"] and scan["counts"]["unknown"] >= 1

    # チャット（SSE）
    with client.stream(
        "POST",
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": f"これは{RARE}の話です"}],
            "max_new_tokens": 8,
            "temperature": 0.5,
        },
    ) as resp:
        assert resp.status_code == 200
        events = []
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    kinds = [e["type"] for e in events]
    assert kinds[0] in ("meta", "start")
    assert "start" in kinds and "delta" in kinds and "done" in kinds
    assert events[-1]["stats"]["engine"].startswith("LFM")

    # 即時学習
    r = client.post("/api/learn", json={"chars": [RARE], "mode": "instant"}).json()
    assert r["ok"] and r["results"][0]["token_id"] >= 0

    # 学習リセット
    r = client.request("DELETE", "/api/learn").json()
    assert r["ok"]
    st = client.get("/api/status").json()["lfm"]
    assert st["learned_chars"] == []
