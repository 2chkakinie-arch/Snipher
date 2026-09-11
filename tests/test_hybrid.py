"""ハイブリッド補正(HybridAssist)とモデル取り込み API のテスト。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher.lfm.assist import AssistConfig, HybridAssist  # noqa: E402

torch = pytest.importorskip("torch")  # polish 経路の検証に使う(無ければ assist のみ)


@pytest.fixture()
def loaded_engine(lfm_env):
    from tests.conftest import fresh_engine

    eng = fresh_engine()
    eng.load()
    assert eng.is_ready
    return eng



# ---------------------------------------------------------------------- #
# ハイブリッド判定
# ---------------------------------------------------------------------- #
def test_assist_draft_is_fast_and_annotated():
    a = HybridAssist()
    d = a.draft("こんにちは")
    assert d["text"]
    assert d["draft_seconds"] < 0.5
    assert "base_text" in d and d["base_text"]


def test_assist_needs_lfm_threshold():
    a = HybridAssist(cfg=AssistConfig(threshold=0.5))
    d_low = {"text": "x", "base_text": "x", "confidence": 0.1}
    d_high = {"text": "x", "base_text": "x", "confidence": 0.9}
    assert a.needs_lfm(d_low) is True
    assert a.needs_lfm(d_high) is False
    a.cfg.enabled = False
    assert a.needs_lfm(d_low) is False


def test_assist_polish_messages():
    a = HybridAssist()
    msgs = a.polish_messages("下書きです。", "ユーザーの入力")
    assert msgs[0]["role"] == "system"
    assert "下書きです。" in msgs[1]["content"]
    assert "ユーザーの入力" in msgs[1]["content"]


# ---------------------------------------------------------------------- #
# LFM 補正経路(小型モデルで実際に回す)
# ---------------------------------------------------------------------- #
def _collect(gen, limit=200):
    events = []
    for i, ev in enumerate(gen):
        events.append(ev)
        if ev.get("type") in ("done", "error") or i > limit:
            break
    return events


def test_assist_polish_with_lfm_stream(lfm_env, loaded_engine):
    a = HybridAssist(cfg=AssistConfig(threshold=0.99))  # 強制的に LFM 補正
    draft = a.draft("変な話題きょくせん話")
    events = _collect(a.polish_with_lfm(loaded_engine, draft, "変な話題きょくせん話"))
    kinds = [e["type"] for e in events]
    assert "delta" in kinds
    assert kinds[-1] in ("delta", "done")  # 最後まで回りきる


def test_assist_polish_returns_text_and_stats(lfm_env, loaded_engine):
    a = HybridAssist(cfg=AssistConfig(threshold=0.99))
    draft = a.draft("変な話題")
    gen = a.polish_with_lfm(loaded_engine, draft, "変な話題")
    text = None
    stats = {}
    events = []
    while True:
        try:
            ev = gen.send(None)
        except StopIteration as stop:
            if stop.value:
                text, stats = stop.value
            break
        events.append(ev)
    assert any(e["type"] == "delta" for e in events)
    assert stats.get("assist") == "lfm"
    assert stats.get("draft") == draft["text"]
    assert text is not None and text != ""  # 空でも下書きへフォールバックしない


# ---------------------------------------------------------------------- #
# モデル取り込み API（HuggingFace に繋がらない環境向け）
# ---------------------------------------------------------------------- #
@pytest.fixture()
def client(tmp_path, monkeypatch):
    """アップロード先を tmp に向けた TestClient。"""
    import snipher.api as api_mod

    monkeypatch.setattr(api_mod, "UPLOAD_DIR", tmp_path / "models" / "upload")
    api_mod._lfm_engine = None
    from fastapi.testclient import TestClient

    return TestClient(api_mod.app)


def _engine_ready(client, timeout=120):
    deadline = time.time() + timeout
    state = None
    while time.time() < deadline:
        st = client.get("/api/status").json().get("lfm") or {}
        state = st.get("state")
        if state in ("ready", "failed"):
            return st
        time.sleep(0.3)
    return {"state": state}


def test_model_import_status_initially_incomplete(client):
    d = client.get("/api/model/import").json()
    assert d["upload"]["complete"] is False
    assert "config.json" in "".join(d["upload"]["missing"])


def test_model_upload_then_load(client, tiny_model):
    src = Path(tiny_model["model_dir"])
    files = [("files", (p.name, open(p, "rb"), "application/octet-stream")) for p in sorted(src.iterdir())]
    r = client.post("/api/model/upload", files=files, data={"activate": "0"}).json()
    assert r["ok"] is True
    assert r["status"]["complete"] is True, r["status"]["missing"]
    names = {f["name"] for f in r["status"]["files"]}
    assert "config.json" in names

    # 取り込み状態が complete になっている
    st = client.get("/api/model/import").json()
    assert st["upload"]["complete"] is True

    # ロード(ホットスワップ)→ ready まで待つ
    r2 = client.post("/api/model/load", json={}).json()
    assert r2["ok"] is True
    eng_st = _engine_ready(client)
    assert eng_st["state"] == "ready", eng_st.get("error")
    assert client.get("/api/status").json()["lfm"]["is_local"] is True


def test_model_upload_rejects_bad_files(client):
    r = client.post(
        "/api/model/upload",
        files=[("files", ("evil.sh", b"#!/bin/sh\n", "text/plain"))],
    ).json()
    assert r["ok"] is True
    assert r["skipped"] == ["evil.sh"]
    assert r["status"]["files"] == []


def test_model_load_missing_dir(client):
    r = client.post("/api/model/load", json={"source": "/nonexistent/model"}).json()
    assert r["ok"] is False
