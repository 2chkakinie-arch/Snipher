"""Snipher Core API（/api/status, /api/chat, /api/complete, /api/model/*）のテスト。

バックエンドを off にしてネットワーク・ torch なしで決定論的に検証する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("SNIPHER_LFM_BACKEND", "off")
    monkeypatch.setenv("SNIPHER_LFM_AUTO_FETCH", "0")
    import snipher.api as api_mod

    api_mod._core = None
    api_mod._lfm_engine = None
    from fastapi.testclient import TestClient

    with TestClient(api_mod.app) as c:
        yield c
    api_mod._core = None


def _sse(client, payload) -> list[dict]:
    events = []
    with client.stream("POST", "/api/chat", json=payload) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events


def test_status_shape_without_neural(client):
    d = client.get("/api/status").json()
    assert "lfm" in d and "acquire" in d and "hybrid" in d
    assert d["fallback"] == "Snipher-mini+"
    assert d["hybrid"]["available"] is False


def test_chat_fallback_is_instant_japanese(client):
    evs = _sse(client, {"messages": [{"role": "user", "content": "こんにちは"}], "mode": "auto"})
    kinds = [e["type"] for e in evs]
    assert kinds[0] == "assist" and "start" in kinds and "delta" in kinds and kinds[-1] == "done"
    done = evs[-1]
    assert done["stats"]["route"] == "fallback"
    assert done["stats"]["fallback_reason"]
    assert done["text"]


def test_chat_fast_mode(client):
    evs = _sse(client, {"messages": [{"role": "user", "content": "ありがとう！"}], "mode": "fast"})
    done = evs[-1]
    assert done["stats"]["assist"] == "rule"
    assert done["stats"]["route"] == "instant"


def test_chat_legacy_hybrid_flag(client):
    # hybrid=False → mode=lfm 相当（ニューラルが無いので結果は fallback 経路）
    evs = _sse(client, {"messages": [{"role": "user", "content": "こんにちは"}], "hybrid": False})
    assert evs[-1]["type"] == "done"


def test_complete_endpoint(client):
    r = client.post("/api/complete", json={"text": "私は猫が好き"}).json()
    assert r["ok"] is True
    assert r["text"]
    assert r["engine"] == "rule"  # ニューラル off なのでルールのみ


def test_model_fetch_endpoint_no_backend(client):
    r = client.post("/api/model/fetch", json={}).json()
    assert r["ok"] is True
    # off → deps_missing になりネットワークには出ない
    st = client.get("/api/model/acquire").json()
    assert st["boot_state"] in ("deps_missing", "idle", "booting", "failed")


def test_acquire_status_endpoint(client):
    d = client.get("/api/model/acquire").json()
    assert "phase" in d and "boot_state" in d


def test_classic_and_root_pages(client):
    assert client.get("/").status_code == 200
    assert "Snipher" in client.get("/").text
    assert client.get("/classic").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}
