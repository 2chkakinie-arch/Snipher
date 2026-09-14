"""v8 再帰的思考モデルの API のテスト。

重み（v8*.npz）が無い環境でも graceful に縮退し、
存在する環境では再帰的思考の SSE 契約（thought / draft / verify / done）を返す。
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
    api_mod._recurrent = None
    from fastapi.testclient import TestClient

    with TestClient(api_mod.app) as c:
        yield c
    api_mod._core = None
    api_mod._recurrent = None


def _v8_available() -> bool:
    return Path("snipher/data/neural/v8.npz").exists()


def test_v8_status_is_graceful(client):
    r = client.get("/api/v8/status")
    assert r.status_code == 200
    data = r.json()
    if _v8_available():
        assert data["available"] is True
        assert data["kind"] == "recurrent-v8"
        assert data["core_B"]["kind"] == "moe"
    else:
        assert data["available"] is False


def test_v8_novel_endpoint(client):
    r = client.post("/api/v8/novel", json={"prompt": "小さな町の話", "max_chars": 64,
                                            "temperature": 0.9, "with_think": True})
    assert r.status_code == 200
    data = r.json()
    if _v8_available():
        assert data["ok"] is True
        assert data["text"]
    else:
        assert data["ok"] is False
        assert "未ビルド" in data.get("reason", "")


def test_chat_mode_v8_streams_recurrent_contract(client):
    """mode=v8 では recurrent の SSE 契約（start→…→done）で流れる。"""
    events = []
    with client.stream("POST", "/api/chat", json={
        "messages": [{"role": "user", "content": "赤信号ではどうする？"}],
        "mode": "v8", "max_new_tokens": 48,
    }) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    if not _v8_available():
        assert events and events[0].get("type") == "error"
        return
    types = [e.get("type") for e in events]
    assert types[0] == "start"
    assert "thought" in types
    assert "draft" in types
    assert "verify" in types
    assert types[-1] == "done"
    done = events[-1]
    assert done["stats"]["route"] == "recurrent"
