"""LFM2.5 フルウェイトを常駐ホストに委譲するバックエンドのテスト。

ネットワークは使わない（`_req` を差し替えて SSE を再現する）。
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher.lfm.config import LfmConfig  # noqa: E402
from snipher.lfm.remote_backend import (  # noqa: E402
    RemoteLfmBackend,
    _extract_text,
    _normalize,
    backend_from_env,
)


class FakeResp(io.BytesIO):
    """urlopen が返すオブジェクトの最小実装（イテレータ = 1 行ずつ）。"""

    def __init__(self, lines: list[bytes], headers: dict | None = None, status: int = 200):
        super().__init__(b"".join(lines))
        self._lines = lines
        self.headers = headers or {"Content-Type": "text/event-stream"}
        self.status = status

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _backend(lines: list[bytes], headers: dict | None = None) -> RemoteLfmBackend:
    b = RemoteLfmBackend("https://vps.example.com", token="t0k")
    b._alive = True  # probe 済み扱い
    b._req = lambda path, payload=None, **kw: FakeResp(lines, headers)  # type: ignore[assignment]
    return b


def test_backend_from_env_requires_url():
    cfg = LfmConfig()
    cfg.remote_url = ""
    assert backend_from_env(cfg) is None
    cfg.remote_url = "http://127.0.0.1:9"
    b = backend_from_env(cfg)
    assert b is not None and b.token == ""


def test_stream_snipher_sse_is_relayed():
    payload = [
        b'data: {"type":"start","engine":"LFM2.5-1.2B-JP"}\n\n',
        b'data: {"type":"delta","text":"\\u304a\\u306f"}\n\n',
        b'data: {"type":"delta","text":"\\u3088\\u3046"}\n\n',
        b'data: {"type":"done","text":"\\u304a\\u306f\\u3088\\u3046","stats":{"tokens_per_second":31.0}}\n\n',
    ]
    b = _backend(payload)
    evs = list(b.stream_chat([{"role": "user", "content": "こんにちは"}], max_new_tokens=32))
    kinds = [e["type"] for e in evs]
    assert kinds == ["start", "delta", "delta", "done"]
    done = evs[-1]
    assert done["text"] == "おはよう"
    assert done["stats"]["tokens_per_second"] == 31.0
    assert done["stats"]["backend"] == "remote"
    assert "vps.example.com" in b.engine_name()


def test_stream_openai_sse_is_normalized():
    payload = [
        'data: {"choices":[{"delta":{"content":"猫"}}]}\n\n'.encode("utf-8"),
        'data: {"choices":[{"delta":{"content":"いいですね"}}]}\n\n'.encode("utf-8"),
        b"data: [DONE]\n\n",
    ]
    b = _backend(payload)
    evs = list(b.stream_chat([{"role": "user", "content": "hi"}]))
    assert "".join(e["text"] for e in evs if e["type"] == "delta") == "猫いいですね"
    assert evs[-1]["type"] == "done"


def test_non_stream_json_is_accepted():
    obj = json.dumps({"text": "そうでしたね。"}).encode("utf-8")
    b = _backend([b""], headers={"Content-Type": "application/json"})
    b._req = lambda path, payload=None, **kw: FakeResp([obj], {"Content-Type": "application/json"})  # type: ignore[assignment]
    evs = list(b.stream_chat([{"role": "user", "content": "x"}]))
    assert evs[-1]["text"] == "そうでしたね。"


def test_unreachable_remote_degrades_without_raising(tmp_path):
    b = RemoteLfmBackend("http://127.0.0.1:1", token="")
    b._alive = False
    b._alive_at = 1e12            # probe をスキップさせないための固定
    evs = list(b.stream_chat([{"role": "user", "content": "x"}]))
    assert evs[0]["type"] == "error"
    r = b.complete("私は猫が好き")
    assert r["changed"] is False and r["text"] == "私は猫が好き"


def test_status_shape():
    b = RemoteLfmBackend("http://localhost:8000", token="x")
    st = b.status()
    assert st["backend"] == "remote-http" and st["is_local"] is False
    assert st["download_required"] is False and st["auth"] is True
    assert b.scan_unknown(" Anything ") is None


def test_normalize_helpers():
    assert _normalize({"type": "delta", "text": "あ"}, "snipher") == {"type": "delta", "text": "あ"}
    assert _normalize({"unknown": 1}, "snipher") == {"type": "skip"}
    assert _normalize({"choices": [{"delta": {"content": "い"}}]}, "openai")["text"] == "い"
    assert _extract_text({"choices": [{"message": {"content": "う"}}]}, "openai") == "う"
    assert _extract_text({"text": "え"}, "snipher") == "え"


def test_core_selects_remote_as_heavy_backend(monkeypatch):
    """リモートが生きていれば「重いコアが居ない環境」でも neural 経路になる。"""
    from snipher.core import ROUTE_NEURAL, SnipherCore

    core = SnipherCore(torch_provider=lambda: None)
    remote = _backend([b'data: {"type":"start"}\n\n',
                       b'data: {"type":"delta","text":"\\u304a\\u306f\\u3088\\u3046\\u3067\\u3059\\u3002"}\n\n',
                       b'data: {"type":"done","text":"\\u304a\\u306f\\u3088\\u3046\\u3067\\u3059\\u3002"}\n\n'])
    core.remote_backend = lambda: remote          # type: ignore[method-assign]
    core.cfg.remote_url = "https://vps.example.com"
    core.disable_light()
    assert core.active_backend() is remote
    draft = core.assist.draft("量子コンピュータの仕組みってどうなってるの")
    assert core.route_of(draft, "auto") == ROUTE_NEURAL
    evs = list(core.stream_reply([{"role": "user", "content": "猫について教えて"}], mode="auto"))
    done = evs[-1]
    assert done["text"].startswith("おはよう")
    assert done["stats"]["route"] == ROUTE_NEURAL
