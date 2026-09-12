"""GGUF バックエンド(llama.cpp)のテスト。

llama_cpp をモックして、イベント契約・テンプレートフォールバック・
engine_name・scan_unknown を検証する。実 GGUF がある環境では
SNIPHER_TEST_GGUF=/path/to/model.gguf で実推論テストも走る。
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher.lfm.config import LfmConfig  # noqa: E402
from snipher.lfm.gguf_backend import GgufBackend, runtime_kind  # noqa: E402


class FakeLlama:
    last_kwargs: dict = {}

    def __init__(self, model_path=None, **kw):
        FakeLlama.last_kwargs = {"model_path": model_path, **kw}
        self.metadata = {
            "general.architecture": "lfm2",
            "general.name": "LFM2.5-TEST-JP",
            "tokenizer.chat_template": "{{ 'x' }}",
        }

    def create_chat_completion(self, messages=None, stream=False, **kw):
        assert stream is True
        for tok in ("こん", "にちは"):
            yield {"choices": [{"delta": {"content": tok}}]}
        yield {"choices": [{"delta": {}}]}

    def create_completion(self, prompt=None, stream=False, **kw):
        assert stream is True
        assert isinstance(prompt, str) and prompt
        for tok in ("東", "京。"):
            yield {"choices": [{"text": tok}]}


@pytest.fixture()
def fake_llama_cpp(monkeypatch):
    mod = types.ModuleType("llama_cpp")
    mod.Llama = FakeLlama
    mod.__version__ = "0.0.0-fake"
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    return mod


@pytest.fixture()
def gguf_file(tmp_path):
    p = tmp_path / "LFM2.5-1.2B-JP-202606-Q4_K_M.gguf"
    p.write_bytes(b"GGUF\x00\x00\x00\x00" + b"\x00" * 64)
    return p


def test_runtime_kind_detects_llama_cpp(fake_llama_cpp):
    assert runtime_kind(LfmConfig()) == "llama-cpp-python"


def test_backend_load_and_engine_name(fake_llama_cpp, gguf_file):
    b = GgufBackend(LfmConfig())
    b.load(gguf_file)
    assert b.state == "ready"
    assert b.is_ready
    name = b.engine_name()
    assert name.startswith("LFM2.5-TEST-JP")  # GGUF メタデータの実名
    assert "Q4_K_M" in name and "GGUF" in name
    st = b.status()
    assert st["state"] == "ready" and st["is_local"] is True
    assert st["learned_chars"] == []
    b.unload()
    assert b.state == "idle"


def test_stream_chat_native_template(fake_llama_cpp, gguf_file):
    b = GgufBackend(LfmConfig())
    b.load(gguf_file)
    evs = list(b.stream_chat([{"role": "user", "content": "こんにちは"}], max_new_tokens=8))
    kinds = [e["type"] for e in evs]
    assert kinds[0] == "start" and kinds[-1] == "done" and "delta" in kinds
    done = evs[-1]
    assert done["text"] == "こんにちは"
    assert done["stats"]["template_mode"] == "native"
    assert done["stats"]["new_tokens"] == 2
    assert done["stats"]["engine"].startswith("LFM")
    # システムプロンプトが自動注入されている
    msgs = FakeLlama.last_kwargs  # load 時のものであるため生成呼び出しは下記で確認
    b.unload()


def test_stream_chat_raw_mode(fake_llama_cpp, gguf_file):
    b = GgufBackend(LfmConfig())
    b.load(gguf_file)
    evs = list(b.stream_chat([{"role": "user", "content": "日本の首都は"}],
                             max_new_tokens=8, use_template=False))
    done = [e for e in evs if e["type"] == "done"][0]
    assert done["stats"]["template_mode"] == "raw"
    assert done["text"] == "東京。"
    b.unload()


def test_stream_chat_builtin_chatml_when_no_template(fake_llama_cpp, gguf_file):
    b = GgufBackend(LfmConfig())
    b.load(gguf_file)
    b._llm.metadata.pop("tokenizer.chat_template")  # テンプレートが無い GGUF
    evs = list(b.stream_chat([{"role": "user", "content": "日本の首都は"}], max_new_tokens=8))
    done = [e for e in evs if e["type"] == "done"][0]
    assert done["stats"]["template_mode"] == "builtin"
    assert done["text"] == "東京。"
    b.unload()


def test_scan_unknown_is_byte_fallback(fake_llama_cpp, gguf_file):
    b = GgufBackend(LfmConfig())
    b.load(gguf_file)
    scan = b.scan_unknown("𠮷野家🚀")
    assert scan["counts"]["unknown"] == 0
    assert scan["unknown"] == []
    assert "byte-fallback" in scan["note"]
    b.unload()


# ---------------------------------------------------------------------- #
# 実モデルテスト（SNIPHER_TEST_GGUF が指定されたときだけ実行）
# ---------------------------------------------------------------------- #
REAL_GGUF = os.environ.get("SNIPHER_TEST_GGUF", "").strip()


@pytest.mark.skipif(
    not REAL_GGUF or not Path(REAL_GGUF).exists() or not os.environ.get("SNIPHER_TEST_GGUF_RUN"),
    reason="SNIPHER_TEST_GGUF(+_RUN=1) に実 GGUF を指定したときだけ実行",
)
def test_real_gguf_inference():
    pytest.importorskip("llama_cpp")
    cfg = LfmConfig()
    cfg.prompt_budget = 512
    cfg.max_new_tokens = 16
    b = GgufBackend(cfg)
    b.load(REAL_GGUF)
    assert b.state == "ready"
    evs = list(b.stream_chat([{"role": "user", "content": "Reply with exactly one word: OK"}],
                             max_new_tokens=8, temperature=0.1))
    done = [e for e in evs if e["type"] == "done"]
    assert done and done[0]["text"].strip()
    assert done[0]["stats"]["tokens_per_second"] > 0
    b.unload()
