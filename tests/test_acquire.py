"""モデル自動取得(ModelAcquirer)のテスト。

ローカル HTTP サーバー（Range 対応）に対して、レジューム・ソースフォールバック・
共有ページ解決・破損検出を検証する。ネットワークには一切出ない。
"""

from __future__ import annotations

import http.server
import socketserver
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snipher.lfm.acquire import FilePlan, FileSource, ModelAcquirer  # noqa: E402


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Range リクエストに対応する最小のファイルサーバー。"""

    root: Path = Path(".")
    fail: set = set()          # 404 にするパス
    no_range: set = set()      # Range を無視するパス

    def log_message(self, *a):  # 静かに
        pass

    def _path(self):
        return self.path.split("?")[0]

    def _send(self, body: bool, method_ok=(200, 206)):
        p = self.root / self._path().lstrip("/")
        if self._path() in self.fail or not p.exists() or not p.is_file():
            self.send_response(404)
            self.end_headers()
            return
        data = p.read_bytes()
        rng = self.headers.get("Range")
        start = 0
        code = 200
        if rng and rng.startswith("bytes=") and self._path() not in self.no_range:
            try:
                start = int(rng[6:].split("-")[0])
                if 0 < start < len(data):
                    code = 206
                    data = data[start:]
                elif start >= len(data):
                    self.send_response(416)
                    self.end_headers()
                    return
            except ValueError:
                pass
        elif rng and self._path() in self.no_range:
            start = 0  # Range 非対応 → 全体を 200 で返す
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        if body:
            self.wfile.write(data)

    def do_GET(self):
        self._send(True)

    def do_HEAD(self):
        self._send(False)


@pytest.fixture()
def httpd(tmp_path):
    root = tmp_path / "www"
    root.mkdir()

    class Handler(_RangeHandler):
        pass

    Handler.root = root
    Handler.fail = set()
    Handler.no_range = set()
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    port = srv.server_address[1]
    yield srv, root, Handler
    srv.shutdown()
    srv.server_close()


def _url(port, name):
    return f"http://127.0.0.1:{port}/{name}"


def _mk(root: Path, name: str, data: bytes) -> Path:
    p = root / name
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------------- #
def test_download_basic(httpd, tmp_path):
    srv, root, _ = httpd
    port = srv.server_address[1]
    payload = b"SNIPHER-MODEL-DATA" * 100
    _mk(root, "model.bin", payload)

    acq = ModelAcquirer(cache_dir=tmp_path / "cache", min_weight_size=10)
    plan = FilePlan(filename="model.bin", repo="x/y",
                    sources=[FileSource(url=_url(port, "model.bin"), label="local")],
                    min_size=10)
    ok, paths, err = acq.ensure_plan([plan], "test")
    assert ok, err
    assert paths[0].read_bytes() == payload
    st = acq.status()
    assert st["phase"] == "complete"
    assert st["files_done"] == 1


def test_resume_from_part(httpd, tmp_path):
    srv, root, _ = httpd
    port = srv.server_address[1]
    payload = bytes(range(256)) * 400  # 102400B
    _mk(root, "w.bin", payload)

    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    half = len(payload) // 2
    (cache / "w.bin.part").write_bytes(payload[:half])  # 中断済みの part

    acq = ModelAcquirer(cache_dir=cache, min_weight_size=10)
    plan = FilePlan(filename="w.bin", repo="x/y",
                    sources=[FileSource(url=_url(port, "w.bin"), label="local")],
                    min_size=10)
    ok, paths, err = acq.ensure_plan([plan], "test")
    assert ok, err
    assert paths[0].read_bytes() == payload  # 続きから取得して完全一致


def test_fallback_to_next_source(httpd, tmp_path):
    srv, root, handler = httpd
    port = srv.server_address[1]
    payload = b"FALLBACK-OK" * 50
    _mk(root, "f.bin", payload)
    handler.fail.add("/missing.bin")

    acq = ModelAcquirer(cache_dir=tmp_path / "cache", min_weight_size=10)
    plan = FilePlan(
        filename="f.bin", repo="x/y",
        sources=[
            FileSource(url=_url(port, "missing.bin"), label="dead"),
            FileSource(url=_url(port, "f.bin"), label="alive"),
        ],
        min_size=10,
    )
    ok, paths, err = acq.ensure_plan([plan], "test")
    assert ok, err
    assert paths[0].read_bytes() == payload
    assert "alive" in "".join(acq.status()["log"])


def test_too_small_file_is_rejected(httpd, tmp_path):
    srv, root, _ = httpd
    port = srv.server_address[1]
    _mk(root, "tiny.bin", b"123")  # min_size 未満（期限切れ共有の HTML 等を弾く想定）

    acq = ModelAcquirer(cache_dir=tmp_path / "cache", min_weight_size=1000)
    plan = FilePlan(filename="tiny.bin", repo="x/y",
                    sources=[FileSource(url=_url(port, "tiny.bin"), label="local")],
                    min_size=1000)
    ok, _, err = acq.ensure_plan([plan], "test")
    assert not ok
    assert "小さすぎ" in err or "small" in err.lower() or err
    assert acq.status()["phase"] == "failed"


def test_dead_host_skipped_fast(tmp_path):
    """接続拒否のホストは死亡扱いになり、後続ファイルで再試行しない。"""
    # 何も listen していないポート
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]

    acq = ModelAcquirer(cache_dir=tmp_path / "cache", min_weight_size=1)
    mk = lambda name: FilePlan(  # noqa: E731
        filename=name, repo="x/y",
        sources=[FileSource(url=f"http://127.0.0.1:{dead_port}/{name}", label="dead-host")],
        min_size=1,
    )
    ok, _, err = acq.ensure_plan([mk("a.bin"), mk("b.bin")], "test")
    assert not ok
    assert any("127.0.0.1" in h for h in acq._dead_hosts) or err


def test_share_page_resolver(httpd, tmp_path):
    """共有ページ(ギガワタス風)から直リンクを自動解決してダウンロードする。"""
    srv, root, _ = httpd
    port = srv.server_address[1]
    payload = b"SAFETENSORS-BODY" * 100
    (root / "get").mkdir(exist_ok=True)
    _mk(root / "get", "abc123", payload)
    page = '<html><body><a class="btn" href="/get/abc123">ダウンロード</a></body></html>'
    (root / "d").mkdir(exist_ok=True)
    (root / "d" / "abc123").write_text(page, encoding="utf-8")

    acq = ModelAcquirer(cache_dir=tmp_path / "cache", min_weight_size=100)
    direct = acq.resolve_share_page(_url(port, "d/abc123"))
    assert direct == _url(port, "get/abc123")

    plan = FilePlan(
        filename="model.safetensors", repo="LiquidAI/LFM2.5-1.2B-JP-202606",
        sources=[FileSource(url=_url(port, "d/abc123"), label="共有ミラー(自動解決)")],
        min_size=100,
    )
    ok, paths, err = acq.ensure_plan([plan], "torch")
    assert ok, err
    assert paths[0].read_bytes() == payload


def test_optional_files_allowed_to_fail(httpd, tmp_path):
    srv, root, handler = httpd
    port = srv.server_address[1]
    _mk(root, "config.json", b'{"ok": true}')
    handler.fail.add("/generation_config.json")

    acq = ModelAcquirer(cache_dir=tmp_path / "cache", min_weight_size=1)
    plans = [
        FilePlan(filename="config.json", repo="x/y",
                 sources=[FileSource(url=_url(port, "config.json"), label="local")]),
        FilePlan(filename="generation_config.json", repo="x/y",
                 sources=[FileSource(url=_url(port, "generation_config.json"), label="local")],
                 optional=True),
    ]
    ok, paths, err = acq.ensure_plan(plans, "torch")
    assert ok, err
    assert [p.name for p in paths] == ["config.json"]


def test_plans_reference_official_sources(tmp_path):
    """既定プランが公式リポジトリ・共有ミラーを参照していること。"""
    from snipher.lfm.config import DEFAULT_GGUF_REPO, DEFAULT_MODEL_ID, LfmConfig

    cfg = LfmConfig()
    acq = ModelAcquirer(
        cache_dir=tmp_path, extra_urls=cfg.extra_urls(),
        share_page="https://giga-watasu.jp/d/aabdc5853ed6a0696ee4a965",
        model_id=DEFAULT_MODEL_ID, gguf_repo=DEFAULT_GGUF_REPO,
        gguf_file="LFM2.5-1.2B-JP-202606-Q4_K_M.gguf",
    )
    g = acq.plan_gguf()
    assert g[0].filename == "LFM2.5-1.2B-JP-202606-Q4_K_M.gguf"
    assert any("huggingface.co" in s.url for s in g[0].sources)
    assert any("hf-mirror.com" in s.url for s in g[0].sources)

    t = acq.plan_torch()
    names = [p.filename for p in t]
    assert "config.json" in names and "tokenizer.json" in names and "model.safetensors" in names
    w = [p for p in t if p.filename == "model.safetensors"][0]
    assert any("共有ミラー" in s.label for s in w.sources)
    assert any("huggingface.co" in s.url for s in w.sources)


def test_cached_file_not_redownloaded(httpd, tmp_path):
    srv, root, _ = httpd
    port = srv.server_address[1]
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "m.bin").write_bytes(b"CACHED" * 10)

    acq = ModelAcquirer(cache_dir=cache, min_weight_size=10)
    plan = FilePlan(filename="m.bin", repo="x/y",
                    sources=[FileSource(url=_url(port, "does-not-exist"), label="dead")],
                    min_size=10)
    ok, paths, err = acq.ensure_plan([plan], "test")
    assert ok, err  # ソースが死んでいてもキャッシュで成功
    assert paths[0].read_bytes() == b"CACHED" * 10
