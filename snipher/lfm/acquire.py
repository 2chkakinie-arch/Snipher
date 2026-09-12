"""モデルの全自動取得（レジューム対応・マルチソース）。

Snipher は起動時に LFM2.5-1.2B-JP の重みを**自分で**取りに行く。
ユーザーがファイルをアップロードしたり、手動でダウンロードする必要はない。

試行するソース（先頭ほど優先）:

1. ローカルキャッシュ / ``SNIPHER_LFM_MODEL`` / ``SNIPHER_LFM_GGUF``（既にあれば再取得しない）
2. ``SNIPHER_LFM_URLS`` で指定された直接 URL（任意数）
3. ギガワタス共有ページ（ユーザー提供の model.safetensors ミラー。
   ページ HTML から直リンクを自動解決する。期限切れなら自動スキップ）
4. HuggingFace 公式リポジトリ
   - GGUF : ``LiquidAI/LFM2.5-1.2B-JP-202606-GGUF``（llama.cpp 用・CPU 最速）
   - native: ``LiquidAI/LFM2.5-1.2B-JP-202606``（transformers 用）
5. hf-mirror.com（公式のミラー）

ダウンロードは Range リクエストによるレジューム対応で、``.part`` に書いてから
アトミックに rename する。進捗（ファイル・バイト・速度・ETA）は ``status()``
から UI / API がポーリングできる。
"""

from __future__ import annotations

import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

CHUNK = 1024 * 512  # 512KB
USER_AGENT = "snipher-acquire/1.0 (+https://github.com/2chkakinie-arch/Snipher)"
PROBE_TIMEOUT = 6
HTTP_TIMEOUT = 60

HF_TEMPLATES = [
    "https://huggingface.co/{repo}/resolve/main/{filename}?download=true",
    "https://hf-mirror.com/{repo}/resolve/main/{filename}?download=true",
]

def _host_of(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return urlparse(url).netloc or url
    except Exception:  # noqa: BLE001
        return url


PHASE_IDLE = "idle"
PHASE_CHECKING = "checking"
PHASE_DOWNLOADING = "downloading"
PHASE_COMPLETE = "complete"
PHASE_FAILED = "failed"
PHASE_WAIT_RETRY = "wait_retry"


@dataclass
class FileSource:
    """1 ファイルの取得元。url が直接 URL か、テンプレート（{repo}/{filename}）。"""

    url: str
    label: str
    filename: str | None = None  # テンプレート展開に使うファイル名を上書き

    def resolve(self, repo: str, filename: str) -> str:
        name = self.filename or filename
        if "{repo}" in self.url or "{filename}" in self.url:
            return self.url.format(repo=repo, filename=name)
        return self.url


@dataclass
class FilePlan:
    """取得対象の 1 ファイル。"""

    filename: str               # キャッシュディレクトリ内の名前
    repo: str                   # HF リポジトリ ID（テンプレート用）
    sources: list[FileSource]
    min_size: int = 1           # これ未満なら破損扱い（再取得）
    optional: bool = False      # 取得できなくても致命的でない


@dataclass
class AcquireStatus:
    phase: str = PHASE_IDLE
    backend: str = ""
    files_total: int = 0
    files_done: int = 0
    current_file: str = ""
    current_source: str = ""
    bytes_done: int = 0          # 現在のファイルの取得済みバイト
    bytes_total: int = 0         # 現在のファイルの総バイト（不明なら 0）
    total_done: int = 0          # プラン全体の取得済みバイト
    total_size: int = 0          # プラン全体の推定バイト
    speed_bps: float = 0.0
    eta_seconds: float | None = None
    error: str = ""
    last_success_path: str = ""
    log: list[str] = field(default_factory=list)
    updated: float = 0.0
    attempts: int = 0

    def to_dict(self) -> dict:
        d = {
            "phase": self.phase,
            "backend": self.backend,
            "files_total": self.files_total,
            "files_done": self.files_done,
            "current_file": self.current_file,
            "current_source": self.current_source,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "total_done": self.total_done,
            "total_size": self.total_size,
            "speed_bps": round(self.speed_bps, 1),
            "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds else None,
            "error": self.error or None,
            "last_success_path": self.last_success_path or None,
            "log": self.log[-12:],
            "attempts": self.attempts,
        }
        pct = None
        if self.bytes_total:
            pct = round(self.bytes_done / self.bytes_total * 100, 1)
        elif self.total_size:
            pct = round(self.total_done / self.total_size * 100, 1)
        d["percent"] = pct
        return d


class ModelAcquirer:
    """プランに従ってモデルファイルを全自動取得する。"""

    def __init__(self, cache_dir: Path, *, extra_urls: list[str] | None = None,
                 share_page: str = "", model_id: str = "", gguf_repo: str = "",
                 gguf_file: str = "", min_weight_size: int = 10 * 1024 * 1024):
        self.cache_dir = Path(cache_dir)
        self.extra_urls = list(extra_urls or [])
        self.share_page = share_page
        self.model_id = model_id
        self.gguf_repo = gguf_repo
        self.gguf_file = gguf_file
        self.min_weight_size = min_weight_size
        self._lock = threading.RLock()
        self._st = AcquireStatus()
        self._cancel = threading.Event()
        self._share_resolved: str | None = None
        self._share_dead = False
        self._dead_hosts: set[str] = set()

    # ------------------------------------------------------------------ #
    # 状態
    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        with self._lock:
            return self._st.to_dict()

    def cancel(self) -> None:
        self._cancel.set()

    def _set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self._st, k, v)
            self._st.updated = time.time()

    def _say(self, msg: str) -> None:
        log.info("[acquire] %s", msg)
        with self._lock:
            self._st.log.append(msg)
            self._st.log = self._st.log[-40:]
            self._st.updated = time.time()

    # ------------------------------------------------------------------ #
    # プラン構築
    # ------------------------------------------------------------------ #
    def _hf_sources(self, repo: str, filename: str | None = None) -> list[FileSource]:
        out = []
        for tmpl in HF_TEMPLATES:
            host = "huggingface.co" if "huggingface.co" in tmpl else "hf-mirror.com"
            out.append(FileSource(url=tmpl, label=host, filename=filename))
        return out

    def _weight_sources(self) -> list[FileSource]:
        """重量ファイル（model.safetensors）のソース列。"""
        srcs: list[FileSource] = []
        for u in self.extra_urls:
            srcs.append(FileSource(url=u, label="env(SNIPHER_LFM_URLS)"))
        if self.share_page:
            srcs.append(FileSource(url=self.share_page, label="共有ミラー(自動解決)", filename="model.safetensors"))
        srcs.extend(self._hf_sources(self.model_id))
        return srcs

    def plan_torch(self) -> list[FilePlan]:
        """transformers バックエンド用: config + tokenizer + model.safetensors。"""
        small = [
            ("config.json", 200),
            ("generation_config.json", 50),
            ("special_tokens_map.json", 2),
            ("tokenizer_config.json", 100),
            ("tokenizer.json", 1024),
            ("chat_template.jinja", 1),
        ]
        plans = [
            FilePlan(filename=name, repo=self.model_id,
                     sources=self._hf_sources(self.model_id), min_size=ms, optional=True)
            for name, ms in small
        ]
        # chat_template.jinja が無いモデルもあるので optional。config/tokenizer は必須扱い
        for p in plans:
            if p.filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
                p.optional = False
        plans.append(FilePlan(
            filename="model.safetensors", repo=self.model_id,
            sources=self._weight_sources(), min_size=self.min_weight_size,
        ))
        return plans

    def plan_gguf(self) -> list[FilePlan]:
        """llama.cpp バックエンド用: 単一 GGUF ファイル。"""
        return [FilePlan(
            filename=self.gguf_file, repo=self.gguf_repo,
            sources=self._hf_sources(self.gguf_repo),
            min_size=self.min_weight_size,
        )]

    # ------------------------------------------------------------------ #
    # ギガワタス共有ページの直リンク解決
    # ------------------------------------------------------------------ #
    def resolve_share_page(self, page_url: str) -> str | None:
        """共有ページ HTML から直接ダウンロード URL を推定する。"""
        if self._share_resolved:
            return self._share_resolved
        if self._share_dead:
            return None
        try:
            req = urllib.request.Request(page_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT * 3) as resp:  # noqa: S310
                html = resp.read(512 * 1024).decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            self._say(f"共有ページに接続できません({exc.__class__.__name__}) → 次のソースへ")
            self._share_dead = True
            return None

        from urllib.parse import urljoin

        candidates: list[str] = []
        # ダウンロードボタン/リンクの href・form action を収集
        for m in re.finditer(r'(?:href|action)\s*=\s*["\']([^"\']+)["\']', html, re.I):
            u = m.group(1)
            if re.search(r"(download|/f/|/get|/dl)", u, re.I):
                candidates.append(urljoin(page_url, u))
        # JS 内の URL 文字列
        for m in re.finditer(r'["\'](https?://[^"\']*(?:download|/f/|/get|/dl)[^"\']*)["\']', html, re.I):
            candidates.append(m.group(1))
        # 同一 id を含むパス
        idm = re.search(r"/d/([A-Za-z0-9]+)", page_url)
        if idm:
            fid = idm.group(1)
            for prefix in ("/f/", "/get/", "/download/", "/dl/"):
                candidates.append(urljoin(page_url, prefix + fid))

        seen = set()
        for c in candidates:
            if c in seen or c == page_url:
                continue
            seen.add(c)
            try:
                req = urllib.request.Request(c, method="HEAD", headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:  # noqa: S310
                    ctype = (resp.headers.get("Content-Type") or "").lower()
                    clen = int(resp.headers.get("Content-Length") or 0)
                    if resp.status == 200 and clen >= self.min_weight_size and "text/html" not in ctype:
                        self._share_resolved = c
                        self._say(f"共有ミラーの直リンクを解決: {c[:80]}")
                        return c
            except Exception:  # noqa: BLE001
                continue
        self._say("共有ページから直リンクを解決できませんでした → 次のソースへ")
        self._share_dead = True
        return None

    # ------------------------------------------------------------------ #
    # ダウンロード（レジューム対応）
    # ------------------------------------------------------------------ #
    def _download_one(self, plan: FilePlan, dest: Path) -> tuple[bool, str]:
        part = dest.with_suffix(dest.suffix + ".part")
        last_err = "no source"
        for src in plan.sources:
            if self._cancel.is_set():
                return False, "cancelled"
            if "共有ミラー" in src.label:
                resolved = self.resolve_share_page(src.url)
                if not resolved:
                    continue
                url = resolved
            else:
                url = src.resolve(plan.repo, plan.filename)
            self._set(current_source=src.label)
            host = _host_of(url)
            if host in self._dead_hosts:
                last_err = f"host unreachable ({host})"
                continue
            self._say(f"{plan.filename} ← {src.label}")
            for attempt in range(2):
                try:
                    self._fetch_url(url, part, plan)
                    size = part.stat().st_size
                    if size < plan.min_size:
                        part.unlink(missing_ok=True)
                        raise IOError(f"サイズが小さすぎます({size}B < {plan.min_size}B)。破損または期限切れ")
                    part.rename(dest)
                    self._set(last_success_path=str(dest))
                    return True, f"ok: {plan.filename} ({size/1e6:.1f}MB) via {src.label}"
                except urllib.error.HTTPError as e:
                    last_err = f"HTTP {e.code} ({src.label})"
                    if e.code in (401, 403, 404, 410):
                        break  # このソースは望み薄 → 次へ
                    if e.code == 416:  # Range 不一致 → part を破棄して最初から
                        part.unlink(missing_ok=True)
                        last_err = "HTTP 416 (range) → 再試行"
                except (urllib.error.URLError, OSError, TimeoutError) as e:
                    # 接続レベルの失敗 → ホストを死亡扱い(後続ファイルを高速スキップ)
                    self._dead_hosts.add(host)
                    last_err = f"{e.__class__.__name__}: {str(e)[:120]} ({src.label})"
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = f"{e.__class__.__name__}: {str(e)[:120]} ({src.label})"
                if self._cancel.is_set():
                    return False, "cancelled"
                time.sleep(1.0 * (attempt + 1))
            # ソースを替える前に part は残す（次のソースが Range 非対応なら最初から）
        return False, last_err

    def _fetch_url(self, url: str, part: Path, plan: FilePlan) -> None:
        done = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if done:
            headers["Range"] = f"bytes={done}-"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:  # noqa: S310
            status = getattr(resp, "status", 200)
            clen = resp.headers.get("Content-Length")
            if status == 206 and clen:
                total = int(clen) + done
                mode = "ab"
            else:
                # Range 未対応 → 最初から
                if done and status == 200:
                    done = 0
                    part.unlink(missing_ok=True)
                total = int(clen) if clen else 0
                mode = "wb"
            self._set(bytes_done=done, bytes_total=total)
            if total:
                with self._lock:
                    self._st.total_size = max(self._st.total_size, self._st.total_done + total)
            t0 = time.time()
            with open(part, mode) as out:
                while True:
                    if self._cancel.is_set():
                        raise IOError("cancelled")
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    el = max(time.time() - t0, 1e-6)
                    speed = (done - (self._st.bytes_done if mode == "ab" else 0)) / el if mode == "ab" else done / el
                    eta = (total - done) / speed if (total and speed > 1) else None
                    self._set(bytes_done=done, speed_bps=max(speed, 0.0), eta_seconds=eta)

    # ------------------------------------------------------------------ #
    # プラン実行
    # ------------------------------------------------------------------ #
    def ensure_plan(self, plans: list[FilePlan], backend: str) -> tuple[bool, list[Path], str]:
        """プランの全ファイルを揃える。→ (成功?, 取得済みパス一覧, エラー)。

        既にキャッシュにあればダウンロードしない（サイズ検証つき）。
        """
        self._cancel.clear()
        self._dead_hosts.clear()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._set(phase=PHASE_CHECKING, backend=backend, error="",
                  files_total=len(plans), files_done=0, total_done=0, total_size=0)
        with self._lock:
            self._st.attempts += 1

        paths: list[Path] = []
        total_done = 0
        for i, plan in enumerate(plans):
            if self._cancel.is_set():
                return False, paths, "cancelled"
            dest = self.cache_dir / plan.filename
            if dest.exists() and dest.stat().st_size >= plan.min_size:
                paths.append(dest)
                total_done += dest.stat().st_size
                self._set(files_done=i + 1, total_done=total_done)
                continue
            if dest.exists():
                dest.unlink()  # 前回の破損ファイル
            self._set(current_file=plan.filename, bytes_done=0, bytes_total=0)
            self._set(phase=PHASE_DOWNLOADING)
            ok, msg = self._download_one(plan, dest)
            if not ok:
                if plan.optional:
                    self._say(f"skip(optional): {msg}")
                    self._set(files_done=i + 1)
                    continue
                self._set(phase=PHASE_FAILED, error=msg)
                self._say(f"FAIL: {msg}")
                return False, paths, msg
            self._say(msg)
            paths.append(dest)
            total_done += dest.stat().st_size
            self._set(files_done=i + 1, total_done=total_done, bytes_done=0, bytes_total=0, speed_bps=0.0)

        self._set(phase=PHASE_COMPLETE, current_file="", error="",
                  last_success_path=str(self.cache_dir))
        return True, paths, ""


def build_acquirer(cfg) -> ModelAcquirer:
    """LfmConfig から ModelAcquirer を組み立てる。"""
    gguf_file = f"LFM2.5-1.2B-JP-202606-{cfg.gguf_quant}.gguf" if hasattr(cfg, "gguf_quant") else None
    if gguf_file is None:
        from .config import DEFAULT_GGUF_QUANT

        gguf_file = f"LFM2.5-1.2B-JP-202606-{DEFAULT_GGUF_QUANT}.gguf"
    return ModelAcquirer(
        cache_dir=Path(cfg.cache_dir),
        extra_urls=cfg.extra_urls(),
        share_page=GIGA_WATASU_SHARE_URL if _share_applies(cfg) else "",
        model_id=cfg.model_id if hasattr(cfg, "model_id") else _model_id(cfg),
        gguf_repo=cfg.gguf_repo if hasattr(cfg, "gguf_repo") else _gguf_repo(),
        gguf_file=gguf_file,
    )


def _share_applies(cfg) -> bool:
    """共有ミラーは公式モデル ID を使うときだけ意味がある。"""
    from .config import DEFAULT_MODEL_ID

    return _model_id(cfg) == DEFAULT_MODEL_ID


def _model_id(cfg) -> str:
    src = cfg.model_source
    if src and not Path(src).exists():
        return src
    from .config import DEFAULT_MODEL_ID

    return DEFAULT_MODEL_ID


def _gguf_repo() -> str:
    from .config import DEFAULT_GGUF_REPO

    return DEFAULT_GGUF_REPO
