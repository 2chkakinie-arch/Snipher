"""GGUF バックエンド: llama.cpp で LFM2.5-1.2B-JP を動かす（CPU 最速経路）。

2 つの実行方式を自動選択する:

1. ``llama-cpp-python``（pip で入る Python バインディング）
2. ``llama-server`` バイナリ（PATH / var/bin / SNIPHER_LLAMA_SERVER を自動探索し、
   OpenAI 互換 API をローカルで起動して使う）

イベント形式は torch バックエンド（engine.LfmEngine.stream_chat）と同一:
    {"type": "start"|"delta"|"done"|"error", ...}

GGUF のトークナイザは byte-fallback BPE なので「未知文字」は原理的に発生しない
（どんな文字・絵文字でもそのまま読み書きできる）。予約トークン方式の未知文字
学習は torch バックエンド専用の機能。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from .config import DEFAULT_SYSTEM_PROMPT, LfmConfig
from .template import MODE_BUILTIN, MODE_NATIVE, MODE_RAW, render_builtin_chatml

log = logging.getLogger(__name__)

STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_FAILED = "failed"

LLAMA_STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]


def find_llama_server(explicit: str = "") -> str | None:
    """llama-server バイナリを探す（明示指定 → PATH → var/bin）。"""
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return str(p)
        return shutil.which(explicit)
    found = shutil.which("llama-server")
    if found:
        return found
    repo_bin = Path(__file__).resolve().parents[2] / "var" / "bin"
    for name in ("llama-server", "llama-server.exe", "server", "llama.cpp-server"):
        cand = repo_bin / name
        if cand.exists():
            return str(cand)
    return None


def runtime_kind(cfg: LfmConfig) -> str | None:
    """利用可能な llama.cpp ランタイム: 'llama-cpp-python' | 'llama-server' | None。"""
    try:
        import llama_cpp  # noqa: F401

        return "llama-cpp-python"
    except Exception:  # noqa: BLE001
        pass
    if find_llama_server(cfg.llama_server_bin):
        return "llama-server"
    return None


class GgufBackend:
    """llama.cpp による LFM2.5-1.2B-JP 推論。LfmEngine と同じ顔を持つ。"""

    kind = "gguf"

    def __init__(self, cfg: LfmConfig):
        self.cfg = cfg
        self.state = STATE_IDLE
        self.error: str | None = None
        self.model_path: str | None = None
        self.model_name: str = ""
        self.quant_name: str = ""
        self.runtime: str | None = None
        self.load_seconds: float | None = None
        self._llm = None                 # llama-cpp-python
        self._server: dict | None = None  # llama-server サブプロセス
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # ライフサイクル
    # ------------------------------------------------------------------ #
    def load(self, gguf_path: str | Path) -> None:
        gguf_path = Path(gguf_path)
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF が見つかりません: {gguf_path}")
        with self._lock:
            self.state = STATE_LOADING
            self.error = None
            t0 = time.time()
            self.model_path = str(gguf_path)
            self.quant_name = _quant_from_name(gguf_path.name)
            self.runtime = runtime_kind(self.cfg)
            if self.runtime == "llama-cpp-python":
                self._load_python_binding(gguf_path)
            elif self.runtime == "llama-server":
                self._load_server(gguf_path)
            else:
                self.state = STATE_FAILED
                self.error = (
                    "llama.cpp ランタイムがありません。pip install llama-cpp-python "
                    "または llama-server バイナリを PATH/var/bin に置いてください。"
                )
                raise RuntimeError(self.error)
            self.load_seconds = round(time.time() - t0, 2)
            self.state = STATE_READY
            log.info("GGUF バックエンド準備完了 (%.1fs, %s)", self.load_seconds, self.runtime)

    def _n_threads(self) -> int:
        if self.cfg.n_threads > 0:
            return self.cfg.n_threads
        return max(1, min(4, (os.cpu_count() or 2)))

    def _load_python_binding(self, gguf_path: Path) -> None:
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=str(gguf_path),
            n_ctx=self.cfg.prompt_budget + self.cfg.max_new_tokens + 64,
            n_threads=self._n_threads(),
            n_batch=min(512, self.cfg.prompt_budget),
            verbose=False,
        )
        try:
            md = getattr(self._llm, "metadata", None) or {}
            self.model_name = md.get("general.name") or self.model_name
        except Exception:  # noqa: BLE001
            pass

    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _load_server(self, gguf_path: Path) -> None:
        binpath = find_llama_server(self.cfg.llama_server_bin)
        port = self._free_port()
        cmd = [
            binpath, "-m", str(gguf_path),
            "--host", "127.0.0.1", "--port", str(port),
            "-c", str(self.cfg.prompt_budget + self.cfg.max_new_tokens + 64),
            "-t", str(self._n_threads()),
            "--no-webui",
        ]
        log.info("llama-server 起動: %s", " ".join(cmd))
        proc = subprocess.Popen(  # noqa: S603
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 180
        while time.time() < deadline:
            if proc.poll() is not None:
                err = (proc.stderr.read() or "")[-2000:] if proc.stderr else ""
                raise RuntimeError(f"llama-server が終了しました: {err}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:  # noqa: S310
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        else:
            proc.terminate()
            raise RuntimeError("llama-server の起動がタイムアウトしました")
        self._server = {"proc": proc, "base": base}

    def unload(self) -> None:
        with self._lock:
            self._llm = None
            if self._server:
                try:
                    self._server["proc"].terminate()
                    self._server["proc"].wait(timeout=5)
                except Exception:  # noqa: BLE001
                    try:
                        self._server["proc"].kill()
                    except Exception:  # noqa: BLE001
                        pass
                self._server = None
            self.state = STATE_IDLE

    @property
    def is_ready(self) -> bool:
        return self.state == STATE_READY

    # ------------------------------------------------------------------ #
    # 状態（LfmEngine.status() と互換のキーを含む）
    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        try:
            import llama_cpp

            versions = {"llama-cpp-python": llama_cpp.__version__}
        except Exception:  # noqa: BLE001
            versions = {"runtime": self.runtime or None}
        return {
            "state": self.state,
            "error": self.error,
            "backend": "gguf",
            "runtime": self.runtime,
            "model_source": self.model_path,
            "is_local": bool(self.model_path and Path(self.model_path).exists()),
            "quantized": True,
            "quant": self.quant_name,
            "base_vocab": None,
            "reserved_tokens": 0,
            "learned_chars": [],
            "template": {"has_native": True, "default_mode": MODE_NATIVE,
                         "builtin": "GGUF 埋め込みテンプレート / 内蔵 ChatML"},
            "versions": versions,
            "autostart": self.cfg.autostart,
        }

    def engine_name(self) -> str:
        q = f" {self.quant_name}" if self.quant_name else ""
        rt = "llama.cpp" if self.runtime else "GGUF"
        name = self.model_name or "LFM2.5-1.2B-JP"
        return f"{name} (GGUF{q} · {rt})"

    def scan_unknown(self, text: str) -> dict | None:
        """GGUF は byte-fallback BPE のため未知文字は発生しない。"""
        if not self.is_ready:
            return None
        return {
            "counts": {"total": len(text), "unknown": 0, "ok": len(text)},
            "unknown": [],
            "note": "byte-fallback トークナイザ: すべての文字をそのまま扱えます（学習不要）",
        }

    # ------------------------------------------------------------------ #
    # 生成（イベント形式は LfmEngine.stream_chat と同一）
    # ------------------------------------------------------------------ #
    def stream_chat(self, messages: list[dict], *, max_new_tokens: int | None = None,
                    min_new_tokens: int | None = None,
                    temperature: float | None = None, top_k: int | None = None,
                    repetition_penalty: float | None = None,
                    use_template: bool = True, system_prompt: str | None = None):
        if not self.is_ready:
            raise RuntimeError("GGUF バックエンドが準備できていません")
        if not self._lock.acquire(timeout=0.2):
            raise RuntimeError("別の生成が進行中です")
        try:
            msgs = _normalize_messages(messages, use_template, system_prompt)
            max_new = int(max_new_tokens or self.cfg.max_new_tokens)
            temp = float(temperature if temperature is not None else self.cfg.temperature)
            top_k = int(top_k or self.cfg.top_k)
            rep = float(repetition_penalty or self.cfg.repetition_penalty)

            if self._llm is not None:
                yield from self._stream_python_binding(msgs, max_new, temp, top_k, rep, use_template)
            else:
                yield from self._stream_server(msgs, max_new, temp, top_k, rep, use_template)
        finally:
            self._lock.release()

    # ---- llama-cpp-python ---- #
    def _stream_python_binding(self, msgs, max_new, temp, top_k, rep, use_template):
        t0 = time.time()
        mode = MODE_NATIVE if use_template else MODE_RAW
        n_pieces = 0
        text_out: list[str] = []
        t_first: float | None = None
        try:
            if use_template and _has_chat_template(self._llm):
                # GGUF 埋め込みの chat template を自動適用
                def _make_stream():
                    return self._llm.create_chat_completion(
                        messages=msgs, stream=True, max_tokens=max_new,
                        temperature=max(temp, 1e-4), top_k=top_k, repeat_penalty=rep,
                    )
                pieces = _iter_chat_stream(_make_stream())
            else:
                # 内蔵 ChatML / テンプレートなし（raw）
                if use_template:
                    prompt = render_builtin_chatml(msgs, bos="<|startoftext|>")
                    prompt += "<|im_start|>assistant\n"
                    mode = MODE_BUILTIN
                else:
                    prompt = "\n".join(str(m.get("content", "")) for m in msgs if m.get("content"))
                def _make_stream():
                    return self._llm.create_completion(
                        prompt, stream=True, max_tokens=max_new,
                        temperature=max(temp, 1e-4), top_k=top_k, repeat_penalty=rep,
                        stop=LLAMA_STOP_TOKENS if use_template else None,
                    )
                pieces = _iter_completion_stream(_make_stream())

            yield {"type": "start", "engine": self.engine_name(), "template_mode": mode,
                   "prompt_tokens": None, "max_new_tokens": max_new}
            for piece in pieces:
                if not piece:
                    continue
                if t_first is None:
                    t_first = time.time()
                n_pieces += 1
                text_out.append(piece)
                yield {"type": "delta", "text": piece}
        except Exception as exc:  # noqa: BLE001
            log.exception("GGUF 生成エラー")
            yield {"type": "error", "message": f"生成エラー: {exc}"}
            return

        t_end = time.time()
        decode_s = (t_end - t_first) if t_first else None
        tps = (n_pieces / decode_s) if (decode_s and n_pieces) else None
        yield {
            "type": "done",
            "text": "".join(text_out),
            "stats": {
                "new_tokens": n_pieces,
                "prefill_seconds": round(t_first - t0, 3) if t_first else None,
                "decode_seconds": round(decode_s, 3) if decode_s else None,
                "tokens_per_second": round(tps, 2) if tps else None,
                "template_mode": mode,
                "engine": self.engine_name(),
            },
        }

    # ---- llama-server サブプロセス ---- #
    def _stream_server(self, msgs, max_new, temp, top_k, rep, use_template):
        base = self._server["base"]
        t0 = time.time()
        mode = MODE_NATIVE if use_template else MODE_RAW
        body: dict = {
            "max_tokens": max_new,
            "temperature": max(temp, 1e-4),
            "top_k": top_k,
            "repeat_penalty": rep,
            "stream": True,
            "stop": LLAMA_STOP_TOKENS if use_template else None,
        }
        if use_template:
            body["messages"] = msgs
            url = base + "/v1/chat/completions"
        else:
            body["prompt"] = "\n".join(str(m.get("content", "")) for m in msgs if m.get("content"))
            body.pop("stop")
            url = base + "/v1/completions"

        n_pieces = 0
        text_out: list[str] = []
        t_first: float | None = None
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            yield {"type": "start", "engine": self.engine_name(), "template_mode": mode,
                   "prompt_tokens": None, "max_new_tokens": max_new}
            with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        ev = json.loads(payload)
                    except Exception:  # noqa: BLE001
                        continue
                    piece = ""
                    if "choices" in ev and ev["choices"]:
                        ch = ev["choices"][0]
                        piece = (ch.get("delta") or {}).get("content") or ch.get("text") or ""
                    if piece:
                        if t_first is None:
                            t_first = time.time()
                        n_pieces += 1
                        text_out.append(piece)
                        yield {"type": "delta", "text": piece}
        except Exception as exc:  # noqa: BLE001
            log.exception("llama-server 生成エラー")
            yield {"type": "error", "message": f"生成エラー: {exc}"}
            return

        t_end = time.time()
        decode_s = (t_end - t_first) if t_first else None
        tps = (n_pieces / decode_s) if (decode_s and n_pieces) else None
        yield {
            "type": "done",
            "text": "".join(text_out),
            "stats": {
                "new_tokens": n_pieces,
                "prefill_seconds": round(t_first - t0, 3) if t_first else None,
                "decode_seconds": round(decode_s, 3) if decode_s else None,
                "tokens_per_second": round(tps, 2) if tps else None,
                "template_mode": mode,
                "engine": self.engine_name(),
            },
        }


# ---------------------------------------------------------------------- #
# ヘルパー
# ---------------------------------------------------------------------- #
def _normalize_messages(messages: list[dict], use_template: bool, system_prompt: str | None) -> list[dict]:
    msgs = [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in messages]
    if use_template and system_prompt:
        msgs = [{"role": "system", "content": system_prompt}] + [m for m in msgs if m["role"] != "system"]
    elif use_template and not any(m["role"] == "system" for m in msgs):
        msgs = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + msgs
    return msgs


def _has_chat_template(llm) -> bool:
    try:
        md = getattr(llm, "metadata", None) or {}
        return bool(md.get("tokenizer.chat_template"))
    except Exception:  # noqa: BLE001
        return False


def _quant_from_name(name: str) -> str:
    import re

    m = re.search(r"(Q\d+_[A-Z0-9_]+|F16|BF16)", name, re.I)
    return m.group(1).upper() if m else ""


def _iter_chat_stream(stream):
    for chunk in stream:
        try:
            delta = chunk["choices"][0].get("delta") or {}
            yield delta.get("content") or ""
        except Exception:  # noqa: BLE001
            continue


def _iter_completion_stream(stream):
    for chunk in stream:
        try:
            yield chunk["choices"][0].get("text") or ""
        except Exception:  # noqa: BLE001
            continue
